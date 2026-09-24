"""Load one batch of processed calls for the management report - and decide, per
call, what the report may say about it.

Three questions are answered here and nowhere else, so that no chart, finding
or table can answer them differently:

* Which calls are in the batch, and in what state. The status envelope in
  results/ is the truth; a scorecard on disk says nothing about whether the
  call's latest run succeeded.
* Which numbers count. Only the success-only, single-rubric cohort that
  aggregation.load_scorecards returns - the same cohort the banker reports and
  the index are built from - so the headline index here and there agree.
* Which calls may contribute TEXT. Only a call whose redacted transcript
  parses, is its own, and says redaction was on. Missing, unreadable or
  disabled all mean "numbers only": the check fails closed, unlike the
  dashboard's `enabled is False`, because this file is made to be forwarded.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path

from pydantic import ValidationError

from callqa.aggregation import load_scorecards
from callqa.ingestion import CALL_ID_RE, is_windows_reserved
from callqa.models import CallMeta, CallResult, Features, RedactedTranscript, ScoreCard
from callqa.rubric import Rubric

logger = logging.getLogger(__name__)

# Judge engines whose scores are not an assessment of anyone.
DEMO_ENGINES = frozenset({"mock", "synthetic-demo"})

# The call types the bank's metadata uses, in the report's language. Anything
# else is shown as written (escaped, re-redacted, capped), and grouped under
# "אחר" in the segment analysis when it is rare.
CALL_TYPE_HE = {
    "service": "שירות",
    "loans": "הלוואות",
    "loan": "הלוואות",
    "cards": "כרטיסי אשראי",
    "card": "כרטיסי אשראי",
    "credit_cards": "כרטיסי אשראי",
    "mortgage": "משכנתאות",
    "mortgages": "משכנתאות",
    "investments": "השקעות",
    "investment": "השקעות",
    "sales": "מכירות",
    "retention": "שימור לקוחות",
    "collections": "גבייה",
    "complaints": "תלונות",
    "business": "עסקים",
}
UNSPECIFIED_TYPE = "לא צוין"
MAX_TYPE_LEN = 40

STAGE_HE = {
    "ingestion": "קליטה",
    "audio": "עיבוד שמע",
    "asr": "תמלול",
    "speakers": "זיהוי דוברים",
    "redaction": "הסתרת פרטים",
    "features": "מדדים אובייקטיביים",
    "judge": "שיפוט",
    "report": "הפקת דוח",
}

# Why a call was held for a human, recognised from the fixed English texts the
# pipeline writes (pipeline.py review_reasons, judge/runner.py). The error text
# itself is never shown: it can quote the model, or a recording's file name.
HELD_REASONS = (
    ("judge output failed after", "תשובת מודל השיפוט לא עברה אימות"),
    ("recording is only", "הקלטה קצרה מדי"),
    ("of speech was detected", "מעט מדי דיבור בהקלטה"),
    ("speaker roles inferred with low confidence", "זיהוי הדוברים בוודאות נמוכה"),
    ("redaction was DISABLED", "הסתרת הפרטים כובתה"),
)
HELD_OTHER = "סיבה אחרת"


@dataclass
class CallRecord:
    """Everything the report knows about one call, already gated."""

    call_id: str
    status: str                      # success | needs_human_review | failed
    banker_id: str
    call_date: date | None
    date_source: str                 # "call" (metadata) | "processed" (judge time) | "none"
    call_type_key: str | None        # as written in the metadata (never displayed raw)
    call_type: str                   # display label (Hebrew, safe)
    duration_sec: float | None
    mono: bool | None                # single-channel recording (roles inferred)
    card: ScoreCard | None           # set only for the scored cohort
    features: Features | None
    text_allowed: bool               # redaction proven on: text may be shown
    redaction_disabled: bool         # redaction explicitly off (loud in the report)
    redaction_counts: dict[str, int] = field(default_factory=dict)
    held_reasons: list[str] = field(default_factory=list)
    failed_reason: str | None = None
    call_report: str | None = None   # relative link from reports/, if the file exists
    banker_report: str | None = None

    @property
    def scored(self) -> bool:
        return self.card is not None


@dataclass
class BatchFilters:
    date_from: date | None = None
    date_to: date | None = None
    call_type: str | None = None
    banker_id: str | None = None
    run_id: str | None = None

    @property
    def active(self) -> bool:
        return any(v is not None for v in (self.date_from, self.date_to, self.call_type,
                                           self.banker_id, self.run_id))


@dataclass
class Batch:
    output_dir: Path
    rubric: Rubric
    filters: BatchFilters
    records: list[CallRecord]
    excluded_other_rubric: int
    calibration: dict | None
    judge_engines: list[str]
    models: list[str]
    prompt_versions: list[str]
    rubric_hashes: list[str]
    scored_at: tuple[str, str] | None     # first/last judge timestamp (ISO)

    @property
    def scored(self) -> list[CallRecord]:
        return [r for r in self.records if r.card is not None]

    @property
    def held(self) -> list[CallRecord]:
        return [r for r in self.records if r.status == "needs_human_review"]

    @property
    def failed(self) -> list[CallRecord]:
        return [r for r in self.records if r.status == "failed"]

    @property
    def is_demo(self) -> bool:
        return any(e in DEMO_ENGINES for e in self.judge_engines)


# -- small tolerant readers ---------------------------------------------------

def _read_json(path: Path) -> object | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _read_model(path: Path, model: type):  # noqa: ANN202 - returns an instance of `model`
    try:
        return model.model_validate_json(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, ValidationError):
        return None


def safe_call_id(value: object) -> str | None:
    """A call id read from JSON, fit for an href, an element id or a file name."""
    if isinstance(value, str) and CALL_ID_RE.match(value) and not is_windows_reserved(value):
        return value
    return None


_DATE_PATTERNS = (
    (re.compile(r"^(\d{4})-(\d{1,2})-(\d{1,2})"), ("y", "m", "d")),
    (re.compile(r"^(\d{4})/(\d{1,2})/(\d{1,2})"), ("y", "m", "d")),
    # Israeli order: day first.
    (re.compile(r"^(\d{1,2})[/.](\d{1,2})[/.](\d{4})"), ("d", "m", "y")),
)


def parse_call_date(value: object) -> date | None:
    """call_date is free text in the bank's CSV: accept ISO and the Israeli
    day-first forms, and nothing that would have to be guessed."""
    if not isinstance(value, str):
        return None
    text = value.strip()
    for pattern, order in _DATE_PATTERNS:
        m = pattern.match(text)
        if m:
            parts = dict(zip(order, (int(g) for g in m.groups()), strict=True))
            try:
                return date(parts["y"], parts["m"], parts["d"])
            except ValueError:
                return None
    return None


def _timestamp_date(value: str | None) -> date | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).date()
    except ValueError:
        return parse_call_date(value)


def call_type_label(raw: str | None) -> str:
    """The display label for a call type: a known id in Hebrew, else the value
    itself, re-redacted and capped - it is free text from a CSV."""
    if raw is None or not str(raw).strip():
        return UNSPECIFIED_TYPE
    key = str(raw).strip()
    known = CALL_TYPE_HE.get(key.casefold())
    if known:
        return known
    from callqa.redaction import redact_text

    cleaned = re.sub(r"\s+", " ", redact_text(key)[0]).strip()
    cleaned = re.sub(r"<[^>]{0,200}>", "", cleaned).strip() or UNSPECIFIED_TYPE
    return cleaned[:MAX_TYPE_LEN]


def held_reasons(error: str | None) -> list[str]:
    reasons = [he for marker, he in HELD_REASONS if error and marker in error]
    return reasons or [HELD_OTHER]


def failed_reason(result: CallResult) -> str:
    error = result.error or ""
    if error.startswith("locked:"):
        return "השיחה הייתה נעולה בידי תהליך אחר"
    if error.startswith("unsafe call_id refused"):
        return "מזהה שיחה לא תקין"
    from callqa.pipeline import STAGES

    done = len(result.stages_completed)
    if done < len(STAGES):
        return f"כשל בשלב {STAGE_HE.get(STAGES[done], STAGES[done])}"
    return "כשל לאחר השלמת השלבים"


# -- loading -------------------------------------------------------------------

def _results(output_dir: Path) -> dict[str, CallResult]:
    out: dict[str, CallResult] = {}
    results_dir = output_dir / "results"
    if not results_dir.is_dir():
        return out
    for path in sorted(results_dir.glob("*.json")):
        result = _read_model(path, CallResult)
        if result is None:
            continue
        call_id = safe_call_id(result.call_id)
        if call_id and path.stem == call_id:
            out[call_id] = result
    return out


def _run_members(output_dir: Path, run_id: str) -> set[str]:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,80}", run_id):
        raise FileNotFoundError(f"not a run id: {run_id!r}")
    data = _read_json(output_dir / "runs" / f"{run_id}.json")
    if not isinstance(data, dict):
        raise FileNotFoundError(f"no run record {run_id} under {output_dir / 'runs'}")
    return {c["call_id"] for c in data.get("calls", [])
            if isinstance(c, dict) and safe_call_id(c.get("call_id"))}


def _text_gate(output_dir: Path, call_id: str) -> tuple[bool, bool, dict[str, int]]:
    """(text allowed, redaction explicitly disabled, redaction counts)."""
    redacted = _read_model(output_dir / "redacted" / f"{call_id}.json", RedactedTranscript)
    if redacted is None or redacted.call_id != call_id:
        return False, False, {}
    counts = {k: int(v) for k, v in redacted.redaction_counts.items()
              if isinstance(v, int) and v >= 0}
    return redacted.enabled is True, redacted.enabled is False, counts


def banker_report_links(output_dir: Path, cohort: list[ScoreCard]) -> dict[str, str]:
    """banker_id -> 'bankers/<slug>.html' for the banker pages that exist.

    `cohort` must be what load_scorecards(output_dir) returned - the set
    `callqa report` wrote the banker pages for. A collision suffix depends on
    the whole set, and slugs computed over a subset (one month, one call type)
    could point a link at the wrong person's page.
    """
    from callqa.reporting.banker_report import _banker_slugs

    slugs = _banker_slugs([c.banker_id for c in cohort])
    links: dict[str, str] = {}
    for banker_id, slug in slugs.items():
        if (output_dir / "reports" / "bankers" / f"{slug}.html").is_file():
            links[banker_id] = f"bankers/{slug}.html"
    return links


def load_batch(output_dir: Path, rubric: Rubric, filters: BatchFilters | None = None) -> Batch:
    filters = filters or BatchFilters()
    results = _results(output_dir)

    # The published cohort: success-only and one rubric/prompt version, exactly
    # as the banker reports compute it. Cards dropped for another rubric are
    # counted so the report can say so.
    all_success = load_scorecards(output_dir) if (output_dir / "scores").is_dir() else []
    cards = {c.call_id: c for c in all_success if safe_call_id(c.call_id)}
    excluded_other_rubric = _count_left_out(output_dir, results, len(all_success))

    members: set[str] | None = None
    if filters.run_id:
        members = _run_members(output_dir, filters.run_id)

    banker_links = banker_report_links(output_dir, all_success) if cards else {}
    reports_dir = output_dir / "reports"

    records: list[CallRecord] = []
    for call_id in sorted(set(results) | set(cards)):
        if members is not None and call_id not in members:
            continue
        result = results.get(call_id)
        card = cards.get(call_id)
        status = result.status if result is not None else "success"
        if status != "success":
            card = None          # a stale scorecard is not this call's result
        meta = _read_model(output_dir / "ingestion" / f"{call_id}.json", CallMeta)
        features = _read_model(output_dir / "features" / f"{call_id}.json", Features)
        if features is not None and features.call_id != call_id:
            features = None
        text_ok, disabled, counts = _text_gate(output_dir, call_id)

        banker_id = (card.banker_id if card else None) or (meta.banker_id if meta else None)
        banker_id = str(banker_id or "unknown")
        call_date = parse_call_date(meta.call_date) if meta else None
        date_source = "call" if call_date else "none"
        if call_date is None and card is not None:
            call_date = _timestamp_date(card.timestamp)
            date_source = "processed" if call_date else "none"
        type_key = (meta.call_type.strip() if meta and meta.call_type else None) or None
        duration = meta.duration_sec if meta else None
        if duration is None and features is not None:
            duration = features.call_duration_sec
        mono = None
        if features is not None:
            mono = not features.overlap_metrics_available
        elif meta is not None:
            mono = meta.channels < 2

        record = CallRecord(
            call_id=call_id, status=status, banker_id=banker_id,
            call_date=call_date, date_source=date_source,
            call_type_key=type_key, call_type=call_type_label(type_key),
            duration_sec=duration, mono=mono, card=card, features=features,
            text_allowed=text_ok, redaction_disabled=disabled, redaction_counts=counts,
        )
        if status == "needs_human_review":
            record.held_reasons = held_reasons(result.error if result else None)
        elif status == "failed" and result is not None:
            record.failed_reason = failed_reason(result)
        if (card is not None and text_ok and result is not None and result.report_path
                and (reports_dir / "calls" / f"{call_id}.html").is_file()):
            record.call_report = f"calls/{call_id}.html"
        record.banker_report = banker_links.get(banker_id)
        if _matches(record, filters):
            records.append(record)

    scored_cards = [r.card for r in records if r.card is not None]
    engines = sorted({c.judge_engine for c in scored_cards})
    models = sorted({Path(c.model).name or c.model for c in scored_cards})
    stamps = sorted(c.timestamp for c in scored_cards if c.timestamp)
    logger.info("executive batch: %d call(s) in scope, %d scored", len(records),
                len(scored_cards))
    return Batch(
        output_dir=output_dir, rubric=rubric, filters=filters, records=records,
        excluded_other_rubric=excluded_other_rubric,
        calibration=_calibration(output_dir),
        judge_engines=engines, models=models,
        prompt_versions=sorted({c.prompt_version for c in scored_cards}),
        rubric_hashes=sorted({c.rubric_sha256 for c in scored_cards if c.rubric_sha256}),
        scored_at=(stamps[0], stamps[-1]) if stamps else None,
    )


def _count_left_out(output_dir: Path, results: dict[str, CallResult], kept: int) -> int:
    """Success scorecards on disk that the cohort left out (another rubric or
    prompt version, or unreadable) - counted, not re-parsed."""
    on_disk = 0
    for path in (output_dir / "scores").glob("*.json"):
        result = results.get(path.stem)
        if result is None or result.status == "success":
            on_disk += 1
    return max(0, on_disk - kept)


def _calibration(output_dir: Path) -> dict | None:
    data = _read_json(output_dir / "reports" / "calibration.json")
    if not isinstance(data, dict):
        return None
    keep = ("n_calls", "overall_qwk", "overall_pass", "qwk_pass_threshold",
            "min_calls_for_pass", "flagged_dimensions")
    return {k: data.get(k) for k in keep}


def _matches(record: CallRecord, f: BatchFilters) -> bool:
    if f.date_from and (record.call_date is None or record.call_date < f.date_from):
        return False
    if f.date_to and (record.call_date is None or record.call_date > f.date_to):
        return False
    if f.call_type is not None:
        wanted = f.call_type.strip().casefold()
        if (record.call_type_key or "").casefold() != wanted and \
                record.call_type.casefold() != wanted:
            return False
    return not (f.banker_id is not None and record.banker_id != f.banker_id)
