"""The whole-system evaluation: run the pipeline over a golden set and report
every metric that matters, with a before/after comparison that can gate a change.
"""

from __future__ import annotations

import json
import logging
import statistics
import subprocess
import sys
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel, Field

from callqa.config import Config
from callqa.eval.metrics import cer, redaction_prf, role_accuracy, wer
from callqa.models import DialogTranscript, ScoreCard
from callqa.ops.provenance import fingerprint

logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_GOLDEN = _REPO_ROOT / "eval" / "golden"

# Regression tolerances for the gate. Rates are on 0..1; judge QWK is noisier.
_TOL = {"rate": 0.02, "qwk": 0.05}


class CallMetrics(BaseModel):
    call_id: str
    wer: float = 0.0
    cer: float = 0.0
    role_accuracy: float = 0.0
    redaction: dict = Field(default_factory=dict)


# What each number in a report actually means, carried INSIDE the report.
#
# On the synthetic golden set the references are the mock pipeline's own
# output, and the mock ASR emits the reference text verbatim. So WER, CER,
# role accuracy and QWK are 0.0 / 1.0 / 1.0 by construction and will stay
# there however good or bad the real system becomes. They are regression
# sentinels - they detect a CHANGE from known-good - and they are not accuracy
# measurements. Only a real, human-labelled golden set can produce those.
#
# Redaction recall and precision are the exception, and the reason the
# distinction is worth writing down rather than leaving in a code comment: the
# gold identifiers are hand-labelled independently of the detector, so those
# two are genuine measurements even here.
#
# This ships in every report because a bare "wer_mean: 0.0" in a JSON file is
# read by a bank as "this system transcribes perfectly". It does not.
SYNTHETIC_SET_NOTE = (
    "On the synthetic golden set, wer/cer/role_accuracy/judge_qwk are REGRESSION "
    "SENTINELS, not accuracy: the references are the mock pipeline's own output, "
    "so these are 0.0/1.0 by construction and only move when behaviour changes. "
    "redaction_recall and redaction_precision ARE real measurements - the gold "
    "identifiers are hand-labelled independently of the detector. Point "
    "`callqa eval --golden-dir` at a human-labelled set of real calls to measure "
    "accuracy."
)


class EvalReport(BaseModel):
    created_at: str
    n_calls: int
    seconds_total: float
    wer_mean: float
    cer_mean: float
    role_accuracy_mean: float
    redaction_recall_mean: float
    redaction_precision_mean: float
    redaction_f1_mean: float
    judge_qwk: float | None = None
    calls: list[CallMetrics] = Field(default_factory=list)
    fingerprint: dict = Field(default_factory=dict)
    # "synthetic" unless a --golden-dir of real, labelled calls was supplied.
    golden_set: str = "synthetic"
    what_these_numbers_mean: str = SYNTHETIC_SET_NOTE


def _load_references(golden_dir: Path) -> dict:
    return json.loads((golden_dir / "references.json").read_text(encoding="utf-8"))


def _generate_synthetic_inputs(dest: Path) -> None:
    subprocess.run(
        [sys.executable, str(_REPO_ROOT / "scripts" / "generate_sample_data.py"),
         "--input-dir", str(dest)],
        check=True, capture_output=True)


def evaluate(config: Config, golden_dir: Path | None = None) -> EvalReport:
    """Run the pipeline over the golden set and score it against the references."""
    from callqa.engines import build_engines
    from callqa.pipeline import process_call
    from callqa.rubric import load_rubric
    from callqa.state import StateDB

    golden_dir = golden_dir or DEFAULT_GOLDEN
    references = _load_references(golden_dir)
    rubric = load_rubric()

    work = Path(tempfile.mkdtemp(prefix="callqa-eval-"))
    is_real_golden_set = (golden_dir / "calls").exists()
    input_dir = golden_dir if is_real_golden_set else work / "input"
    if not is_real_golden_set:
        _generate_synthetic_inputs(input_dir)
    out_dir = work / "output"

    # Evaluate against the pipeline the config selects (mock in CI, real on a
    # bank's golden set), pointed at a throwaway output dir.
    eval_config = config.model_copy(deep=True)
    eval_config.paths.input_dir = input_dir
    eval_config.paths.output_dir = out_dir
    eval_config.paths.state_db = work / "state.db"
    engines = build_engines(eval_config)
    state = StateDB(eval_config.paths.state_db)

    from callqa.ingestion import load_metadata

    meta = load_metadata(input_dir / "metadata.csv", input_dir / "calls")
    from callqa.cli import _call_input  # thin helper; not the pipeline core

    started = time.time()
    scorecards: list[ScoreCard] = []
    gold_ratings: dict[str, list[dict]] = {}
    call_metrics: list[CallMetrics] = []

    for cid, ref in sorted(references.items()):
        row = meta.rows.get(cid)
        if row is None:
            logger.warning("golden call %s not in the generated input; skipping", cid)
            continue
        call = _call_input(input_dir / "calls" / row["file_name"], eval_config)
        call = call.model_copy(update={"call_id": cid})
        process_call(call, engines, state)

        dialog = DialogTranscript.model_validate_json(
            (out_dir / "transcripts" / f"{cid}.dialog.json").read_text(encoding="utf-8"))
        produced_text = " ".join(t.text for t in dialog.turns)
        produced_turns = [{"speaker": t.speaker} for t in dialog.turns]

        card_path = out_dir / "scores" / f"{cid}.json"
        if card_path.exists():
            card = ScoreCard.model_validate_json(card_path.read_text(encoding="utf-8"))
            scorecards.append(card)
            if ref.get("scores"):
                gold_ratings[cid] = [ref["scores"]]

        call_metrics.append(CallMetrics(
            call_id=cid,
            wer=wer(ref["transcript"], produced_text),
            cer=cer(ref["transcript"], produced_text),
            role_accuracy=role_accuracy(ref.get("speakers", []), produced_turns),
            redaction=redaction_prf(ref["transcript"], ref.get("pii", [])),
        ))

    seconds_total = round(time.time() - started, 3)
    judge_qwk = _judge_qwk(scorecards, gold_ratings, rubric)

    def mean(values: list[float]) -> float:
        return round(statistics.mean(values), 4) if values else 0.0

    return EvalReport(
        created_at=datetime.now(UTC).isoformat(timespec="seconds"),
        n_calls=len(call_metrics),
        seconds_total=seconds_total,
        wer_mean=mean([c.wer for c in call_metrics]),
        cer_mean=mean([c.cer for c in call_metrics]),
        role_accuracy_mean=mean([c.role_accuracy for c in call_metrics]),
        redaction_recall_mean=mean([c.redaction["recall"] for c in call_metrics]),
        redaction_precision_mean=mean([c.redaction["precision"] for c in call_metrics]),
        redaction_f1_mean=mean([c.redaction["f1"] for c in call_metrics]),
        judge_qwk=judge_qwk,
        calls=call_metrics,
        fingerprint=fingerprint(eval_config, rubric.sha256),
        golden_set="real" if is_real_golden_set else "synthetic",
        what_these_numbers_mean=(
            "Measured against a supplied golden set of labelled calls."
            if is_real_golden_set else SYNTHETIC_SET_NOTE
        ),
    )


def _judge_qwk(scorecards, gold_ratings, rubric):  # noqa: ANN001
    if not scorecards or not gold_ratings:
        return None
    try:
        from callqa.calibration import calibrate
        result = calibrate(scorecards, gold_ratings, rubric)
        return result.overall_qwk
    except Exception as exc:  # noqa: BLE001 - QWK is undefined on too-few/constant calls
        logger.debug("judge QWK not computed: %s", exc)
        return None


# ----------------------------------------------------------------- comparison

# metric -> (higher_is_better, tolerance-key)
_METRICS = {
    "wer_mean": (False, "rate"),
    "cer_mean": (False, "rate"),
    "role_accuracy_mean": (True, "rate"),
    "redaction_recall_mean": (True, "rate"),
    "redaction_precision_mean": (True, "rate"),
    "redaction_f1_mean": (True, "rate"),
    "judge_qwk": (True, "qwk"),
}


def compare(report: EvalReport, baseline: EvalReport) -> list[dict]:
    """Regressions of `report` against `baseline`, each {metric, was, now, delta}."""
    regressions: list[dict] = []
    for metric, (higher_better, tol_key) in _METRICS.items():
        now = getattr(report, metric)
        was = getattr(baseline, metric)
        if now is None or was is None:
            continue
        tol = _TOL[tol_key]
        worse = (now < was - tol) if higher_better else (now > was + tol)
        if worse:
            regressions.append({"metric": metric, "was": was, "now": now,
                                "delta": round(now - was, 4)})
    return regressions
