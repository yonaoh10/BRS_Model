"""The management (executive) report: numbers, privacy gates, scope, output, CLI.

The batch here is written straight to disk as the pipeline's artifacts -
scores, results, redacted transcripts, ingestion and features - so every
path the report reads is exercised without running the pipeline, and hostile
content can be planted exactly where a real batch could carry it.
"""

from __future__ import annotations

import json
import re
import statistics
import time
from datetime import date, timedelta
from html.parser import HTMLParser
from pathlib import Path

import pytest

from callqa.aggregation import load_scorecards
from callqa.judge.prompts import PROMPT_VERSION
from callqa.models import (
    CallMeta,
    CallResult,
    DimensionScore,
    Evidence,
    Features,
    RedactedTranscript,
    RedactedTurn,
    ScoreCard,
    SpeechRateWPM,
)
from callqa.pipeline import STAGES
from callqa.reporting.executive import BatchFilters, build_executive_report
from callqa.reporting.executive.analysis import analyse
from callqa.reporting.executive.dataset import load_batch, parse_call_date
from callqa.reporting.executive.render import report_name, script_json
from callqa.rubric import load_rubric, weighted_total

RAW_ID = "123456782"            # a valid-checksum Israeli ID: must never appear
RAW_NAME = "ישראל ישראלי"
HOSTILE_QUOTE = "</script><script>alert(1)</script>"
ERROR_SENTINEL = "050-7654321 שם_קובץ_פרטי"


def _write(path: Path, model) -> None:  # noqa: ANN001
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(model.model_dump_json(indent=2), encoding="utf-8")


def _scores(rubric, levels: dict[str, int], quote: str, when: str = "00:10") -> dict:  # noqa: ANN001
    return {
        d.id: DimensionScore(
            score=levels.get(d.id, 4),
            reasoning_he=f"נימוק לממד {d.name_he}",
            evidence=[Evidence(quote=quote, timestamp=when, speaker="banker")],
        )
        for d in rubric.dimensions
    }


def write_call(out: Path, rubric, call_id: str, *, banker: str = "B001",  # noqa: ANN001, PLR0913
               levels: dict[str, int] | None = None, status: str = "success",
               call_date: str | None = "2026-08-02", call_type: str | None = "service",
               duration: float = 300.0, redaction: str = "on", engine: str = "vllm",
               summary: str = "סיכום השיחה", quote: str = "שלום, במה אפשר לעזור?",
               talk_ratio: float = 0.5, error: str | None = None,
               with_card: bool | None = None, report_file: bool = False) -> None:
    levels = levels or {}
    turns = [RedactedTurn(speaker="banker", start=0.0, end=4.0, text=quote),
             RedactedTurn(speaker="customer", start=4.0, end=8.0, text='תודה, ת"ז <ת"ז:████>')]
    if redaction != "missing":
        _write(out / "redacted" / f"{call_id}.json", RedactedTranscript(
            call_id=call_id, engine="test", turns=turns,
            redaction_counts={"ISRAELI_ID": 1}, enabled=(redaction == "on")))
    _write(out / "ingestion" / f"{call_id}.json", CallMeta(
        call_id=call_id, banker_id=banker, file_name=f"{call_id}.wav", duration_sec=duration,
        channels=2, sample_rate=16000, call_date=call_date, call_type=call_type,
        banker_channel="L"))
    _write(out / "features" / f"{call_id}.json", Features(
        call_id=call_id, talk_ratio=talk_ratio, longest_banker_monologue_sec=20.0,
        interruptions_by_banker=1, interruptions_by_customer=0, patience_median_sec=1.0,
        banker_question_count=5, banker_questions_per_minute=1.0,
        speech_rate_wpm=SpeechRateWPM(banker=150, customer=140), dead_air_total_sec=3.0,
        call_duration_sec=duration, banker_speech_sec=duration * talk_ratio,
        customer_speech_sec=duration * (1 - talk_ratio)))
    scores = _scores(rubric, levels, quote)
    total, gate, failed = weighted_total(rubric, scores)
    if with_card is None:
        with_card = status != "failed"
    if with_card:
        _write(out / "scores" / f"{call_id}.json", ScoreCard(
            call_id=call_id, banker_id=banker, scores=scores, weighted_total=total,
            gate_failed=gate, failed_gates=failed, strengths_he=["פתיחה אדיבה"],
            development_area_he="לחזק את הסיכום", summary_he=summary,
            judge_engine=engine, model="/models/some-judge", prompt_sha256="a" * 64,
            prompt_version=PROMPT_VERSION, rubric_sha256=rubric.sha256,
            timestamp="2026-09-01T10:00:00+00:00"))
    report_path = None
    if report_file:
        path = out / "reports" / "calls" / f"{call_id}.html"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("<html></html>", encoding="utf-8")
        report_path = str(path)
    done = STAGES if status != "failed" else STAGES[:3]
    _write(out / "results" / f"{call_id}.json", CallResult(
        call_id=call_id, status=status, error=error, stages_completed=list(done),
        report_path=report_path))


@pytest.fixture(scope="module")
def rubric():  # noqa: ANN201
    return load_rubric()


def make_batch(out: Path, rubric, n: int = 60) -> None:  # noqa: ANN001
    """A batch with structure: B001 strong, B006 weak, loans harder, a trend."""
    start = date(2026, 6, 7)
    for i in range(n):
        banker = f"B00{i % 6 + 1}"
        skill = {"B001": 1, "B006": -2}.get(banker, 0)
        # Round-based variation (i // 6) hits every banker alike; i % 3 would
        # have given one banker a bonus on every call.
        base = 3 + skill + (1 if (i // 6) % 3 == 0 else 0) - (1 if (i // 6) % 7 == 0 else 0)
        levels = {d.id: max(1, min(5, base + (1 if j % 3 == 0 else 0)))
                  for j, d in enumerate(rubric.dimensions)}
        if i % 11 == 0:
            levels["identification"] = 2          # a gate failure
        ctype = "loans" if i % 4 == 0 else "service"
        if ctype == "loans":
            levels["clarity"] = max(1, levels["clarity"] - 1)
        write_call(out, rubric, f"C{i:04d}", banker=banker, levels=levels,
                   call_date=(start + timedelta(days=i)).isoformat(), call_type=ctype,
                   duration=120 + 11 * i, talk_ratio=0.4 + (i % 5) * 0.08,
                   report_file=(i % 2 == 0))
    # Planted hazards.
    write_call(out, rubric, "RAWCALL", redaction="off", summary=f"הלקוח {RAW_NAME} ת.ז {RAW_ID}",
               quote=f"מספר הזהות שלי {RAW_ID}", report_file=True)
    write_call(out, rubric, "NOREDACT", redaction="missing", summary=f"סיכום עם {RAW_NAME}",
               quote=f"שם הלקוח {RAW_NAME}")
    write_call(out, rubric, "HOSTILE", call_type="<script>alert(2)</script>", quote=HOSTILE_QUOTE,
               summary="סיכום <b>מודגש</b> ו-</script>")
    write_call(out, rubric, "HELD1", status="needs_human_review", with_card=True,
               error="speaker roles inferred with low confidence (0.40 < 0.60)")
    write_call(out, rubric, "HELD2", status="needs_human_review", with_card=False,
               error=f"judge output failed after 3 attempts: {ERROR_SENTINEL}")
    write_call(out, rubric, "FAIL1", status="failed", error=f"RuntimeError: {ERROR_SENTINEL}")


@pytest.fixture
def batch_out(tmp_path: Path, rubric) -> Path:  # noqa: ANN001
    out = tmp_path / "output"
    make_batch(out, rubric)
    return out


class _Parser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.ids: set[str] = set()
        self.scripts = 0
        self.hrefs: list[str] = []

    def handle_starttag(self, tag, attrs):  # noqa: ANN001, ANN201
        d = dict(attrs)
        if "id" in d:
            self.ids.add(d["id"])
        if tag == "script":
            self.scripts += 1
        if tag == "a" and d.get("href"):
            self.hrefs.append(d["href"])


def _parse(html: str) -> _Parser:
    p = _Parser()
    p.feed(html)
    p.close()
    return p


# -- structure and numbers ------------------------------------------------------

def test_report_has_three_levels_and_matches_the_published_cohort(batch_out, rubric):  # noqa: ANN001
    report = build_executive_report(batch_out, rubric)
    html = report.html.read_text(encoding="utf-8")
    parsed = _parse(html)
    for section in ("level1", "level2", "level3", "method", "explorer", "sec-dims", "sec-trend",
                    "sec-bankers", "sec-risk", "sec-coverage", "sec-cases", "sec-attention"):
        assert section in parsed.ids, section
    assert parsed.scripts == 2        # the data block and the behaviour: nothing injected
    cohort = load_scorecards(batch_out)
    a = report.analysis
    assert a.n_scored == len(cohort)
    assert a.total.mean == pytest.approx(statistics.mean(c.weighted_total for c in cohort))
    assert a.n_held == 2 and a.n_failed == 1
    assert a.n_scope == a.n_scored + a.n_held + a.n_failed


def test_lost_points_add_up_to_the_gap_below_100(batch_out, rubric):  # noqa: ANN001
    a = analyse(load_batch(batch_out, rubric))
    lost = sum(d.lost_mean for d in a.dims) + a.gate_penalty_mean
    # weighted_total is stored rounded to 0.1, so allow that much.
    assert lost == pytest.approx(100 - a.total.mean, abs=0.06)
    assert sum(d.lost_share for d in a.dims) <= 1.0 + 1e-9


def test_held_and_failed_calls_are_counted_but_never_scored(batch_out, rubric):  # noqa: ANN001
    a = analyse(load_batch(batch_out, rubric))
    held = {r.call_id: r for r in a.batch.held}
    assert set(held) == {"HELD1", "HELD2"}
    assert all(r.card is None for r in held.values())      # HELD1's scorecard is not used
    assert "זיהוי הדוברים בוודאות נמוכה" in held["HELD1"].held_reasons
    assert "תשובת מודל השיפוט לא עברה אימות" in held["HELD2"].held_reasons
    failed = a.batch.failed[0]
    assert failed.failed_reason == "כשל בשלב זיהוי דוברים"


def test_banker_flags_find_the_weak_and_strong_banker(tmp_path, rubric):  # noqa: ANN001
    out = tmp_path / "output"
    make_batch(out, rubric, n=120)
    a = analyse(load_batch(out, rubric))
    flags = {b.key: b.flag for b in a.bankers}
    assert flags["B006"] == "bad"
    assert flags["B001"] == "ok"
    assert {b.key for b in a.bankers if b.flag in ("bad", "ok")} <= {"B001", "B006"}


def test_small_batches_render_without_statistics(tmp_path, rubric):  # noqa: ANN001
    out = tmp_path / "output"
    write_call(out, rubric, "ONLY1")
    report = build_executive_report(out, rubric)
    a = report.analysis
    assert a.n_scored == 1 and a.trend is None and all(b.flag == "few" for b in a.bankers)
    assert "</html>" in report.html.read_text(encoding="utf-8")


def test_an_empty_batch_renders_an_explanation_not_an_error(tmp_path, rubric):  # noqa: ANN001
    out = tmp_path / "output"
    write_call(out, rubric, "HELDONLY", status="needs_human_review", with_card=False,
               error="judge output failed after 2 attempts: x")
    report = build_executive_report(out, rubric)
    html = report.html.read_text(encoding="utf-8")
    assert report.analysis.n_scored == 0
    assert "לא נמצאו בתקופה זו שיחות שנוקדו" in html


# -- privacy ----------------------------------------------------------------------

def _all_outputs(report) -> str:  # noqa: ANN001
    return "\n".join(p.read_text(encoding="utf-8") for p in (report.html, report.csv, report.json))


def test_no_raw_identifier_or_ungated_text_reaches_any_output(batch_out, rubric):  # noqa: ANN001
    report = build_executive_report(batch_out, rubric)
    text = _all_outputs(report)
    assert RAW_ID not in text
    assert RAW_NAME not in text                  # the missing-redaction call's text too
    assert ERROR_SENTINEL not in text            # results.error is never shown
    assert "050-7654321" not in text
    assert "calls/RAWCALL.html" not in text      # no link to a raw per-call report
    assert "/models/some-judge" not in text      # only the model's base name
    # Anti-vacuity: gated calls DO contribute their text.
    assert "שלום, במה אפשר לעזור?" in text


def test_hostile_text_cannot_break_out_of_the_page(batch_out, rubric):  # noqa: ANN001
    report = build_executive_report(batch_out, rubric)
    html = report.html.read_text(encoding="utf-8")
    assert "<script>alert" not in html
    assert HOSTILE_QUOTE not in html
    assert _parse(html).scripts == 2
    block = re.search(r'<script type="application/json" id="xr-data">(.*?)</script>', html, re.S)
    data = json.loads(block.group(1))
    hostile = next(c for c in data["calls"] if c["id"] == "HOSTILE")
    assert "<" not in hostile["t"]               # the free-text call type is sanitised
    assert "alert(1)" in json.dumps(data["text"]["HOSTILE"], ensure_ascii=False)  # shown, inert


def test_hebrew_stays_literal_in_the_embedded_data(batch_out, rubric):  # noqa: ANN001
    html = build_executive_report(batch_out, rubric).html.read_text(encoding="utf-8")
    block = re.search(r'id="xr-data">(.*?)</script>', html, re.S).group(1)
    assert "\\u05" not in block                  # the leak sweep can read it
    assert "סיכום השיחה" in block


def test_script_json_escapes_everything_that_ends_a_script():
    out = script_json({"a": "</script><!-- &     >"})
    assert "<" not in out and ">" not in out and "&" not in out
    assert " " not in out and " " not in out
    assert json.loads(out) == {"a": "</script><!-- &     >"}


def test_no_quotes_variant_carries_no_call_text(batch_out, rubric):  # noqa: ANN001
    report = build_executive_report(batch_out, rubric, with_text=False)
    html = report.html.read_text(encoding="utf-8")
    assert "שלום, במה אפשר לעזור?" not in html
    assert "נימוק לממד" not in html
    assert "סיכום השיחה" not in html
    assert "גרסה להפצה רחבה" in html


def test_csv_is_excel_safe(batch_out, rubric):  # noqa: ANN001
    write_call(batch_out, rubric, "FORMULA", banker="=HYPERLINK(1)")
    report = build_executive_report(batch_out, rubric)
    content = report.csv.read_text(encoding="utf-8")
    assert content.startswith("﻿")
    assert ",'=HYPERLINK(1)," in content
    assert ",=HYPERLINK" not in content


# -- scope, names, links -----------------------------------------------------------

def test_filters_narrow_the_scope(batch_out, rubric):  # noqa: ANN001
    all_calls = load_batch(batch_out, rubric).records
    loans = load_batch(batch_out, rubric, BatchFilters(call_type="loans")).records
    assert loans and all(r.call_type_key == "loans" for r in loans)
    july = load_batch(batch_out, rubric, BatchFilters(date_from=date(2026, 7, 1),
                                                      date_to=date(2026, 7, 31))).records
    assert july and all(date(2026, 7, 1) <= r.call_date <= date(2026, 7, 31) for r in july)
    one = load_batch(batch_out, rubric, BatchFilters(banker_id="B002")).records
    assert one and {r.banker_id for r in one} == {"B002"}
    assert len(loans) < len(all_calls)


def test_scoped_reports_get_their_own_file(batch_out, rubric):  # noqa: ANN001
    report = build_executive_report(batch_out, rubric, BatchFilters(call_type="loans"))
    assert report.html.name == "executive-loans.html"
    assert report_name(BatchFilters(), None) == "executive"
    assert report_name(BatchFilters(), "q3") == "executive-q3"
    with pytest.raises(ValueError):
        report_name(BatchFilters(), "../x")


def test_links_point_only_at_files_that_exist(batch_out, rubric):  # noqa: ANN001
    report = build_executive_report(batch_out, rubric)
    html = report.html.read_text(encoding="utf-8")
    block = re.search(r'id="xr-data">(.*?)</script>', html, re.S).group(1)
    data = json.loads(block)
    reports = batch_out / "reports"
    for call in data["calls"]:
        if call["rep"]:
            assert (reports / call["rep"]).is_file()
    for href in _parse(html).hrefs:
        if not href.startswith("#"):
            assert (reports / href).is_file(), href


@pytest.mark.parametrize(("text", "expected"), [
    ("2026-08-31", date(2026, 8, 31)), ("31/08/2026", date(2026, 8, 31)),
    ("31.8.2026", date(2026, 8, 31)), ("2026/08/31", date(2026, 8, 31)),
    ("31/13/2026", None), ("yesterday", None), ("", None), (None, None),
])
def test_call_dates_in_the_forms_a_bank_csv_uses(text, expected):  # noqa: ANN001
    assert parse_call_date(text) == expected


def test_demo_data_is_labelled_as_such(tmp_path, rubric):  # noqa: ANN001
    out = tmp_path / "output"
    write_call(out, rubric, "M1", engine="mock")
    html = build_executive_report(out, rubric).html.read_text(encoding="utf-8")
    assert "אין להשתמש בדוח זה להערכת עובדים" in html


def test_a_thousand_calls_render_in_reasonable_time(tmp_path, rubric):  # noqa: ANN001
    out = tmp_path / "output"
    make_batch(out, rubric, n=1000)
    started = time.perf_counter()
    report = build_executive_report(out, rubric)
    elapsed = time.perf_counter() - started
    size = report.html.stat().st_size
    assert report.analysis.n_scored >= 1000
    assert elapsed < 30, elapsed
    assert size < 8_000_000, size


# -- CLI ----------------------------------------------------------------------------

def test_cli_report_writes_the_management_report_and_links_it(batch_out, monkeypatch):  # noqa: ANN001
    from callqa.cli import main

    monkeypatch.setenv("CALLQA_PATHS__OUTPUT_DIR", str(batch_out))
    monkeypatch.setenv("CALLQA_PATHS__STATE_DB", str(batch_out.parent / "state.db"))
    assert main(["report"]) == 0
    assert (batch_out / "reports" / "executive.html").is_file()
    index = (batch_out / "reports" / "index.html").read_text(encoding="utf-8")
    assert 'href="executive.html"' in index
    assert main(["executive-report", "--from", "01/07/2026", "--to", "2026-07-31",
                 "--no-quotes"]) == 0
    assert (batch_out / "reports" / "executive-2026-07-01_2026-07-31.html").is_file()
    assert main(["executive-report", "--from", "not-a-date"]) == 2
    assert main(["executive-report", "--from", "2026-08-01", "--to", "2026-07-01"]) == 2


# -- the page in a real browser ------------------------------------------------------

def _chrome() -> str | None:
    import glob
    import os
    import shutil

    local = os.environ.get("LOCALAPPDATA", "")
    for pattern in ("/opt/pw-browsers/chromium*/chrome-linux/chrome",
                    str(Path.home() / ".cache/ms-playwright/chromium*/chrome-linux/chrome"),
                    str(Path(local) / "ms-playwright/chromium*/chrome-win/chrome.exe")):
        matches = sorted(glob.glob(pattern))
        if matches:
            return matches[-1]
    edge = [Path(os.environ.get(v, "")) / "Microsoft/Edge/Application/msedge.exe"
            for v in ("ProgramFiles(x86)", "ProgramFiles")]
    return (shutil.which("chromium") or shutil.which("google-chrome")
            or next((str(e) for e in edge if e.is_file()), None))


@pytest.mark.skipif(_chrome() is None, reason="no Chromium or Edge available")
def test_the_page_works_in_a_browser(batch_out, rubric):  # noqa: ANN001
    """Filters, drill-down and links run without a script error, from disk, and
    nothing ungated appears in the rendered text."""
    playwright = pytest.importorskip("playwright.sync_api")
    report = build_executive_report(batch_out, rubric)
    with playwright.sync_playwright() as p:
        browser = p.chromium.launch(executable_path=_chrome())
        page = browser.new_context(viewport={"width": 1280, "height": 900},
                                   locale="he-IL").new_page()
        errors: list[str] = []
        page.on("pageerror", lambda e: errors.append(str(e)))
        page.goto(report.html.as_uri())
        page.wait_for_timeout(300)
        assert page.evaluate("document.documentElement.getAttribute('data-explorer')") == "ready"
        scored = report.analysis.n_scored
        assert page.evaluate("document.querySelectorAll('#xp-body tr.row').length") == \
            min(50, scored)
        # A finding's link filters the explorer to exactly its calls.
        page.click("#sec-risk a[data-filter] >> nth=0")
        page.wait_for_timeout(150)
        gate_calls = sum(1 for r in report.analysis.batch.scored if r.card.gate_failed)
        assert page.inner_text("#xp-count").startswith(str(gate_calls))
        # A row opens its drill-down with the per-dimension scores.
        page.click("#xp-body tr.row >> nth=0")
        page.wait_for_timeout(150)
        assert page.evaluate("document.querySelectorAll('#xp-body tr.drawer .dcard').length") == \
            len(report.analysis.dims)
        # One banker: the profile appears.
        page.click("#xp-reset")
        page.select_option("#xp-banker", "B002")
        page.wait_for_timeout(150)
        assert "B002" in page.inner_text("#xp-profile")
        # The raw call is listed by number only.
        page.click("#xp-reset")
        page.fill("#xp-q", "RAWCALL")
        page.wait_for_timeout(150)
        page.click("#xp-body tr.row >> nth=0")
        page.wait_for_timeout(150)
        text = page.evaluate("document.body.innerText")
        assert RAW_ID not in text and RAW_NAME not in text
        assert "לא ניתן היה לאמת" in text
        page.click("#btn-theme")
        page.emulate_media(media="print")
        page.wait_for_timeout(100)
        assert not errors, errors
        browser.close()
