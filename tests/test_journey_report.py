"""The journey report: rendering, privacy, the content layer, charts, findings."""

from __future__ import annotations

import csv
import importlib.util
import io
import json
import re
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from callqa.config import load_config
from callqa.journey.analysis import PENDING, analyse
from callqa.journey.findings import build_opinion
from callqa.journey.importers.atlas import attach_atlas
from callqa.journey.importers.workbook import import_workbook
from callqa.journey.models import (
    Commitment,
    ContentLayer,
    Evidence,
    InteractionCard,
    ReturnJudgement,
    StoryVerdict,
)
from callqa.journey.store import load_dataset, save_content, save_dataset
from callqa.journey.vocab import load_taxonomy, load_units
from callqa.reporting.journey import charts
from callqa.reporting.journey.render import (
    JOURNEY_REPORT_FILE_RE,
    build_journey_report,
    report_name,
)
from tests.journey_fixtures import RAW_ACCOUNT_NUMBERS, build_atlas, build_workbook

REPO_ROOT = Path(__file__).resolve().parent.parent
FORBIDDEN_WORD = "אצלכם"
FAKE_ID = "123456782"          # a valid-checksum Israeli ID the redactor must catch


def _data(html: str) -> dict:
    m = re.search(r'<script type="application/json" id="xr-data">(.*?)</script>', html, re.S)
    assert m
    return json.loads(m.group(1))


@pytest.fixture
def config(tmp_path):
    cfg = load_config(None, {"paths": {"output_dir": str(tmp_path / "out")}})
    xlsx, zip_path = build_workbook(tmp_path / "in")
    ds = import_workbook(xlsx, audio=zip_path).dataset
    attach_atlas(ds, build_atlas(tmp_path / "atlas"))
    save_dataset(cfg, ds)
    return cfg


def _content(config) -> ContentLayer:
    ds = load_dataset(config)
    recorded = [i for i in ds.interactions if i.recorded]
    first = recorded[0]
    quote = f"כבר התקשרתי אתמול, תעודת זהות {FAKE_ID}, ואף אחד לא חזר <b>אליי</b>"
    ev = Evidence(interaction_id=first.interaction_id, line=3, quote=quote)
    cards = {i.interaction_id: InteractionCard(
        interaction_id=i.interaction_id, topic="loans_mortgages", outcome="not_resolved",
        retold="yes", retold_ev=ev, commitments=[Commitment(kind="callback", by="bank")])
        for i in recorded}
    judgements = {i.interaction_id: ReturnJudgement(
        interaction_id=i.interaction_id, category="unclosed_loop", basis="content",
        decided_by="llm", objective_class="content", reason_he="נימוק ייחודי שכתב המודל",
        quotes=[ev]) for i in recorded}
    verdicts = {s.story_key: StoryVerdict(story_key=s.story_key, topic="loans_mortgages",
                                          status="open", status_basis="content",
                                          headline_he="לקוח שחיכה לטלפון",
                                          narrative_he=f"הלקוח (ת.ז. {FAKE_ID}) פנה שלוש פעמים.")
                for s in ds.stories}
    return ContentLayer(engine="mock", created_at=datetime(2026, 9, 1), cards=cards,
                        judgements=judgements, verdicts=verdicts)


# ---------------------------------------------------------------- the page


def test_report_without_content_is_facts_only(config):
    report = build_journey_report(config)
    html = report.html.read_text(encoding="utf-8")
    for anchor in ('id="level1"', 'id="level2"', 'id="level3"', 'id="method"', 'id="explorer"'):
        assert anchor in html
    assert "סיווג לפי תוכן עוד לא הורץ" in html
    assert "ממתין לסיווג לפי תוכן" in html          # content returns are pending, not unclassifiable
    data = _data(html)
    assert len(data["stories"]) == 3
    assert report.stories_csv.exists() and report.returns_csv.exists() and report.json.exists()
    assert (report.html.parent.parent / "journey").is_dir()
    assert report.analysis.categories_strict.get(PENDING)
    # every story has its card and its timeline
    assert html.count('class="card story"') == 3
    assert html.count('class="ch ch-timeline"') == 3


def test_no_account_numbers_and_no_forbidden_word_anywhere(config):
    save_content(config, load_dataset(config).dataset_id, _content(config))
    report = build_journey_report(config)
    for path in (report.html, report.stories_csv, report.returns_csv, report.json):
        text = path.read_text(encoding="utf-8")
        for raw in RAW_ACCOUNT_NUMBERS:
            assert raw not in text, path.name
        assert FORBIDDEN_WORD not in text, path.name
        assert FAKE_ID not in text, path.name


def test_content_layer_fills_categories_quotes_and_narratives(config):
    save_content(config, load_dataset(config).dataset_id, _content(config))
    report = build_journey_report(config)
    html = report.html.read_text(encoding="utf-8")
    a = report.analysis
    assert a.categories_strict.get("unclosed_loop")
    assert not a.categories_strict.get(PENDING)
    assert a.metrics["failure_rate"].k == a.metrics["failure_rate"].n > 0
    assert "לקוח שחיכה לטלפון" in html
    assert "כבר התקשרתי אתמול" in html
    assert "<b>אליי</b>" not in html                   # markup from the model is stripped
    assert "שכבת תוכן: mock" in html
    assert "דוח הדגמה" in html                          # a mock engine is labelled as such


def test_no_quotes_version_carries_no_text_from_calls(config):
    save_content(config, load_dataset(config).dataset_id, _content(config))
    report = build_journey_report(config, with_text=False, name="wide")
    html = report.html.read_text(encoding="utf-8")
    assert report.html.name == "journey-wide.html"
    assert "כבר התקשרתי אתמול" not in html
    assert "נימוק ייחודי שכתב המודל" not in html
    assert "לקוח שחיכה לטלפון" not in html
    assert "ללא ציטוטים ונימוקים" in html


def test_csv_exports_are_excel_safe(config):
    report = build_journey_report(config)
    text = report.stories_csv.read_text(encoding="utf-8")
    assert text.startswith("﻿")
    rows = list(csv.DictReader(io.StringIO(text.lstrip("﻿"))))
    assert [int(r["story_no"]) for r in rows] == [1, 2, 3]
    assert not any(v.startswith(("=", "+", "-", "@")) for r in rows for v in r.values() if v)


def test_report_names():
    assert report_name(None) == "journey"
    assert report_name("q3") == "journey-q3"
    assert report_name("journey-q3") == "journey-q3"
    with pytest.raises(ValueError):
        report_name("../x")
    assert JOURNEY_REPORT_FILE_RE.fullmatch("journey-q3.html")
    assert not JOURNEY_REPORT_FILE_RE.fullmatch("journey-.html")


def test_cli_report(config, monkeypatch, capsys):
    from callqa.cli import main
    monkeypatch.setenv("CALLQA_PATHS__OUTPUT_DIR", str(config.paths.output_dir))
    assert main(["journey", "report", "--no-quotes"]) == 0
    assert "journey.html" in capsys.readouterr().out


def test_cli_report_without_a_dataset_fails_cleanly(tmp_path, monkeypatch, capsys):
    from callqa.cli import main
    monkeypatch.setenv("CALLQA_PATHS__OUTPUT_DIR", str(tmp_path / "empty"))
    assert main(["journey", "report"]) != 0
    assert "no journey dataset" in capsys.readouterr().err


def test_index_links_the_journey_report(config, tmp_path):
    from callqa.reporting.banker_report import _write_index
    build_journey_report(config)
    index = _write_index(config.paths.output_dir, [], {}, {}, "")
    assert 'href="journey.html"' in index.read_text(encoding="utf-8")


# ---------------------------------------------------------------- findings


def test_findings_mark_every_number(config):
    save_content(config, load_dataset(config).dataset_id, _content(config))
    ds = load_dataset(config)
    tax = load_taxonomy()
    a, _ = analyse(ds, taxonomy=tax, units=load_units(), min_rate_n=1, min_firm_n=1)
    opinion = build_opinion(a, tax)
    assert opinion.findings
    for f in opinion.findings:
        for text in (f.title, f.text):
            bare = re.sub(r"⟦[^⟧]*⟧", "", text)
            assert not re.search(r"\d", bare.replace("Kaplan-Meier", "")), text
    assert opinion == build_opinion(a, tax)             # deterministic


def test_small_batch_is_called_preliminary(config):
    ds = load_dataset(config)
    tax = load_taxonomy()
    a, _ = analyse(ds, taxonomy=tax, units=load_units())
    assert "ראשונית" in build_opinion(a, tax).assessment


# ---------------------------------------------------------------- charts


T0 = datetime(2026, 7, 5, 9, 0)


def test_story_timeline_draws_both_lanes_and_links():
    contacts = [{"index": i, "at": T0 + timedelta(minutes=m), "kind": k, "category_cls": "ch-cat-first",
                 "category": "x", "direction": "inbound", "title": f"c{i}"}
                for i, (m, k) in enumerate([(0, "recorded_call"), (1, "abandoned"),
                                            (60 * 50, "message"), (60 * 51, "unrecorded_answered")])]
    sessions = [{"start": T0 + timedelta(minutes=2), "end": T0 + timedelta(minutes=9),
                 "unit_kind": "center", "execute": True, "view_only": False, "contact_index": 0,
                 "title": "מרכז"}]
    promises = [{"from_index": 0, "to": T0 + timedelta(hours=50), "outcome": "broken"}]
    svg = str(charts.story_timeline(contacts, sessions, label="סיפור 001", anchor="s1",
                                    promises=promises))
    assert svg.count('href="#s1-c') == 4
    assert "ch-sess ch-unit-center ch-sess-exec" in svg
    assert "ch-arc-bad" in svg and "ch-mark-cross" in svg
    assert "ימים" in svg                               # the long gap was compressed and labelled
    # marks one minute apart are pushed apart, not drawn on top of each other
    xs = [float(x) for x in re.findall(r'<circle class="ch-mark [^"]*" cx="([\d.]+)"', svg)]
    assert len(xs) == 1
    crosses = re.findall(r'class="ch-mark-cross" d="M([\d.]+),', svg)
    assert abs(float(crosses[0]) + 6 - xs[0]) >= 16


def test_charts_handle_empty_input():
    assert "ch-none" in str(charts.hbars([], label="x"))
    assert "ch-none" in str(charts.stacked_bars([], [("a", "A", "ch-bar")], label="x"))
    assert "ch-none" in str(charts.columns([("a", 0)], label="x"))
    assert "ch-none" in str(charts.km_chart([], label="x"))
    assert "ch-none" in str(charts.story_timeline([], [], label="x"))


def test_chart_labels_are_escaped():
    out = str(charts.hbars([{"label": "<script>x</script>", "value": 3}], label="<b>t</b>"))
    assert "<script>" not in out and "<b>t</b>" not in out


def test_stacked_bars_show_n_and_shares():
    out = str(charts.stacked_bars([{"label": "שיחה", "counts": {"a": 3, "b": 1}}],
                                  [("a", "A", "ch-bar"), ("b", "B", "ch-bar-bad")], label="x"))
    assert "n=4" in out and "75%" in out


# ---------------------------------------------------------------- demo generator


def _load_demo():
    spec = importlib.util.spec_from_file_location(
        "generate_journey_demo", REPO_ROOT / "scripts" / "generate_journey_demo.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_demo_generator_end_to_end(tmp_path, monkeypatch):
    demo = _load_demo()
    ws = tmp_path / "jdemo"
    summary = demo.generate(ws, stories=25, seed=3)
    assert summary["stories"] == 25
    from callqa.cli import main
    monkeypatch.setenv("CALLQA_PATHS__OUTPUT_DIR", str(ws / "output"))
    assert main(["journey", "import", "--xlsx", str(ws / "input" / "handoff.xlsx"),
                 "--atlas", str(ws / "input" / "atlas")]) == 0
    assert main(["journey", "report"]) == 0
    html = (ws / "output" / "reports" / "journey.html").read_text(encoding="utf-8")
    assert len(_data(html)["stories"]) == 25
    assert FORBIDDEN_WORD not in html
