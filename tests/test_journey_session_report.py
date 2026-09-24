"""The report's two analysis levels: a single contact and the banker session
(the bank's Atlas report, ATL_R03), the page for one contact, the exports,
and the command that checks a dataset against the published figures."""

from __future__ import annotations

import csv
import io
import json
import re

import pytest

from callqa.config import load_config
from callqa.journey.importers.atlas import attach_atlas
from callqa.journey.importers.workbook import import_workbook
from callqa.journey.store import load_dataset, save_content, save_dataset
from callqa.reporting.journey import build_contact_report, build_journey_report
from tests.journey_fixtures import RAW_ACCOUNT_NUMBERS, build_atlas, build_workbook
from tests.test_journey_report import FAKE_ID, FORBIDDEN_WORD, _content


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


def test_both_levels_are_in_the_report(config):
    report = build_journey_report(config)
    html = report.html.read_text(encoding="utf-8")
    for anchor in ('id="sec-calls"', 'id="sec-atlas"', 'id="xc"', 'id="xs"',
                   'class="lvl-tabs', "דף המספרים", "נספח — סיווג הפעולות"):
        assert anchor in html, anchor
    data = _data(html)
    assert len(data["contacts"]) == sum(s.contacts for s in report.analysis.stories)
    assert len(data["sessions"]) == 2
    # the page of numbers: fifteen lines, as in ATL_R02
    assert html.count('<tr><td class="num n">') >= 15
    # each session row opens its story card at the session
    assert 'id="s1-x1"' in html and 'id="s1-x2"' in html
    s = report.sessions
    assert s is not None and len(s.head) == 15 and s.sessions_from == "export"


def test_level_option_keeps_one_level(config):
    calls = build_journey_report(config, name="calls", level="call")
    html = calls.html.read_text(encoding="utf-8")
    assert 'id="sec-calls"' in html and 'id="xc"' in html
    assert 'id="sec-atlas"' not in html and 'id="xs"' not in html
    assert "sessions" not in _data(html)
    sessions = build_journey_report(config, name="sessions", level="session")
    html = sessions.html.read_text(encoding="utf-8")
    assert 'id="sec-atlas"' in html and 'id="xs"' in html and 'id="xc"' not in html
    with pytest.raises(ValueError):
        build_journey_report(config, level="everything")


def test_without_atlas_there_is_no_session_level(tmp_path):
    cfg = load_config(None, {"paths": {"output_dir": str(tmp_path / "out")}})
    xlsx, zip_path = build_workbook(tmp_path / "in")
    save_dataset(cfg, import_workbook(xlsx, audio=zip_path).dataset)
    report = build_journey_report(cfg)
    html = report.html.read_text(encoding="utf-8")
    assert 'id="sec-atlas"' not in html and 'id="xs"' not in html
    assert report.sessions_csv is None and report.contacts_csv.exists()


def test_exports_and_privacy(config):
    save_content(config, load_dataset(config).dataset_id, _content(config))
    report = build_journey_report(config)
    contacts = list(csv.DictReader(io.StringIO(
        report.contacts_csv.read_text(encoding="utf-8").lstrip("﻿"))))
    sessions = list(csv.DictReader(io.StringIO(
        report.sessions_csv.read_text(encoding="utf-8").lstrip("﻿"))))
    codes = list(csv.DictReader(io.StringIO(
        report.codes_csv.read_text(encoding="utf-8").lstrip("﻿"))))
    assert {r["type"] for r in contacts} >= {"REC", "ABN", "UM"}
    assert [r["kind"] for r in sessions] == ["info", "execute"]
    assert sessions[1]["contact_no"] == "" and sessions[0]["contact_no"] == "1"
    assert {r["code"] for r in codes} == {"201", "11", "332", "970"}
    data = json.loads(report.json.read_text(encoding="utf-8"))
    assert len(data["sessions"]["head"]) == 15
    for path in (report.html, report.contacts_csv, report.sessions_csv, report.codes_csv,
                 report.json):
        text = path.read_text(encoding="utf-8")
        for raw in RAW_ACCOUNT_NUMBERS:
            assert raw not in text, path.name
        assert FORBIDDEN_WORD not in text and FAKE_ID not in text, path.name


def test_one_contact_page(config):
    save_content(config, load_dataset(config).dataset_id, _content(config))
    path = build_contact_report(config, None, 1, 1)
    assert path.name == "journey-call-001-1.html"
    html = path.read_text(encoding="utf-8")
    for text in ("שכבת הטבלה", "שכבת התוכן", "שכבת אטלס", "הפנייה בתוך הסיפור",
                 'class="compat-note"'):
        assert text in html, text
    # the session behind the call, with its operations and their categories
    assert "מידע ושאילתות" in html and "B0001" in html and "332" in html
    for raw in RAW_ACCOUNT_NUMBERS:
        assert raw not in html
    assert FAKE_ID not in html and FORBIDDEN_WORD not in html
    quiet = build_contact_report(config, None, 1, 1, with_text=False).read_text(encoding="utf-8")
    assert "כבר התקשרתי אתמול" not in quiet and "נימוק ייחודי שכתב המודל" not in quiet
    with pytest.raises(ValueError):
        build_contact_report(config, None, 1, 99)


def test_cli_contact_and_level(config, monkeypatch, capsys):
    from callqa.cli import main
    monkeypatch.setenv("CALLQA_PATHS__OUTPUT_DIR", str(config.paths.output_dir))
    assert main(["journey", "report", "--contact", "1:2"]) == 0
    assert "journey-call-001-2.html" in capsys.readouterr().out
    assert main(["journey", "report", "--contact", "x"]) != 0
    assert main(["journey", "report", "--level", "session"]) == 0


def test_atlas_check_against_expected_figures(config, monkeypatch, capsys, tmp_path):
    from callqa.cli import main
    monkeypatch.setenv("CALLQA_PATHS__OUTPUT_DIR", str(config.paths.output_dir))
    assert main(["journey", "atlas-check"]) == 0
    out = capsys.readouterr().out
    assert "15." in out and "מקור הסשנים" in out
    good = tmp_path / "good.yaml"
    good.write_text("counts: {stories: 3, sessions: 2, ops: 5}\n"
                    "kinds: {execute: 1, info: 1}\n"
                    "rows_by_category: {open: 2, info: 2, execute: 1}\n", encoding="utf-8")
    assert main(["journey", "atlas-check", "--expect", str(good)]) == 0
    bad = tmp_path / "bad.yaml"
    bad.write_text("counts: {sessions: 767}\nhead: {bankers: '7.4 / 6.0'}\n", encoding="utf-8")
    assert main(["journey", "atlas-check", "--expect", str(bad)]) != 0
    out = capsys.readouterr().out
    assert "counts.sessions: 2 (expected 767)" in out and "head.bankers" in out


def test_the_published_expectations_file_is_complete():
    from pathlib import Path

    from callqa.resources import load_yaml
    exp = load_yaml(Path(__file__).resolve().parent.parent / "eval" / "atlas_r02_expected.yaml")
    assert exp["counts"] == {"stories": 96, "sessions": 767, "ops": 3406, "covered": 64,
                             "covered_active": 61, "codes": 69}
    assert len(exp["head"]) == 14            # line 12 needs the vendor's status
    assert sum(exp["kinds"].values()) == sum(exp["classes"].values()) == 739
    assert sum(exp["rows_by_category"].values()) == 3406
