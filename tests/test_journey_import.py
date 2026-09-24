"""Journey inputs: the workbook reader, time parsing, the importers, privacy."""

from __future__ import annotations

import json
import locale
import zipfile
from datetime import date, datetime

import pytest

from callqa.config import load_config
from callqa.journey.importers.atlas import attach_atlas
from callqa.journey.importers.contract import import_contract
from callqa.journey.importers.workbook import classify, import_workbook
from callqa.journey.pseudo import account_key, normalise_account
from callqa.journey.timeparse import (
    TimeParseError,
    add_business_days,
    parse_dt,
)
from callqa.journey.xlsx import XlsxError, column_index, read_xlsx, write_xlsx
from tests.journey_fixtures import RAW_ACCOUNT_NUMBERS, build_atlas, build_workbook

# ---------------------------------------------------------------- time parsing


@pytest.mark.parametrize("value,expected", [
    ("02Aug2026 16:04:34", datetime(2026, 8, 2, 16, 4, 34)),
    ("02AUG2026:16:04:34", datetime(2026, 8, 2, 16, 4, 34)),
    ("15Jul2026 11:00:28.557", datetime(2026, 7, 15, 11, 0, 28, 557000)),
    ("2026-08-02 16:04:34.123", datetime(2026, 8, 2, 16, 4, 34, 123000)),
    ("2026-08-02T16:04:34", datetime(2026, 8, 2, 16, 4, 34)),
    ("02/08/2026 16:04", datetime(2026, 8, 2, 16, 4)),
    ("2.8.2026", datetime(2026, 8, 2)),
    ("2026-08-02", datetime(2026, 8, 2)),
    (46236.66983796296, datetime(2026, 8, 2, 16, 4, 34)),       # Excel serial
    (2_101_305_874.0, datetime(2026, 8, 2, 16, 4, 34)),         # SAS seconds
    (date(2026, 8, 2), datetime(2026, 8, 2)),
])
def test_parse_dt_shapes(value, expected):
    assert parse_dt(value) == expected


def test_parse_dt_does_not_depend_on_the_locale(monkeypatch):
    # strptime("%b") would follow the process locale; the parser must not.
    monkeypatch.setattr(locale, "getlocale", lambda *a: ("he_IL", "UTF-8"))
    assert parse_dt("02Aug2026 16:04:34").month == 8


@pytest.mark.parametrize("bad", ["", "yesterday", "31Foo2026 10:00:00", "2026-02-30", True])
def test_parse_dt_rejects(bad):
    with pytest.raises(TimeParseError):
        parse_dt(bad)


def test_business_days_skip_friday_and_saturday():
    thursday = datetime(2026, 7, 2, 10, 0)            # a Thursday
    assert add_business_days(thursday, 1) == datetime(2026, 7, 5, 10, 0)   # Sunday
    assert add_business_days(thursday, 2, frozenset({date(2026, 7, 5)})) == \
        datetime(2026, 7, 7, 10, 0)


# ---------------------------------------------------------------- xlsx


def test_xlsx_round_trip_with_dates_gaps_and_hebrew(tmp_path):
    path = tmp_path / "t.xlsx"
    write_xlsx(path, [("גיליון", [["א", None, 3.5], [datetime(2026, 8, 2, 16, 4, 34), True]])])
    wb = read_xlsx(path)
    sheet = wb.sheets[0]
    assert sheet.name == "גיליון"
    assert sheet.rows[0] == ["א", None, 3.5]
    assert sheet.rows[1][0] == datetime(2026, 8, 2, 16, 4, 34)
    assert sheet.rows[1][1] is True


def test_column_index():
    assert [column_index(r) for r in ("A1", "Z9", "AA1", "AZ3")] == [0, 25, 26, 51]


def test_xlsx_refuses_a_zip_bomb(tmp_path):
    path = tmp_path / "bomb.xlsx"
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("xl/workbook.xml", "<a/>" + " " * 5_000_000)
    with pytest.raises(XlsxError):
        read_xlsx(path)


def test_xlsx_refuses_a_non_workbook(tmp_path):
    path = tmp_path / "x.xlsx"
    path.write_bytes(b"not a zip")
    with pytest.raises(XlsxError):
        read_xlsx(path)


# ---------------------------------------------------------------- workbook import


@pytest.fixture
def workbook(tmp_path):
    return build_workbook(tmp_path / "in")


def test_sheets_are_recognised_by_content_not_order(workbook):
    xlsx, _zip = workbook
    found = classify(read_xlsx(xlsx))
    assert found["files"].name == "ATL_REPEAT_0000"
    assert found["messages"].name == "ATL_REPEAT_0001"
    assert found["interactions"].name == "ATL_REPEAT_0002"


def test_workbook_import_counts(workbook):
    xlsx, zip_path = workbook
    built = import_workbook(xlsx, audio=zip_path)
    ds = built.dataset
    c = ds.report.counts
    assert (c.stories, c.interactions, c.returns) == (3, 8, 5)
    assert (c.calls, c.recorded_calls, c.unrecorded_calls) == (6, 4, 2)
    assert (c.correspondences, c.messages) == (2, 3)
    assert (c.audio_files_mapped, c.multi_file_calls) == (7, 2)
    assert ds.report.ok


def test_split_call_segments_are_ordered_and_case_insensitive(workbook):
    xlsx, zip_path = workbook
    ds = import_workbook(xlsx, audio=zip_path).dataset
    call = ds.calls["007203b059935cda"]
    names = [s.file_name[-2:] for s in call.segments]
    assert names == ["01", "02", "03"]
    assert [s.seq for s in call.segments] == [1, 2, 3]
    assert call.complete


def test_orphan_file_and_declared_count_are_reported(workbook):
    xlsx, zip_path = workbook
    ds = import_workbook(xlsx, audio=zip_path).dataset
    codes = {i.code: i for i in ds.report.issues}
    assert codes["audio_orphans"].count == 1
    assert "1_9999" not in json.dumps(codes["audio_orphans"].examples)
    assert codes["declared_repeats_differ"].count == 1


def test_message_direction_and_redaction(workbook):
    xlsx, zip_path = workbook
    ds = import_workbook(xlsx, audio=zip_path).dataset
    um = [i for i in ds.interactions if i.channel == "message"]
    assert {i.direction for i in um} == {"inbound"}
    body = next(m.body for m in ds.messages if m.message_id == "COR-5081090")
    assert "123456782" not in body and "050-1234567" not in body


def test_no_account_number_leaves_the_private_map(workbook, tmp_path):
    xlsx, zip_path = workbook
    built = import_workbook(xlsx, audio=zip_path)
    dumped = built.dataset.model_dump_json()
    for raw in RAW_ACCOUNT_NUMBERS:
        assert raw not in dumped
    assert {r[3] for r in built.private_rows} == set(RAW_ACCOUNT_NUMBERS)


def test_story_numbers_follow_first_contact(workbook):
    xlsx, _ = workbook
    ds = import_workbook(xlsx).dataset
    firsts = [s.first_at for s in sorted(ds.stories, key=lambda s: s.story_no)]
    assert firsts == sorted(firsts)


def test_account_key_normalises_leading_zeros():
    assert normalise_account("083", "0335145") == normalise_account(83, "335145.0")
    assert account_key("83", "335145") == account_key("083", "0335145")
    assert account_key("83", "335145") != account_key("84", "335145")


# ---------------------------------------------------------------- atlas


def test_atlas_attaches_facts_sessions_and_coverage(workbook, tmp_path):
    xlsx, zip_path = workbook
    ds = import_workbook(xlsx, audio=zip_path).dataset
    attach_atlas(ds, build_atlas(tmp_path / "atlas"))
    abandoned = [i for i in ds.interactions if i.answer == "abandoned"]
    assert len(abandoned) == 1 and abandoned[0].call_key == "007203b62c453f87"
    assert len(ds.atlas_sessions) == 2
    first, second = ds.atlas_sessions
    assert first.unit_code == "109" and first.matched_interaction_id is not None
    assert [op.op_category for op in first.ops] == ["open", "info", "info"]
    assert second.has_execute and not second.view_only
    assert second.unit_code == "83"
    story1 = min(ds.stories, key=lambda s: s.story_no)
    assert story1.atlas_coverage == "partial"
    assert "335145" not in ds.model_dump_json()


# ---------------------------------------------------------------- contract


def test_contract_import_drops_unknown_columns(tmp_path):
    folder = tmp_path / "contract"
    folder.mkdir()
    (folder / "interactions.csv").write_text(
        "interaction_id,account_ref,started_at,channel,direction,status,call_key,national_id\n"
        "i1,17/335145,2026-08-01 10:00:00,call,inbound,answered,abc123abc123,123456782\n"
        "i2,17/335145,2026-08-01 12:00:00,call,inbound,abandoned,,123456782\n"
        "i3,17/335145,2026-08-02 09:00:00,message,outbound,,,123456782\n",
        encoding="utf-8")
    (folder / "call_segments.csv").write_text(
        "call_key,seq,file_name\nabc123abc123,2,b.wav\nabc123abc123,1,a.wav\n", encoding="utf-8")
    built = import_contract(folder)
    ds = built.dataset
    assert ds.report.dropped_columns["interactions.csv"] == ["national_id"]
    assert "123456782" not in ds.model_dump_json()
    assert [s.file_name for s in ds.calls["abc123abc123"].segments] == ["a.wav", "b.wav"]
    assert sum(i.answer == "abandoned" for i in ds.interactions) == 1


def test_contract_missing_required_column_is_an_error(tmp_path):
    folder = tmp_path / "c"
    folder.mkdir()
    (folder / "interactions.csv").write_text("interaction_id,started_at\nx,2026-08-01\n",
                                             encoding="utf-8")
    assert not import_contract(folder).dataset.report.ok


# ---------------------------------------------------------------- CLI


def _cli(args, monkeypatch, tmp_path):
    from callqa.cli import main
    monkeypatch.setenv("CALLQA_PATHS__OUTPUT_DIR", str(tmp_path / "out"))
    return main(args)


def test_cli_import_dry_run_writes_nothing(workbook, monkeypatch, tmp_path, capsys):
    xlsx, zip_path = workbook
    code = _cli(["journey", "import", "--xlsx", str(xlsx), "--audio", str(zip_path),
                 "--dry-run"], monkeypatch, tmp_path)
    out = capsys.readouterr().out
    assert code == 0
    assert "dry run" in out and not (tmp_path / "out" / "journey").exists()
    for raw in RAW_ACCOUNT_NUMBERS:
        assert raw not in out


def test_cli_import_then_reveal(workbook, monkeypatch, tmp_path, capsys):
    xlsx, zip_path = workbook
    assert _cli(["journey", "import", "--xlsx", str(xlsx), "--audio", str(zip_path)],
                monkeypatch, tmp_path) == 0
    config = load_config(None, {"paths": {"output_dir": str(tmp_path / "out")}})
    from callqa.journey.store import load_dataset, private_dir
    ds = load_dataset(config)
    assert ds.report.counts.stories == 3
    folder = private_dir(config, ds.dataset_id)
    from callqa.portable import private_to_owner
    assert private_to_owner(folder)
    for path in (tmp_path / "out" / "journey").rglob("*"):
        if path.is_file() and "private" not in path.parts:
            content = path.read_text(encoding="utf-8")
            for raw in RAW_ACCOUNT_NUMBERS:
                assert raw not in content, path
    capsys.readouterr()
    assert _cli(["journey", "reveal", "1"], monkeypatch, tmp_path) == 0
    assert "335145" in capsys.readouterr().out
    assert "story 1" in (folder / "reveal.log").read_text(encoding="utf-8")
