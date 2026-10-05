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
    detect_day_first,
    parse_dt,
    tz_suffix,
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


def test_eight_digits_are_a_date_not_sas_seconds():
    assert parse_dt("20260802") == datetime(2026, 8, 2)
    with pytest.raises(TimeParseError, match="bad_date"):
        parse_dt("20261340")


def test_a_small_numeric_string_is_refused_as_ambiguous_but_a_typed_cell_is_not():
    # 24300 is 1966-07-12 as an Excel day count and 2026-07-12 as SAS days
    with pytest.raises(TimeParseError, match="ambiguous_numeric_date"):
        parse_dt("24300")
    assert parse_dt(24300) == datetime(1966, 7, 12)     # a typed xlsx cell: Excel
    assert parse_dt("2101305874").year == 2026          # SAS seconds as text


def test_a_zone_suffix_is_dropped_and_the_clock_kept():
    assert parse_dt("2026-08-03T10:00:00+03:00") == datetime(2026, 8, 3, 10, 0)
    assert parse_dt("2026-08-03T10:00:00Z") == datetime(2026, 8, 3, 10, 0)
    assert tz_suffix("2026-08-03T10:00:00+03:00") and not tz_suffix("2026-08-03 10:00:00")


def test_month_first_dates_are_read_when_the_column_says_so():
    assert parse_dt("08/02/2026 16:04", day_first=False) == datetime(2026, 8, 2, 16, 4)
    assert detect_day_first(["08/02/2026", "08/15/2026"]) is False
    assert detect_day_first(["02/08/2026", "15/08/2026"]) is True
    assert detect_day_first(["02/08/2026", "03/09/2026"]) is None      # nothing decides
    assert detect_day_first(["2026-08-02", datetime(2026, 8, 2)]) is None
    assert detect_day_first(["15/08/2026", "08/15/2026"]) is None     # inconsistent


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
    # ATL_R02: covered when the story's first contact is on or after the first
    # DAY the log holds - here the same day as the first row
    assert story1.atlas_coverage == "full"
    assert ds.atlas_rules.sessions_from == "export"
    assert "335145" not in ds.model_dump_json()
    # the exported sessions are exactly the ones their rows make
    assert any(i.code == "atlas_sessions_rebuilt" for i in ds.report.issues)


# ---------------------------------------------------------------- contract


def test_contract_import_drops_unknown_columns(tmp_path):
    folder = tmp_path / "contract"
    folder.mkdir()
    (folder / "interactions.csv").write_text(
        "interaction_id,account_ref,started_at,channel,direction,status,call_key,notes\n"
        "i1,17/335145,2026-08-01 10:00:00,call,inbound,answered,abc123abc123,x123456782\n"
        "i2,17/335145,2026-08-01 12:00:00,call,inbound,abandoned,,x123456782\n"
        "i3,17/335145,2026-08-02 09:00:00,message,outbound,,,x123456782\n",
        encoding="utf-8")
    (folder / "call_segments.csv").write_text(
        "call_key,seq,file_name\nabc123abc123,2,b.wav\nabc123abc123,1,a.wav\n", encoding="utf-8")
    built = import_contract(folder)
    ds = built.dataset
    assert ds.report.dropped_columns["interactions.csv"] == ["notes"]
    assert "123456782" not in ds.model_dump_json()
    assert [s.file_name for s in ds.calls["abc123abc123"].segments] == ["a.wav", "b.wav"]
    assert sum(i.answer == "abandoned" for i in ds.interactions) == 1


@pytest.mark.parametrize("column", ["national_id", "National ID", "phone", "שם לקוח", "ת\"ז"])
def test_a_column_naming_a_persons_identity_fails_the_import(tmp_path, column):
    folder = tmp_path / "c"
    folder.mkdir()
    (folder / "interactions.csv").write_text(
        f"interaction_id,account_ref,started_at,channel,{column}\n"
        "i1,17/335145,2026-08-01 10:00:00,call,123456782\n", encoding="utf-8")
    ds = import_contract(folder).dataset
    issue = next(i for i in ds.report.issues if i.code == "forbidden_columns")
    assert not ds.report.ok and column.lower() in issue.message
    assert "123456782" not in issue.message
    assert not ds.interactions


def _contract(tmp_path, interactions: str, messages: str | None = None):
    folder = tmp_path / "c"
    folder.mkdir()
    (folder / "interactions.csv").write_text(
        "interaction_id,account_ref,started_at,channel,direction\n" + interactions,
        encoding="utf-8")
    if messages is not None:
        (folder / "messages.csv").write_text(
            "correspondence_id,message_id,sent_at,direction,body\n" + messages, encoding="utf-8")
    return import_contract(folder).dataset


def _codes(ds) -> dict[str, int]:
    return {i.code: i.count for i in ds.report.issues}


def test_contract_month_first_dates_are_read_when_the_file_shows_it(tmp_path):
    ds = _contract(tmp_path, "i1,17/335145,08/15/2026 10:00,call,inbound\n"
                             "i2,17/335145,08/02/2026 12:00,call,inbound\n")
    assert ds.report.ok and sorted(i.at.month for i in ds.interactions) == [8, 8]
    assert {i.at.day for i in ds.interactions} == {2, 15}


def test_contract_refuses_slashed_dates_nothing_decides(tmp_path):
    ds = _contract(tmp_path, "i1,17/335145,02/08/2026 10:00,call,inbound\n"
                             "i2,17/335145,03/09/2026 12:00,call,inbound\n")
    assert not ds.report.ok and "date_order_ambiguous" in _codes(ds)


def test_contract_counts_zone_suffixes_ambiguous_numbers_and_bad_message_times(tmp_path):
    ds = _contract(tmp_path,
                   "i1,17/335145,2026-08-03T10:00:00+03:00,call,inbound\n"
                   "i2,17/335145,24300,call,inbound\n",
                   "COR-1,UM-1,2026-08-03T11:00:00Z,inbound,hello\n"
                   "COR-1,UM-2,not a time,outbound,world\n")
    codes = _codes(ds)
    assert codes["tz_stripped"] == 1 and codes["ambiguous_numeric_date"] == 1
    assert codes["bad_time"] == 1 and codes["bad_message_time"] == 1
    assert not ds.report.ok                          # the ambiguous number is an error
    assert [i.at for i in ds.interactions] == [datetime(2026, 8, 3, 10, 0)]


def test_duplicate_rows_of_one_contact_are_folded_and_counted(tmp_path):
    folder = tmp_path / "c"
    folder.mkdir()
    (folder / "interactions.csv").write_text(
        "interaction_id,account_ref,started_at,channel,direction,call_key,talk_seconds\n"
        "i1,17/335145,2026-08-01 10:00:00,call,,abc123abc123,\n"
        "i1,17/335145,2026-08-01 10:00:30,call,inbound,abc123abc123,240\n"   # the same call
        "i2,17/335145,2026-08-01 12:00:00,call,inbound,abc123abc124,\n"
        "i3,17/335145,2026-08-01 12:00:10,call,inbound,,\n",                 # no id: kept
        encoding="utf-8")
    ds = import_contract(folder).dataset
    assert _codes(ds).get("duplicate_contacts") == 1
    assert len(ds.interactions) == 3 and ds.report.counts.returns == 2
    first = min(ds.interactions, key=lambda i: i.at)
    assert first.direction == "inbound" and first.talk_seconds == 240   # taken from the twin


def test_a_manifest_declares_date_order_data_end_and_holidays(tmp_path):
    folder = tmp_path / "c"
    folder.mkdir()
    (folder / "manifest.yaml").write_text(
        "contract_version: 2\nangles: [vendor]\ndate_order: month_first\n"
        "data_end: 2026-08-31 23:59:59\ncoverage_from: 2026-06-25\n"
        "windows: {call_tol_sec: 30, in_after_min: 45}\n", encoding="utf-8")
    (folder / "holidays.yaml").write_text("2026: [2026-09-23, 2026-09-24]\n", encoding="utf-8")
    (folder / "interactions.csv").write_text(
        "interaction_id,account_ref,started_at,channel\n"
        "i1,17/335145,08/02/2026 10:00,call\n", encoding="utf-8")
    ds = import_contract(folder).dataset
    assert ds.report.ok and ds.contract_version == 2 and ds.source == "journey-contract-v2"
    assert ds.interactions[0].at == datetime(2026, 8, 2, 10, 0)      # month first, as declared
    assert ds.data_end == datetime(2026, 8, 31, 23, 59, 59)
    assert ds.holidays == [date(2026, 9, 23), date(2026, 9, 24)]
    from callqa.journey.importers.manifest import rules_from_manifest
    from callqa.journey.models import AtlasRules
    rules = rules_from_manifest(ds.manifest, AtlasRules())
    assert rules.coverage_from == datetime(2026, 6, 25) and rules.call_in_after == 45


def test_a_declared_layer_the_batch_lacks_is_an_error(tmp_path):
    folder = tmp_path / "c"
    folder.mkdir()
    (folder / "manifest.yaml").write_text("contract_version: 2\nangles: [vendor, atlas]\n",
                                          encoding="utf-8")
    (folder / "interactions.csv").write_text(
        "interaction_id,account_ref,started_at,channel\ni1,17/335145,2026-08-01,call\n",
        encoding="utf-8")
    ds = import_contract(folder).dataset
    assert not ds.report.ok and "declared_layer_missing" in _codes(ds)
    assert import_contract(folder, atlas=True).dataset.report.ok


def test_unknown_manifest_fields_and_versions_are_errors(tmp_path):
    folder = tmp_path / "c"
    folder.mkdir()
    (folder / "interactions.csv").write_text(
        "interaction_id,account_ref,started_at,channel\ni1,17/335145,2026-08-01,call\n",
        encoding="utf-8")
    (folder / "manifest.yaml").write_text("contract_version: 2\ncolour: blue\n", encoding="utf-8")
    assert "manifest_invalid" in _codes(import_contract(folder).dataset)
    (folder / "manifest.yaml").write_text("contract_version: 3\n", encoding="utf-8")
    assert "contract_version" in _codes(import_contract(folder).dataset)


def test_exporter_row_counts_and_stray_files_are_checked(tmp_path):
    folder = tmp_path / "c"
    folder.mkdir()
    (folder / "interactions.csv").write_text(
        "interaction_id,account_ref,started_at,channel\ni1,17/335145,2026-08-01,call\n"
        "i2,17/335145,2026-08-02,call\n", encoding="utf-8")
    (folder / "export_manifest.csv").write_text(
        "file,source_tables,rows\ninteractions.csv,T4418,3\n", encoding="utf-8")
    (folder / "ATLR_INT.csv").write_text("x\n", encoding="utf-8")
    ds = import_contract(folder).dataset
    codes = _codes(ds)
    assert codes["row_count_mismatch"] == 1 and not ds.report.ok
    assert codes["unknown_files"] == 1
    assert next(i for i in ds.report.issues if i.code == "unknown_files").examples == ["ATLR_INT.csv"]


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
