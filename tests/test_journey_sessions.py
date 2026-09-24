"""Banker sessions built and tied to contacts exactly as the bank's Atlas
project does (ATL_R01 block 6, ATL_R02 blocks 2-4). Every expected value
below is worked out by hand from those rules."""

from __future__ import annotations

from datetime import datetime, timedelta

from callqa.journey.importers.atlas import attach_atlas
from callqa.journey.importers.workbook import import_workbook
from callqa.journey.models import (
    AtlasOp,
    AtlasRules,
    AtlasSession,
    ImportReport,
    Interaction,
    JourneyDataset,
    Message,
    Story,
)
from callqa.journey.sessions import AtlasRow, build_sessions, unit_class
from callqa.journey.timeline import build_timelines
from callqa.journey.vocab import load_atlas_codes, load_units, normalise_op_code
from tests.journey_fixtures import build_atlas, build_workbook

T = datetime(2026, 7, 5, 10, 0)


def _row(minute: float, op: str = "11", banker: str = "B1", acc: str = "1", order: int = 0,
         unit: str = "109") -> AtlasRow:
    return AtlasRow(account=acc, at=T + timedelta(minutes=minute), banker=banker, unit=unit,
                    op=op, order=order)


def _spans(sessions):
    return [[int((r.at - T).total_seconds() // 60) for r in s] for s in sessions]


# ------------------------------------------------------------ building


def test_a_pause_of_exactly_gap_minutes_keeps_the_session():
    rows = [_row(0, "201"), _row(10), _row(40), _row(70.5)]
    # 10 -> 40 is 30 minutes: not MORE than 30, same session; 40 -> 70.5 is 30.5
    assert _spans(build_sessions(rows)) == [[0, 10, 40], [70]]


def test_the_opening_code_starts_a_session_even_mid_run():
    rows = [_row(0, "201"), _row(1), _row(2, "201"), _row(3)]
    assert _spans(build_sessions(rows)) == [[0, 1], [2, 3]]


def test_a_banker_change_starts_a_session_and_so_does_the_way_back():
    rows = [_row(0, banker="B1"), _row(1, banker="B2"), _row(2, banker="B1")]
    assert _spans(build_sessions(rows)) == [[0], [1], [2]]


def test_accounts_never_share_a_session():
    rows = [_row(0, acc="1"), _row(1, acc="2")]
    assert len(build_sessions(rows)) == 2


def test_rows_of_one_second_keep_the_export_order():
    # two bankers at the same second: the export's order decides, as in R02
    a = build_sessions([_row(0, banker="B1", order=1), _row(0, banker="B2", order=2),
                        _row(1, banker="B2", order=3)])
    b = build_sessions([_row(0, banker="B2", order=1), _row(0, banker="B1", order=2),
                        _row(1, banker="B2", order=3)])
    assert [len(s) for s in a] == [1, 2] and [len(s) for s in b] == [1, 1, 1]


def test_codes_are_compared_without_leading_zeros():
    assert normalise_op_code("035") == "35" == normalise_op_code(35)
    assert normalise_op_code("201.0") == "201" and normalise_op_code("000") == "0"
    codes = load_atlas_codes()
    assert codes["201"] == "open" and codes["990"] == "not_customer"
    assert codes["11"] == "info" and codes["970"] == "execute" and codes["1"] == "execute"
    assert sum(v == "info" for v in codes.values()) == 19
    assert sum(v == "execute" for v in codes.values()) == 36


# ------------------------------------------------------------ what a session is


def _sess(cats: list[str]) -> AtlasSession:
    return AtlasSession(session_id="x", story_key="s", banker_code="B1", unit_code="109",
                        start=T, end=T, ops=[AtlasOp(at=T, op_code="1", op_category=c)
                                             for c in cats])


def test_session_kind_is_its_strongest_operation():
    assert _sess(["open", "info", "execute"]).kind == "execute"
    assert _sess(["open", "unclassified", "info"]).kind == "info"
    assert _sess(["open", "unclassified"]).kind == "unclassified"
    assert _sess(["open"]).kind == "open"
    assert _sess(["open", "not_customer"]).kind == "open"
    assert _sess(["not_customer"]).kind == "not_customer"
    assert _sess([]).kind == "open"          # R02: a session without classified rows


def test_unit_class_centre_back_office_own_branch_other():
    units = load_units()
    assert unit_class(units, "109", "109") == "center"      # the centre wins over own
    assert unit_class(units, "136", "83") == "back"
    assert unit_class(units, "083", "83") == "own"
    assert unit_class(units, "55", "83") == "other"
    assert unit_class(units, "308", "83") == "other"        # a headquarters unit
    assert unit_class(units, "171", "171") == "own"         # a listed branch that is the account's


# ------------------------------------------------------------ tying to contacts


def _dataset(contacts, sessions, messages=()):
    inter = [Interaction(interaction_id=f"i{n}", story_key="s1", at=T + c["at"],
                         channel=c.get("channel", "call"), direction=c.get("direction", "inbound"),
                         talk_seconds=c.get("talk"), correspondence_id=c.get("cor"))
             for n, c in enumerate(contacts)]
    story = Story(story_key="s1", story_no=1, branch="83", first_at=inter[0].at,
                  last_at=inter[-1].at, atlas_coverage="full")
    return JourneyDataset(dataset_id="ds-20260705-00000000", source="t", created_at=T,
                          stories=[story], interactions=inter, messages=list(messages),
                          atlas_sessions=[AtlasSession(session_id=f"x{n}", story_key="s1",
                                                       banker_code="B1", unit_code="109",
                                                       start=T + m, end=T + m)
                                          for n, m in enumerate(sessions)],
                          report=ImportReport(source="t"))


def _linked(ds):
    tl = build_timelines(ds)[0]
    return ({s.session_id: c.index for c in tl.contacts for s in c.sessions},
            {s.session_id for s in tl.background})


def test_call_windows():
    m = timedelta(minutes=1)
    ds = _dataset([{"at": 60 * m, "direction": "outbound", "talk": 120},
                   {"at": 300 * m, "direction": "inbound", "talk": 600},
                   {"at": 600 * m, "direction": "unknown", "talk": 600}],
                  [40 * m, 39 * m, 62 * m, 300 * m, 340 * m, 341 * m, 580 * m, 630 * m, 631 * m])
    linked, bg = _linked(ds)
    # outbound: 20 before the dial .. its end (2 min of talk)
    assert linked["x0"] == 0 and "x1" in bg and linked["x2"] == 0
    # inbound: its start .. 30 after its end (10 min of talk)
    assert linked["x3"] == 1 and linked["x4"] == 1 and "x5" in bg
    # unknown direction: 20 before .. 30 after the MOMENT, not the end
    assert linked["x6"] == 2 and linked["x7"] == 2 and "x8" in bg


def test_each_message_is_its_own_touch_point():
    m = timedelta(minutes=1)
    msgs = [Message(message_id="a", correspondence_id="UM-1", at=T, direction="inbound"),
            Message(message_id="b", correspondence_id="UM-1", at=T + 24 * 60 * m,
                    direction="outbound")]
    ds = _dataset([{"at": timedelta(0), "channel": "message", "cor": "UM-1",
                    "direction": "inbound"}],
                  [30 * m, 24 * 60 * m - 10 * m, 12 * 60 * m], messages=msgs)
    linked, bg = _linked(ds)
    # the reply a day later carries its own 30-minute window
    assert linked["x0"] == 0 and linked["x1"] == 0 and "x2" in bg


def test_a_session_in_two_windows_goes_to_the_nearest_then_the_earlier():
    m = timedelta(minutes=1)
    ds = _dataset([{"at": 0 * m, "direction": "unknown"},
                   {"at": 20 * m, "direction": "unknown"}],
                  [10 * m, 16 * m])
    linked, _ = _linked(ds)
    assert linked["x0"] == 0          # 10 from both: the earlier contact
    assert linked["x1"] == 1          # 16 vs 4


def test_windows_follow_the_dataset_rules():
    m = timedelta(minutes=1)
    ds = _dataset([{"at": 0 * m, "direction": "inbound", "talk": 0}], [40 * m])
    assert _linked(ds)[1] == {"x0"}
    ds.atlas_rules = AtlasRules(call_in_after=45)
    assert _linked(ds)[0] == {"x0": 0}


# ------------------------------------------------------------ import


def _rows_only(folder):
    atlas = build_atlas(folder)
    (atlas / "ATLR_SESS.csv").unlink()
    (atlas / "ATLR_CODECAT.csv").unlink()
    return atlas


def test_sessions_are_built_from_log_rows_when_the_export_has_none(tmp_path):
    xlsx, zip_path = build_workbook(tmp_path / "in")
    ds = import_workbook(xlsx, audio=zip_path).dataset
    attach_atlas(ds, _rows_only(tmp_path / "atlas"))
    assert ds.report.ok and ds.atlas_rules.sessions_from == "rows"
    a, b = ds.atlas_sessions
    assert (a.banker_code, a.unit_code, a.n_ops, a.first_op) == ("B0001", "109", 3, "201")
    assert a.kind == "info" and a.minutes == 6
    assert (b.banker_code, b.unit_code, b.kind) == ("B0002", "83", "execute")
    assert a.session_id.endswith("-1") and b.session_id.endswith("-2")


def test_a_peek_is_one_row_of_the_opening_code(tmp_path):
    xlsx, zip_path = build_workbook(tmp_path / "in")
    ds = import_workbook(xlsx, audio=zip_path).dataset
    atlas = _rows_only(tmp_path / "atlas")
    with (atlas / "ATLR_ROWS.csv").open("a", encoding="utf-8") as f:
        f.write("6,1,05JUL2026:09:00:00,109,B0003,201,מצב לקוח\n")
    attach_atlas(ds, atlas)
    peeks = [s for s in ds.atlas_sessions if s.peek]
    assert len(peeks) == 1 and peeks[0].banker_code == "B0003" and peeks[0].kind == "open"


def test_exported_sessions_that_the_rows_do_not_rebuild_are_reported(tmp_path):
    xlsx, zip_path = build_workbook(tmp_path / "in")
    ds = import_workbook(xlsx, audio=zip_path).dataset
    atlas = build_atlas(tmp_path / "atlas")
    text = (atlas / "ATLR_SESS.csv").read_text(encoding="utf-8")
    (atlas / "ATLR_SESS.csv").write_text(text.replace("B0002,83", "B0009,83"), encoding="utf-8")
    attach_atlas(ds, atlas)
    issue = next(i for i in ds.report.issues if i.code == "atlas_sessions_differ")
    assert issue.count == 1 and ds.atlas_rules.sessions_from == "export"


def test_coverage_from_is_a_day(tmp_path):
    xlsx, zip_path = build_workbook(tmp_path / "in")
    ds = import_workbook(xlsx, audio=zip_path).dataset
    attach_atlas(ds, build_atlas(tmp_path / "atlas"),
                 rules=AtlasRules(coverage_from=datetime(2026, 7, 5, 18, 0)))
    by_no = {s.story_no: s.atlas_coverage for s in ds.stories}
    # story 1 began 2.7 (partial); story 2 on the day itself, before 18:00 (full)
    assert by_no[1] == "partial" and by_no[2] == "full" and by_no[3] == "full"


# ------------------------------------------------------------ the two levels


def _one_story():
    """One story (branch 83), worked out by hand:
         10:00 inbound recorded call, 5 min, two files
         12:00 abandoned call
         15:00 outbound answered call, not recorded
       sessions: 10:02 centre B1 info 4' | 10:20 centre B2 info 2' |
                 13:00 own branch B3 execute 10' | 14:45 centre B1 peek |
                 next day 09:00 back office B4 execute 5'"""
    from callqa.journey.models import CallAudio, Segment
    h = timedelta(hours=1)
    m = timedelta(minutes=1)
    inter = [
        Interaction(interaction_id="c1", story_key="s1", at=T, channel="call", recorded=True,
                    direction="inbound", answer="answered", talk_seconds=300, call_key="k1",
                    call_id="call-k1"),
        Interaction(interaction_id="c2", story_key="s1", at=T + 2 * h, channel="call",
                    direction="inbound", answer="abandoned", talk_seconds=0),
        Interaction(interaction_id="c3", story_key="s1", at=T + 5 * h, channel="call",
                    direction="outbound", answer="answered", talk_seconds=120),
    ]

    def sess(n, start, minutes, unit, banker, cats, descs=()):
        ops = [AtlasOp(at=start + i * m, op_code=str(i), op_category=c,
                       description=(descs[i] if i < len(descs) else ""))
               for i, c in enumerate(cats)]
        return AtlasSession(session_id=f"s1-{n}", story_key="s1", banker_code=banker,
                            unit_code=unit, start=start, end=start + minutes * m, ops=ops,
                            n_ops=len(ops), first_op="201", peek=len(ops) == 1)
    sessions = [
        sess(1, T + 2 * m, 4, "109", "B1", ["open", "info", "info"],
             ["מצב לקוח", "תנועות", "תנועות"]),
        sess(2, T + 20 * m, 2, "109", "B2", ["open", "info"]),
        sess(3, T + 3 * h, 10, "083", "B3", ["open", "execute", "info", "unclassified", "info"],
             ["מצב לקוח", "העברה", "תנועות", "קוד 5", "פרטי חשבון"]),
        sess(4, T + 4 * h + 45 * m, 0, "109", "B1", ["open"]),
        sess(5, T + 23 * h, 5, "136", "B4", ["open", "execute"]),
    ]
    story = Story(story_key="s1", story_no=7, branch="83", first_at=T, last_at=T + 5 * h,
                  atlas_coverage="full")
    return JourneyDataset(
        dataset_id="ds-20260705-00000000", source="t", created_at=T, stories=[story],
        interactions=inter, atlas_sessions=sessions, report=ImportReport(source="t"),
        calls={"k1": CallAudio(call_key="k1", call_id="call-k1",
                               segments=[Segment(seq=1, file_name="a"),
                                         Segment(seq=2, file_name="b")])})


def test_story_sessions_by_the_r02_formulas():
    from callqa.journey.session_analysis import story_sessions
    tl = build_timelines(_one_story())[0]
    ss = story_sessions(tl, load_units())
    assert (ss.n_sess, ss.n_bankers, ss.n_uclass) == (5, 4, 3)
    assert ss.n_handoff == 3 and ss.x_cross            # c c own c back
    assert (ss.n_do_sess, ss.n_nodo_sess) == (2, 3)
    assert ss.banker_min == 21 and ss.bg_min == 15 and (ss.n_bg, ss.n_bg_br) == (2, 1)
    assert ss.hrs_to_do == 3.0 and ss.do_after and ss.n_do_after == 1
    assert ss.n_by_class == {"center": 3, "back": 1, "own": 1, "other": 0}


def test_contacts_and_sessions_rows():
    from callqa.journey.session_analysis import analyse_sessions
    ds = _one_story()
    a = analyse_sessions(ds, build_timelines(ds), load_units(), status_of={"s1": "unclear"})
    c1, c2, c3 = a.contacts
    assert (c1.int_type, c1.n_sess, c1.n_bankers, c1.banker_min, c1.segments) == ("REC", 2, 2, 6, 2)
    assert c1.has_center and not c1.has_execute and not c1.has_branch
    assert round(c1.resp_hours * 60) == 2
    assert (c2.int_type, c2.n_sess, c2.first_move, c2.resp_hours) == ("ABN", 0, "bank", 1.0)
    assert (c3.int_type, c3.n_sess, c3.resp_hours) == ("TALK", 1, None)
    rows = {s.no: s for s in a.sessions}
    assert rows[3].contact_n is None and rows[3].unit_class == "own" and rows[3].kind == "execute"
    # up to three distinct operations, never the screen opening
    assert rows[3].ops_txt == "העברה | תנועות | קוד 5"
    assert rows[1].ops_txt == "תנועות" and rows[4].peek and rows[4].contact_n == 3
    assert rows[5].after_last_contact and not rows[3].after_last_contact
    assert a.after_abandon == {"bank": 1, "customer": 0, "none": 0}
    assert a.kinds["execute"] == 2 and a.classes["center"] == 3


def test_page_of_numbers_and_tables():
    from callqa.journey.session_analysis import analyse_sessions
    ds = _one_story()
    a = analyse_sessions(ds, build_timelines(ds), load_units(), status_of={"s1": "unclear"})
    head = {h.key: h.value_txt for h in a.head}
    assert len(a.head) == 15
    assert head["covered"] == "1 / 1"
    assert head["bankers"] == "4.0 / 4.0"
    assert head["three_bankers"] == "1 מתוך 1"            # below the rate base: a count
    assert head["no_execute"] == "3 מתוך 5"
    assert head["background_branch"] == "1 מתוך 2"
    assert head["talk"] == "1 / 0 / 0 מתוך 1"
    assert head["cross"] == "1 מתוך 1 / 3.0"
    assert head["after_abandon"] == "1 / 0 / 0" and head["bank_hours"] == "1.00"
    assert head["unclear_execute"] == "1 מתוך 1"
    assert head["hours_to_execute"] == "3.0"
    assert head["multi_segment"] == "1 מתוך 1"
    assert head["segment_bankers"] == "— / 2.00"
    by_type = {r["type"]: r for r in a.by_type}
    assert by_type["REC"]["pct_center"] == 100.0 and by_type["ABN"]["pct_activity"] == 0.0
    assert a.by_status[0]["key"] == "unclear" and a.by_status[0]["pct_execute_after"] == 100.0
    assert a.units[0]["unit"] == "109" and a.units[0]["sessions"] == 3
    # the unclear story with an execution after its last contact is drawn
    assert [c["story_no"] for c in a.cases] == [7]
    rows = a.cases[0]["rows"]
    assert [r["event"] for r in rows[:3]] == ["פנייה", "אטלס", "אטלס"]
    assert all(r["banker"] in ("", "B1", "B2", "B3", "B4") for r in rows)
