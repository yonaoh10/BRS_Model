"""The deterministic journey engine: timelines, rules, statistics, analysis."""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from callqa.journey.analysis import analyse
from callqa.journey.importers.atlas import attach_atlas
from callqa.journey.importers.workbook import import_workbook
from callqa.journey.models import (
    AtlasOp,
    AtlasSession,
    Commitment,
    ImportReport,
    Interaction,
    InteractionCard,
    JourneyDataset,
    Story,
)
from callqa.journey.rules import RuleSettings, apply_rules, check_promise, objective_class
from callqa.journey.stats import cluster_bootstrap_share, kaplan_meier, km_median, mcnemar_exact
from callqa.journey.timeline import build_timelines
from callqa.journey.vocab import load_taxonomy, load_units
from tests.journey_fixtures import build_atlas, build_workbook

T0 = datetime(2026, 7, 5, 10, 0)   # a Sunday


def _story_dataset(contacts: list[dict], sessions: list[AtlasSession] | None = None,
                   coverage: str = "full") -> JourneyDataset:
    inter = []
    for n, c in enumerate(contacts):
        inter.append(Interaction(interaction_id=f"i{n}", story_key="s1", at=T0 + c["at"],
                                 channel=c.get("channel", "call"), recorded=c.get("recorded", False),
                                 direction=c.get("direction", "inbound"),
                                 answer=c.get("answer", "answered"),
                                 talk_seconds=c.get("talk", 60)))
    story = Story(story_key="s1", story_no=1, branch="83", first_at=inter[0].at,
                  last_at=inter[-1].at, atlas_coverage=coverage)
    return JourneyDataset(dataset_id="ds-20260705-00000000", source="test",
                          created_at=T0, stories=[story], interactions=inter,
                          atlas_sessions=sessions or [], report=ImportReport(source="test"))


def _session(start: timedelta, minutes: float, ops: list[str], unit="109", banker="B1"):
    s0 = T0 + start
    return AtlasSession(session_id=f"x{start}", story_key="s1", banker_code=banker,
                        unit_code=unit, start=s0, end=s0 + timedelta(minutes=minutes),
                        ops=[AtlasOp(at=s0 + timedelta(minutes=i), op_code="1", op_category=c)
                             for i, c in enumerate(ops)])


# ------------------------------------------------------------------ timeline


def test_sessions_attach_to_the_contact_they_serve():
    ds = _story_dataset([{"at": timedelta(0)}, {"at": timedelta(hours=5)}],
                        sessions=[_session(timedelta(minutes=10), 5, ["info"]),
                                  _session(timedelta(hours=2), 5, ["execute"]),
                                  _session(timedelta(hours=5, minutes=5), 5, ["info"])])
    tl = build_timelines(ds)[0]
    assert [len(c.sessions) for c in tl.contacts] == [1, 1]
    assert len(tl.background) == 1           # the session at +2h served no contact


def test_objective_classes():
    ds = _story_dataset([{"at": timedelta(0)},
                         {"at": timedelta(hours=1), "answer": "abandoned", "talk": 0},
                         {"at": timedelta(hours=2), "direction": "outbound"},
                         {"at": timedelta(hours=3), "recorded": True},
                         {"at": timedelta(hours=4)}],
                        sessions=[_session(timedelta(hours=4, minutes=1), 3, ["open", "execute"])])
    tl = build_timelines(ds)[0]
    got = [objective_class(c, True) for c in tl.contacts[1:]]
    assert got == ["abandoned", "bank_initiated", "content", "answered_execute"]
    assert objective_class(tl.contacts[4], False) == "unrecorded_no_cover"


# ------------------------------------------------------------------ promises


def test_a_promise_is_kept_when_the_bank_acts_first():
    ds = _story_dataset([{"at": timedelta(0), "recorded": True},
                         {"at": timedelta(days=1), "direction": "outbound"}])
    tl = build_timelines(ds)[0]
    p = check_promise(tl, tl.contacts[0], "callback", RuleSettings())
    assert p.outcome == "kept" and p.settled_by == "bank_contact"


def test_a_promise_is_broken_when_the_customer_comes_back_first():
    ds = _story_dataset([{"at": timedelta(0), "recorded": True},
                         {"at": timedelta(hours=20)},
                         {"at": timedelta(days=1), "direction": "outbound"}])
    tl = build_timelines(ds)[0]
    p = check_promise(tl, tl.contacts[0], "callback", RuleSettings())
    assert p.outcome == "broken" and p.settled_by == "customer_returned"


def test_an_atlas_execution_keeps_a_promise():
    ds = _story_dataset([{"at": timedelta(0), "recorded": True}, {"at": timedelta(days=5)}],
                        sessions=[_session(timedelta(hours=3), 5, ["open", "execute"], unit="83")])
    tl = build_timelines(ds)[0]
    p = check_promise(tl, tl.contacts[0], "execute_action", RuleSettings())
    assert p.outcome == "kept" and p.settled_by == "atlas_execute"


def test_deadline_counts_business_days_only():
    # promise on Thursday; customer back on Sunday = within 2 business days -> broken
    thursday = datetime(2026, 7, 2, 12, 0)
    ds = _story_dataset([{"at": thursday - T0, "recorded": True},
                         {"at": thursday - T0 + timedelta(days=3)}])
    tl = build_timelines(ds)[0]
    p = check_promise(tl, tl.contacts[0], "callback", RuleSettings(callback_business_days=2))
    assert p.due.weekday() == 0          # Monday: Friday and Saturday were skipped
    assert p.outcome == "broken"


def test_no_cover_no_verdict():
    ds = _story_dataset([{"at": timedelta(0), "recorded": True}], coverage="partial")
    tl = build_timelines(ds)[0]
    p = check_promise(tl, tl.contacts[0], "callback",
                      RuleSettings(data_end=T0 + timedelta(days=30)))
    assert p.outcome == "unknown"


def test_rules_mark_bank_initiated_and_leave_content_to_the_model():
    ds = _story_dataset([{"at": timedelta(0), "recorded": True},
                         {"at": timedelta(hours=3), "direction": "outbound"},
                         {"at": timedelta(hours=6), "recorded": True}])
    tl = build_timelines(ds)[0]
    card = InteractionCard(interaction_id="i0", commitments=[
        Commitment(kind="callback", by="bank")], outcome="not_resolved")
    facts = apply_rules(tl, {"i0": card}, RuleSettings())
    assert facts.judgements["i1"].category == "bank_initiated"
    assert facts.judgements["i1"].decided_by == "rule"
    assert facts.judgements["i2"].decided_by == "none"
    assert facts.promises[0].outcome == "kept"


def test_status_inference():
    ds = _story_dataset([{"at": timedelta(0)}, {"at": timedelta(hours=1)}],
                        sessions=[_session(timedelta(hours=2), 5, ["execute"], unit="83")])
    tl = build_timelines(ds)[0]
    facts = apply_rules(tl, {}, RuleSettings(data_end=T0 + timedelta(days=20)))
    assert (facts.verdict.status, facts.verdict.status_basis) == ("closed", "inference")
    facts = apply_rules(tl, {}, RuleSettings(data_end=T0 + timedelta(days=2)))
    assert facts.verdict.status == "unclear"


# ------------------------------------------------------------------ statistics


def test_kaplan_meier_matches_a_hand_computation():
    steps = kaplan_meier([1, 2, 2, 3, 4], [True, True, False, True, False])
    surv = [(s.t, round(s.survival, 4)) for s in steps]
    assert surv == [(0.0, 1.0), (1, 0.8), (2, 0.6), (3, 0.3), (4, 0.3)]
    assert km_median(steps) == 3


def test_cluster_bootstrap_is_deterministic_and_wider_than_naive():
    clusters = [(3, 3), (0, 3), (3, 3), (0, 3), (3, 3), (0, 3), (1, 3), (2, 3)]
    a = cluster_bootstrap_share(clusters)
    assert a == cluster_bootstrap_share(clusters)
    assert a.p == pytest.approx(12 / 24)
    from callqa.reporting.executive.stats import proportion_ci
    naive = proportion_ci(12, 24)
    assert (a.high - a.low) > (naive.high - naive.low)


def test_mcnemar():
    assert mcnemar_exact(0, 0) == 1.0
    assert mcnemar_exact(10, 0) == pytest.approx(0.001953125)


# ------------------------------------------------------------------ analysis


def test_analysis_over_the_fixture_workbook(tmp_path):
    xlsx, zip_path = build_workbook(tmp_path / "in")
    ds = import_workbook(xlsx, audio=zip_path).dataset
    attach_atlas(ds, build_atlas(tmp_path / "atlas"))
    analysis, facts = analyse(ds, taxonomy=load_taxonomy(), units=load_units())
    m = analysis.metrics
    assert m["stories"].value == 3 and m["returns"].value == 5
    # three stories only: every rate is computed but none is shown as a rate
    assert m["abandoned"].k == 1 and not m["abandoned"].shown
    assert m["abandoned"].preliminary
    s1 = next(s for s in analysis.stories if s.story_no == 1)
    assert s1.abandoned == 1 and s1.coverage == "full"
    assert s1.bankers == 2 and s1.crossings == 1
    assert sum(analysis.objective_classes.values()) == 5
    assert all(metric.definition_he for metric in m.values())
    # every story began on or after the first day the log holds (ATL_R02)
    assert analysis.coverage == {"full": 3}


def test_every_metric_says_what_would_make_it_wrong_where_it_matters(tmp_path):
    xlsx, zip_path = build_workbook(tmp_path / "in")
    ds = import_workbook(xlsx, audio=zip_path).dataset
    analysis, _ = analyse(ds, taxonomy=load_taxonomy(), units=load_units())
    for key in ("failure_rate", "retold", "promises_broken", "abandoned", "closed"):
        assert analysis.metrics[key].wrong_if_he, key


def test_units_config():
    units = load_units()
    assert units.kind("109", "83") == "center"
    assert units.kind("083", "83") == "own_branch"
    assert units.kind("55", "83") == "branch"
    assert units.label("136", None) == "תפעול עורפי"
