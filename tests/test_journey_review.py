"""Regressions from the review of the journey feature: each test is a defect
that was found by building its failing input."""

from __future__ import annotations

import json
import os
from datetime import datetime, timedelta

import pytest

from callqa.journey.analysis import analyse
from callqa.journey.engine import DatasetLock, ProcessLocked
from callqa.journey.llm.tasks import parse_card
from callqa.journey.llm.verify import verify_evidence
from callqa.journey.models import (
    Commitment,
    Evidence,
    ImportReport,
    Interaction,
    InteractionCard,
    JourneyDataset,
    Message,
    ReturnJudgement,
    Story,
)
from callqa.journey.rules import RuleSettings, apply_rules
from callqa.journey.timeline import build_timelines
from callqa.journey.transcript_view import ContentView, Line
from callqa.journey.vocab import load_taxonomy, load_units

TAX = load_taxonomy()
UNITS = load_units()
T0 = datetime(2026, 7, 5, 9, 0)   # a Sunday


def _one(text: str, **kw) -> ContentView:
    return ContentView("i1", "call", [Line(1, "bank", text)], **kw)


# ---------------------------------------------------------------- 1. the verifier

@pytest.mark.parametrize("line,quote", [
    ("הכסף יגיע תוך 15 ימים", "יגיע תוך 5 ימים"),
    ("העמלה היא 150 שקל", "העמלה היא 15"),
    ("נחזור אליך תוך 24 שעות", "נחזור אליך תוך 2"),
])
def test_a_quote_is_matched_word_by_word(line, quote):
    assert verify_evidence(_one(line), 1, quote)[0] is None


def test_a_broken_word_is_stored_as_the_lines_own_words_or_refused():
    ev, _ = verify_evidence(_one("הבקשה לא אושרה בכלל"), 1, "א אושרה בכלל")
    assert ev is None or ev.quote == "לא אושרה בכלל"


@pytest.mark.parametrize("line,quote,stored", [
    ("אי אפשר לבטל את העסקה", "אפשר לבטל את העסקה", "אי אפשר לבטל את העסקה"),
    ("הכרטיס שלך אינו פעיל כרגע במערכת", "פעיל כרגע במערכת", "אינו פעיל כרגע במערכת"),
    ("ההלוואה איננה מאושרת עדיין", "מאושרת עדיין", "איננה מאושרת עדיין"),
    ("לא ממש עזרו לי בסניף", "עזרו לי בסניף", "לא ממש עזרו לי בסניף"),
    ("אני אבדוק ואחזור אלייך, עד מחר", "אחזור אלייך עד מחר", "ואחזור אלייך, עד מחר"),
])
def test_the_stored_quote_is_the_lines_words_with_its_negation(line, quote, stored):
    ev, _ = verify_evidence(_one(line), 1, quote)
    assert ev is not None and ev.quote == stored


def test_a_line_the_model_was_not_shown_is_not_quotable():
    view = ContentView("i1", "call", [Line(1, "bank", "שורה ראשונה ארוכה מספיק"),
                                      Line(2, "bank", "שורה שנייה ארוכה מספיק")], shown=[1])
    assert verify_evidence(view, 2, "שורה שנייה ארוכה")[0] is None
    assert verify_evidence(view, 1, "שורה ראשונה ארוכה")[0] is not None


# ---------------------------------------------------------------- 8. evidence as an object

def test_evidence_given_as_an_object_is_used_not_dropped():
    view = _one("אני אבדוק ואחזור אלייך עד מחר בצהריים")
    raw = json.dumps({
        "topic": "other", "issue_he": "", "customer_request_he": "", "outcome": "not_resolved",
        "outcome_ev": [], "commitments": [{"kind": "callback", "by": "bank", "when_he": "",
                                           "ev": {"line": 1, "quote": "ואחזור אלייך עד מחר"}}],
        "prior_contact_mentioned": False, "prior_ev": "L1", "retold": "no", "retold_ev": [],
        "banker_aware_of_history": "unclear", "redirect": "none", "frustration": 1,
        "confidence": "low"}, ensure_ascii=False)
    res = parse_card(raw, view, TAX, first=False)
    assert len(res.card.commitments) == 1
    assert res.failed_quotes and "prior_ev" in res.card.problems   # a string is reported


# ---------------------------------------------------------------- 3. Kaplan-Meier censoring

def _dataset(stories: list[list[dict]]) -> JourneyDataset:
    inter, sts = [], []
    for s_no, contacts in enumerate(stories, start=1):
        key = f"s{s_no}"
        for n, c in enumerate(contacts):
            inter.append(Interaction(interaction_id=f"{key}-{n}", story_key=key, at=c["at"],
                                     channel=c.get("channel", "call"),
                                     recorded=c.get("recorded", False),
                                     direction=c.get("direction", "inbound"),
                                     answer=c.get("answer", "answered"), talk_seconds=60,
                                     correspondence_id=c.get("cor")))
        ats = [c["at"] for c in contacts]
        sts.append(Story(story_key=key, story_no=s_no, branch="83", first_at=min(ats),
                         last_at=max(ats), atlas_coverage=contacts[0].get("cover", "full")))
    return JourneyDataset(dataset_id="ds-20260705-00000000", source="test", created_at=T0,
                          stories=sts, interactions=inter, report=ImportReport(source="test"))


def test_open_stories_are_censored_at_the_end_of_the_data():
    stories = [[{"at": T0 + timedelta(days=i)}] for i in range(10)]            # open, quiet
    stories += [[{"at": T0}, {"at": T0 + timedelta(days=5)}] for _ in range(4)]
    stories.append([{"at": T0 + timedelta(days=60)}])                            # data runs 60 days
    ds = _dataset(stories)
    closed = {f"s{n}" for n in range(11, 15)}
    from callqa.journey.models import StoryVerdict
    verdicts = {k: StoryVerdict(story_key=k, status="closed", status_basis="content")
                for k in closed}
    a, _ = analyse(ds, taxonomy=TAX, units=UNITS, verdicts=verdicts)
    at5 = [s["s"] for s in a.km if s["t"] <= 5][-1]
    assert at5 > 0.7                    # 4 of 15 closed by day 5, not 75%
    assert a.metrics["median_days_to_resolution"].value is None


# ---------------------------------------------------------------- 4. retold on its own base

def test_retold_never_exceeds_its_base():
    # each story: a recorded first call, then an OUTBOUND recorded call whose card says retold
    stories = [[{"at": T0 + timedelta(days=i), "recorded": True},
                {"at": T0 + timedelta(days=i, hours=3), "recorded": True, "direction": "outbound"}]
               for i in range(12)]
    ds = _dataset(stories)
    cards = {f"s{n}-1": InteractionCard(interaction_id=f"s{n}-1", retold="yes")
             for n in range(1, 13)}
    a, _ = analyse(ds, taxonomy=TAX, units=UNITS, cards=cards)
    m = a.metrics["retold"]
    assert (m.k or 0) <= (m.n or 0)


# ---------------------------------------------------------------- 5. promises inside a thread

def test_a_promise_in_a_thread_is_dated_by_its_message_and_kept_by_a_later_reply():
    cor = "COR-1"
    ds = _dataset([[{"at": T0, "channel": "message", "recorded": True, "cor": cor}]])
    ds.messages = [
        Message(message_id="m1", correspondence_id=cor, at=T0, direction="inbound", body="שאלה"),
        Message(message_id="m2", correspondence_id=cor, at=T0 + timedelta(days=4),
                direction="outbound", body="נבדוק ונחזור אליך"),
        Message(message_id="m3", correspondence_id=cor, at=T0 + timedelta(days=5),
                direction="outbound", body="בדקנו, הכול תקין")]
    tl = build_timelines(ds)[0]
    card = InteractionCard(interaction_id="s1-0", commitments=[Commitment(
        kind="callback", by="bank", evidence=Evidence(interaction_id="s1-0", line=2,
                                                      quote="נבדוק ונחזור אליך"))])
    facts = apply_rules(tl, {"s1-0": card}, RuleSettings(data_end=T0 + timedelta(days=30)))
    p = facts.promises[0]
    assert p.made_at == T0 + timedelta(days=4)
    assert p.outcome == "kept" and p.settled_by == "bank_contact"


# ---------------------------------------------------------------- 9. the lock

def test_an_empty_lock_file_is_a_lock_being_taken(tmp_path):
    (tmp_path / "process.lock").write_text("", encoding="utf-8")
    with pytest.raises(ProcessLocked):
        with DatasetLock(tmp_path):
            pass


def test_only_the_owner_removes_the_lock(tmp_path):
    lock = DatasetLock(tmp_path)
    with lock:
        # someone else's lock now in its place (e.g. after a stale takeover)
        (tmp_path / "process.lock").write_text(str(os.getpid() + 1), encoding="utf-8")
    assert (tmp_path / "process.lock").exists()


def test_a_dead_holder_is_replaced(tmp_path):
    (tmp_path / "process.lock").write_text("999999999", encoding="utf-8")
    old = (tmp_path / "process.lock").stat().st_mtime - 3600
    os.utime(tmp_path / "process.lock", (old, old))
    with DatasetLock(tmp_path):
        assert (tmp_path / "process.lock").read_text(encoding="utf-8") == str(os.getpid())
    assert not (tmp_path / "process.lock").exists()


# ---------------------------------------------------------------- 10. small bases

def test_small_batches_show_no_rates():
    ds = _dataset([[{"at": T0}, {"at": T0 + timedelta(hours=1)}],
                   [{"at": T0 + timedelta(days=1)}]])
    a, _ = analyse(ds, taxonomy=TAX, units=UNITS)
    from callqa.journey.findings import build_opinion
    op = build_opinion(a, TAX)
    assert not any(f.key == "visibility" for f in op.findings)
    assert not a.metrics["median_days_to_resolution"].shown


# ---------------------------------------------------------------- judgement base of B

def test_rule_judgements_are_kept_for_outbound_returns():
    ds = _dataset([[{"at": T0, "recorded": True},
                    {"at": T0 + timedelta(hours=2), "recorded": True, "direction": "outbound"}]])
    judged = {"s1-1": ReturnJudgement(interaction_id="s1-1", category="unclosed_loop",
                                      decided_by="llm", basis="content")}
    a, facts = analyse(ds, taxonomy=TAX, units=UNITS, judgements=judged)
    assert facts[0].judgements["s1-1"].category == "bank_initiated"
