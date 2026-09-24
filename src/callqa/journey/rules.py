"""What can be decided without reading a word: rules over the event sequence.

Everything here is measured, not guessed, and says so:

objective_class of a return (the contact's own facts)
    abandoned              the customer called and nobody answered
    bank_initiated         the bank called or wrote (outbound)
    content                a recorded call or a correspondence: its reason is
                           read from what was said (the language model's task)
    answered_execute       an answered call with no recording, after which a
                           banker performed an operation on the account
    answered_info          ... after which a banker only looked
    answered_no_trace      ... with no banker activity behind it (full Atlas cover)
    unrecorded_no_cover    a call with no recording and no Atlas cover to read

Promises (made by the bank, found by the language model in the transcript)
are judged here: kept when the bank acted - an outbound contact, or an Atlas
operation that executes something - before the customer came back and within
`callback_business_days` Sunday-to-Thursday days; broken when the customer
came back first, or the deadline passed with the account covered; unknown
otherwise.

A story's end status is closed / open / unclear, with its basis: content
(what was said), inference (the bank executed something after the last
contact and nothing followed for `quiet_days`), or none.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta

from callqa.journey.models import (
    InteractionCard,
    PromiseCheck,
    ReturnJudgement,
    StoryVerdict,
)
from callqa.journey.timeline import Contact, StoryTimeline
from callqa.journey.timeparse import add_business_days

BANK_PROMISES = {"callback", "send_document", "execute_action", "check_and_update"}


@dataclass
class RuleSettings:
    callback_business_days: int = 2
    quiet_days: int = 7
    data_end: datetime | None = None          # the last moment the data covers


@dataclass
class StoryFacts:
    timeline: StoryTimeline
    judgements: dict[str, ReturnJudgement] = field(default_factory=dict)   # by interaction id
    promises: list[PromiseCheck] = field(default_factory=list)
    verdict: StoryVerdict | None = None


def objective_class(c: Contact, covered: bool) -> str:
    if c.kind == "abandoned":
        return "abandoned"
    if c.direction == "outbound":
        return "bank_initiated"
    if c.has_content:
        return "content"
    if c.kind in ("unrecorded_answered", "unrecorded_unknown") and covered:
        if any(s.has_execute for s in c.sessions):
            return "answered_execute"
        if c.sessions:
            return "answered_info"
        return "answered_no_trace" if c.kind == "unrecorded_answered" else "unrecorded_no_cover"
    return "unrecorded_no_cover"


def _bank_acted(tl: StoryTimeline, after: datetime, before: datetime) -> tuple[str, datetime] | None:
    """The first bank action on the account in (after, before]: an outbound
    contact, or an Atlas session with an executing operation."""
    events: list[tuple[datetime, str]] = []
    for c in tl.contacts:
        if c.direction == "outbound" and after < c.at <= before:
            events.append((c.at, "bank_contact"))
        for m in c.messages:              # the bank's replies inside a thread
            if m.direction == "outbound" and after < m.at <= before:
                events.append((m.at, "bank_contact"))
    for s in tl.sessions:
        for op in s.ops:
            if op.op_category == "execute" and after < op.at <= before:
                events.append((op.at, "atlas_execute"))
                break
    if not events:
        return None
    at, how = min(events)
    return how, at


def promise_time(made_in: Contact, evidence=None):  # noqa: ANN001, ANN201
    """When a promise was made: the end of the call; in a correspondence, the
    time of the message holding it (the view shows one line per message, in
    time order)."""
    if made_in.kind == "message" and made_in.messages:
        msgs = sorted(made_in.messages, key=lambda m: m.at)
        line = getattr(evidence, "line", None)
        if isinstance(line, int) and 1 <= line <= len(msgs):
            return msgs[line - 1].at
        return msgs[-1].at
    return made_in.end


def check_promise(tl: StoryTimeline, made_in: Contact, kind: str, settings: RuleSettings,
                  evidence=None) -> PromiseCheck:
    made_at = promise_time(made_in, evidence)
    due = add_business_days(made_at, settings.callback_business_days)
    nxt = tl.next_contact_after(made_at, inbound_only=True)
    horizon = min(due, nxt.at) if nxt else due
    acted = _bank_acted(tl, made_at, horizon)
    if acted:
        how, at = acted
        return PromiseCheck(made_in=made_in.interaction.interaction_id, kind=kind,
                            made_at=made_at, due=due, outcome="kept", settled_by=how,
                            settled_at=at, evidence=evidence)
    if nxt is not None and nxt.at <= due:
        return PromiseCheck(made_in=made_in.interaction.interaction_id, kind=kind,
                            made_at=made_at, due=due, outcome="broken",
                            settled_by="customer_returned", settled_at=nxt.at, evidence=evidence)
    covered = tl.story.atlas_coverage == "full"
    if covered and settings.data_end is not None and settings.data_end >= due:
        return PromiseCheck(made_in=made_in.interaction.interaction_id, kind=kind,
                            made_at=made_at, due=due, outcome="broken", settled_by="deadline",
                            settled_at=due, evidence=evidence)
    return PromiseCheck(made_in=made_in.interaction.interaction_id, kind=kind, made_at=made_at,
                        due=due, outcome="unknown", evidence=evidence)


def _inferred(c: Contact, prev_broken: bool, same_day_index: int, unit_changed: bool) -> str | None:
    """A category the event sequence suggests for a return with no content.
    Always shown as an inference, never counted as a content classification."""
    if c.kind == "abandoned":
        return None
    if prev_broken:
        return "unclosed_loop"
    if same_day_index >= 2 and unit_changed:
        return "excessive_runaround"
    return None


def apply_rules(tl: StoryTimeline, cards: dict[str, InteractionCard], settings: RuleSettings
                ) -> StoryFacts:
    facts = StoryFacts(timeline=tl)
    covered = tl.story.atlas_coverage == "full"

    # promises the bank made in each contact with content
    for c in tl.contacts:
        card = cards.get(c.interaction.interaction_id)
        if card is None:
            continue
        for commitment in card.commitments:
            if commitment.by == "bank" and commitment.kind in BANK_PROMISES:
                facts.promises.append(check_promise(tl, c, commitment.kind, settings,
                                                    commitment.evidence))
    broken_before: dict[str, bool] = {}
    for c in tl.contacts:
        broken_before[c.interaction.interaction_id] = any(
            p.outcome == "broken" and p.settled_by == "customer_returned"
            and p.settled_at == c.at for p in facts.promises)

    # a return's facts; the content category is left to the language model
    day_counts: dict[str, int] = {}
    last_units: set[str] = set()
    for c in tl.contacts:
        day = c.at.date().isoformat()
        day_counts[day] = day_counts.get(day, 0) + 1
        units = {s.unit_code for s in c.sessions}
        unit_changed = bool(units and last_units and units != last_units)
        if units:
            last_units = units
        if not c.is_return:
            continue
        oc = objective_class(c, covered)
        if oc == "bank_initiated":
            judgement = ReturnJudgement(interaction_id=c.interaction.interaction_id,
                                        category="bank_initiated", basis="fact",
                                        decided_by="rule", objective_class=oc)
        elif oc == "content":
            judgement = ReturnJudgement(interaction_id=c.interaction.interaction_id,
                                        category="unclassifiable", basis="none",
                                        decided_by="none", objective_class=oc)
        else:
            judgement = ReturnJudgement(
                interaction_id=c.interaction.interaction_id, category="unclassifiable",
                basis="fact", decided_by="rule", objective_class=oc,
                inferred_category=_inferred(c, broken_before[c.interaction.interaction_id],
                                            day_counts[day], unit_changed))
        facts.judgements[c.interaction.interaction_id] = judgement

    facts.verdict = infer_status(tl, cards, facts.promises, settings)
    return facts


def infer_status(tl: StoryTimeline, cards: dict[str, InteractionCard],
                 promises: list[PromiseCheck], settings: RuleSettings) -> StoryVerdict:
    verdict = StoryVerdict(story_key=tl.story.story_key)
    last = tl.contacts[-1]
    last_card = None
    for c in reversed(tl.contacts):
        if c.interaction.interaction_id in cards:
            last_card = cards[c.interaction.interaction_id]
            break
    last_is_content = last_card is not None and last_card.interaction_id == last.interaction.interaction_id
    pending = [p for p in promises if p.outcome != "kept" and p.made_at >= last.at - timedelta(seconds=1)]
    if last_is_content and last_card.outcome == "resolved" and not pending:
        verdict.status, verdict.status_basis = "closed", "content"
        verdict.status_note_he = "הפנייה האחרונה הסתיימה בפתרון"
        return verdict
    if last_is_content and (last_card.outcome == "not_resolved" or pending):
        verdict.status, verdict.status_basis = "open", "content"
        verdict.status_note_he = ("בפנייה האחרונה נשארה הבטחה פתוחה" if pending
                                  else "הפנייה האחרונה הסתיימה בלי פתרון")
        return verdict
    executed_after = [op.at for s in tl.sessions for op in s.ops
                      if op.op_category == "execute" and op.at > last.at]
    end = settings.data_end
    if executed_after and end is not None and end - max(executed_after) >= timedelta(days=settings.quiet_days):
        verdict.status, verdict.status_basis = "closed", "inference"
        verdict.status_note_he = (f"אחרי הפנייה האחרונה בוצעה פעולה בחשבון, ולא הייתה פנייה נוספת "
                                  f"{settings.quiet_days} ימים ומעלה")
        return verdict
    if last.kind == "abandoned":
        verdict.status, verdict.status_basis = "open", "fact"
        verdict.status_note_he = "המגע האחרון הוא שיחה שננטשה"
        return verdict
    verdict.status, verdict.status_basis = "unclear", "none"
    verdict.status_note_he = "אין תוכן או פעולה מתועדת שמאפשרים לקבוע איך הסתיים"
    return verdict
