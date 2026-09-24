"""Each account's contacts and the bankers' actions, on one timeline.

A Contact is something the customer or the bank did toward the other: a
recorded call, a call nobody recorded (answered or abandoned), a written
correspondence. A banker session (Atlas) is attached to the contact it served
or left as work with no documented contact, exactly as the bank's Atlas
project does it (ATL_R01 block 6):

The touch points are every call, every MESSAGE of a correspondence (each at
its own time and in its own direction), and a correspondence with no messages
at its own time. Each touch point has a window (the minutes are
`AtlasRules`, kept with the dataset):

    outbound call      from 20 minutes before the dial to the end of the call
    inbound call       from its start to 30 minutes after it ended
    outbound message   from 30 minutes before it was sent
    inbound message    up to 60 minutes after it arrived
    unknown direction  20 minutes before to 30 minutes after the moment itself

A session is attached to the touch point whose window its START falls in; in
several windows, to the nearest touch point, and on a tie to the earlier one.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta

from callqa.journey.models import (
    AtlasRules,
    AtlasSession,
    Interaction,
    JourneyDataset,
    Message,
    Story,
)


@dataclass
class Contact:
    interaction: Interaction
    index: int                                   # 0 = the first contact of the story
    kind: str                                    # see contact_kind()
    messages: list[Message] = field(default_factory=list)
    sessions: list[AtlasSession] = field(default_factory=list)

    @property
    def at(self) -> datetime:
        return self.interaction.at

    @property
    def end(self) -> datetime:
        talk = self.interaction.talk_seconds or 0.0
        return self.interaction.at + timedelta(seconds=talk)

    @property
    def is_return(self) -> bool:
        return self.index > 0

    @property
    def has_content(self) -> bool:
        return self.kind in ("recorded_call", "message")

    @property
    def direction(self) -> str:
        return self.interaction.direction


def contact_kind(i: Interaction) -> str:
    if i.channel == "message":
        return "message"
    if i.channel == "call":
        if i.recorded:
            return "recorded_call"
        if i.answer == "abandoned":
            return "abandoned"
        if i.answer == "answered":
            return "unrecorded_answered"
        return "unrecorded_unknown"
    return i.channel                              # branch / other


@dataclass
class StoryTimeline:
    story: Story
    contacts: list[Contact]
    sessions: list[AtlasSession]                  # every banker session of the account
    background: list[AtlasSession]                # sessions tied to no contact

    @property
    def returns(self) -> list[Contact]:
        return [c for c in self.contacts if c.is_return]

    def next_contact_after(self, moment: datetime, *, inbound_only: bool = False) -> Contact | None:
        for c in self.contacts:
            if c.at > moment and (not inbound_only or c.direction != "outbound"):
                return c
        return None


@dataclass(frozen=True)
class TouchPoint:
    at: datetime
    lo: datetime
    hi: datetime


def touch_points(c: Contact, rules: AtlasRules) -> list[TouchPoint]:
    """The moments of a contact that a banker session can serve (ATL_R01)."""
    def mins(x: float) -> timedelta:
        return timedelta(minutes=x)

    def unknown(at: datetime) -> TouchPoint:
        return TouchPoint(at, at - mins(rules.unk_before), at + mins(rules.unk_after))

    i = c.interaction
    if i.channel == "call":
        talk = timedelta(seconds=i.talk_seconds or 0.0)
        if c.direction == "outbound":
            return [TouchPoint(c.at, c.at - mins(rules.call_out_before), c.at + talk)]
        if c.direction == "inbound":
            return [TouchPoint(c.at, c.at, c.at + talk + mins(rules.call_in_after))]
        return [unknown(c.at)]
    if i.channel == "message" and c.messages:
        points = []
        for m in c.messages:
            if m.direction == "outbound":
                points.append(TouchPoint(m.at, m.at - mins(rules.msg_out_before), m.at))
            elif m.direction == "inbound":
                points.append(TouchPoint(m.at, m.at, m.at + mins(rules.msg_in_after)))
            else:
                points.append(unknown(m.at))
        return points
    return [unknown(c.at)]


def attach_sessions(contacts: list[Contact], sessions: list[AtlasSession],
                    rules: AtlasRules | None = None) -> list[AtlasSession]:
    """Attach each session to its contact; return the ones tied to none. A
    session already matched upstream (the Atlas export's own link) keeps it."""
    rules = rules or AtlasRules()
    by_id = {c.interaction.interaction_id: c for c in contacts}
    points = [(p, c) for c in contacts for p in touch_points(c, rules)]
    background = []
    for s in sessions:
        target = by_id.get(s.matched_interaction_id or "")
        if target is None:
            candidates = [(abs((s.start - p.at).total_seconds()), p.at, c.index, c)
                          for p, c in points if p.lo <= s.start <= p.hi]
            if candidates:
                target = min(candidates, key=lambda t: t[:3])[3]
        if target is None:
            background.append(s)
        else:
            target.sessions.append(s)
    return background


def build_timelines(dataset: JourneyDataset) -> list[StoryTimeline]:
    messages_of: dict[str, list[Message]] = {}
    for m in dataset.messages:
        messages_of.setdefault(m.correspondence_id.upper(), []).append(m)
    sessions_of: dict[str, list[AtlasSession]] = {}
    for s in dataset.atlas_sessions:
        sessions_of.setdefault(s.story_key, []).append(s)
    out = []
    for story in sorted(dataset.stories, key=lambda s: s.story_no):
        items = dataset.interactions_of(story.story_key)
        contacts = [Contact(interaction=i, index=n, kind=contact_kind(i),
                            messages=sorted(messages_of.get((i.correspondence_id or "").upper(), []),
                                            key=lambda m: m.at))
                    for n, i in enumerate(items)]
        sessions = sorted(sessions_of.get(story.story_key, []), key=lambda s: s.start)
        background = attach_sessions(contacts, sessions, dataset.atlas_rules)
        out.append(StoryTimeline(story=story, contacts=contacts, sessions=sessions,
                                 background=background))
    return out
