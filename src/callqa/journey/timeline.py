"""Each account's contacts and the bankers' actions, on one timeline.

A Contact is something the customer or the bank did toward the other: a
recorded call, a call nobody recorded (answered or abandoned), a written
correspondence. A banker session (Atlas) is attached to the contact it served
- the one whose window it starts in - or left as work with no documented
contact. The windows are the ones the Atlas project measured:

    outbound call      from 20 minutes before the dial to the end of the call
    inbound call       from the call to 30 minutes after it ended
    outbound message   from 30 minutes before it was sent
    inbound message    up to 60 minutes after it arrived
    unknown direction  20 minutes before to 30 minutes after
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta

from callqa.journey.models import AtlasSession, Interaction, JourneyDataset, Message, Story

WINDOWS = {  # (minutes before, minutes after the end)
    ("call", "outbound"): (20, 0),
    ("call", "inbound"): (0, 30),
    ("message", "outbound"): (30, 0),
    ("message", "inbound"): (0, 60),
}
DEFAULT_WINDOW = (20, 30)


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


def _window(c: Contact) -> tuple[datetime, datetime]:
    before, after = WINDOWS.get((c.interaction.channel, c.direction), DEFAULT_WINDOW)
    return c.at - timedelta(minutes=before), c.end + timedelta(minutes=after)


def attach_sessions(contacts: list[Contact], sessions: list[AtlasSession]
                    ) -> list[AtlasSession]:
    """Attach each session to its contact; return the ones tied to none. A
    session already matched upstream (the Atlas export's own link) keeps it."""
    by_id = {c.interaction.interaction_id: c for c in contacts}
    background = []
    for s in sessions:
        target = by_id.get(s.matched_interaction_id or "")
        if target is None:
            candidates = []
            for c in contacts:
                lo, hi = _window(c)
                if lo <= s.start <= hi:
                    candidates.append((abs((s.start - c.at).total_seconds()), c.index, c))
            if candidates:
                target = min(candidates, key=lambda t: (t[0], t[1]))[2]
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
        background = attach_sessions(contacts, sessions)
        out.append(StoryTimeline(story=story, contacts=contacts, sessions=sessions,
                                 background=background))
    return out
