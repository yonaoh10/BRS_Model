"""Deciding which anonymous speaker is the banker and which is the customer.

Diarization answers "who spoke when" with arbitrary labels. Everything this
system reports about a banker - talk ratio, monologue length, interruptions,
the rubric itself - depends on getting those two labels the right way round,
and a swap does not look like an error: it produces a complete, plausible,
inverted report. So the decision is made from several independent signals, it
records which ones fired and how they voted, and it publishes how decisive the
result was rather than asserting it.

The signals are the ones the literature on call-centre role identification
relies on, adapted to Hebrew retail banking: the opening seconds carry the
most information (an agent answers with a scripted greeting and identifies the
desk), the agent drives identification and asks most of the questions, and the
customer is the party who reads out identifying numbers.

Nothing here is an LLM call. This stage runs before redaction, so the raw
transcript must not leave the process.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

from callqa.models import TranscriptSegment

logger = logging.getLogger(__name__)

OPENING_WINDOW_SEC = 45.0
CLOSING_WINDOW_SEC = 45.0

# An agent answers the phone for the bank, not for themselves.
AGENT_OPENING = (
    "הגעת ל", "הגעתם ל", "מדבר ", "מדברת ", "מוקד", "שירות לקוחות", "נציג",
    "במה אפשר לעזור", "איך אפשר לעזור", "במה אוכל לעזור", "איך אוכל לעזור",
    "השיחה מוקלטת", "השיחה הזאת מוקלטת", "לצורכי בקרת איכות", "שמי ",
)
# Only the bank asks for these.
IDENTITY_REQUEST = (
    "תעודת זהות", "תעודת הזהות", "מספר זהות", "ת.ז", 'ת"ז', "מספר חשבון",
    "לאימות", "לזהות אותך", "אזהה אותך", "נזהה אותך", "הזיהוי",
    "ארבע הספרות האחרונות", "שם האם", "תאריך לידה", "מה מספר",
)
# Service register: the side that is helping.
SERVICE_LANGUAGE = (
    "בשמחה", "אשמח", "אבדוק", "אני בודק", "אני בודקת", "אעביר אותך",
    "לסיכום", "יש עוד משהו", "אני מבין", "אני מבינה", "אני מתנצל",
    "אני מתנצלת", "נטפל", "אני אעזור", "רגע אחד",
)
AGENT_CLOSING = (
    "תודה שפנית", "תודה שפניתם", "יום נעים", "יום טוב", "נשמח לעמוד לשירותך",
    "ערב טוב ותודה", "נשמח לעזור",
)
# Same list the feature stage counts questions with, so the two agree.
HEBREW_INTERROGATIVES = ("האם", "מה", "מתי", "איך", "כמה", "למה", "איפה", "מי")

DIGIT_RUN = re.compile(r"(?<!\d)\d{5,}(?!\d)")

# Weights are ordered by how much each signal is worth on its own. The opening
# is worth most because it is scripted; talk time is not used at all, because
# in banking either party can dominate a call.
SIGNAL_WEIGHTS = {
    "opening": 3.0,
    "identity_request": 2.5,
    "identity_supply": 2.0,     # votes for the OTHER speaker
    "question_rate": 1.5,
    "service_language": 1.0,
    "closing": 1.0,
    "first_speaker": 0.5,
}


TOTAL_SIGNAL_WEIGHT = sum(SIGNAL_WEIGHTS.values())


@dataclass
class RoleSignal:
    """One piece of evidence, kept for the audit trail."""

    name: str
    weight: float
    votes_for: int          # speaker index this signal points at
    detail: str

    def as_dict(self) -> dict[str, object]:
        return {"name": self.name, "weight": self.weight,
                "votes_for": self.votes_for, "detail": self.detail}


@dataclass
class RoleDecision:
    banker_index: int
    confidence: float        # 0.0 = coin flip, 1.0 = every signal agreed
    signals: list[RoleSignal]

    @property
    def decided(self) -> bool:
        return self.confidence > 0.0


def _count_markers(text: str, markers: tuple[str, ...]) -> int:
    return sum(text.count(marker) for marker in markers)


def _question_count(text: str) -> int:
    count = text.count("?")
    for sentence in re.split(r"[.!?\n]", text):
        stripped = sentence.strip()
        if stripped.startswith(HEBREW_INTERROGATIVES):
            count += 1
    return count


def _vote(signals: list[RoleSignal], name: str, per_speaker: dict[int, float],
          invert: bool = False) -> None:
    """Record a vote for whichever speaker scores higher on this signal."""
    a, b = per_speaker.get(0, 0.0), per_speaker.get(1, 0.0)
    if a == b:
        return                                   # no information, no vote
    winner = 0 if a > b else 1
    if invert:
        winner = 1 - winner
    signals.append(RoleSignal(
        name=name,
        weight=SIGNAL_WEIGHTS[name],
        votes_for=winner,
        detail=f"speaker0={a:g} speaker1={b:g}" + (" (inverted)" if invert else ""),
    ))


def infer_roles(
    labeled: list[tuple[int, TranscriptSegment]], call_id: str = "-"
) -> RoleDecision:
    """Decide which speaker index is the banker.

    `labeled` is the output of speaker attribution: (speaker_index, segment)
    pairs covering the whole call, time-ordered.
    """
    if not labeled:
        return RoleDecision(banker_index=0, confidence=0.0, signals=[])

    indices = {idx for idx, _ in labeled}
    if len(indices) < 2:
        logger.warning("call_id=%s: only one speaker in the transcript; "
                       "roles cannot be inferred", call_id)
        return RoleDecision(banker_index=next(iter(indices)), confidence=0.0, signals=[])

    end_of_call = max(seg.end for _, seg in labeled)
    text: dict[int, str] = {0: "", 1: ""}
    opening: dict[int, str] = {0: "", 1: ""}
    closing: dict[int, str] = {0: "", 1: ""}
    words: dict[int, int] = {0: 0, 1: 0}
    for idx, seg in labeled:
        if idx not in text:
            continue
        text[idx] += " " + seg.text
        words[idx] += len(seg.text.split())
        if seg.start <= OPENING_WINDOW_SEC:
            opening[idx] += " " + seg.text
        if seg.end >= end_of_call - CLOSING_WINDOW_SEC:
            closing[idx] += " " + seg.text

    signals: list[RoleSignal] = []
    _vote(signals, "opening",
          {i: _count_markers(opening[i], AGENT_OPENING) for i in (0, 1)})
    _vote(signals, "identity_request",
          {i: _count_markers(text[i], IDENTITY_REQUEST) for i in (0, 1)})
    # The party who reads out a long run of digits is supplying identification,
    # which makes them the customer - so this signal votes the other way.
    _vote(signals, "identity_supply",
          {i: len(DIGIT_RUN.findall(text[i])) for i in (0, 1)}, invert=True)
    _vote(signals, "question_rate",
          {i: round(_question_count(text[i]) / max(words[i], 1) * 100, 2) for i in (0, 1)})
    _vote(signals, "service_language",
          {i: _count_markers(text[i], SERVICE_LANGUAGE) for i in (0, 1)})
    _vote(signals, "closing",
          {i: _count_markers(closing[i], AGENT_CLOSING) for i in (0, 1)})
    first_speaker = labeled[0][0]
    signals.append(RoleSignal(
        name="first_speaker", weight=SIGNAL_WEIGHTS["first_speaker"],
        votes_for=first_speaker,
        detail=f"speaker{first_speaker} opened the call",
    ))

    tally = {0: 0.0, 1: 0.0}
    for signal in signals:
        tally[signal.votes_for] += signal.weight
    banker_index = 0 if tally[0] >= tally[1] else 1
    # Normalised by every signal this design looks for, not only the ones that
    # happened to fire. Dividing by the fired weight would report a call
    # decided solely by "who spoke first" as fully confident, and a swapped
    # report is the one failure that looks exactly like a correct one.
    confidence = round(abs(tally[0] - tally[1]) / TOTAL_SIGNAL_WEIGHT, 3)

    logger.info(
        "call_id=%s: banker=speaker%d confidence=%.2f (%s)",
        call_id, banker_index, confidence,
        ", ".join(f"{s.name}->{s.votes_for}" for s in signals),
    )
    return RoleDecision(banker_index=banker_index, confidence=confidence, signals=signals)
