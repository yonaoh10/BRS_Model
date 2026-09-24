"""Speaker attribution (stage 4).

Stereo path (primary): each channel was transcribed separately with a known
role - merge the two transcripts into one time-ordered DialogTranscript.

Mono path: attribute each WORD to a diarized speaker, split transcript
segments where the speaker changes (speakers/diarization.py), then decide
which of the two anonymous speakers is the banker (speakers/roles.py).
"""

from __future__ import annotations

import logging

from callqa.models import (
    DialogTranscript,
    DialogTurn,
    DiarizationQualityRecord,
    RoleSignalRecord,
    Speaker,
    Transcript,
    VADSegment,
)
from callqa.speakers.diarization import DiarizedSegment, attribute_segments
from callqa.speakers.roles import infer_roles

logger = logging.getLogger(__name__)


def merge_stereo(
    call_id: str, banker_transcript: Transcript, customer_transcript: Transcript
) -> DialogTranscript:
    """Merge two per-channel transcripts into one time-ordered dialog."""
    turns: list[DialogTurn] = []
    for role, transcript in (("banker", banker_transcript), ("customer", customer_transcript)):
        for seg in transcript.segments:
            if not seg.text.strip():
                continue
            turns.append(
                DialogTurn(
                    speaker=role,  # type: ignore[arg-type]
                    start=seg.start,
                    end=seg.end,
                    text=seg.text.strip(),
                    words=seg.words,
                )
            )
    turns.sort(key=lambda t: (t.start, t.end))
    return DialogTranscript(
        call_id=call_id, attribution_mode="stereo", role_confidence=1.0, turns=turns
    )


def assign_mono_roles(
    call_id: str,
    transcript: Transcript,
    diarized: list[DiarizedSegment],
    parts: list[tuple[int, float, float]] | None = None,
) -> DialogTranscript:
    """Build a role-labelled dialog from a mono transcript and a diarization.

    Two separate decisions, kept separate on purpose: attribution (which of the
    two anonymous speakers said each word) and roles (which one is the banker).
    Both record how they were reached, because a mono call's report rests on
    inferences a stereo call's does not.
    """
    attribution = attribute_segments(transcript.segments, diarized)
    decision = infer_roles(attribution.segments, call_id)
    conflicts = part_role_conflicts(attribution.segments, decision.banker_index, parts or [],
                                    call_id)

    turns = [
        DialogTurn(
            speaker=_speaker_for("banker" if idx == decision.banker_index else "customer"),
            start=seg.start,
            end=seg.end,
            text=seg.text.strip(),
            words=seg.words,
        )
        for idx, seg in attribution.segments
        if seg.text.strip()
    ]
    turns.sort(key=lambda t: (t.start, t.end))
    return DialogTranscript(
        call_id=call_id,
        attribution_mode="mono_diarized",
        role_confidence=decision.confidence,
        turns=turns,
        banker_index=decision.banker_index,
        role_signals=[RoleSignalRecord(**signal.as_dict()) for signal in decision.signals],
        diarization=DiarizationQualityRecord(**attribution.quality.as_dict()),
        segment_role_conflicts=conflicts,
    )


# A part must disagree this clearly before it is flagged: a short part (a
# transfer greeting) carries little evidence either way.
PART_CONFLICT_CONFIDENCE = 0.3


def part_role_conflicts(labeled: list, banker_index: int,
                        parts: list[tuple[int, float, float]], call_id: str) -> list[int]:
    """Parts of a multi-part call whose own evidence names the other speaker
    as the banker. The whole call is diarized at once, so one voice keeps one
    label across parts; a part that votes the other way is where the labels
    are least trustworthy, and its lines are marked uncertain downstream."""
    if len(parts) < 2:
        return []
    conflicts = []
    for seq, start, end in parts:
        inside = [(i, seg) for i, seg in labeled if start <= seg.start < end]
        if len({i for i, _ in inside}) < 2:
            continue
        local = infer_roles(inside, f"{call_id}#part{seq}")
        if local.banker_index != banker_index and local.confidence >= PART_CONFLICT_CONFIDENCE:
            conflicts.append(seq)
    return conflicts


def verify_stereo_roles(call_id: str, dialog: DialogTranscript) -> DialogTranscript:
    """A two-channel call whose roles no metadata states (an assembled NICE
    recording): check from what each channel says which one is the banker,
    and swap the labels when the evidence says the other channel is."""
    from callqa.models import TranscriptSegment

    labeled = [
        (0 if t.speaker == "banker" else 1,
         TranscriptSegment(speaker=t.speaker, start=t.start, end=t.end, text=t.text, words=[]))
        for t in dialog.turns
    ]
    decision = infer_roles(labeled, call_id)
    swapped = decision.banker_index == 1
    turns = dialog.turns
    if swapped:
        turns = [t.model_copy(update={"speaker": "customer" if t.speaker == "banker"
                                      else "banker"}) for t in turns]
        logger.info("call_id=%s: channels were the other way round; roles swapped", call_id)
    return dialog.model_copy(update={
        "attribution_mode": "stereo_inferred", "turns": turns,
        "role_confidence": decision.confidence, "roles_swapped": swapped,
        "role_signals": [RoleSignalRecord(**s.as_dict()) for s in decision.signals],
    })


def speaker_segments_from_dialog(
    dialog: DialogTranscript,
) -> tuple[list[VADSegment], list[VADSegment]]:
    """Derive per-role speech segments from dialog turns (mono path features)."""
    banker = [VADSegment(start=t.start, end=t.end) for t in dialog.turns if t.speaker == "banker"]
    customer = [
        VADSegment(start=t.start, end=t.end) for t in dialog.turns if t.speaker == "customer"
    ]
    return banker, customer


def _speaker_for(role: str) -> Speaker:
    return "banker" if role == "banker" else "customer"
