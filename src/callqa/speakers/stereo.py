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
) -> DialogTranscript:
    """Build a role-labelled dialog from a mono transcript and a diarization.

    Two separate decisions, kept separate on purpose: attribution (which of the
    two anonymous speakers said each word) and roles (which one is the banker).
    Both record how they were reached, because a mono call's report rests on
    inferences a stereo call's does not.
    """
    attribution = attribute_segments(transcript.segments, diarized)
    decision = infer_roles(attribution.segments, call_id)

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
        role_signals=[RoleSignalRecord(**signal.as_dict()) for signal in decision.signals],
        diarization=DiarizationQualityRecord(**attribution.quality.as_dict()),
    )


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
