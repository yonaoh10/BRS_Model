"""Speaker attribution (stage 4).

Stereo path (primary): each channel was transcribed separately with a known
role - merge the two transcripts into one time-ordered DialogTranscript.

Mono path (fallback): assign diarized speaker indices to roles using a
documented heuristic, then merge.

Role heuristic (mono): within the diarized transcript, the speaker with more
question marks plus greeting keywords in the first 30 seconds is the banker.
Rationale: bankers open the call with a scripted greeting and drive the
identification questions. role_confidence reflects how decisive the signal was.
"""

from __future__ import annotations

import logging

from callqa.models import (
    DialogTranscript,
    DialogTurn,
    Speaker,
    Transcript,
    TranscriptSegment,
    VADSegment,
)

logger = logging.getLogger(__name__)

GREETING_KEYWORDS = ["שלום", "בוקר טוב", "ערב טוב", "במה אפשר לעזור", "מדבר", "מדברת", "הגעת"]


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


def _overlap(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    return max(0.0, min(a_end, b_end) - max(a_start, b_start))


def assign_mono_roles(
    call_id: str,
    transcript: Transcript,
    diarized: list[tuple[int, VADSegment]],
) -> DialogTranscript:
    """Attach diarized speaker indices to transcript segments, then map the
    two indices to banker/customer roles via the documented heuristic."""
    labeled: list[tuple[int, TranscriptSegment]] = []
    for seg in transcript.segments:
        best_idx, best_ov = 0, -1.0
        for idx, dseg in diarized:
            ov = _overlap(seg.start, seg.end, dseg.start, dseg.end)
            if ov > best_ov:
                best_idx, best_ov = idx, ov
        labeled.append((best_idx, seg))

    # Heuristic scoring: question marks overall + greeting keywords in first 30s.
    scores = {0: 0.0, 1: 0.0}
    for idx, seg in labeled:
        scores[idx] = scores.get(idx, 0.0) + seg.text.count("?")
        if seg.start <= 30.0:
            scores[idx] += sum(2.0 for kw in GREETING_KEYWORDS if kw in seg.text)
    banker_idx = max(scores, key=lambda k: scores[k])
    total = sum(scores.values())
    role_confidence = round(scores[banker_idx] / total, 3) if total > 0 else 0.5

    turns = [
        DialogTurn(
            speaker="banker" if idx == banker_idx else "customer",
            start=seg.start,
            end=seg.end,
            text=seg.text.strip(),
            words=seg.words,
        )
        for idx, seg in labeled
        if seg.text.strip()
    ]
    turns.sort(key=lambda t: (t.start, t.end))
    logger.info(
        "mono role heuristic: call_id=%s banker_idx=%d confidence=%.2f",
        call_id, banker_idx, role_confidence,
    )
    return DialogTranscript(
        call_id=call_id,
        attribution_mode="mono_diarized",
        role_confidence=role_confidence,
        turns=turns,
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
