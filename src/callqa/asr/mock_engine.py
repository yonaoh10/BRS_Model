"""Deterministic mock ASR engine.

Returns canned Hebrew dialogs with plausible timestamps derived from the
call's VAD segments, so downstream stages behave like the real pipeline.
"""

from __future__ import annotations

from pathlib import Path

from callqa.asr.fixtures import dialog_for_call
from callqa.models import (
    Speaker,
    Transcript,
    TranscriptQuality,
    TranscriptSegment,
    VADSegment,
    Word,
)

DEFAULT_DURATION_SEC = 120.0


def _time_slots(n_lines: int, duration: float) -> list[tuple[float, float]]:
    """Evenly spaced (start, end) slots over the call, one per dialog line."""
    slot = duration / max(n_lines, 1)
    return [(round(i * slot, 2), round(i * slot + slot * 0.8, 2)) for i in range(n_lines)]


def _words_for(text: str, start: float, end: float) -> list[Word]:
    tokens = text.split()
    if not tokens:
        return []
    step = (end - start) / len(tokens)
    return [
        Word(
            word=tok,
            start=round(start + i * step, 2),
            end=round(start + (i + 1) * step, 2),
            probability=0.9,
        )
        for i, tok in enumerate(tokens)
    ]


class MockASREngine:
    """Mock ASR: same interface as the real engine, zero model dependencies."""

    name = "mock"

    def transcribe(
        self,
        wav_path: Path,
        *,
        call_id: str,
        role: Speaker | None,
        vad_segments: list[VADSegment],
    ) -> Transcript:
        dialog = dialog_for_call(call_id)
        duration = max((s.end for s in vad_segments), default=DEFAULT_DURATION_SEC)
        duration = max(duration, 30.0)
        slots = _time_slots(len(dialog), duration)

        segments: list[TranscriptSegment] = []
        for i, ((speaker, text), (start, end)) in enumerate(zip(dialog, slots, strict=True)):
            if role is not None and speaker != role:
                continue  # channel transcription: only this channel's lines
            # One deliberately low-confidence segment per call exercises the
            # transcript quality block.
            avg_logprob = -1.6 if i == len(dialog) - 2 else -0.25
            segments.append(
                TranscriptSegment(
                    speaker=speaker if role is not None else None,  # type: ignore[arg-type]
                    start=start,
                    end=end,
                    text=text,
                    words=_words_for(text, start, end),
                    avg_logprob=avg_logprob,
                )
            )
        return _with_quality(
            Transcript(call_id=call_id, language="he", engine=self.name, segments=segments),
            low_confidence_logprob=-1.0,
        )


def _with_quality(transcript: Transcript, low_confidence_logprob: float) -> Transcript:
    logprobs = [s.avg_logprob for s in transcript.segments if s.avg_logprob is not None]
    low = [
        i
        for i, s in enumerate(transcript.segments)
        if s.avg_logprob is not None and s.avg_logprob < low_confidence_logprob
    ]
    transcript.quality = TranscriptQuality(
        low_confidence_segments=low,
        mean_logprob=round(sum(logprobs) / len(logprobs), 4) if logprobs else None,
        low_confidence_ratio=round(len(low) / len(transcript.segments), 4)
        if transcript.segments
        else 0.0,
    )
    return transcript
