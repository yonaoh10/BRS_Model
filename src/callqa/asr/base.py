"""ASR engine protocol."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable

from callqa.models import Speaker, Transcript, VADSegment


@runtime_checkable
class ASREngine(Protocol):
    """Transcribes one WAV file (a single channel, or the mono mix)."""

    def transcribe(
        self,
        wav_path: Path,
        *,
        call_id: str,
        role: Speaker | None,
        vad_segments: list[VADSegment],
    ) -> Transcript:
        """role is the known speaker for this channel (stereo path), or None
        for a mono mix (speakers attributed later by diarization)."""
        ...
