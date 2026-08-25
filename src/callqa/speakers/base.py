"""Speaker attribution protocol (mono diarization fallback)."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable

from callqa.models import VADSegment


@runtime_checkable
class MonoDiarizer(Protocol):
    """Diarizes a mono WAV into two anonymous speakers (0 and 1)."""

    def diarize(self, wav_path: Path, call_id: str) -> list[tuple[int, VADSegment]]:
        """Returns (speaker_index, segment) tuples, time-ordered."""
        ...
