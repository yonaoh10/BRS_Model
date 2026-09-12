"""Speaker attribution protocol (used when a recording is not per-channel)."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable

from callqa.speakers.diarization import DiarizedSegment


@runtime_checkable
class MonoDiarizer(Protocol):
    """Splits a single-channel recording into anonymous speaker segments.

    Labels are arbitrary strings; deciding which one is the banker is a
    separate problem, handled in speakers/roles.py.
    """

    def diarize(self, wav_path: Path, call_id: str) -> list[DiarizedSegment]:
        """Returns time-ordered segments, each attributed to one speaker label."""
        ...
