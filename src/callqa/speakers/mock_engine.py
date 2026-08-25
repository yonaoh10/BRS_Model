"""Mock mono diarizer: alternates two speakers over fixed-length slots."""

from __future__ import annotations

from pathlib import Path

from callqa.models import VADSegment


class MockMonoDiarizer:
    def diarize(self, wav_path: Path, call_id: str) -> list[tuple[int, VADSegment]]:
        # Deterministic alternating 8-second turns over a 2-minute window.
        result = []
        for i in range(15):
            start = i * 8.0
            result.append((i % 2, VADSegment(start=start, end=start + 7.0)))
        return result
