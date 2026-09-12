"""Mock diarizer: alternating two speakers, so the mono path runs with no model."""

from __future__ import annotations

from pathlib import Path

from callqa.speakers.diarization import DiarizedSegment


class MockMonoDiarizer:
    name = "mock"

    def diarize(self, wav_path: Path, call_id: str) -> list[DiarizedSegment]:
        # Deterministic alternating 8-second turns over a 2-minute window.
        return [
            DiarizedSegment(f"SPEAKER_{i % 2:02d}", i * 8.0, i * 8.0 + 7.0)
            for i in range(15)
        ]
