"""Mono diarization via pyannote (lazy; gated model, needs HF_TOKEN on server).

Only needed when recordings turn out to be mono. The pipeline builds this
wrapper eagerly but loads the pyannote pipeline on first use, so stereo-only
deployments never touch pyannote at all.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

from callqa.config import SpeakersConfig
from callqa.models import VADSegment

logger = logging.getLogger(__name__)


class LazyPyannoteDiarizer:
    def __init__(self, config: SpeakersConfig) -> None:
        self.config = config
        self._pipeline = None

    def _load(self):  # noqa: ANN202
        if self._pipeline is None:
            try:
                from pyannote.audio import Pipeline  # lazy import
            except ImportError as exc:
                raise ImportError(
                    "pyannote.audio is not installed but a mono recording needs "
                    "diarization. Install requirements-server.txt and run "
                    "scripts/download_models.py --diarization (requires HF_TOKEN)."
                ) from exc
            token = os.environ.get("HF_TOKEN")
            self._pipeline = Pipeline.from_pretrained(
                self.config.pyannote_model, use_auth_token=token
            )
            logger.info("pyannote diarization pipeline loaded")
        return self._pipeline

    def diarize(self, wav_path: Path, call_id: str) -> list[tuple[int, VADSegment]]:
        pipeline = self._load()
        diarization = pipeline(str(wav_path), num_speakers=2)
        speaker_ids: dict[str, int] = {}
        result: list[tuple[int, VADSegment]] = []
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            idx = speaker_ids.setdefault(speaker, len(speaker_ids) % 2)
            result.append((idx, VADSegment(start=float(turn.start), end=float(turn.end))))
        result.sort(key=lambda item: item[1].start)
        return result
