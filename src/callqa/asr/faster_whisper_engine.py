"""Real ASR engine: faster-whisper with the ivrit.ai CT2 model.

Imported lazily - this module must never be imported in mock mode.
Model weights are expected in config.asr.model_dir, populated by
scripts/download_models.py on the bank server (never downloaded here).
"""

from __future__ import annotations

import logging
from pathlib import Path

from callqa.asr.mock_engine import _with_quality
from callqa.config import ASRConfig
from callqa.models import Speaker, Transcript, TranscriptSegment, VADSegment, Word

logger = logging.getLogger(__name__)


class FasterWhisperEngine:
    name = "faster_whisper"

    def __init__(self, config: ASRConfig) -> None:
        model_dir = Path(config.model_dir)
        if not model_dir.exists() or not any(model_dir.iterdir()):
            raise FileNotFoundError(
                f"ASR model directory is missing or empty: {model_dir}. "
                "Run scripts/download_models.py --asr on the server first."
            )
        # find_spec, not an import: merely importing faster_whisper pulls in
        # ctranslate2 and its OpenMP runtime, which must stay out of processes
        # that will also load torch (see the `model` property).
        import importlib.util

        if importlib.util.find_spec("faster_whisper") is None:
            raise ImportError(
                "faster-whisper is not installed. Install requirements-server.txt "
                "on the server (see README deployment section)."
            )
        self.config = config
        self._model_dir = model_dir
        self._model = None

    @property
    def model(self):  # noqa: ANN201 - WhisperModel is a lazy import
        """Loaded on first transcription, not at engine construction.

        A resumed run whose ASR stage is already complete never calls
        transcribe(), and eagerly holding the CT2 model there wasted ~1 GB
        and - on Intel macOS - put ctranslate2's OpenMP runtime and torch's
        into one process for nothing, which is exactly the duplicate-libiomp
        setup that aborts or segfaults.
        """
        if self._model is None:
            from faster_whisper import WhisperModel  # lazy import

            # local_files_only guards against any accidental network access.
            self._model = WhisperModel(
                str(self._model_dir),
                compute_type=self.config.compute_type,
                local_files_only=True,
            )
            logger.info("faster-whisper model loaded from %s", self._model_dir)
        return self._model

    def transcribe(
        self,
        wav_path: Path,
        *,
        call_id: str,
        role: Speaker | None,
        vad_segments: list[VADSegment],
    ) -> Transcript:
        segments_iter, _info = self.model.transcribe(
            str(wav_path),
            language=self.config.language,  # forced - never autodetect
            word_timestamps=self.config.word_timestamps,
            vad_filter=self.config.vad_filter,
        )
        segments: list[TranscriptSegment] = []
        for seg in segments_iter:
            words = [
                Word(
                    word=w.word.strip(),
                    start=float(w.start),
                    end=float(w.end),
                    probability=float(w.probability) if w.probability is not None else None,
                )
                for w in (seg.words or [])
            ]
            segments.append(
                TranscriptSegment(
                    speaker=role,
                    start=float(seg.start),
                    end=float(seg.end),
                    text=seg.text.strip(),
                    words=words,
                    avg_logprob=float(seg.avg_logprob),
                )
            )
        transcript = Transcript(
            call_id=call_id, language=self.config.language, engine=self.name, segments=segments
        )
        return _with_quality(transcript, self.config.low_confidence_logprob)
