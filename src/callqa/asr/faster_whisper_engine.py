"""Real ASR engine: faster-whisper with the ivrit.ai CT2 model.

Imported lazily - this module must never be imported in mock mode.
Model weights are expected in config.asr.model_dir, populated by
scripts/download_models.py on the bank server (never downloaded here).
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path

from callqa.asr.mock_engine import _with_quality
from callqa.config import ASRConfig
from callqa.models import Speaker, Transcript, TranscriptSegment, VADSegment, Word

logger = logging.getLogger(__name__)


# Half-precision types CTranslate2 only computes on a CUDA GPU.
_GPU_ONLY_COMPUTE = {"float16", "int8_float16", "bfloat16", "int8_bfloat16"}


def _usable_compute_type(requested: str) -> str:
    """The shipped default (float16) is for the GPU server. On a machine with
    no CUDA GPU - every bank VDI desktop - CTranslate2 refuses it with
    "Requested float16 compute type, but the target device or backend do not
    support efficient float16 computation", and every call failed at ASR
    until someone found the one config line to change. int8 is what the CPU
    runs best; the substitution is logged, never silent."""
    if requested not in _GPU_ONLY_COMPUTE:
        return requested
    try:
        import ctranslate2  # lazy: comes with faster-whisper

        if ctranslate2.get_cuda_device_count() > 0:
            return requested
    except (ImportError, AttributeError, RuntimeError):
        return requested     # cannot tell; let the engine report its own error
    logger.warning("no CUDA GPU on this machine: asr.compute_type=%s needs one; "
                   "using int8 on the CPU (set asr.compute_type: int8 to silence this)",
                   requested)
    return "int8"


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
        self._load_lock = threading.Lock()
        # Intel macOS: ctranslate2 and torch each bundle libiomp5. LOAD ORDER
        # DECIDES SURVIVAL: ct2-then-torch ran a whole night of real calls;
        # torch-then-ct2 (torch arrives with silero VAD in the audio stage,
        # before the first transcribe) segfaults inside __kmp_create_worker
        # during int8 quantization even with KMP_DUPLICATE_LIB_OK set - the
        # crash report is in qa/rescued-2026-09-15/. So on darwin, if torch
        # has not been imported yet, claim the OpenMP runtime for ct2 NOW by
        # loading the model eagerly. Everywhere else the load stays deferred
        # to the first transcribe() (a resumed run never pays for it).
        import sys

        if sys.platform == "darwin":
            if "torch" not in sys.modules:
                _ = self.model
            else:
                # torch is ALREADY resident, so ct2 will load second on the
                # first transcribe - the order that segfaulted. We only survive
                # here on KMP_DUPLICATE_LIB_OK. Make the fragility visible
                # instead of silent-until-crash: if a future import pulls torch
                # in before this engine is built, this log is the warning that
                # the load order was inverted (QA round 3 F11).
                logger.warning("faster-whisper engine built with torch already "
                               "imported on darwin; relying on KMP_DUPLICATE_LIB_OK "
                               "to survive the ct2/torch OpenMP clash")

    @property
    def model(self):  # noqa: ANN201 - WhisperModel is a lazy import
        """Loaded on first transcription, not at engine construction.

        A resumed run whose ASR stage is already complete never calls
        transcribe(), and eagerly holding the CT2 model there wasted ~1 GB
        and - on Intel macOS - put ctranslate2's OpenMP runtime and torch's
        into one process for nothing, which is exactly the duplicate-libiomp
        setup that aborts or segfaults.
        """
        # Double-checked under a lock. `run --max-workers N` shares ONE Engines
        # across N threads, and this load is a multi-second C-extension call
        # between the check and the assignment: two workers arriving together
        # both loaded the model - ~1.5 GB each, on a GPU sized for one.
        if self._model is not None:
            return self._model
        with self._load_lock:
            if self._model is not None:
                return self._model
            import os
            import sys

            if sys.platform in ("darwin", "win32"):
                # Windows too: the ctranslate2 and torch wheels each ship
                # their own libiomp5md.dll, and the second to initialise in
                # one process aborts with OMP Error #15 unless told this is
                # expected. Intel macOS: ctranslate2 and torch each bundle libiomp5, and
                # once torch is resident (silero VAD runs before ASR on every
                # fresh call) importing faster_whisper aborts the interpreter
                # with OMP Error #15. This is Intel's own documented workaround;
                # the bank's Linux target never takes this branch. An operator
                # who has set the variable keeps their value.
                os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
            from faster_whisper import WhisperModel  # lazy import

            # local_files_only guards against any accidental network access.
            self._model = WhisperModel(
                str(self._model_dir),
                compute_type=_usable_compute_type(self.config.compute_type),
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
