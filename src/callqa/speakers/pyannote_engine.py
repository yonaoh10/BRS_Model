"""Mono diarization via pyannote (lazy import; the model is downloaded once).

Used whenever a recording does not separate the two parties onto their own
channels, which is every recording made on a single device. The wrapper is
built eagerly and the pipeline is loaded on first use, so a stereo-only
deployment never touches pyannote at all.

TWO PYANNOTE GENERATIONS ARE SUPPORTED, because which one a machine has
depends on when it was provisioned:

- pyannote.audio 4.x with `speaker-diarization-community-1` (the default, and
  the better model: the 3.1 pipeline's published error rate on two-speaker
  audio is roughly twice community-1's). Its output object also exposes
  `exclusive_speaker_diarization`, which forces one speaker per moment and is
  built precisely for reconciling diarization against ASR word timestamps.
  That is what this pipeline needs, so it is preferred when present.
- pyannote.audio 3.x with `speaker-diarization-3.1`, whose pipeline returns an
  Annotation iterated through `itertracks`.

Both are gated on Hugging Face: the account must accept the model conditions
once, and HF_TOKEN must be set when the files are first fetched. After that
the weights live in the HF cache and the machine can run with HF_HUB_OFFLINE=1,
which is how the bank server is meant to run.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

from callqa.config import SpeakersConfig
from callqa.speakers.diarization import DiarizedSegment

logger = logging.getLogger(__name__)


class DiarizationUnavailableError(RuntimeError):
    """Raised when a mono recording arrives and diarization cannot run."""


def _resolve_device(requested: str) -> str:
    if requested != "auto":
        return requested
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


class LazyPyannoteDiarizer:
    name = "pyannote"

    def __init__(self, config: SpeakersConfig) -> None:
        self.config = config
        self._pipeline = None

    # -- loading ----------------------------------------------------------

    def _load(self):  # noqa: ANN202 - the pyannote Pipeline type is a lazy import
        if self._pipeline is not None:
            return self._pipeline
        try:
            from pyannote.audio import Pipeline
        except ImportError as exc:
            raise DiarizationUnavailableError(
                "This recording does not separate the two speakers onto different "
                "channels, so it needs diarization, but pyannote.audio is not "
                "installed. Install requirements-server.txt and run "
                "scripts/download_models.py --diarization."
            ) from exc

        model = self.config.diarization_model
        token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
        source = Path(model)
        try:
            if source.exists():
                # A local pipeline config, for an air-gapped machine.
                self._pipeline = Pipeline.from_pretrained(str(source))
            else:
                self._pipeline = _from_pretrained_with_token(Pipeline, model, token)
        except Exception as exc:  # noqa: BLE001 - surfaced as one clear message
            raise DiarizationUnavailableError(
                f"could not load the diarization model '{model}': {exc}. "
                "Accept the model conditions on Hugging Face once, set HF_TOKEN, "
                "and run scripts/download_models.py --diarization. On an offline "
                "machine, copy the Hugging Face cache across and set HF_HUB_OFFLINE=1."
            ) from exc

        device = _resolve_device(self.config.device)
        if device == "cuda":
            try:
                import torch
                self._pipeline.to(torch.device("cuda"))
            except Exception as exc:  # noqa: BLE001 - CPU is a valid outcome
                logger.warning("could not move the diarizer to GPU (%s); staying on CPU", exc)
                device = "cpu"
        logger.info("diarization pipeline loaded: model=%s device=%s", model, device)
        return self._pipeline

    # -- inference --------------------------------------------------------

    def diarize(self, wav_path: Path, call_id: str) -> list[DiarizedSegment]:
        pipeline = self._load()
        kwargs: dict[str, int] = {}
        if self.config.num_speakers:
            # Telling the model there are exactly two parties is the single
            # cheapest accuracy win available on a two-party call.
            kwargs["num_speakers"] = self.config.num_speakers
        output = pipeline(str(wav_path), **kwargs)
        segments = _segments_from_output(output, prefer_exclusive=self.config.exclusive)
        if not segments:
            logger.warning("call_id=%s: diarization returned no speech", call_id)
        logger.info("diarization done: call_id=%s segments=%d speakers=%d",
                    call_id, len(segments), len({s.label for s in segments}))
        return segments


def _from_pretrained_with_token(pipeline_cls, model: str, token: str | None):  # noqa: ANN001,ANN202
    """`token=` in pyannote.audio 4.x, `use_auth_token=` in 3.x."""
    try:
        return pipeline_cls.from_pretrained(model, token=token)
    except TypeError:
        return pipeline_cls.from_pretrained(model, use_auth_token=token)


def _segments_from_output(output, prefer_exclusive: bool = True) -> list[DiarizedSegment]:  # noqa: ANN001
    """Normalise pyannote 3.x and 4.x results into DiarizedSegment.

    4.x returns an object carrying several views of the result; the exclusive
    one assigns a single speaker to every moment, which is what word-level
    attribution wants. 3.x returns an Annotation directly.
    """
    annotation = output
    if prefer_exclusive and hasattr(output, "exclusive_speaker_diarization"):
        annotation = output.exclusive_speaker_diarization
    elif hasattr(output, "speaker_diarization"):
        annotation = output.speaker_diarization

    segments: list[DiarizedSegment] = []
    if hasattr(annotation, "itertracks"):
        for turn, _, label in annotation.itertracks(yield_label=True):
            segments.append(DiarizedSegment(str(label), float(turn.start), float(turn.end)))
    else:
        # pyannote 4.x iterates (segment, label) pairs directly.
        for item in annotation:
            turn, label = (item[0], item[-1]) if isinstance(item, tuple) else (item, "SPEAKER_00")
            segments.append(DiarizedSegment(str(label), float(turn.start), float(turn.end)))
    segments.sort(key=lambda s: (s.start, s.end))
    return segments
