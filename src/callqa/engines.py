"""Engines: the dependency-injection container for all model-backed components.

Built ONCE per process from config and reused across calls - model loading is
per-process, never per-call. Real engines are imported lazily inside their
builder branches so the package imports cleanly on machines without
torch / faster-whisper / presidio / vLLM.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

from callqa.config import Config
from callqa.resources import find_config
from callqa.rubric import Rubric, load_rubric

if TYPE_CHECKING:
    from callqa.asr.base import ASREngine
    from callqa.judge.base import Judge
    from callqa.redaction import Redactor
    from callqa.speakers.base import MonoDiarizer

logger = logging.getLogger(__name__)


@dataclass
class Engines:
    """Container of long-lived engine instances shared by all calls in a process."""

    config: Config
    rubric: Rubric
    asr: ASREngine
    mono_diarizer: MonoDiarizer | None
    redactor: Redactor
    judge: Judge
    mock: bool


def build_engines(config: Config, rubric_path: str | None = None) -> Engines:
    """Build all engines once, honoring mock mode and lazy imports."""
    mock = config.run.mock
    rubric = load_rubric(find_config("rubric.yaml", rubric_path))

    if mock or config.asr.engine == "mock":
        from callqa.asr.mock_engine import MockASREngine

        asr: ASREngine = MockASREngine()
    else:
        from callqa.asr.faster_whisper_engine import FasterWhisperEngine

        asr = FasterWhisperEngine(config.asr)

    mono_diarizer: MonoDiarizer | None
    if mock:
        from callqa.speakers.mock_engine import MockMonoDiarizer

        mono_diarizer = MockMonoDiarizer()
    else:
        # Built lazily on first mono file: pyannote is gated and only needed
        # if recordings turn out to be mono.
        from callqa.speakers.pyannote_engine import LazyPyannoteDiarizer

        mono_diarizer = LazyPyannoteDiarizer(config.speakers)

    from callqa.redaction import build_redactor

    redactor = build_redactor(config.redaction, mock=mock,
                              models_dir=config.paths.models_dir)

    if mock or config.judge.engine == "mock":
        from callqa.judge.mock_judge import MockJudge

        judge: Judge = MockJudge()
    else:
        from callqa.judge.vllm_judge import VLLMJudge

        judge = VLLMJudge(config.judge)
        judge.check_connectivity()

    logger.info(
        "engines built: asr=%s judge=%s mock=%s",
        type(asr).__name__,
        type(judge).__name__,
        mock,
    )
    return Engines(
        config=config,
        rubric=rubric,
        asr=asr,
        mono_diarizer=mono_diarizer,
        redactor=redactor,
        judge=judge,
        mock=mock,
    )
