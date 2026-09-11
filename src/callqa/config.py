"""Configuration loading and validation.

A single YAML file (config/config.yaml) loaded into pydantic models.
Every field is overridable via environment variables with the CALLQA_ prefix,
nested keys separated by double underscores (e.g. CALLQA_JUDGE__BASE_URL).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator

ENV_PREFIX = "CALLQA_"
ENV_NESTED_DELIMITER = "__"


class PathsConfig(BaseModel):
    input_dir: Path = Path("data/input")
    output_dir: Path = Path("data/output")
    models_dir: Path = Path("models")
    state_db: Path = Path("data/callqa_state.db")


class RunConfig(BaseModel):
    mock: bool = False
    force: bool = False
    max_workers: int = Field(default=1, ge=1, le=32)


class WatchConfig(BaseModel):
    poll_seconds: float = Field(default=30, gt=0)
    stable_seconds: float = Field(default=10, ge=0)
    move_processed: bool = True


class AudioConfig(BaseModel):
    target_sample_rate: int = Field(default=16000, gt=0)
    vad: Literal["silero", "energy"] = "silero"
    min_speech_ms: int = Field(default=250, ge=0)


class ASRConfig(BaseModel):
    # faster_whisper = in-process (bank server). remote = HTTP client to a
    # cloud-hosted ASR server (DEV ONLY; see cloud/README.md). mock = fake.
    engine: Literal["faster_whisper", "remote", "mock"] = "faster_whisper"
    model_dir: str = "{models_dir}/ivrit-whisper-large-v3-turbo-ct2"
    language: str = "he"
    compute_type: str = "float16"
    word_timestamps: bool = True
    vad_filter: bool = True
    low_confidence_logprob: float = -1.0
    # --- remote engine only (dev phase); ignored by faster_whisper/mock ---
    base_url: str = ""
    api_key: str | None = None
    timeout_sec: float = 900.0

    @field_validator("language")
    @classmethod
    def _language_must_be_forced(cls, v: str) -> str:
        if not v or v == "auto":
            raise ValueError("asr.language must be an explicit language code (never autodetect)")
        return v

    @model_validator(mode="after")
    def _remote_needs_base_url(self) -> ASRConfig:
        if self.engine == "remote" and not self.base_url:
            raise ValueError("asr.engine='remote' requires asr.base_url (see cloud/README.md)")
        return self


class SpeakersConfig(BaseModel):
    mode: Literal["auto", "stereo", "mono"] = "auto"
    banker_channel: Literal["L", "R", "from_metadata"] = "from_metadata"
    pyannote_model: str = "pyannote/speaker-diarization-3.1"


class RedactionConfig(BaseModel):
    enabled: bool = True
    ner: bool = False


class JudgeConfig(BaseModel):
    engine: Literal["vllm", "mock"] = "vllm"
    base_url: str = "http://localhost:8000/v1"
    model: str = "<LLM_MODEL_ID_PLACEHOLDER>"
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    max_tokens: int = Field(default=2500, gt=0)
    n_samples: int = Field(default=1, ge=1, le=9)
    max_retries: int = Field(default=3, ge=0, le=10)
    # Transcript token budget (approx.) before the long-call chunking rule kicks in.
    max_transcript_chars: int = Field(default=24000, gt=0)
    # Optional bearer token sent as `Authorization: Bearer ...` (vLLM --api-key,
    # or any remote OpenAI-compatible endpoint). Prefer the env var
    # CALLQA_JUDGE__API_KEY over writing secrets into config.yaml.
    api_key: str | None = None


class ReportingConfig(BaseModel):
    group_comparison: Literal["median", "mean"] = "median"
    language: str = "he"


class Config(BaseModel):
    paths: PathsConfig = PathsConfig()
    run: RunConfig = RunConfig()
    watch: WatchConfig = WatchConfig()
    audio: AudioConfig = AudioConfig()
    asr: ASRConfig = ASRConfig()
    speakers: SpeakersConfig = SpeakersConfig()
    redaction: RedactionConfig = RedactionConfig()
    judge: JudgeConfig = JudgeConfig()
    reporting: ReportingConfig = ReportingConfig()

    @model_validator(mode="after")
    def _interpolate_model_dir(self) -> Config:
        # "{models_dir}" placeholder in asr.model_dir resolves against paths.models_dir.
        if "{models_dir}" in self.asr.model_dir:
            self.asr.model_dir = self.asr.model_dir.replace(
                "{models_dir}", str(self.paths.models_dir)
            )
        return self


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _env_overrides(environ: dict[str, str] | None = None) -> dict[str, Any]:
    """Collect CALLQA_SECTION__FIELD=value overrides into a nested dict."""
    environ = environ if environ is not None else dict(os.environ)
    overrides: dict[str, Any] = {}
    for raw_key, value in environ.items():
        if not raw_key.startswith(ENV_PREFIX):
            continue
        path = raw_key[len(ENV_PREFIX):].lower().split(ENV_NESTED_DELIMITER)
        node = overrides
        for part in path[:-1]:
            node = node.setdefault(part, {})
        node[path[-1]] = value
    return overrides


def load_config(
    config_path: str | Path | None = None,
    cli_overrides: dict[str, Any] | None = None,
) -> Config:
    """Load config.yaml, apply env-var and CLI overrides, validate.

    Precedence (low to high): yaml file < CALLQA_ env vars < cli_overrides.
    """
    data: dict[str, Any] = {}
    if config_path is not None:
        path = Path(config_path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")
        loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
        if loaded is not None:
            if not isinstance(loaded, dict):
                raise ValueError(f"Config file must contain a YAML mapping: {path}")
            data = loaded
    data = _deep_merge(data, _env_overrides())
    if cli_overrides:
        data = _deep_merge(data, cli_overrides)
    return Config.model_validate(data)
