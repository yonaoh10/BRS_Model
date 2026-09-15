"""Configuration loading and validation.

A single YAML file (config/config.yaml) loaded into pydantic models.
Every field is overridable via environment variables with the CALLQA_ prefix,
nested keys separated by double underscores (e.g. CALLQA_JUDGE__BASE_URL).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlparse

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator

ENV_PREFIX = "CALLQA_"
ENV_NESTED_DELIMITER = "__"

LOOPBACK_HOSTS = {"localhost", "127.0.0.1", "::1", "[::1]", "0.0.0.0"}


def validate_endpoint(url: str, field_name: str) -> str:
    """Reject an endpoint that would send call data somewhere unintended.

    This pipeline handles recordings of bank customers. An endpoint is a
    complete egress path for a transcript or for the audio itself, and a
    mistyped or injected value is indistinguishable from a deliberate one, so
    the shape is checked rather than trusted: a real scheme, and plaintext
    only when it stays on this machine.
    """
    if not url:
        return url
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise ValueError(
            f"{field_name} must be an http:// or https:// URL, got {url!r}"
        )
    host = (parsed.hostname or "").lower()
    if not host:
        raise ValueError(f"{field_name} has no host: {url!r}")
    if parsed.scheme == "http" and host not in LOOPBACK_HOSTS:
        raise ValueError(
            f"{field_name} sends data in clear text to {host}. Use https, or "
            f"a loopback address for a server on this machine."
        )
    return url


class StrictModel(BaseModel):
    """Config sections reject unknown keys.

    A typo in config.yaml was silently ignored, so a setting the operator
    believed they had changed simply never applied.
    """

    model_config = {"extra": "forbid"}


class PathsConfig(StrictModel):
    input_dir: Path = Path("data/input")
    output_dir: Path = Path("data/output")
    models_dir: Path = Path("models")
    state_db: Path = Path("data/callqa_state.db")


class RunConfig(StrictModel):
    mock: bool = False
    force: bool = False
    max_workers: int = Field(default=1, ge=1, le=32)


class WatchConfig(StrictModel):
    poll_seconds: float = Field(default=30, gt=0)
    stable_seconds: float = Field(default=10, ge=0)
    move_processed: bool = True


class AudioConfig(StrictModel):
    target_sample_rate: int = Field(default=16000, gt=0)
    vad: Literal["silero", "energy"] = "silero"
    min_speech_ms: int = Field(default=250, ge=0)


class ASRConfig(StrictModel):
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

    @field_validator("base_url")
    @classmethod
    def _safe_base_url(cls, value: str) -> str:
        return validate_endpoint(value, "asr.base_url")

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


class SpeakersConfig(StrictModel):
    mode: Literal["auto", "stereo", "mono"] = "auto"
    banker_channel: Literal["L", "R", "from_metadata"] = "from_metadata"
    # community-1 is the current pyannote open pipeline and roughly halves the
    # two-speaker error rate of 3.1; a filesystem path loads a local pipeline
    # config instead, for an air-gapped machine.
    diarization_model: str = "pyannote/speaker-diarization-community-1"
    num_speakers: int = 2          # 0 lets the model estimate the count
    device: Literal["auto", "cpu", "cuda"] = "auto"
    exclusive: bool = True         # one speaker per moment, for word alignment
    # Diarization needs only the audio, so on a mono call it runs in a
    # separate process while ASR transcribes; on the dev machine that hides
    # ~170 s of the slower stage behind the other. A separate process because
    # ctranslate2 and torch each bundle libiomp5 and cannot share one.
    parallel_diarization: bool = True
    # Below this, who-is-who was a near coin flip and the call is held for a
    # human rather than reported as fact.
    min_role_confidence: float = 0.34


class RedactionConfig(StrictModel):
    enabled: bool = True
    ner: bool = False


class JudgeConfig(StrictModel):
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

    @field_validator("base_url")
    @classmethod
    def _safe_base_url(cls, value: str) -> str:
        return validate_endpoint(value, "judge.base_url")


class ReportingConfig(StrictModel):
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
