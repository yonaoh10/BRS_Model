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

from pydantic import BaseModel, Field, field_validator, model_validator

from callqa.resources import load_yaml

ENV_PREFIX = "CALLQA_"
ENV_NESTED_DELIMITER = "__"

# Not 0.0.0.0: it is what a server's "listening on" line prints, not an
# address anything can connect to (Windows refuses it outright).
LOOPBACK_HOSTS = {"localhost", "127.0.0.1", "::1", "[::1]"}


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

    @model_validator(mode="after")
    def _silero_rate(self) -> AudioConfig:
        # Silero only knows 8 and 16 kHz. Caught here, at config load, rather
        # than on the first call of a batch.
        if self.vad == "silero" and self.target_sample_rate not in (8000, 16000):
            raise ValueError(
                f"audio.vad=silero supports target_sample_rate 8000 or 16000, "
                f"not {self.target_sample_rate}"
            )
        return self


class ASRConfig(StrictModel):
    # faster_whisper = in-process on the machine running the pipeline. mock =
    # deterministic fake, for a run with no models and no GPU.
    engine: Literal["faster_whisper", "mock"] = "faster_whisper"
    model_dir: str = "{models_dir}/ivrit-whisper-large-v3-turbo-ct2"
    language: str = "he"
    compute_type: str = "float16"
    # auto = the GPU when one can actually run CTranslate2, else the CPU.
    device: Literal["auto", "cpu", "cuda"] = "auto"
    word_timestamps: bool = True
    vad_filter: bool = True
    low_confidence_logprob: float = -1.0
    # Beam search width. 5 is faster-whisper's own default; 1 (greedy) is
    # roughly twice as fast on a CPU for a small loss in accuracy.
    beam_size: int = Field(default=5, ge=1, le=10)

    @field_validator("language")
    @classmethod
    def _language_must_be_forced(cls, v: str) -> str:
        if not v or v == "auto":
            raise ValueError("asr.language must be an explicit language code (never autodetect)")
        return v


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
    # separate process while ASR transcribes; measured on the 6-core dev
    # machine this takes the pair from ~401 s sequential to ~347 s. A separate
    # process because ctranslate2 and torch each bundle libiomp5 and cannot
    # share one.
    parallel_diarization: bool = True
    # Below this, who-is-who was a near coin flip and the call is held for a
    # human rather than reported as fact.
    min_role_confidence: float = 0.34


class RedactionConfig(StrictModel):
    enabled: bool = True
    # Hebrew NER for person names (DictaBERT-NER, downloaded separately). Costs
    # a model load and inference time per call; catches names nobody asked for,
    # which the question-driven rules cannot.
    ner: bool = False
    # Presidio, layered on top of the built-in recognizers. OFF by default, and
    # the default is the important part: constructing presidio's AnalyzerEngine
    # loads a spaCy pipeline, and presidio DOWNLOADS that model when it is
    # missing - an outbound network call, at runtime, on a machine that is
    # supposed to be air-gapped. It also adds almost nothing here: the built-in
    # pass already detects payment cards (Luhn), e-mail and IBAN itself, so
    # presidio contributes IP addresses and little else on Hebrew text.
    # Turn it on only on a machine where en_core_web_lg is already installed.
    presidio: bool = False


class JudgeConfig(StrictModel):
    # "vllm" names the WIRE PROTOCOL, not a required product: the judge is an
    # OpenAI-compatible /v1/chat/completions client and nothing more. Any server
    # that speaks it works - vLLM, TGI, llama.cpp's server, or an internal
    # gateway. The pipeline never launches, manages or assumes a specific
    # server; it only calls `base_url` and checks that it answers.
    engine: Literal["vllm", "mock"] = "vllm"
    base_url: str = "http://127.0.0.1:8000/v1"
    model: str = "<LLM_MODEL_ID_PLACEHOLDER>"
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    max_tokens: int = Field(default=2500, gt=0)
    n_samples: int = Field(default=1, ge=1, le=9)
    max_retries: int = Field(default=3, ge=0, le=10)
    # Per-request wall-clock timeout, seconds. A hung/slow endpoint otherwise
    # ties up a worker for (max_retries+1) x this; make it an operator knob
    # rather than a hardcoded 600 so a stuck local vLLM can be given up on fast.
    request_timeout_sec: float = Field(default=600.0, gt=0)
    # Transcript token budget (approx.) before the long-call chunking rule kicks in.
    max_transcript_chars: int = Field(default=24000, gt=0)
    # A judge endpoint on the PUBLIC internet is refused unless this is set:
    # redacted transcripts are still customer conversations, and a mistyped or
    # injected base_url must not quietly send them to a hosted API. Checked
    # when the connection is first made (the name is resolved then).
    allow_public_endpoint: bool = False
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


class MonitoringConfig(StrictModel):
    """Knobs for the ops layer (run records, drift). Additive; the pipeline core
    never reads these."""

    # On-prem cost is machine time, so a run's cost is estimated as GPU-hours ×
    # this rate. 0 (the default) means "report GPU-hours, no currency figure".
    gpu_cost_per_hour: float = Field(default=0.0, ge=0.0)


class RetentionConfig(StrictModel):
    """Lifecycle of RAW, PII-bearing artifacts (raw transcripts + raw audio).
    Derived non-PII (redacted transcripts, scores, reports) is always kept."""

    raw_days: int = Field(default=90, ge=1)


class JourneyAtlasConfig(StrictModel):
    # A contact in the Atlas export is the dataset's contact when the ids match
    # and the times are within this many seconds (the bank's calls table and
    # the vendor workbook agree to the second in 460 of 504 calls, and within
    # a minute in all of them).
    join_tolerance_sec: float = Field(default=60.0, ge=0.0)


class JourneyNMFConfig(StrictModel):
    # Which NICE stream is the banker when a recording carries two:
    # auto = decide from what is said in each (opening, identification).
    banker_stream: Literal["auto", "first", "second"] = "auto"
    # A codec code the parser does not know, mapped by hand: {12: "alaw"}.
    codec_overrides: dict[int, str] = Field(default_factory=dict)


class JourneyLLMConfig(StrictModel):
    # cpu = a small local model (llama-server on this machine); gpu = the
    # bank's internal GPU server. One setting moves every default below.
    profile: Literal["cpu", "gpu"] = "cpu"
    # inherit = the judge's engine/endpoint/model; mock = deterministic.
    engine: Literal["inherit", "vllm", "mock"] = "inherit"
    base_url: str | None = None
    gpu_base_url: str | None = None
    model: str | None = None
    ctx_tokens: int | None = Field(default=None, ge=2048)
    concurrency: int | None = Field(default=None, ge=1, le=32)
    transcript_mode: Literal["compressed", "full"] | None = None
    memo: bool | None = None
    max_retries: int = Field(default=2, ge=0, le=5)

    @field_validator("base_url", "gpu_base_url")
    @classmethod
    def _safe_urls(cls, value: str | None) -> str | None:
        return validate_endpoint(value, "journey.llm endpoint") if value else value


class JourneyConfig(StrictModel):
    """Customer journeys (repeat contacts): `callqa journey ...`."""

    # Where datasets and their analysis live; relative = under paths.output_dir.
    root: Path = Path("journey")
    taxonomy: str = "journey_taxonomy.yaml"
    units: str = "journey_units.yaml"
    lexicon: str = "journey_lexicon_he.yaml"
    # A return is any contact after the first; a second, narrower definition
    # counts only returns within this many days of the previous contact.
    return_window_days: int = Field(default=30, ge=1)
    # A bank promise ("we'll call you back") is kept when the bank acts before
    # the customer comes back and within this many Sunday-Thursday days.
    callback_business_days: int = Field(default=2, ge=1, le=30)
    # No contact for this many days after the bank's last action = settled.
    quiet_days: int = Field(default=7, ge=1)
    # Rates are shown only for at least min_rate_n cases; below min_firm_n a
    # finding is labelled preliminary.
    min_rate_n: int = Field(default=10, ge=1)
    min_firm_n: int = Field(default=30, ge=1)
    # Silence inserted between the recorded parts of one call.
    segment_gap_sec: float = Field(default=1.0, ge=0.0, le=10.0)
    # A multi-part call often changes banker; 0 = let diarization estimate
    # (clamped to 2-4).
    multi_segment_num_speakers: int = Field(default=0, ge=0, le=6)
    uncertain_word_prob: float = Field(default=0.5, ge=0.0, le=1.0)
    # Message bodies through the same redaction as transcripts, at import.
    redact_messages: bool = True
    show_call_ids: Literal["short", "full"] = "short"
    # Money per banker-minute for the cost of failed returns; 0 = hours only.
    cost_per_banker_minute: float = Field(default=0.0, ge=0.0)
    atlas: JourneyAtlasConfig = JourneyAtlasConfig()
    nmf: JourneyNMFConfig = JourneyNMFConfig()
    llm: JourneyLLMConfig = JourneyLLMConfig()


class Config(StrictModel):
    # StrictModel (not BaseModel): a MISSPELLED SECTION must fail loudly, not be
    # silently dropped. A plain BaseModel ignored e.g. CALLQA_JUGDE__BASE_URL
    # (typo'd "judge") and a mistyped top-level key in config.yaml, so the
    # operator believed they had redirected the judge endpoint and nothing
    # applied - defeating the whole point of the per-section StrictModels.
    paths: PathsConfig = PathsConfig()
    run: RunConfig = RunConfig()
    watch: WatchConfig = WatchConfig()
    audio: AudioConfig = AudioConfig()
    asr: ASRConfig = ASRConfig()
    speakers: SpeakersConfig = SpeakersConfig()
    redaction: RedactionConfig = RedactionConfig()
    judge: JudgeConfig = JudgeConfig()
    reporting: ReportingConfig = ReportingConfig()
    monitoring: MonitoringConfig = MonitoringConfig()
    retention: RetentionConfig = RetentionConfig()
    journey: JourneyConfig = JourneyConfig()

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
        # A config override is CALLQA_<section>__<field>; without the section
        # delimiter it is not one. CALLQA_DASHBOARD_TOKEN, CALLQA_ASR_API_KEY
        # (single underscore) and CALLQA_CONFIG_DIR belong to the dashboard, the
        # ASR server and the resource loader, not to Config - now that Config is
        # strict they would each be rejected as an unknown top-level key.
        if len(path) < 2:
            continue
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
        loaded = load_yaml(path)
        if loaded is not None:
            if not isinstance(loaded, dict):
                raise ValueError(f"Config file must contain a YAML mapping: {path}")
            data = loaded
    data = _deep_merge(data, _env_overrides())
    if cli_overrides:
        data = _deep_merge(data, cli_overrides)
    return Config.model_validate(data)
