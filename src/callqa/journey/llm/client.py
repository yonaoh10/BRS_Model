"""Who answers the reading tasks, and with which budget.

One setting, journey.llm.profile, moves every default:

    cpu   a small local model (llama-server on this machine): 8K context, one
          request at a time, long conversations shortened around cue phrases
    gpu   the bank's internal GPU server: 32K context, six requests at once,
          whole conversations

Explicit values under journey.llm override the profile. The endpoint and model
default to the judge's (engine: inherit), so a machine that already scores
calls needs nothing new.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from callqa.config import Config, JudgeConfig
from callqa.journey.llm.tasks import Prompt

PROFILES = {
    "cpu": {"ctx_tokens": 8192, "concurrency": 1, "transcript_mode": "compressed",
            "max_tokens": 1200},
    "gpu": {"ctx_tokens": 32768, "concurrency": 6, "transcript_mode": "full",
            "max_tokens": 2000},
}
CHARS_PER_TOKEN = 2.2          # Hebrew on the tokenizers we use, measured roughly
PROMPT_RESERVE_TOKENS = 1800   # instructions, story context, and the answer


@dataclass(frozen=True)
class Profile:
    name: str
    ctx_tokens: int
    concurrency: int
    transcript_mode: str
    max_tokens: int
    max_retries: int

    @property
    def transcript_chars(self) -> int:
        """How much transcript fits beside the instructions and the answer."""
        return int(max(1000, self.ctx_tokens - PROMPT_RESERVE_TOKENS - self.max_tokens)
                   * CHARS_PER_TOKEN)


def resolve_profile(config: Config, override: str | None = None) -> Profile:
    llm = config.journey.llm
    name = override or llm.profile
    base = PROFILES[name]
    return Profile(name=name, ctx_tokens=llm.ctx_tokens or base["ctx_tokens"],
                   concurrency=llm.concurrency or base["concurrency"],
                   transcript_mode=llm.transcript_mode or base["transcript_mode"],
                   max_tokens=base["max_tokens"], max_retries=llm.max_retries)


class ContentEngine(Protocol):
    name: str
    model: str

    def answer(self, prompt: Prompt, feedback: str | None = None) -> str:
        """The raw JSON text answering the prompt."""
        ...


class LLMContentEngine:
    """The reading tasks over the same OpenAI-compatible client as the judge."""

    name = "vllm"

    def __init__(self, judge_config: JudgeConfig) -> None:
        from callqa.judge.vllm_judge import VLLMJudge

        self.model = judge_config.model
        self._judge = VLLMJudge(judge_config)

    def check(self) -> None:
        self._judge.check_connectivity()

    def answer(self, prompt: Prompt, feedback: str | None = None) -> str:
        user = prompt.user
        if feedback:
            user += (f"\n\nהתשובה הקודמת נדחתה: {feedback}\n"
                     "תקן והחזר את ה-JSON המלא מחדש.")
        return self._judge.chat_json(prompt.system, user, prompt.schema, prompt.task)


def judge_config_for(config: Config, profile: Profile) -> JudgeConfig:
    """The judge's connection settings, with journey.llm's overrides."""
    llm = config.journey.llm
    base_url = llm.base_url
    if profile.name == "gpu" and llm.gpu_base_url:
        base_url = llm.gpu_base_url
    update = {"max_tokens": profile.max_tokens, "temperature": 0.0}
    if base_url:
        update["base_url"] = base_url
    if llm.model:
        update["model"] = llm.model
    return config.judge.model_copy(update=update)


def make_engine(config: Config, profile: Profile, *, mock: bool = False) -> ContentEngine:
    llm = config.journey.llm
    use_mock = mock or llm.engine == "mock" or (llm.engine == "inherit"
                                                and config.judge.engine == "mock")
    if use_mock:
        from callqa.journey.llm.mock import MockContentEngine
        return MockContentEngine()
    return LLMContentEngine(judge_config_for(config, profile))
