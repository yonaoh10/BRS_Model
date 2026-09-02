"""Real judge: OpenAI-compatible client against a local vLLM endpoint.

Uses stdlib urllib only (no extra dependency). The only permitted network
target at runtime is the configured local vLLM base_url. The pipeline never
launches vLLM itself; it only checks connectivity at startup.
"""

from __future__ import annotations

import json
import logging
import urllib.error
import urllib.request

from callqa.config import JudgeConfig
from callqa.judge.base import JudgeRequest

logger = logging.getLogger(__name__)


class VLLMJudgeError(RuntimeError):
    pass


class VLLMJudge:
    name = "vllm"

    def __init__(self, config: JudgeConfig) -> None:
        self.config = config
        if "<" in config.model:
            raise ValueError(
                "judge.model is still the placeholder. Set the served model id in "
                "config/config.yaml (see scripts/download_models.py --llm)."
            )

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"
        return headers

    def check_connectivity(self) -> None:
        url = f"{self.config.base_url.rstrip('/')}/models"
        req = urllib.request.Request(url, headers=self._headers(), method="GET")
        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                if resp.status != 200:
                    raise VLLMJudgeError(f"vLLM /models returned HTTP {resp.status}")
        except (urllib.error.URLError, OSError) as exc:
            raise VLLMJudgeError(
                f"vLLM endpoint unreachable at {self.config.base_url}. "
                "Start it with scripts/start_vllm.sh."
            ) from exc
        logger.info("vLLM endpoint reachable at %s", self.config.base_url)

    def complete(self, request: JudgeRequest) -> str:
        payload = {
            "model": self.config.model,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": request.system_prompt},
                {"role": "user", "content": request.user_prompt},
            ],
        }
        url = f"{self.config.base_url.rstrip('/')}/chat/completions"
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers=self._headers(),
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=600) as resp:
                body = json.loads(resp.read().decode("utf-8"))
        except (urllib.error.URLError, OSError, json.JSONDecodeError) as exc:
            raise VLLMJudgeError(f"vLLM request failed: {exc}") from exc
        try:
            return body["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise VLLMJudgeError(f"unexpected vLLM response shape: {exc}") from exc
