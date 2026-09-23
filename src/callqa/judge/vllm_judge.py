"""Real judge: OpenAI-compatible client against a local vLLM endpoint.

Uses stdlib urllib only (no extra dependency). The only permitted network
target at runtime is the configured local vLLM base_url. The pipeline never
launches vLLM itself; it only checks connectivity at startup.
"""

from __future__ import annotations

import ipaddress
import json
import logging
import re
import socket
import time
import urllib.error
import urllib.request
from urllib.parse import urlparse

from callqa.config import LOOPBACK_HOSTS, JudgeConfig
from callqa.judge.base import JudgeRequest

logger = logging.getLogger(__name__)


class VLLMJudgeError(RuntimeError):
    pass


class _StillLoading(Exception):
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
        host = (urlparse(config.base_url).hostname or "").lower()
        # A server on this machine is never reached through a proxy. Python
        # applies the Windows system proxy (set by group policy on most bank
        # desktops) to every URL, and its "<local>" bypass only covers names
        # without a dot - so 127.0.0.1 went to the corporate proxy. Remote
        # servers follow the system's proxy settings, as the browser does.
        handlers = [urllib.request.ProxyHandler({})] if host in LOOPBACK_HOSTS else []
        self._open = urllib.request.build_opener(*handlers).open

    def _headers(self) -> dict[str, str]:
        # A real User-Agent, because CDNs in front of hosted endpoints
        # (many reverse proxies do) reject urllib's default
        # Python-urllib/x.y with a 403 before the request reaches vLLM.
        headers = {"Content-Type": "application/json", "User-Agent": "callqa-judge/1.0"}
        if self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"
        return headers

    def _refuse_public_endpoint(self) -> None:
        """Transcripts stay inside the bank's network unless told otherwise."""
        if self.config.allow_public_endpoint:
            return
        host = urlparse(self.config.base_url).hostname or ""
        try:
            addresses = {info[4][0] for info in socket.getaddrinfo(host, None)}
        except OSError:
            return            # unresolvable: the connection fails on its own
        public = sorted(a for a in addresses
                        if ipaddress.ip_address(a.split("%")[0]).is_global)
        if public:
            raise VLLMJudgeError(
                f"judge.base_url points at {host}, a public internet address ({public[0]}). "
                "Transcripts do not leave the bank's network: use the bank's own judge "
                "server, or set judge.allow_public_endpoint: true if this is really "
                "intended. Nothing was sent.")

    def check_connectivity(self, wait_for_loading: float = 180.0) -> None:
        self._refuse_public_endpoint()
        url = f"{self.config.base_url.rstrip('/')}/models"
        deadline = time.monotonic() + wait_for_loading
        while True:
            try:
                self._models_listing(url)
                return
            except _StillLoading:
                if time.monotonic() >= deadline:
                    raise VLLMJudgeError(
                        f"the judge server at {self.config.base_url} is still loading its "
                        f"model after {wait_for_loading:.0f}s. Wait for it to finish and "
                        "run again.") from None
                logger.info("judge server is loading its model; waiting ...")
                time.sleep(5)

    def _models_listing(self, url: str) -> None:
        req = urllib.request.Request(url, headers=self._headers(), method="GET")
        try:
            with self._open(req, timeout=10) as resp:
                if resp.status != 200:
                    raise VLLMJudgeError(f"judge /models returned HTTP {resp.status}")
                listing = resp.read()
        except urllib.error.HTTPError as exc:
            # Caught BEFORE URLError, which it subclasses. An HTTP status means
            # the server is up and answered; reporting it as "unreachable, start
            # it" sent the operator to restart a server that was running fine
            # and was simply refusing a missing or wrong API key.
            if exc.code == 503:
                # llama-server answers 503 "Loading model" until its model is
                # in memory - a minute or two for a 4 GB file an antivirus is
                # scanning. That is not a wrong URL.
                raise _StillLoading() from exc
            if exc.code in (401, 403):
                raise VLLMJudgeError(
                    f"the judge endpoint at {self.config.base_url} is running but "
                    f"refused the request (HTTP {exc.code}): the API key is missing "
                    "or wrong. Set CALLQA_JUDGE__API_KEY to the key the server was "
                    "started with."
                ) from exc
            raise VLLMJudgeError(
                f"the judge endpoint at {self.config.base_url} answered HTTP "
                f"{exc.code} to GET /models; check that judge.base_url ends in /v1."
            ) from exc
        except (urllib.error.URLError, OSError) as exc:
            raise VLLMJudgeError(
                f"judge endpoint unreachable at {self.config.base_url}: nothing is "
                "listening there. Start your model server (scripts/start_llama_server.py "
                "and scripts/start_vllm.sh are examples) and check judge.base_url."
            ) from exc
        self._check_it_serves_our_model(listing)
        logger.info("vLLM endpoint reachable at %s", self.config.base_url)

    def _check_it_serves_our_model(self, listing: bytes) -> None:
        """Refuse a server that is not serving the configured model.

        Before any transcript is sent. On a multi-session Windows host the
        loopback interface is shared by everyone logged in: when a colleague's
        judge already holds the port, ours cannot start, and the pipeline would
        otherwise send this user's calls to theirs. The model a server reports
        is the file it was started with, which differs. Compared by the last
        path component, so a relative and an absolute spelling of one file
        match; a server that lists nothing is not second-guessed.
        """
        try:
            ids = [str(m.get("id", "")) for m in json.loads(listing).get("data", [])]
        except (ValueError, AttributeError, TypeError):
            return
        ids = [i for i in ids if i]
        if not ids:
            return

        def leaf(name: str) -> str:
            return re.split(r"[\\/]", name.rstrip("\\/"))[-1].casefold()

        if leaf(self.config.model) not in {leaf(i) for i in ids}:
            raise VLLMJudgeError(
                f"the judge server at {self.config.base_url} serves {ids}, not "
                f"judge.model '{self.config.model}'. It may be another user's server "
                "on the same machine, or judge.model is out of date. Nothing was sent."
            )

    @staticmethod
    def _scorecard_schema(dimension_ids: list[str]) -> dict:
        """JSON schema enforced by vLLM's structured output (xgrammar).

        `json_object` mode only guarantees syntax, and models at the 7-14B
        tier were observed breaking even that (an unescaped '"' inside a
        value derails guided decoding), inventing extra keys, and moving
        top-level fields into `scores`. Pinning the full shape removes every
        layout failure mode; only the CONTENT of strings is left to the
        model, and evidence verification already polices that.
        """
        # Length caps are part of the grammar, not a polite request: on the
        # first real call dictalm2.0 spent the whole 4000-token budget on
        # reasoning prose across six retries and never finished the JSON.
        # Bounded strings and arrays make the complete scorecard physically
        # fit inside the model's context budget.
        evidence = {
            "type": "object",
            "properties": {
                # minLength too: dictalm quoted the call's opening grunt
                # ("אה?") as evidence for three dimensions across six
                # retries; a quote below the verifier's minimum is now
                # unrepresentable rather than politely discouraged.
                "quote": {"type": "string", "minLength": 12, "maxLength": 160},
                "timestamp": {"type": "string", "pattern": "^[0-9]{1,4}:[0-5][0-9]$"},
                "speaker": {"type": "string", "enum": ["banker", "customer"]},
            },
            "required": ["quote", "timestamp", "speaker"],
            "additionalProperties": False,
        }
        dimension = {
            "type": "object",
            "properties": {
                "score": {"type": "integer", "minimum": 1, "maximum": 5},
                "reasoning_he": {"type": "string", "maxLength": 400},
                "evidence": {"type": "array", "items": evidence,
                             "minItems": 1, "maxItems": 2},
            },
            "required": ["score", "reasoning_he", "evidence"],
            "additionalProperties": False,
        }
        return {
            "type": "object",
            "properties": {
                "scores": {
                    "type": "object",
                    "properties": dict.fromkeys(dimension_ids, dimension),
                    "required": list(dimension_ids),
                    "additionalProperties": False,
                },
                "strengths_he": {"type": "array", "maxItems": 4,
                                 "items": {"type": "string", "maxLength": 200}},
                "development_area_he": {"type": "string", "maxLength": 500},
                "summary_he": {"type": "string", "maxLength": 500},
            },
            "required": ["scores", "strengths_he", "development_area_he", "summary_he"],
            "additionalProperties": False,
        }

    def complete(self, request: JudgeRequest) -> str:
        messages = [
            {"role": "system", "content": request.system_prompt},
            {"role": "user", "content": request.user_prompt},
        ]
        schema = self._scorecard_schema([d.id for d in request.dimensions])
        response_format = {
            "type": "json_schema",
            "json_schema": {"name": "scorecard", "schema": schema},
        }
        try:
            return self._chat(messages, response_format)
        except VLLMJudgeError as exc:
            # Older/other OpenAI-compatible servers may not implement
            # json_schema; degrade to plain json_object mode.
            text = str(exc)
            if "json_schema" in text or "response_format" in text:
                logger.warning("endpoint rejected json_schema structured output; "
                               "falling back to json_object mode")
                return self._chat_with_role_fallback(
                    request, {"type": "json_object"})
            if "alternate" in text or "system role" in text.lower():
                merged = f"{request.system_prompt}\n\n{request.user_prompt}"
                return self._chat([{"role": "user", "content": merged}], response_format)
            raise

    def _chat_with_role_fallback(self, request: JudgeRequest,
                                 response_format: dict) -> str:
        messages = [
            {"role": "system", "content": request.system_prompt},
            {"role": "user", "content": request.user_prompt},
        ]
        try:
            return self._chat(messages, response_format)
        except VLLMJudgeError as exc:
            # Some chat templates (Mistral-family, e.g. dictalm2.0-instruct)
            # accept no system role at all and vLLM rejects the request with
            # "roles must alternate". The instructions still apply - they
            # just have to travel inside the user turn.
            text = str(exc)
            if "alternate" not in text and "system role" not in text.lower():
                raise
            logger.info("model's chat template rejects a system message; "
                        "resending with the system prompt merged into the user turn")
            merged = f"{request.system_prompt}\n\n{request.user_prompt}"
            return self._chat([{"role": "user", "content": merged}], response_format)

    _CTX_BUDGET_RE = re.compile(
        r"maximum context length is (\d+) tokens and your request has (\d+) input tokens"
    )

    @staticmethod
    def _llamacpp_budget(detail: str) -> tuple[int, int] | None:
        """llama.cpp's form of the same 400: a JSON error of type
        exceed_context_size_error carrying n_ctx and n_prompt_tokens."""
        ctx = re.search(r'"n_ctx"\s*:\s*(\d+)', detail)
        used = re.search(r'"n_prompt_tokens"\s*:\s*(\d+)', detail)
        return (int(ctx.group(1)), int(used.group(1))) if ctx and used else None

    def _chat(self, messages: list[dict[str, str]], response_format: dict,
              max_tokens: int | None = None) -> str:
        payload = {
            "model": self.config.model,
            "temperature": self.config.temperature,
            "max_tokens": max_tokens or self.config.max_tokens,
            "response_format": response_format,
            "messages": messages,
        }
        url = f"{self.config.base_url.rstrip('/')}/chat/completions"
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers=self._headers(),
            method="POST",
        )
        try:
            with self._open(req, timeout=self.config.request_timeout_sec) as resp:
                body = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            # The response body is where vLLM says WHY (context length,
            # template restrictions); losing it made 400s undebuggable.
            detail = ""
            try:
                detail = exc.read().decode("utf-8", errors="replace")[:300]
            except OSError:
                pass
            # Retry prompts grow (validation feedback is appended), and a
            # fixed max_tokens eventually no longer fits the model context.
            # vLLM's 400 names the exact budget - shrink to it and resend.
            budget = self._CTX_BUDGET_RE.search(detail)
            ctx_used = ((int(budget.group(1)), int(budget.group(2))) if budget
                        else self._llamacpp_budget(detail))
            if ctx_used and max_tokens is None:
                ctx, used = ctx_used
                available = ctx - used - 16
                if available >= 512:
                    logger.info("max_tokens %d does not fit (%d input / %d ctx); "
                                "retrying with %d",
                                self.config.max_tokens, used, ctx, available)
                    return self._chat(messages, response_format, max_tokens=available)
            raise VLLMJudgeError(f"vLLM request failed: {exc}: {detail}") from exc
        except (urllib.error.URLError, OSError, json.JSONDecodeError) as exc:
            raise VLLMJudgeError(f"vLLM request failed: {exc}") from exc
        try:
            choice = body["choices"][0]
            content = choice["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise VLLMJudgeError(f"unexpected vLLM response shape: {exc}") from exc
        if not isinstance(content, str):
            # Some servers send content:null (e.g. a pure tool-call turn). The
            # method's contract is -> str; returning None here would violate it
            # and only be caught two layers down. Fail loudly and retry.
            raise VLLMJudgeError("vLLM response had no text content")
        if choice.get("finish_reason") == "length":
            # A structured-output response cut at max_tokens is a valid JSON
            # PREFIX, which then fails parsing with a misleading error. Name
            # the real problem so the retry prompt asks for brevity.
            raise VLLMJudgeError(
                "the model hit max_tokens before finishing the JSON; "
                "keep reasoning_he short and quote less"
            )
        return content
