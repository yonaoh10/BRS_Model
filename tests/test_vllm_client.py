"""Unit tests: the vLLM/OpenAI-compatible judge client against a local fake server.

Exercises the real HTTP path (URL building, auth header, JSON payload, response
parsing, connectivity check) without vLLM installed.
"""

from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from callqa.config import JudgeConfig
from callqa.judge.base import JudgeRequest
from callqa.judge.vllm_judge import VLLMJudge, VLLMJudgeError
from callqa.models import Features, RedactedTranscript, SpeechRateWPM


class _FakeVLLM(BaseHTTPRequestHandler):
    """Records every request; answers /models and /chat/completions."""

    seen: list[dict] = []
    require_key: str | None = None

    def _authorized(self) -> bool:
        if self.require_key is None:
            return True
        return self.headers.get("Authorization") == f"Bearer {self.require_key}"

    def do_GET(self) -> None:  # noqa: N802
        self.seen.append({"method": "GET", "path": self.path,
                          "auth": self.headers.get("Authorization")})
        if not self._authorized():
            self.send_response(401)
            self.end_headers()
            return
        body = json.dumps({"data": [{"id": "test-model"}]}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self) -> None:  # noqa: N802
        length = int(self.headers.get("Content-Length", 0))
        payload = json.loads(self.rfile.read(length) or b"{}")
        self.seen.append({"method": "POST", "path": self.path,
                          "auth": self.headers.get("Authorization"), "payload": payload})
        if not self._authorized():
            self.send_response(401)
            self.end_headers()
            return
        body = json.dumps({"choices": [{"message": {"content": '{"ok": true}'}}]}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args) -> None:  # silence
        return


@pytest.fixture()
def fake_server():
    _FakeVLLM.seen = []
    _FakeVLLM.require_key = None
    server = HTTPServer(("127.0.0.1", 0), _FakeVLLM)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield server, f"http://127.0.0.1:{server.server_port}/v1"
    server.shutdown()


def _request(call_id: str = "C1") -> JudgeRequest:
    redacted = RedactedTranscript(call_id=call_id, engine="regex", turns=[])
    features = Features(
        call_id=call_id, talk_ratio=0.5, longest_banker_monologue_sec=1, interruptions_by_banker=0,
        interruptions_by_customer=0, patience_median_sec=None, banker_question_count=0,
        banker_questions_per_minute=0, speech_rate_wpm=SpeechRateWPM(), dead_air_total_sec=0,
        call_duration_sec=60, banker_speech_sec=30, customer_speech_sec=30,
    )
    return JudgeRequest(call_id=call_id, system_prompt="sys", user_prompt="user",
                        dimensions=[], redacted=redacted, features=features)


def test_placeholder_model_rejected() -> None:
    with pytest.raises(ValueError, match="placeholder"):
        VLLMJudge(JudgeConfig(model="<LLM_MODEL_ID_PLACEHOLDER>"))


def test_complete_sends_openai_payload_without_auth(fake_server) -> None:
    _server, base_url = fake_server
    judge = VLLMJudge(JudgeConfig(model="test-model", base_url=base_url, max_tokens=123))
    judge.check_connectivity()
    raw = judge.complete(_request())
    assert raw == '{"ok": true}'
    get, post = _FakeVLLM.seen
    assert get["path"] == "/v1/models" and get["auth"] is None
    assert post["path"] == "/v1/chat/completions" and post["auth"] is None
    p = post["payload"]
    assert p["model"] == "test-model"
    assert p["temperature"] == 0.0 and p["max_tokens"] == 123
    # Structured output: the full scorecard schema is pinned server-side so
    # layout failures (stray keys, quote-derailed strings) cannot happen.
    assert p["response_format"]["type"] == "json_schema"
    schema = p["response_format"]["json_schema"]["schema"]
    assert schema["properties"]["scores"]["required"] == []  # helper request has no dims
    assert [m["role"] for m in p["messages"]] == ["system", "user"]


def test_bearer_token_sent_when_configured(fake_server) -> None:
    _server, base_url = fake_server
    _FakeVLLM.require_key = "s3cret"
    judge = VLLMJudge(JudgeConfig(model="test-model", base_url=base_url, api_key="s3cret"))
    judge.check_connectivity()
    assert judge.complete(_request()) == '{"ok": true}'
    assert all(r["auth"] == "Bearer s3cret" for r in _FakeVLLM.seen)


def test_missing_token_is_rejected_by_server(fake_server) -> None:
    _server, base_url = fake_server
    _FakeVLLM.require_key = "s3cret"
    judge = VLLMJudge(JudgeConfig(model="test-model", base_url=base_url))
    with pytest.raises(VLLMJudgeError):
        judge.check_connectivity()


def test_unreachable_endpoint_gives_clear_error() -> None:
    judge = VLLMJudge(JudgeConfig(model="test-model", base_url="http://127.0.0.1:9/v1"))
    with pytest.raises(VLLMJudgeError, match="unreachable"):
        judge.check_connectivity()


def test_api_key_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    from callqa.config import load_config

    monkeypatch.setenv("CALLQA_JUDGE__API_KEY", "from-env")
    assert load_config(None).judge.api_key == "from-env"


def test_a_refused_api_key_is_not_reported_as_an_unreachable_server() -> None:
    """HTTPError subclasses URLError, so a 401 from a running, correctly secured
    server came out as "vLLM endpoint unreachable ... start it with
    start_vllm.sh" - sending the operator to restart a server that was fine and
    was only refusing a missing key."""
    import io
    import urllib.error
    from unittest import mock

    import pytest

    from callqa.config import JudgeConfig
    from callqa.judge.vllm_judge import VLLMJudge, VLLMJudgeError

    refused = urllib.error.HTTPError("http://x/v1/models", 401, "Unauthorized",
                                     {}, io.BytesIO(b""))
    judge = VLLMJudge(JudgeConfig(base_url="http://127.0.0.1:8000/v1", model="m"))
    with mock.patch.object(judge, "_open", side_effect=refused):
        with pytest.raises(VLLMJudgeError, match="API key") as err:
            judge.check_connectivity()
    assert "unreachable" not in str(err.value)


def test_a_server_serving_another_model_gets_no_transcript(fake_server) -> None:
    """On a shared Windows host a colleague's judge can hold the port. Its
    /models names its own file, and the pipeline refuses before sending data."""
    _server, base_url = fake_server
    judge = VLLMJudge(JudgeConfig(model="C:/Users/me/models/mine.gguf", base_url=base_url))
    with pytest.raises(VLLMJudgeError, match="Nothing was sent"):
        judge.check_connectivity()
    same_file = VLLMJudge(JudgeConfig(model="models\\x\\test-model", base_url=base_url))
    same_file.check_connectivity()                  # a path to the served model is fine


def test_a_local_judge_is_never_reached_through_a_proxy(monkeypatch) -> None:
    """Windows applies the system proxy to every URL, and its <local> bypass
    misses 127.0.0.1 - judge requests went to the corporate proxy."""
    monkeypatch.setenv("HTTP_PROXY", "http://proxy.invalid:3128")
    judge = VLLMJudge(JudgeConfig(model="m", base_url="http://127.0.0.1:8000/v1"))
    handlers = judge._open.__self__.handlers
    assert not any(isinstance(h, urllib.request.ProxyHandler) and h.proxies for h in handlers)
    # ...whereas the default opener, which it replaced, would have used it:
    default = urllib.request.build_opener().handlers
    assert any(isinstance(h, urllib.request.ProxyHandler) and h.proxies for h in default)


def test_a_public_judge_endpoint_is_refused_before_anything_is_sent(monkeypatch) -> None:
    import socket

    monkeypatch.setattr(socket, "getaddrinfo",
                        lambda host, port: [(None, None, None, "", ("8.8.8.8", 0))])
    judge = VLLMJudge(JudgeConfig(model="m", base_url="https://api.example.com/v1"))
    with pytest.raises(VLLMJudgeError, match="public internet"):
        judge.check_connectivity()
    allowed = VLLMJudge(JudgeConfig(model="m", base_url="https://api.example.com/v1",
                                    allow_public_endpoint=True))
    allowed._refuse_public_endpoint()                 # explicitly allowed: no refusal


def test_a_judge_still_loading_its_model_is_waited_for(fake_server, monkeypatch) -> None:
    """llama-server answers 503 until its model is loaded; that is not a wrong URL."""
    import io

    _server, base_url = fake_server
    judge = VLLMJudge(JudgeConfig(model="test-model", base_url=base_url))
    real_open = judge._open
    calls = {"n": 0}

    def loading_then_up(req, timeout=0):  # noqa: ANN001
        calls["n"] += 1
        if calls["n"] == 1:
            raise urllib.error.HTTPError(req.full_url, 503, "Loading model", {},
                                         io.BytesIO(b'{"message":"Loading model"}'))
        return real_open(req, timeout=timeout)

    monkeypatch.setattr(judge, "_open", loading_then_up)
    monkeypatch.setattr("callqa.judge.vllm_judge.time.sleep", lambda s: None)
    judge.check_connectivity()
    assert calls["n"] == 2


def test_llama_cpp_context_overflow_shrinks_max_tokens() -> None:
    detail = ('{"error":{"code":400,"message":"the request exceeds the available context '
              'size","type":"exceed_context_size_error","n_prompt_tokens":9000,"n_ctx":16384}}')
    assert VLLMJudge._llamacpp_budget(detail) == (16384, 9000)
