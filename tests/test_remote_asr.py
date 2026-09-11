"""Unit tests: the remote (cloud) ASR client + the server's multipart parser.

Exercises the real HTTP path against a local fake server - multipart
encoding, auth header, Transcript round-trip, and every failure mode - with
no GPU, no model and no cloud account.
"""

from __future__ import annotations

import json
import sys
import threading
import wave
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest

from callqa.asr.remote_engine import RemoteASREngine, RemoteASRError, _encode_multipart
from callqa.config import ASRConfig, load_config
from callqa.models import Transcript, TranscriptSegment, VADSegment, Word

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "cloud"))
from asr_server import parse_multipart  # noqa: E402

SAMPLE_TRANSCRIPT = Transcript(
    call_id="C1",
    language="he",
    engine="faster_whisper",
    segments=[
        TranscriptSegment(
            speaker="banker", start=0.0, end=3.0, text="שלום, במה אפשר לעזור?",
            words=[Word(word="שלום", start=0.0, end=0.5, probability=0.9)],
            avg_logprob=-0.3,
        )
    ],
)


class _FakeASRServer(BaseHTTPRequestHandler):
    seen: list[dict] = []
    require_key: str | None = None
    model_loaded: bool = True
    fail_with: int | None = None

    def _authorized(self) -> bool:
        if self.require_key is None:
            return True
        return self.headers.get("Authorization") == f"Bearer {self.require_key}"

    def _json(self, code: int, body: str) -> None:
        data = body.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self) -> None:  # noqa: N802
        self.seen.append({"method": "GET", "path": self.path,
                          "auth": self.headers.get("Authorization")})
        if not self._authorized():
            self._json(401, '{"error":"unauthorized"}')
            return
        self._json(200, json.dumps({"status": "ok", "model_loaded": self.model_loaded}))

    def do_POST(self) -> None:  # noqa: N802
        length = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(length)
        fields, files = parse_multipart(raw, self.headers.get("Content-Type", ""))
        self.seen.append({"method": "POST", "path": self.path,
                          "auth": self.headers.get("Authorization"),
                          "fields": fields, "file_bytes": len(files.get("file", b""))})
        if not self._authorized():
            self._json(401, '{"error":"unauthorized"}')
            return
        if self.fail_with:
            self._json(self.fail_with, '{"error":"boom"}')
            return
        self._json(200, SAMPLE_TRANSCRIPT.model_dump_json())

    def log_message(self, *args) -> None:
        return


@pytest.fixture()
def fake_asr():
    _FakeASRServer.seen = []
    _FakeASRServer.require_key = None
    _FakeASRServer.model_loaded = True
    _FakeASRServer.fail_with = None
    server = HTTPServer(("127.0.0.1", 0), _FakeASRServer)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield server, f"http://127.0.0.1:{server.server_port}"
    server.shutdown()


@pytest.fixture()
def wav_file(tmp_path: Path) -> Path:
    path = tmp_path / "chunk.wav"
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        wf.writeframes(b"\x00\x01" * 16000)
    return path


def _engine(base_url: str, **kw) -> RemoteASREngine:
    return RemoteASREngine(ASRConfig(engine="remote", base_url=base_url, **kw))


# -- multipart round trip ---------------------------------------------------

def test_multipart_roundtrip(wav_file: Path) -> None:
    body, content_type = _encode_multipart(
        {"call_id": "C9", "role": "banker", "language": "he"}, "file", wav_file
    )
    fields, files = parse_multipart(body, content_type)
    assert fields == {"call_id": "C9", "role": "banker", "language": "he"}
    assert files["file"] == wav_file.read_bytes()


def test_multipart_rejects_missing_boundary() -> None:
    with pytest.raises(ValueError, match="boundary"):
        parse_multipart(b"whatever", "multipart/form-data")


# -- client behaviour -------------------------------------------------------

def test_transcribe_roundtrip(fake_asr, wav_file: Path) -> None:
    _server, base_url = fake_asr
    engine = _engine(base_url)
    engine.check_connectivity()
    transcript = engine.transcribe(
        wav_file, call_id="C1", role="banker", vad_segments=[VADSegment(start=0, end=3)]
    )
    assert transcript.segments[0].text == "שלום, במה אפשר לעזור?"
    assert transcript.segments[0].avg_logprob == -0.3
    assert transcript.segments[0].words[0].word == "שלום"

    post = [r for r in _FakeASRServer.seen if r["method"] == "POST"][0]
    assert post["path"] == "/transcribe"
    assert post["fields"] == {"call_id": "C1", "role": "banker", "language": "he"}
    assert post["file_bytes"] == wav_file.stat().st_size


def test_mono_role_is_sent_empty(fake_asr, wav_file: Path) -> None:
    _server, base_url = fake_asr
    _engine(base_url).transcribe(wav_file, call_id="M1", role=None, vad_segments=[])
    post = [r for r in _FakeASRServer.seen if r["method"] == "POST"][0]
    assert post["fields"]["role"] == ""


def test_bearer_token_sent(fake_asr, wav_file: Path) -> None:
    _server, base_url = fake_asr
    _FakeASRServer.require_key = "cloud-secret"
    engine = _engine(base_url, api_key="cloud-secret")
    engine.check_connectivity()
    engine.transcribe(wav_file, call_id="C1", role="banker", vad_segments=[])
    assert all(r["auth"] == "Bearer cloud-secret" for r in _FakeASRServer.seen)


def test_missing_token_rejected(fake_asr) -> None:
    _server, base_url = fake_asr
    _FakeASRServer.require_key = "cloud-secret"
    with pytest.raises(RemoteASRError):
        _engine(base_url).check_connectivity()


def test_unloaded_model_fails_fast(fake_asr) -> None:
    _server, base_url = fake_asr
    _FakeASRServer.model_loaded = False
    with pytest.raises(RemoteASRError, match="no model loaded"):
        _engine(base_url).check_connectivity()


def test_unreachable_server_message() -> None:
    with pytest.raises(RemoteASRError, match="unreachable"):
        _engine("http://127.0.0.1:9").check_connectivity()


def test_server_error_surfaces_status(fake_asr, wav_file: Path) -> None:
    _server, base_url = fake_asr
    _FakeASRServer.fail_with = 500
    with pytest.raises(RemoteASRError, match="HTTP 500"):
        _engine(base_url).transcribe(wav_file, call_id="C1", role="banker", vad_segments=[])


# -- config wiring ----------------------------------------------------------

def test_remote_engine_requires_base_url() -> None:
    with pytest.raises(ValueError, match="base_url"):
        load_config(None, {"asr": {"engine": "remote"}})


def test_cloud_config_file_is_valid() -> None:
    config = load_config("config/config.cloud.yaml",
                         {"asr": {"base_url": "https://example.invalid"}})
    assert config.asr.engine == "remote"
    assert config.judge.engine == "vllm"
    assert config.audio.vad == "energy"


def test_cloud_credentials_come_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CALLQA_ASR__BASE_URL", "https://pod-8001.proxy.runpod.net")
    monkeypatch.setenv("CALLQA_ASR__API_KEY", "asr-key")
    monkeypatch.setenv("CALLQA_JUDGE__API_KEY", "judge-key")
    config = load_config("config/config.cloud.yaml")
    assert config.asr.base_url == "https://pod-8001.proxy.runpod.net"
    assert config.asr.api_key == "asr-key"
    assert config.judge.api_key == "judge-key"
