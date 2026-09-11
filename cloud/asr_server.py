#!/usr/bin/env python3
"""ASR HTTP server - runs ON the rented GPU machine (DEV PHASE ONLY).

Wraps the project's own FasterWhisperEngine in a tiny HTTP API so a laptop
can use a cloud GPU for transcription. Because it reuses the same engine
class the bank server runs in-process, the transcripts it returns are
identical to an on-prem run - the only difference is the network hop.

Deliberately stdlib-only (http.server): no FastAPI/uvicorn to install on the
pod, one less thing to break, and nothing here ships to the bank.

Auth: every request must carry `Authorization: Bearer $CALLQA_ASR_API_KEY`.
The RunPod proxy URL is public, so the server refuses to start without a key.

Run:
    CALLQA_ASR_API_KEY=... python cloud/asr_server.py --port 8001 \
        --model-dir models/ivrit-whisper-large-v3-turbo-ct2
"""

from __future__ import annotations

import argparse
import hmac
import json
import logging
import os
import sys
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from callqa.config import ASRConfig  # noqa: E402
from callqa.models import Transcript  # noqa: E402

logger = logging.getLogger("callqa.cloud.asr")

# Transcription is GPU-bound: serialise it so concurrent callers queue
# instead of fighting over VRAM.
_GPU_LOCK = threading.Lock()
_ENGINE = None
_ENGINE_ERROR: str | None = None


def load_engine(config: ASRConfig):  # noqa: ANN201
    """Load the model once at startup (never per request)."""
    global _ENGINE, _ENGINE_ERROR
    try:
        from callqa.asr.faster_whisper_engine import FasterWhisperEngine

        _ENGINE = FasterWhisperEngine(config)
        logger.info("ASR model loaded from %s", config.model_dir)
    except Exception as exc:  # noqa: BLE001 - report through /health
        _ENGINE_ERROR = f"{type(exc).__name__}: {exc}"
        logger.error("ASR model failed to load: %s", _ENGINE_ERROR)
    return _ENGINE



def parse_multipart(body: bytes, content_type: str) -> tuple[dict[str, str], dict[str, bytes]]:
    """Minimal multipart/form-data parser.

    Self-contained on purpose: the stdlib `cgi` module was removed in Python
    3.13 and the rented machine's Python version is not ours to choose.
    Returns (text fields, file parts keyed by field name).
    """
    marker = "boundary="
    if marker not in content_type:
        raise ValueError("Content-Type is missing the multipart boundary")
    boundary = content_type.split(marker, 1)[1].split(";")[0].strip().strip('"')
    delimiter = b"--" + boundary.encode()

    fields: dict[str, str] = {}
    files: dict[str, bytes] = {}
    for chunk in body.split(delimiter):
        if chunk in (b"", b"--", b"--\r\n", b"\r\n"):
            continue
        chunk = chunk.lstrip(b"\r\n")
        if b"\r\n\r\n" not in chunk:
            continue
        raw_headers, payload = chunk.split(b"\r\n\r\n", 1)
        payload = payload[:-2] if payload.endswith(b"\r\n") else payload
        disposition = ""
        for line in raw_headers.decode("utf-8", errors="replace").split("\r\n"):
            if line.lower().startswith("content-disposition:"):
                disposition = line
        if 'name="' not in disposition:
            continue
        name = disposition.split('name="', 1)[1].split('"', 1)[0]
        if "filename=" in disposition:
            files[name] = payload
        else:
            fields[name] = payload.decode("utf-8", errors="replace")
    return fields, files


class Handler(BaseHTTPRequestHandler):
    api_key = ""
    last_request_ts = time.monotonic()

    server_version = "callqa-asr/1.0"

    # -- helpers --------------------------------------------------------

    def _send(self, code: int, payload: dict | str, raw_json: str | None = None) -> None:
        body = (raw_json if raw_json is not None else json.dumps(payload, ensure_ascii=False))
        data = body.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _authorized(self) -> bool:
        provided = self.headers.get("Authorization", "")
        expected = f"Bearer {type(self).api_key}"
        return hmac.compare_digest(provided, expected)

    def log_message(self, fmt: str, *a) -> None:
        # Never log request bodies: audio and transcripts are sensitive.
        logger.info("%s - %s", self.address_string(), fmt % a)

    # -- routes ---------------------------------------------------------

    def do_GET(self) -> None:  # noqa: N802
        if self.path.split("?")[0] != "/health":
            self._send(404, {"error": "not found"})
            return
        if not self._authorized():
            self._send(401, {"error": "unauthorized"})
            return
        self._send(200, {
            "status": "ok",
            "model_loaded": _ENGINE is not None,
            "model_error": _ENGINE_ERROR,
            "idle_seconds": round(time.monotonic() - type(self).last_request_ts, 1),
        })

    def do_POST(self) -> None:  # noqa: N802
        if self.path.split("?")[0] != "/transcribe":
            self._send(404, {"error": "not found"})
            return
        if not self._authorized():
            self._send(401, {"error": "unauthorized"})
            return
        if _ENGINE is None:
            self._send(503, {"error": f"model not loaded: {_ENGINE_ERROR}"})
            return
        type(self).last_request_ts = time.monotonic()

        try:
            length = int(self.headers.get("Content-Length", 0))
            if length <= 0:
                raise ValueError("empty body")
            fields, files = parse_multipart(
                self.rfile.read(length), self.headers.get("Content-Type", "")
            )
            call_id = fields.get("call_id") or "unknown"
            role = (fields.get("role") or "").strip() or None
            audio_bytes = files["file"]
        except Exception as exc:  # noqa: BLE001
            self._send(400, {"error": f"malformed multipart request: {exc}"})
            return

        tmp_dir = Path(tempfile.mkdtemp(prefix="callqa-asr-"))
        wav_path = tmp_dir / "audio.wav"
        try:
            wav_path.write_bytes(audio_bytes)
            started = time.monotonic()
            with _GPU_LOCK:
                transcript: Transcript = _ENGINE.transcribe(
                    wav_path, call_id=call_id, role=role, vad_segments=[]
                )
            logger.info(
                "transcribed call_id=%s role=%s segments=%d in %.1fs",
                call_id, role or "mono", len(transcript.segments), time.monotonic() - started,
            )
            self._send(200, {}, raw_json=transcript.model_dump_json())
        except Exception as exc:  # noqa: BLE001
            logger.exception("transcription failed for call_id=%s", call_id)
            self._send(500, {"error": f"{type(exc).__name__}: {exc}"})
        finally:
            # Audio is never retained on the rented machine.
            try:
                wav_path.unlink(missing_ok=True)
                tmp_dir.rmdir()
            except OSError:
                pass
            type(self).last_request_ts = time.monotonic()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--host", default="0.0.0.0")  # noqa: S104 - behind the RunPod proxy
    parser.add_argument("--model-dir", default="models/ivrit-whisper-large-v3-turbo-ct2")
    parser.add_argument("--language", default="he")
    parser.add_argument("--compute-type", default="float16")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    api_key = os.environ.get("CALLQA_ASR_API_KEY", "")
    if len(api_key) < 16:
        print("ERROR: CALLQA_ASR_API_KEY must be set to a secret of at least 16 characters.",
              file=sys.stderr)
        print("The pod's HTTP port is reachable from the public internet.", file=sys.stderr)
        return 2
    Handler.api_key = api_key

    load_engine(ASRConfig(
        engine="faster_whisper",
        model_dir=args.model_dir,
        language=args.language,
        compute_type=args.compute_type,
    ))

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    logger.info("callqa ASR server listening on %s:%d", args.host, args.port)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("shutting down")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
