"""Remote ASR engine: HTTP client for a cloud-hosted transcription server.

DEV-PHASE ONLY. This exists so the pipeline can run on a laptop while the
GPU work happens on a rented cloud machine. The bank server runs
`asr.engine: faster_whisper` (in-process) and never uses this file, so the
whole cloud option is removed by deleting `cloud/` and this module.

Wire protocol: multipart POST to `{base_url}/transcribe`, response is the
pipeline's own `Transcript` JSON. The server (cloud/asr_server.py) produces
it with the very same FasterWhisperEngine the bank server will run, so the
transcript is byte-identical to an on-prem run.

Only redacted text ever leaves the pipeline elsewhere; note that AUDIO does
leave the machine here, which is why the dev phase uses synthetic or
consented recordings only (see cloud/README.md).
"""

from __future__ import annotations

import json
import logging
import mimetypes
import urllib.error
import urllib.request
import uuid
from pathlib import Path

from callqa.config import ASRConfig
from callqa.models import Speaker, Transcript, VADSegment

logger = logging.getLogger(__name__)


class RemoteASRError(RuntimeError):
    pass


def _encode_multipart(
    fields: dict[str, str], file_field: str, file_path: Path
) -> tuple[bytes, str]:
    """Build a multipart/form-data body with stdlib only."""
    boundary = f"----callqa{uuid.uuid4().hex}"
    line_end = b"\r\n"
    parts: list[bytes] = []
    for name, value in fields.items():
        parts += [
            f"--{boundary}".encode(),
            f'Content-Disposition: form-data; name="{name}"'.encode(),
            b"",
            str(value).encode("utf-8"),
        ]
    content_type = mimetypes.guess_type(file_path.name)[0] or "application/octet-stream"
    parts += [
        f"--{boundary}".encode(),
        f'Content-Disposition: form-data; name="{file_field}"; '
        f'filename="{file_path.name}"'.encode(),
        f"Content-Type: {content_type}".encode(),
        b"",
    ]
    body = line_end.join(parts) + line_end + file_path.read_bytes() + line_end
    body += f"--{boundary}--".encode() + line_end
    return body, f"multipart/form-data; boundary={boundary}"


class RemoteASREngine:
    name = "remote"

    def __init__(self, config: ASRConfig) -> None:
        if not config.base_url:
            raise ValueError("asr.base_url is required for the remote ASR engine")
        self.config = config
        self.base_url = config.base_url.rstrip("/")

    def _headers(self) -> dict[str, str]:
        headers: dict[str, str] = {}
        if self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"
        return headers

    def check_connectivity(self) -> None:
        """Fail fast at startup rather than mid-batch."""
        url = f"{self.base_url}/health"
        req = urllib.request.Request(url, headers=self._headers(), method="GET")
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                if resp.status != 200:
                    raise RemoteASRError(f"ASR server /health returned HTTP {resp.status}")
                body = json.loads(resp.read().decode("utf-8"))
        except (urllib.error.URLError, OSError, json.JSONDecodeError) as exc:
            raise RemoteASRError(
                f"remote ASR server unreachable at {self.base_url}. "
                "Start it with: python cloud/runpod_cli.py up"
            ) from exc
        if not body.get("model_loaded"):
            raise RemoteASRError(
                f"remote ASR server at {self.base_url} is up but has no model loaded "
                f"(status: {body})"
            )
        logger.info("remote ASR server reachable at %s", self.base_url)

    def transcribe(
        self,
        wav_path: Path,
        *,
        call_id: str,
        role: Speaker | None,
        vad_segments: list[VADSegment],
    ) -> Transcript:
        fields = {"call_id": call_id, "role": role or "", "language": self.config.language}
        body, content_type = _encode_multipart(fields, "file", wav_path)
        headers = {**self._headers(), "Content-Type": content_type}
        req = urllib.request.Request(
            f"{self.base_url}/transcribe", data=body, headers=headers, method="POST"
        )
        try:
            with urllib.request.urlopen(req, timeout=self.config.timeout_sec) as resp:
                payload = resp.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:300]
            raise RemoteASRError(f"remote ASR failed (HTTP {exc.code}): {detail}") from exc
        except (urllib.error.URLError, OSError) as exc:
            raise RemoteASRError(f"remote ASR request failed: {exc}") from exc
        # The server speaks our own Transcript schema, so this is lossless.
        transcript = Transcript.model_validate_json(payload)
        logger.info(
            "remote asr done: call_id=%s role=%s segments=%d",
            call_id, role or "mono", len(transcript.segments),
        )
        return transcript
