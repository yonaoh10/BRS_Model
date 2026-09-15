"""Run pyannote diarization in its own PROCESS, for overlap with ASR.

Diarization needs nothing from the transcript, so on a mono recording it can
run while the ASR stage is still transcribing - measured on the dev machine
that turns a ~401 s sequential pair into ~347 s of combined wall. Contention
stretches both engines, so the win is the gap between them, not the whole
ASR time.

A separate process, not a thread, and that is load-bearing: ctranslate2
(faster-whisper) and torch (pyannote) each bundle their own libiomp5, and one
process holding both aborts or segfaults at random on Intel macOS - the exact
failure `FasterWhisperEngine` defers its imports to dodge. Two processes each
hold exactly one OpenMP runtime.

Invoked as:

    python -m callqa.speakers.diar_worker <wav> <call_id> <out_json>

with the SpeakersConfig JSON on stdin (it never touches argv, which `ps` can
read). Writes [{"label": ..., "start": ..., "end": ...}, ...] to <out_json>
atomically; any failure is a nonzero exit with a sanitized message on stderr,
and the parent falls back to in-process diarization.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def main(argv: list[str]) -> int:
    if len(argv) != 3:
        print("usage: diar_worker <wav> <call_id> <out_json> (config on stdin)",
              file=sys.stderr)
        return 2
    wav, call_id, out_json = Path(argv[0]), argv[1], Path(argv[2])
    from callqa.config import SpeakersConfig
    from callqa.redaction import sanitize_error
    from callqa.speakers.pyannote_engine import LazyPyannoteDiarizer
    from callqa.state import atomic_write_text
    try:
        config = SpeakersConfig.model_validate_json(sys.stdin.read())
        segments = LazyPyannoteDiarizer(config).diarize(wav, call_id)
        atomic_write_text(out_json, json.dumps(
            [{"label": s.label, "start": s.start, "end": s.end} for s in segments]
        ))
    except Exception as exc:  # noqa: BLE001 - the parent decides what to do
        print(f"diarization worker failed: {sanitize_error(exc)}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
