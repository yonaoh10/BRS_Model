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
import os
import sys
from pathlib import Path

from callqa.portable import configure_stdio, exit_when_process_ends


def _pipeline_pid() -> int:
    """The pipeline process that started this worker.

    Passed explicitly because the direct parent is not always it: inside a
    Windows virtualenv, .venv\\Scripts\\python.exe is a small launcher that
    starts the real interpreter as ITS child.
    """
    try:
        return int(os.environ["CALLQA_PARENT_PID"])
    except (KeyError, ValueError):
        return os.getppid()


def main(argv: list[str]) -> int:
    # Exit if the pipeline dies. macOS has no PR_SET_PDEATHSIG and Windows never
    # reparents, so a killed pipeline would otherwise orphan this worker - it
    # would run pyannote to completion holding the GPU for minutes after the
    # operator thinks the job is gone.
    exit_when_process_ends(_pipeline_pid())
    configure_stdio()
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
