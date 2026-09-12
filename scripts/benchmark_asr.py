#!/usr/bin/env python3
"""Measure how long transcription actually takes on THIS machine.

Transcription is the one stage whose cost cannot be predicted from the code:
it depends on the CPU, the thread count, the quantisation and whether there is
a GPU. Everything else in the pipeline is a couple of seconds. So rather than
quoting a number from somebody else's benchmark, run this once on a real
recording and read the number off your own machine:

    python scripts/benchmark_asr.py --audio path/to/call.m4a

It reports the realtime factor (audio seconds processed per wall-clock second)
and projects a batch from it. Model load time is measured separately, because
it is paid once per process, not once per call - which is exactly why `watch`
holds one process open instead of starting one per file.

Needs faster-whisper installed and the model present:
    pip install faster-whisper
    python scripts/download_models.py --asr
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

DEFAULT_MODEL = REPO_ROOT / "models" / "ivrit-whisper-large-v3-turbo-ct2"


def audio_duration(path: Path) -> float:
    proc = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=nk=1:nw=1", str(path)],
        capture_output=True, text=True)
    if proc.returncode != 0:
        raise SystemExit(f"ffprobe failed on {path}:\n{proc.stderr.strip()}")
    return float(proc.stdout.strip())


def detect_device(requested: str) -> tuple[str, str]:
    """Return (device, compute_type). int8 on CPU, float16 on GPU."""
    if requested != "auto":
        return requested, ("float16" if requested == "cuda" else "int8")
    try:
        import ctranslate2
        if ctranslate2.get_cuda_device_count() > 0:
            return "cuda", "float16"
    except Exception:  # noqa: BLE001 - absence of CUDA is not an error here
        pass
    return "cpu", "int8"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--audio", type=Path, required=True)
    ap.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL)
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--compute-type", default=None,
                    help="override: int8, int8_float16, float16, float32")
    ap.add_argument("--threads", type=int, default=0, help="0 = let ctranslate2 decide")
    ap.add_argument("--batch-size", type=int, default=8,
                    help="batched decoding; 1 disables it")
    ap.add_argument("--project", type=int, default=100,
                    help="project the total for this many calls")
    ap.add_argument("--json", action="store_true", help="machine-readable output")
    args = ap.parse_args()

    if not args.audio.exists():
        print(f"ERROR: no such file: {args.audio}", file=sys.stderr)
        return 2
    if not args.model_dir.exists():
        print(f"ERROR: model not found at {args.model_dir}\n"
              f"       run: python scripts/download_models.py --asr", file=sys.stderr)
        return 2
    try:
        from faster_whisper import WhisperModel
    except ImportError:
        print("ERROR: faster-whisper is not installed\n"
              "       run: pip install faster-whisper", file=sys.stderr)
        return 2

    device, compute_type = detect_device(args.device)
    compute_type = args.compute_type or compute_type
    duration = audio_duration(args.audio)

    print(f"audio       {args.audio.name}  ({duration:.1f}s)")
    print(f"device      {device} / {compute_type}"
          + (f" / {args.threads} threads" if args.threads else ""))
    print("loading model ...", flush=True)

    t0 = time.perf_counter()
    kwargs = {"device": device, "compute_type": compute_type}
    if args.threads:
        kwargs["cpu_threads"] = args.threads
    model = WhisperModel(str(args.model_dir), **kwargs)
    load_sec = time.perf_counter() - t0
    print(f"model load  {load_sec:.1f}s  (paid once per process, not per call)")

    print("transcribing ...", flush=True)
    t0 = time.perf_counter()
    segments, info = model.transcribe(
        str(args.audio), language="he", word_timestamps=True, vad_filter=True,
        **({"batch_size": args.batch_size} if args.batch_size > 1 else {}),
    )
    segments = list(segments)          # faster-whisper is lazy; this is the work
    transcribe_sec = time.perf_counter() - t0

    words = sum(len(s.text.split()) for s in segments)
    rtf = duration / transcribe_sec if transcribe_sec else 0.0
    per_call = transcribe_sec + 2.5    # the rest of the pipeline, measured
    result = {
        "audio_sec": round(duration, 1),
        "device": device,
        "compute_type": compute_type,
        "model_load_sec": round(load_sec, 1),
        "transcribe_sec": round(transcribe_sec, 1),
        "realtime_factor": round(rtf, 2),
        "segments": len(segments),
        "words": words,
        "projected_sec_per_call": round(per_call, 1),
        "projected_sec_for_batch": round(load_sec + per_call * args.project, 1),
        "batch_size": args.project,
    }
    if args.json:
        print(json.dumps(result, indent=2))
        return 0

    print(f"transcribe  {transcribe_sec:.1f}s  ->  {rtf:.1f}x realtime "
          f"({len(segments)} segments, {words} words)")
    print()
    print(f"so one call of this length costs about {per_call:.0f}s end to end,")
    print(f"and {args.project} of them about "
          f"{(load_sec + per_call * args.project) / 60:.0f} minutes in one process.")
    print("\nfirst lines of the transcript, to check it is actually Hebrew:")
    for seg in segments[:3]:
        print(f"  [{seg.start:6.1f}] {seg.text.strip()[:90]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
