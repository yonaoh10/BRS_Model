#!/usr/bin/env python3
"""Turn a recorded simulation call into an input the pipeline can process.

The pipeline's primary speaker-attribution path is a stereo channel split:
banker on one channel, customer on the other. A phone call recorded on one
device is mono, which routes to the pyannote diarizer instead - a gated model
that needs HF_TOKEN and a download. For a test call there is a much simpler
route: each side records itself locally, and the two mono tracks are merged
here into one stereo file.

    # two tracks (recommended): banker left, customer right
    python scripts/prepare_real_call.py --call-id REAL001 \
        --banker-track ~/rec/me.m4a --customer-track ~/rec/friend.m4a

    # single mono recording (needs HF_TOKEN at processing time)
    python scripts/prepare_real_call.py --call-id REAL001 --mono ~/rec/call.m4a

Writes data/input/calls/<call_id>.wav at 16 kHz and adds the matching row to
data/input/metadata.csv (rewriting the row if the call_id is already there).

The two tracks do not need to start together: --offset shifts the customer
track, and without it the tracks are simply aligned at zero. Perfect sync is
not required for redaction or the rubric, but it does matter for the
interruption and patience metrics, so align on a shared cue (both sides clap
or say "התחלנו" at the top of the call).
"""

from __future__ import annotations

import argparse
import csv
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from callqa.ingestion import sanitize_call_id  # noqa: E402

COLUMNS = ["call_id", "banker_id", "file_name", "call_date", "call_type",
           "banker_channel", "banker_name"]
SAMPLE_RATE = "16000"


def _ffmpeg(args: list[str]) -> None:
    proc = subprocess.run(["ffmpeg", "-y", "-loglevel", "error", *args],
                          capture_output=True, text=True)
    if proc.returncode != 0:
        raise SystemExit(f"ffmpeg failed:\n{proc.stderr.strip()}")


def _duration(path: Path) -> float:
    proc = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=nk=1:nw=1", str(path)],
        capture_output=True, text=True)
    if proc.returncode != 0:
        raise SystemExit(f"ffprobe failed on {path}:\n{proc.stderr.strip()}")
    return float(proc.stdout.strip())


def build_stereo(banker: Path, customer: Path, out: Path, offset: float) -> None:
    """Banker -> left, customer -> right, 16 kHz, 16-bit PCM."""
    delay = ""
    if offset > 0:                      # customer started late: pad its head
        delay = f"adelay={int(offset * 1000)}|{int(offset * 1000)},"
    elif offset < 0:                    # customer started early: trim its head
        delay = f"atrim=start={-offset},asetpts=PTS-STARTPTS,"
    # join ends with its SHORTEST input, which would silently truncate whichever
    # side talked last. Pad both to silence and cut at the real end instead.
    total = max(_duration(banker), _duration(customer) + offset)
    _ffmpeg([
        "-i", str(banker), "-i", str(customer),
        "-filter_complex",
        f"[0:a]aformat=channel_layouts=mono,apad[l];"
        f"[1:a]{delay}aformat=channel_layouts=mono,apad[r];"
        f"[l][r]join=inputs=2:channel_layout=stereo[a]",
        "-map", "[a]", "-t", f"{total:.3f}",
        "-ar", SAMPLE_RATE, "-c:a", "pcm_s16le", str(out),
    ])


def build_mono(source: Path, out: Path) -> None:
    _ffmpeg(["-i", str(source), "-ac", "1", "-ar", SAMPLE_RATE,
             "-c:a", "pcm_s16le", str(out)])


def upsert_metadata(path: Path, row: dict[str, str]) -> None:
    """Add or replace one row, keeping every column the file already had.

    Rewriting with only this script's seven known columns silently deleted any
    the bank had added, and doing it in place left no way back if the write
    failed halfway.
    """
    rows: list[dict[str, str]] = []
    fieldnames = list(COLUMNS)
    if path.exists():
        with path.open(newline="", encoding="utf-8-sig") as fh:
            reader = csv.DictReader(fh)
            existing = [c for c in (reader.fieldnames or []) if c]
            fieldnames = existing + [c for c in COLUMNS if c not in existing]
            rows = [r for r in reader if r.get("call_id") != row["call_id"]]
        backup = path.with_suffix(path.suffix + ".bak")
        backup.write_bytes(path.read_bytes())
    rows.append(row)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow({c: (r.get(c) or "") for c in fieldnames})
    tmp.replace(path)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--call-id", required=True)
    ap.add_argument("--banker-track", type=Path, help="the banker's own recording")
    ap.add_argument("--customer-track", type=Path, help="the customer's own recording")
    ap.add_argument("--mono", type=Path, help="a single recording of the whole call")
    ap.add_argument("--offset", type=float, default=0.0,
                    help="seconds the customer track lags the banker track (may be negative)")
    ap.add_argument("--banker-id", default="B900")
    ap.add_argument("--banker-name", default="")
    ap.add_argument("--call-date", default="")
    ap.add_argument("--call-type", default="simulation")
    ap.add_argument("--input-dir", type=Path, default=REPO_ROOT / "data" / "input")
    args = ap.parse_args()

    if not shutil.which("ffmpeg"):
        print("ERROR: ffmpeg not found on PATH", file=sys.stderr)
        return 2
    call_id = sanitize_call_id(args.call_id)
    if call_id != call_id:
        print(f"note: using call id {call_id!r} (a call id becomes a file name)")
    two_tracks = bool(args.banker_track and args.customer_track)
    if two_tracks == bool(args.mono):
        print("ERROR: give either --banker-track with --customer-track, or --mono",
              file=sys.stderr)
        return 2

    calls_dir = args.input_dir / "calls"
    calls_dir.mkdir(parents=True, exist_ok=True)
    out = calls_dir / f"{call_id}.wav"

    if two_tracks:
        for p in (args.banker_track, args.customer_track):
            if not p.exists():
                print(f"ERROR: no such file: {p}", file=sys.stderr)
                return 2
        build_stereo(args.banker_track, args.customer_track, out, args.offset)
        channel = "L"
    else:
        if not args.mono.exists():
            print(f"ERROR: no such file: {args.mono}", file=sys.stderr)
            return 2
        build_mono(args.mono, out)
        channel = ""

    upsert_metadata(args.input_dir / "metadata.csv", {
        "call_id": call_id, "banker_id": args.banker_id,
        "file_name": out.name, "call_date": args.call_date,
        "call_type": args.call_type, "banker_channel": channel,
        "banker_name": args.banker_name,
    })

    print(f"wrote {out}")
    print(f"updated {args.input_dir / 'metadata.csv'}")
    if channel:
        print("stereo: banker on the left channel, customer on the right")
    else:
        print("mono: speaker attribution will need pyannote (set HF_TOKEN before processing)")
    print(f"\nnext:\n  python -m callqa validate-inputs\n"
          f"  python -m callqa process --audio {out} --call-id {call_id} "
          f"--banker-id {args.banker_id}"
          + (f" --banker-channel {channel}" if channel else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
