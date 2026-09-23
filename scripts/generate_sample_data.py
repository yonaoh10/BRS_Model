#!/usr/bin/env python3
"""Generate synthetic sample data for mock/E2E runs.

Creates:
- 6 synthetic stereo WAVs (distinct tone/noise bursts per channel, so VAD
  finds alternating speech segments per side),
- data/input/metadata.csv (3 bankers),
- data/input/human_ratings.csv (2 raters on some calls, seeded, correlated
  with the deterministic mock-judge scores so calibration is meaningful).

No models, no network - pure numpy + stdlib.
"""

from __future__ import annotations

import argparse
import csv
import sys
import wave
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from callqa.judge.mock_judge import _stable_int  # noqa: E402

SAMPLE_RATE = 16000
DURATION_SEC = 60
SLOT_SEC = 8.0
BURST_SEC = 6.0

DIMENSIONS = [
    "identification", "compliance", "empathy", "listening",
    "clarity", "resolution", "suitability", "closure",
]

CALLS = [
    ("CALL001", "B001", "רון לוי", "2026-08-01", "service"),
    ("CALL002", "B001", "רון לוי", "2026-08-03", "service"),
    ("CALL003", "B002", "דנה כהן", "2026-08-05", "loans"),
    ("CALL004", "B002", "דנה כהן", "2026-08-08", "service"),
    ("CALL005", "B003", "יוסי מזרחי", "2026-08-10", "cards"),
    ("CALL006", "B003", "יוסי מזרחי", "2026-08-12", "service"),
]


def synth_channel(active_slots: list[int], tone_hz: float, rng: np.random.Generator) -> np.ndarray:
    n = SAMPLE_RATE * DURATION_SEC
    signal = rng.normal(0, 0.002, n).astype(np.float32)  # silent noise floor
    t = np.arange(int(SAMPLE_RATE * BURST_SEC)) / SAMPLE_RATE
    for slot in active_slots:
        start = int(slot * SLOT_SEC * SAMPLE_RATE)
        end = min(start + len(t), n)
        if start >= n:
            break
        # Tone + amplitude modulation + noise: speech-ish energy envelope.
        burst = 0.35 * np.sin(2 * np.pi * tone_hz * t) * (0.6 + 0.4 * np.sin(2 * np.pi * 3.1 * t))
        burst += rng.normal(0, 0.02, len(t))
        signal[start:end] += burst[: end - start].astype(np.float32)
    return signal


def write_stereo_wav(path: Path, left: np.ndarray, right: np.ndarray) -> None:
    stereo = np.stack([left, right], axis=1)
    pcm = (np.clip(stereo, -1, 1) * 32767).astype(np.int16)
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(2)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(pcm.tobytes())


def mock_score(call_id: str, dim: str) -> int:
    return 2 + _stable_int(call_id, dim) % 4


def main() -> int:
    # Printing a Hebrew path to a redirected stream is a UnicodeEncodeError
    # under the Windows ANSI code page (callqa.portable.configure_stdio, inlined
    # because this script may run before callqa is installed).
    for stream in (sys.stdout, sys.stderr):
        if (getattr(stream, "encoding", "") or "").lower().replace("-", "") != "utf8":
            try:
                stream.reconfigure(encoding="utf-8", errors="backslashreplace")
            except (AttributeError, ValueError, OSError):
                pass
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", default="data/input", type=Path)
    parser.add_argument("--seed", default=42, type=int)
    args = parser.parse_args()

    calls_dir = args.input_dir / "calls"
    n_slots = int(DURATION_SEC / SLOT_SEC)

    for i, (call_id, _banker_id, _name, _date, _ctype) in enumerate(CALLS):
        rng = np.random.default_rng(args.seed + i)
        # Banker on L (even slots), customer on R (odd slots); distinct tones.
        left = synth_channel([s for s in range(n_slots) if s % 2 == 0], 220 + 30 * i, rng)
        right = synth_channel([s for s in range(n_slots) if s % 2 == 1], 440 + 30 * i, rng)
        write_stereo_wav(calls_dir / f"{call_id}.wav", left, right)

    # utf-8-sig so Excel opens them as UTF-8 (and saves them back that way).
    with (args.input_dir / "metadata.csv").open("w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            ["call_id", "banker_id", "file_name", "call_date", "call_type",
             "banker_channel", "banker_name"]
        )
        for call_id, banker_id, name, date, ctype in CALLS:
            writer.writerow([call_id, banker_id, f"{call_id}.wav", date, ctype, "L", name])

    # Human ratings: rater R1 on all calls, rater R2 on half (doubly-rated).
    rng = np.random.default_rng(args.seed)
    with (args.input_dir / "human_ratings.csv").open("w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.writer(fh)
        writer.writerow(["call_id", "rater_id", *DIMENSIONS])
        for call_id, *_ in CALLS:
            for rater in ("R1", "R2"):
                if rater == "R2" and int(call_id[-1]) % 2 == 0:
                    continue
                row = [call_id, rater]
                for dim in DIMENSIONS:
                    noise = int(rng.integers(-1, 2))  # -1, 0, or 1
                    row.append(int(np.clip(mock_score(call_id, dim) + noise, 1, 5)))
                writer.writerow(row)

    print(f"Sample data written under {args.input_dir}: "
          f"{len(CALLS)} stereo WAVs, metadata.csv, human_ratings.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
