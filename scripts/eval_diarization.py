#!/usr/bin/env python3
"""Score speaker attribution on a single-channel recording against the truth.

Everything this system reports about a banker rests on who-said-what, and on a
single-channel recording that is inferred rather than observed. An inference
nobody measures is a guess, so this script measures it.

It reports three numbers, and the gap between the second and third is the one
that matters most here:

  DER              the standard diarization error rate: missed speech plus
                   false alarm plus speaker confusion, over reference speech.
  DER (best pair)  the same, with the two hypothesis speakers matched to the
                   reference in whichever direction scores better. This is the
                   quality of the SEPARATION alone.
  role accuracy    the share of reference speech where the role this pipeline
                   assigned (banker or customer) is the right one.

If DER (best pair) is good and role accuracy is near zero, the separation
worked and the roles came out backwards, which is a completely different bug
from a diarizer that cannot tell the speakers apart.

Reference format, either:
  RTTM  - the standard, one `SPEAKER <file> 1 <start> <dur> <NA> <NA> <label>`
          line per turn; labels may be anything, or banker/customer.
  CSV   - `start,end,speaker` with speaker in {banker, customer}, which is what
          a person can produce by hand from the report in a few minutes.

Usage:
  python scripts/eval_diarization.py --dialog data/output/transcripts/REAL001.dialog.json \
                                     --reference refs/REAL001.csv
  python scripts/eval_diarization.py --output-dir data/output --reference-dir refs
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path

FRAME_SEC = 0.01          # 10 ms scoring frames
DEFAULT_COLLAR = 0.25     # NIST convention: ignore +/- 250ms around a boundary


@dataclass
class Turn:
    start: float
    end: float
    speaker: str


# ---------------------------------------------------------------- loading

def load_reference(path: Path) -> list[Turn]:
    if path.suffix.lower() == ".rttm":
        turns = []
        for line in path.read_text(encoding="utf-8").splitlines():
            parts = line.split()
            if len(parts) < 8 or parts[0] != "SPEAKER":
                continue
            start, duration = float(parts[3]), float(parts[4])
            turns.append(Turn(start, start + duration, parts[7]))
        return sorted(turns, key=lambda t: t.start)

    turns = []
    with path.open(newline="", encoding="utf-8-sig") as fh:
        for row in csv.DictReader(fh):
            turns.append(Turn(float(row["start"]), float(row["end"]),
                              (row.get("speaker") or "").strip()))
    return sorted(turns, key=lambda t: t.start)


def load_hypothesis(path: Path) -> list[Turn]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return sorted(
        (Turn(float(t["start"]), float(t["end"]), t["speaker"]) for t in data["turns"]),
        key=lambda t: t.start,
    )


# ---------------------------------------------------------------- scoring

def _frames(turns: list[Turn], n_frames: int) -> list[str | None]:
    """One label per frame; later turns win an overlap, which matches the
    exclusive attribution this pipeline produces."""
    grid: list[str | None] = [None] * n_frames
    for turn in turns:
        lo = max(0, int(turn.start / FRAME_SEC))
        hi = min(n_frames, int(round(turn.end / FRAME_SEC)))
        for i in range(lo, hi):
            grid[i] = turn.speaker
    return grid


def _collar_mask(turns: list[Turn], n_frames: int, collar: float) -> list[bool]:
    """False for frames too close to a reference boundary to be scored."""
    mask = [True] * n_frames
    width = int(collar / FRAME_SEC)
    for turn in turns:
        for boundary in (turn.start, turn.end):
            centre = int(boundary / FRAME_SEC)
            for i in range(max(0, centre - width), min(n_frames, centre + width + 1)):
                mask[i] = False
    return mask


def score(reference: list[Turn], hypothesis: list[Turn],
          collar: float = DEFAULT_COLLAR) -> dict:
    end = max([t.end for t in reference + hypothesis] or [0.0])
    n = int(end / FRAME_SEC) + 1
    ref, hyp = _frames(reference, n), _frames(hypothesis, n)
    scored = _collar_mask(reference, n, collar)

    ref_labels = sorted({t.speaker for t in reference})
    hyp_labels = sorted({t.speaker for t in hypothesis})

    def errors(mapping: dict[str, str]) -> tuple[int, int, int, int]:
        miss = false_alarm = confusion = total = 0
        for i in range(n):
            if not scored[i]:
                continue
            r, h = ref[i], hyp[i]
            if r is not None:
                total += 1
            if r is not None and h is None:
                miss += 1
            elif r is None and h is not None:
                false_alarm += 1
            elif r is not None and h is not None and mapping.get(h, h) != r:
                confusion += 1
        return miss, false_alarm, confusion, total

    identity = {label: label for label in hyp_labels}
    miss, fa, conf, total = errors(identity)
    as_given = (miss + fa + conf) / total if total else 0.0

    # Best of the two possible pairings, which isolates separation quality
    # from whether the roles ended up the right way round.
    best = as_given
    if len(ref_labels) == 2 and len(hyp_labels) == 2:
        swapped = {hyp_labels[0]: ref_labels[1], hyp_labels[1]: ref_labels[0]}
        straight = {hyp_labels[0]: ref_labels[0], hyp_labels[1]: ref_labels[1]}
        best = min(
            (sum(errors(m)[:3]) / total if total else 0.0) for m in (straight, swapped)
        )

    # Role accuracy only means anything when the reference names the roles.
    role_total = role_right = 0
    if set(ref_labels) <= {"banker", "customer"}:
        for i in range(n):
            if not scored[i] or ref[i] is None:
                continue
            role_total += 1
            role_right += int(hyp[i] == ref[i])

    return {
        "der": round(as_given, 4),
        "der_best_pairing": round(best, 4),
        "missed_speech": round(miss / total, 4) if total else 0.0,
        "false_alarm": round(fa / total, 4) if total else 0.0,
        "speaker_confusion": round(conf / total, 4) if total else 0.0,
        "role_accuracy": round(role_right / role_total, 4) if role_total else None,
        "reference_speech_sec": round(total * FRAME_SEC, 1),
        "collar_sec": collar,
    }


# ---------------------------------------------------------------- cli

def _print(call_id: str, result: dict) -> None:
    role = result["role_accuracy"]
    print(f"{call_id}")
    print(f"  DER                {result['der'] * 100:6.2f}%")
    print(f"  DER (best pairing) {result['der_best_pairing'] * 100:6.2f}%")
    print(f"    missed speech    {result['missed_speech'] * 100:6.2f}%")
    print(f"    false alarm      {result['false_alarm'] * 100:6.2f}%")
    print(f"    confusion        {result['speaker_confusion'] * 100:6.2f}%")
    if role is not None:
        print(f"  role accuracy      {role * 100:6.2f}%"
              + ("   <-- roles are inverted" if role < 0.25 else ""))
    print(f"  scored speech      {result['reference_speech_sec']}s"
          f" (collar {result['collar_sec']}s)")


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
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dialog", type=Path, help="one <call>.dialog.json")
    ap.add_argument("--reference", type=Path, help="matching .rttm or .csv")
    ap.add_argument("--output-dir", type=Path, help="score every call with a reference")
    ap.add_argument("--reference-dir", type=Path)
    ap.add_argument("--collar", type=float, default=DEFAULT_COLLAR)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    pairs: list[tuple[str, Path, Path]] = []
    if args.dialog and args.reference:
        pairs.append((args.dialog.name.split(".")[0], args.dialog, args.reference))
    elif args.output_dir and args.reference_dir:
        for dialog in sorted((args.output_dir / "transcripts").glob("*.dialog.json")):
            call_id = dialog.name.split(".")[0]
            for suffix in (".rttm", ".csv"):
                candidate = args.reference_dir / f"{call_id}{suffix}"
                if candidate.exists():
                    pairs.append((call_id, dialog, candidate))
                    break
    else:
        ap.error("pass --dialog with --reference, or --output-dir with --reference-dir")

    if not pairs:
        print("no call had a reference file; nothing to score", file=sys.stderr)
        return 1

    results = {}
    for call_id, dialog, reference in pairs:
        results[call_id] = score(load_reference(reference), load_hypothesis(dialog),
                                 args.collar)
        if not args.json:
            _print(call_id, results[call_id])

    if args.json:
        print(json.dumps(results, indent=2))
    elif len(results) > 1:
        mean_der = sum(r["der"] for r in results.values()) / len(results)
        print(f"\n{len(results)} calls, mean DER {mean_der * 100:.2f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
