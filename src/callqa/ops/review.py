"""The feedback loop: capture a reviewer's verdict on a held call and feed it
into the calibration set.

A call held for `needs_human_review` is the most valuable thing the system
produces — a human's judgement on a hard case — and today it goes nowhere. This
appends the reviewer's per-dimension scores to human_ratings.csv, the exact file
`callqa calibrate` already consumes, so every review makes the next measurement
better. It is a CLI capability, not a dashboard POST: the dashboard stays
read-only and deletable.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path


class ReviewError(ValueError):
    pass


def _load_results(output_dir: Path) -> dict[str, str]:
    statuses: dict[str, str] = {}
    for p in sorted((output_dir / "results").glob("*.json")):
        try:
            r = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if isinstance(r, dict) and r.get("call_id"):
            statuses[r["call_id"]] = r.get("status", "success")
    return statuses


def _existing_verdicts(ratings_csv: Path) -> set[tuple[str, str]]:
    """(call_id, rater_id) pairs already recorded."""
    pairs: set[tuple[str, str]] = set()
    if not ratings_csv.exists():
        return pairs
    with ratings_csv.open(encoding="utf-8-sig", newline="") as fh:
        for row in csv.DictReader(fh):
            cid, rater = (row.get("call_id") or "").strip(), (row.get("rater_id") or "").strip()
            if cid and rater:
                pairs.add((cid, rater))
    return pairs


def pending_reviews(output_dir: Path, ratings_csv: Path, rater: str | None = None) -> list[str]:
    """Calls held for review that have no recorded verdict (from `rater` if given)."""
    held = [cid for cid, status in _load_results(output_dir).items()
            if status == "needs_human_review"]
    verdicts = _existing_verdicts(ratings_csv)
    rated = {cid for cid, r in verdicts if rater is None or r == rater}
    return sorted(cid for cid in held if cid not in rated)


def record_review(ratings_csv: Path, call_id: str, rater_id: str,
                  scores: dict[str, int], dim_ids: list[str]) -> None:
    """Validate and append one reviewer verdict to human_ratings.csv."""
    call_id, rater_id = call_id.strip(), rater_id.strip()
    if not call_id or not rater_id:
        raise ReviewError("call_id and rater_id are required")
    missing = [d for d in dim_ids if d not in scores]
    if missing:
        raise ReviewError(f"missing scores for: {', '.join(missing)}")
    for dim, value in scores.items():
        if dim not in dim_ids:
            raise ReviewError(f"unknown dimension: {dim}")
        if not (isinstance(value, int) and 1 <= value <= 5):
            raise ReviewError(f"{dim}={value} is not an integer 1..5")
    if (call_id, rater_id) in _existing_verdicts(ratings_csv):
        raise ReviewError(f"{rater_id} has already reviewed {call_id}")

    header = ["call_id", "rater_id", *dim_ids]
    # "Has a header already" is not the same question as "exists". A file that
    # exists but is EMPTY - created by a touch, an interrupted write, or a
    # spreadsheet saving nothing - took the append path and wrote the row with
    # no header line. csv.DictReader then read that first verdict AS the header,
    # so the reviewer's judgement vanished and every later row was misaligned.
    # The file has a header only if its first line is one.
    existing_header = None
    if ratings_csv.exists() and ratings_csv.stat().st_size > 0:
        with ratings_csv.open(encoding="utf-8-sig", newline="") as fh:
            existing_header = next(csv.reader(fh), None)
    has_header = bool(existing_header)
    if has_header:  # align to the file's own column order
        # ...but never by throwing a reviewer's score away. When the rubric has
        # gained a dimension since the file was started, the file's header does
        # not have that column, and the row used to be written without it - the
        # score silently dropped while the CLI printed "recorded". A human's
        # judgement on a held call is the scarcest input this system has.
        missing = [c for c in ["call_id", "rater_id", *dim_ids] if c not in existing_header]
        if missing:
            raise ReviewError(
                f"{ratings_csv.name} has no column for {', '.join(missing)}: the rubric "
                "has changed since this file was started, and writing the verdict "
                "would drop those scores. Add the column(s) to the file's header "
                "(existing rows may leave them empty), then record the review again."
            )
        header = existing_header
    ratings_csv.parent.mkdir(parents=True, exist_ok=True)
    row = {"call_id": call_id, "rater_id": rater_id, **{d: scores[d] for d in dim_ids}}
    with ratings_csv.open("a", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=header)
        if not has_header:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in header})
