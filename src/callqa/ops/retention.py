"""Retention: retire RAW customer data on a schedule, with an audit trail.

Raw transcripts and raw audio contain customer identifiers; the bank must be
able to say how long they live and how they are destroyed. This destroys the
RAW artifacts older than `retention.raw_days` and KEEPS the derived, non-PII
outputs — redacted transcripts, redacted audio, scores, reports, calibration —
writing every deletion to a log for audit.

Note on destruction: on an SSD an unlink does not guarantee the bytes are
overwritten; the bank's storage layer (e.g. full-disk encryption) is the
backstop. This tool removes the file and records that it did.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from callqa.portable import remove_file

logger = logging.getLogger(__name__)

# RAW, PII-bearing artifacts. "transcripts/*.json" catches both the raw
# transcript and the raw *.dialog.json. redacted/, scores/, reports/,
# redacted_audio/ are DERIVED non-PII and are deliberately NOT listed.
# EVERYTHING in these two directories, not just the expected names. Every file
# here is raw by definition, and "*.json" missed the one kind most likely to be
# forgotten: the temp file an atomic write leaves behind when the process is
# killed mid-write (SIGKILL, an OOM kill, power loss). It is named
# ".<call>.dialog.json.<random>.tmp", holds the complete unredacted transcript,
# and survived every retention run because of its suffix.
RAW_GLOBS = ["transcripts/*", "audio/wav/*"]
# The ORIGINAL recordings the `watch` driver moved out of the drop directory
# once it had processed them. These are the rawest customer data in the whole
# system - the voice itself, unredacted - and retention used to sweep only the
# output tree, so every recording `watch` had ever handled was kept forever.
# Only the two folders the pipeline itself creates are touched: never
# input/calls/, which belongs to the bank's recording system and its own
# retention policy.
RAW_INPUT_GLOBS = ["processed/*", "failed/*"]
RETENTION_DIR = "retention"


@dataclass
class Expired:
    path: Path
    age_days: int
    bytes: int


def _sweep_roots(output_dir: Path, input_dir: Path | None) -> list[tuple[Path, str]]:
    roots = [(output_dir, g) for g in RAW_GLOBS]
    if input_dir is not None:
        roots += [(input_dir, g) for g in RAW_INPUT_GLOBS]
    return roots


def find_expired(output_dir: Path, raw_days: int, now: float | None = None,
                 input_dir: Path | None = None) -> list[Expired]:
    now = now if now is not None else time.time()
    cutoff = raw_days * 86400
    out: list[Expired] = []
    for root, glob in _sweep_roots(output_dir, input_dir):
        for f in sorted(root.glob(glob)):
            if not f.is_file():
                continue
            age = now - f.stat().st_mtime
            if age > cutoff:
                out.append(Expired(f, int(age // 86400), f.stat().st_size))
    return out


def _label(path: Path, output_dir: Path, input_dir: Path | None) -> str:
    for root, prefix in ((output_dir, ""), (input_dir, "input/")):
        if root is not None:
            try:
                return prefix + path.relative_to(root).as_posix()
            except ValueError:
                continue
    return str(path)                                  # pragma: no cover


def apply_retention(output_dir: Path, raw_days: int, now: float | None = None,
                    input_dir: Path | None = None) -> list[dict]:
    """Destroy expired RAW artifacts; append each to retention/log.jsonl."""
    expired = find_expired(output_dir, raw_days, now, input_dir)
    log_path = output_dir / RETENTION_DIR / "log.jsonl"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    destroyed: list[dict] = []
    stamp = datetime.now(UTC).isoformat(timespec="seconds")
    with log_path.open("a", encoding="utf-8") as fh:
        for e in expired:
            rel = _label(e.path, output_dir, input_dir)
            try:
                remove_file(e.path)
            except OSError as exc:
                # The label, not the full path: a recording's file name is
                # where a customer's number turns up.
                logger.error("could not destroy %s: %s", rel, type(exc).__name__)
                continue
            rec = {"at": stamp, "path": rel, "age_days": e.age_days, "bytes": e.bytes}
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            destroyed.append(rec)
    return destroyed
