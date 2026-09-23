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
RETENTION_DIR = "retention"


@dataclass
class Expired:
    path: Path
    age_days: int
    bytes: int


def find_expired(output_dir: Path, raw_days: int, now: float | None = None) -> list[Expired]:
    now = now if now is not None else time.time()
    cutoff = raw_days * 86400
    out: list[Expired] = []
    for glob in RAW_GLOBS:
        for f in sorted(output_dir.glob(glob)):
            if not f.is_file():
                continue
            age = now - f.stat().st_mtime
            if age > cutoff:
                out.append(Expired(f, int(age // 86400), f.stat().st_size))
    return out


def apply_retention(output_dir: Path, raw_days: int, now: float | None = None) -> list[dict]:
    """Destroy expired RAW artifacts; append each to retention/log.jsonl."""
    expired = find_expired(output_dir, raw_days, now)
    log_path = output_dir / RETENTION_DIR / "log.jsonl"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    destroyed: list[dict] = []
    stamp = datetime.now(UTC).isoformat(timespec="seconds")
    with log_path.open("a", encoding="utf-8") as fh:
        for e in expired:
            try:
                rel = str(e.path.relative_to(output_dir))
                e.path.unlink()
            except OSError as exc:  # pragma: no cover - defensive
                logger.warning("could not destroy %s: %s", e.path, exc)
                continue
            rec = {"at": stamp, "path": rel, "age_days": e.age_days, "bytes": e.bytes}
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            destroyed.append(rec)
    return destroyed
