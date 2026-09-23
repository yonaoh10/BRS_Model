"""Retention: raw PII-bearing artifacts are retired on schedule; derived
non-PII is kept; every destruction is logged."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

from callqa.ops.retention import apply_retention, find_expired


def _write(path: Path, content: str, age_days: float) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    old = time.time() - age_days * 86400
    os.utime(path, (old, old))
    return path


def _populate(out: Path) -> None:
    # RAW (PII) — old ones should be destroyed
    _write(out / "transcripts" / "OLD.json", "raw transcript", 120)
    _write(out / "transcripts" / "OLD.dialog.json", "raw dialog", 120)
    _write(out / "audio" / "wav" / "OLD.mono.wav", "raw audio", 120)
    # RAW but recent — kept
    _write(out / "transcripts" / "NEW.json", "raw transcript", 10)
    # DERIVED non-PII — always kept, even when old
    _write(out / "redacted" / "OLD.json", "redacted", 120)
    _write(out / "scores" / "OLD.json", "score", 120)
    _write(out / "reports" / "calls" / "OLD.html", "report", 120)
    _write(out / "redacted_audio" / "OLD.wav", "redacted audio", 120)


def test_find_expired_lists_only_old_raw(tmp_path: Path) -> None:
    out = tmp_path / "output"
    _populate(out)
    names = {e.path.name for e in find_expired(out, raw_days=90)}
    assert names == {"OLD.json", "OLD.dialog.json", "OLD.mono.wav"}
    # NEW.json is recent; redacted/scores/reports/redacted_audio are derived
    assert "NEW.json" not in names


def test_apply_destroys_raw_keeps_derived_and_logs(tmp_path: Path) -> None:
    out = tmp_path / "output"
    _populate(out)
    destroyed = apply_retention(out, raw_days=90)

    # raw, old -> gone
    assert not (out / "transcripts" / "OLD.json").exists()
    assert not (out / "transcripts" / "OLD.dialog.json").exists()
    assert not (out / "audio" / "wav" / "OLD.mono.wav").exists()
    # recent raw and ALL derived non-PII -> kept
    assert (out / "transcripts" / "NEW.json").exists()
    assert (out / "redacted" / "OLD.json").exists()
    assert (out / "scores" / "OLD.json").exists()
    assert (out / "reports" / "calls" / "OLD.html").exists()
    assert (out / "redacted_audio" / "OLD.wav").exists()

    # audit log records each destruction
    log = (out / "retention" / "log.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(log) == 3 == len(destroyed)
    rec = json.loads(log[0])
    assert {"at", "path", "age_days", "bytes"} <= rec.keys()


def test_nothing_destroyed_when_all_recent(tmp_path: Path) -> None:
    out = tmp_path / "output"
    _write(out / "transcripts" / "R.json", "raw", 5)
    _write(out / "audio" / "wav" / "R.mono.wav", "raw", 5)
    assert apply_retention(out, raw_days=90) == []
    assert (out / "transcripts" / "R.json").exists()


def test_a_crash_orphaned_raw_temp_file_is_retired_too(tmp_path: Path) -> None:
    """An atomic write that is killed mid-flight (SIGKILL, OOM, power loss)
    leaves ".<call>.dialog.json.<random>.tmp" holding the complete unredacted
    transcript. "transcripts/*.json" could never match it, so it outlived every
    retention run while the command reported the raw data destroyed."""
    out = tmp_path / "output"
    orphan = _write(out / "transcripts" / ".C1.dialog.json.k3j2.tmp",
                    '{"turns": [{"text": "raw"}]}', age_days=120)
    kept = _write(out / "redacted" / "C1.json", '{"turns": []}', age_days=120)

    expired = {e.path for e in find_expired(out, raw_days=90)}
    assert orphan in expired
    assert kept not in expired                    # derived, non-PII: kept
