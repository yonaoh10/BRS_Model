"""Unit tests: SQLite state tracking, per-call locks, atomic writes."""

from __future__ import annotations

from pathlib import Path

import pytest

from callqa.state import CallLockedError, StateDB, atomic_write_text


def test_stage_tracking_roundtrip(tmp_path: Path) -> None:
    db = StateDB(tmp_path / "state.db")
    artifact = tmp_path / "artifact.json"
    artifact.write_text("{}")
    assert not db.is_stage_done("C1", "asr")
    db.mark_stage_done("C1", "asr", artifact)
    assert db.is_stage_done("C1", "asr")
    assert db.completed_stages("C1") == ["asr"]
    # A recorded stage whose artifact vanished no longer counts as done.
    artifact.unlink()
    assert not db.is_stage_done("C1", "asr")


def test_clear_call(tmp_path: Path) -> None:
    db = StateDB(tmp_path / "state.db")
    artifact = tmp_path / "a.json"
    artifact.write_text("{}")
    db.mark_stage_done("C1", "asr", artifact)
    db.clear_call("C1")
    assert db.completed_stages("C1") == []


def test_lock_refuses_second_acquire(tmp_path: Path) -> None:
    db = StateDB(tmp_path / "state.db")
    db.acquire_lock("C1")
    with pytest.raises(CallLockedError):
        db.acquire_lock("C1")
    db.release_lock("C1")
    db.acquire_lock("C1")  # re-acquirable after release
    db.release_lock("C1")


def test_stale_lock_from_dead_pid_is_stolen(tmp_path: Path) -> None:
    db = StateDB(tmp_path / "state.db")
    import os
    import sqlite3

    with sqlite3.connect(db.db_path) as conn:
        conn.execute(
            "INSERT INTO locks (call_id, pid, hostname, acquired_at) VALUES (?, ?, ?, 0)",
            ("C1", 999999999, os.uname().nodename),
        )
    db.acquire_lock("C1")  # dead pid -> stolen, no exception
    db.release_lock("C1")


def test_atomic_write(tmp_path: Path) -> None:
    target = tmp_path / "nested" / "file.txt"
    atomic_write_text(target, "hello")
    assert target.read_text() == "hello"
    atomic_write_text(target, "world")
    assert target.read_text() == "world"
    # No stray temp files left behind.
    assert list(target.parent.glob("*.tmp")) == []


def test_concurrent_writes_do_not_lock_the_database(tmp_path: Path) -> None:
    """Under a thread pool the state DB is hammered from many connections at
    once. Re-asserting PRAGMA journal_mode=WAL on every connection raced into
    'database is locked' and aborted the whole run; WAL is now set once."""
    import threading

    db = StateDB(tmp_path / "state.db")
    errors: list[Exception] = []

    def worker(n: int) -> None:
        try:
            for i in range(8):
                db.mark_stage_done(f"C{n}", f"stage{i}", None)
                db.completed_stages(f"C{n}")
        except Exception as exc:  # noqa: BLE001 - capturing is the point
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(n,)) for n in range(10)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, f"concurrent access raised: {errors[:3]}"
    assert len(db.completed_stages("C0")) == 8
