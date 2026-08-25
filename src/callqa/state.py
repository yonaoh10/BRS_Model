"""SQLite run-state: per call x stage completion tracking + per-call locks.

Also provides the atomic-write helper used for every artifact.
Re-running the pipeline skips stages recorded as done (unless --force).
"""

from __future__ import annotations

import json
import os
import sqlite3
import tempfile
import time
from pathlib import Path

from pydantic import BaseModel

_SCHEMA = """
CREATE TABLE IF NOT EXISTS stages (
    call_id TEXT NOT NULL,
    stage TEXT NOT NULL,
    status TEXT NOT NULL,
    artifact_path TEXT,
    completed_at REAL NOT NULL,
    PRIMARY KEY (call_id, stage)
);
CREATE TABLE IF NOT EXISTS locks (
    call_id TEXT PRIMARY KEY,
    pid INTEGER NOT NULL,
    hostname TEXT NOT NULL,
    acquired_at REAL NOT NULL
);
"""


class CallLockedError(RuntimeError):
    """Raised when another process holds the lock for a call_id."""


def atomic_write_text(path: Path, content: str) -> None:
    """Write a file atomically: tmp file in the same directory + rename."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(content)
        os.replace(tmp_name, path)
    except BaseException:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


def atomic_write_model(path: Path, model: BaseModel) -> None:
    atomic_write_text(path, model.model_dump_json(indent=2))


def atomic_write_json(path: Path, data: object) -> None:
    atomic_write_text(path, json.dumps(data, ensure_ascii=False, indent=2))


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


class StateDB:
    """Thin wrapper around the SQLite state database.

    Connections are opened per operation so the object is safe to share
    across threads (the `run` driver may use a thread pool).
    """

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.executescript(_SCHEMA)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=30)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=30000")
        return conn

    # -- stage tracking -------------------------------------------------

    def mark_stage_done(self, call_id: str, stage: str, artifact_path: str | Path | None) -> None:
        with self._connect() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO stages (call_id, stage, status, artifact_path, completed_at)"
                " VALUES (?, ?, 'done', ?, ?)",
                (call_id, stage, str(artifact_path) if artifact_path else None, time.time()),
            )

    def is_stage_done(self, call_id: str, stage: str) -> bool:
        """A stage counts as done only if recorded AND its artifact still exists."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT artifact_path FROM stages WHERE call_id=? AND stage=? AND status='done'",
                (call_id, stage),
            ).fetchone()
        if row is None:
            return False
        artifact_path = row[0]
        if artifact_path and not Path(artifact_path).exists():
            return False
        return True

    def stage_artifact(self, call_id: str, stage: str) -> Path | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT artifact_path FROM stages WHERE call_id=? AND stage=? AND status='done'",
                (call_id, stage),
            ).fetchone()
        if row and row[0] and Path(row[0]).exists():
            return Path(row[0])
        return None

    def clear_call(self, call_id: str) -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM stages WHERE call_id=?", (call_id,))

    def completed_stages(self, call_id: str) -> list[str]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT stage FROM stages WHERE call_id=? AND status='done' ORDER BY completed_at",
                (call_id,),
            ).fetchall()
        return [r[0] for r in rows]

    # -- per-call locks -------------------------------------------------

    def acquire_lock(self, call_id: str) -> None:
        """Acquire the processing lock for call_id or raise CallLockedError.

        A lock whose owning pid is dead (same host) is considered stale and
        is stolen.
        """
        hostname = os.uname().nodename
        with self._connect() as conn:
            try:
                conn.execute(
                    "INSERT INTO locks (call_id, pid, hostname, acquired_at) VALUES (?, ?, ?, ?)",
                    (call_id, os.getpid(), hostname, time.time()),
                )
                return
            except sqlite3.IntegrityError:
                row = conn.execute(
                    "SELECT pid, hostname FROM locks WHERE call_id=?", (call_id,)
                ).fetchone()
        if row is not None:
            pid, lock_host = row
            if lock_host == hostname and pid != os.getpid() and not _pid_alive(pid):
                # Stale lock from a dead process: steal it.
                with self._connect() as conn:
                    conn.execute(
                        "UPDATE locks SET pid=?, hostname=?, acquired_at=? WHERE call_id=?",
                        (os.getpid(), hostname, time.time(), call_id),
                    )
                return
        raise CallLockedError(
            f"call_id={call_id} is already being processed (locked by pid {row[0] if row else '?'})"
        )

    def release_lock(self, call_id: str) -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM locks WHERE call_id=?", (call_id,))
