"""SQLite run-state: per call x stage completion tracking + per-call locks.

Also provides the atomic-write helper used for every artifact.
Re-running the pipeline skips stages recorded as done (unless --force).
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import tempfile
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from pydantic import BaseModel

from callqa.portable import hostname as _hostname
from callqa.portable import is_network_path, pid_alive, replace, started_after

logger = logging.getLogger(__name__)

# A lock older than this is assumed abandoned. Processing one call takes
# minutes; a lock held for hours means the process that took it is gone, and on
# another host its liveness cannot be checked at all.
LOCK_TTL_SECONDS = 6 * 3600

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
        replace(tmp_name, path)
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
    # Never os.kill(pid, 0) here: on Windows signal 0 is CTRL_C_EVENT and any
    # other value is TerminateProcess - the liveness probe would interrupt or
    # kill the very process it was asking about.
    return pid_alive(pid)


class StateDB:
    """Thin wrapper around the SQLite state database.

    Connections are opened per operation so the object is safe to share
    across threads (the `run` driver may use a thread pool).
    """

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        # WAL is a PERSISTENT property of the database file, so set it once here.
        # Re-issuing PRAGMA journal_mode=WAL on every connection needs an
        # exclusive header lock that the busy-timeout does not cover, so under a
        # thread pool the connections raced and raised "database is locked",
        # aborting the whole run.
        conn = sqlite3.connect(self.db_path, timeout=30)
        # WAL needs shared memory between every process using the database,
        # which a network share cannot give: SQLite documents WAL as unsafe
        # there. Preflight refuses such a location; if someone runs anyway,
        # the rollback journal at least stays correct.
        journal = "DELETE" if is_network_path(self.db_path.parent) else "WAL"
        try:
            conn.execute(f"PRAGMA journal_mode={journal}")
            conn.executescript(_SCHEMA)
            conn.commit()
        finally:
            conn.close()

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        """A connection that is committed AND closed.

        `with sqlite3.connect(...)` commits but does not close, so every call
        leaked a file descriptor; a long watch run accumulated hundreds.
        """
        conn = sqlite3.connect(self.db_path, timeout=30)
        try:
            conn.execute("PRAGMA busy_timeout=30000")
            with conn:
                yield conn
        finally:
            conn.close()

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
                "SELECT stage FROM stages WHERE call_id=? AND status='done' "
                "ORDER BY completed_at, rowid",
                (call_id,),
            ).fetchall()
        return [r[0] for r in rows]

    # -- per-call locks -------------------------------------------------

    def _lock_age(self, call_id: str) -> float:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT acquired_at FROM locks WHERE call_id=?", (call_id,)
            ).fetchone()
        return time.time() - row[0] if row else 0.0

    def acquire_lock(self, call_id: str) -> None:
        """Acquire the processing lock for call_id or raise CallLockedError.

        A lock whose owning pid is dead (same host) is considered stale and
        is stolen.
        """
        hostname = _hostname()
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
            acquired_at = time.time() - self._lock_age(call_id)
            stale_by_death = (
                lock_host == hostname and pid != os.getpid()
                and (not _pid_alive(pid) or started_after(pid, acquired_at))
            )
            # A lock left behind by a container that no longer exists can never
            # be shown dead from here, because the pid belongs to another host.
            # Without an age limit that call is unprocessable forever.
            stale_by_age = self._lock_age(call_id) > LOCK_TTL_SECONDS
            if stale_by_death or stale_by_age:
                # One conditional UPDATE, not read-then-write: every racing
                # process otherwise saw the same dead pid and every one of them
                # took the lock, which is exactly the case this guards.
                with self._connect() as conn:
                    changed = conn.execute(
                        "UPDATE locks SET pid=?, hostname=?, acquired_at=? "
                        "WHERE call_id=? AND pid=? AND hostname=?",
                        (os.getpid(), hostname, time.time(), call_id, pid, lock_host),
                    ).rowcount
                if changed:
                    if stale_by_age and not stale_by_death:
                        logger.warning(
                            "call_id=%s: stole a lock older than %ds held by pid %s on %s",
                            call_id, LOCK_TTL_SECONDS, pid, lock_host,
                        )
                    return
        raise CallLockedError(
            f"call_id={call_id} is already being processed (locked by pid {row[0] if row else '?'})"
        )

    def release_lock(self, call_id: str) -> None:
        # ONLY our own lock. An unconditional delete would remove a lock that a
        # DIFFERENT process now owns: if this process's lock was stolen (by age
        # or a recycled pid) while it was still alive, its finally-block release
        # must not clear the new owner's fresh lock and let a third process
        # acquire the same call concurrently. Matching pid+hostname makes a
        # release a no-op once the lock is no longer ours.
        with self._connect() as conn:
            conn.execute(
                "DELETE FROM locks WHERE call_id=? AND pid=? AND hostname=?",
                (call_id, os.getpid(), _hostname()),
            )
