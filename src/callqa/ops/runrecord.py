"""The run manifest: a machine-readable record of a batch.

Written by the drivers (which already wrap process_call), never by process_call
itself. Per-stage timing is read from the state DB's `completed_at` column that
the core already writes — a read-only query, so state.py is untouched.
"""

from __future__ import annotations

import logging
import sqlite3
import threading
from datetime import UTC, datetime
from pathlib import Path

from callqa.config import Config
from callqa.models import CallResult
from callqa.ops.models import CallRunLink, CallRunSummary, RunManifest
from callqa.ops.provenance import fingerprint
from callqa.state import StateDB, atomic_write_model

logger = logging.getLogger(__name__)

RUNS_DIR = "runs"


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _run_id(git_sha: str) -> str:
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return f"{stamp}-{(git_sha or 'none')[:8]}"


def _stage_seconds(db_path: Path, call_id: str, call_start: float) -> dict[str, float]:
    """Per-stage durations from the state DB's completed_at timestamps.

    The first stage is anchored by the call's start time; each later stage is
    the gap since the previous stage completed. Read-only; if the DB can't be
    read the run record simply carries no per-stage breakdown for that call.
    """
    try:
        with sqlite3.connect(str(db_path)) as conn:
            rows = conn.execute(
                "SELECT stage, completed_at FROM stages WHERE call_id=? AND status='done' "
                "ORDER BY completed_at", (call_id,)).fetchall()
    except sqlite3.Error:  # pragma: no cover - defensive
        return {}
    out: dict[str, float] = {}
    prev = call_start
    for stage, done in rows:
        # Only stages that ran in THIS run. A resumed call keeps the earlier
        # run's completed_at for every stage it skips, and differencing those
        # against this run's start fabricated durations: the whole gap between
        # the two runs - hours, a weekend - landed on one stage, and stages that
        # did no work at all were reported with a time. Skipped stages simply
        # have no entry; the run record says what this run did.
        if done is None or done < call_start:
            continue
        out[stage] = round(max(0.0, done - prev), 3)
        prev = done
    return out


class RunRecorder:
    """Accumulates per-call outcomes for one batch and writes the manifest.

    Thread-safe: `cmd_run` records from a thread pool. Every method is
    best-effort — a failure to record must never affect the batch.
    """

    def __init__(self, config: Config, rubric_sha256: str, command: str) -> None:
        self.config = config
        self.command = command
        self.fingerprint = fingerprint(config, rubric_sha256)
        self.run_id = _run_id(self.fingerprint.get("git_sha", "none"))
        self.started_at = _now()
        self._calls: list[CallRunSummary] = []
        self._proc_seconds = 0.0
        self._lock = threading.Lock()

    def record(self, result: CallResult, seconds: float, call_start: float,
               state: StateDB) -> None:
        try:
            stage_seconds = _stage_seconds(state.db_path, result.call_id, call_start)
            summary = CallRunSummary(
                call_id=result.call_id, status=result.status, error=result.error,
                seconds=round(seconds, 3), stage_seconds=stage_seconds)
            with self._lock:
                self._calls.append(summary)
                self._proc_seconds += seconds
        except Exception as exc:  # noqa: BLE001 - recording must not break a run
            logger.debug("run-record: could not record %s: %s", result.call_id, exc)

    def write(self, output_dir: Path) -> Path | None:
        try:
            with self._lock:
                calls = list(self._calls)
                proc_seconds = self._proc_seconds
            counts: dict[str, int] = {}
            for c in calls:
                counts[c.status] = counts.get(c.status, 0) + 1
            gpu_hours = round(proc_seconds / 3600.0, 4)
            rate = self.config.monitoring.gpu_cost_per_hour
            cost = {"gpu_hours": gpu_hours, "rate_per_hour": rate,
                    "estimated": round(gpu_hours * rate, 4)}
            manifest = RunManifest(
                run_id=self.run_id, command=self.command, started_at=self.started_at,
                finished_at=_now(), fingerprint=self.fingerprint, counts=counts,
                cost=cost, calls=calls)
            runs = output_dir / RUNS_DIR
            path = runs / f"{self.run_id}.json"
            atomic_write_model(path, manifest)
            for c in calls:
                atomic_write_model(
                    runs / "by-call" / f"{c.call_id}.json",
                    CallRunLink(call_id=c.call_id, run_id=self.run_id,
                                fingerprint=self.fingerprint))
            return path
        except Exception as exc:  # noqa: BLE001
            logger.warning("run-record: could not write manifest: %s", exc)
            return None
