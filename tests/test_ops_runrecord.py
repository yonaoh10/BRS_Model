"""Run manifests + reproducibility verification (the ops provenance layer)."""

from __future__ import annotations

import wave
from pathlib import Path

import numpy as np

from callqa.models import CallInput
from callqa.ops.models import RunManifest
from callqa.ops.provenance import config_sha256, fingerprint
from callqa.ops.runrecord import RunRecorder
from callqa.ops.verify import verify_call
from callqa.pipeline import process_call
from callqa.rubric import load_rubric
from callqa.state import StateDB

RATE = 16000


def _speech_wav(path: Path, seconds: float = 45.0) -> Path:
    t = np.arange(int(RATE * seconds)) / RATE
    sig = np.zeros_like(t, dtype=np.float32)
    for i in range(int(seconds // 6)):
        lo, hi = int(i * 6 * RATE), int((i * 6 + 5) * RATE)
        sig[lo:hi] = (0.3 * np.sin(2 * np.pi * (160 if i % 2 == 0 else 220) * t[lo:hi])).astype(np.float32)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(RATE)
        wf.writeframes((np.clip(sig, -1, 1) * 32767).astype(np.int16).tobytes())
    return path


def test_fingerprint_reflects_config_and_rubric(workspace) -> None:  # noqa: ANN001
    fp = fingerprint(workspace, "RUBRICSHA")
    assert fp["rubric_sha256"] == "RUBRICSHA"
    assert fp["code_version"] and fp["prompt_version"] and fp["python"]
    # a changed setting changes the config hash
    before = config_sha256(workspace)
    tweaked = workspace.model_copy(deep=True)
    tweaked.judge.temperature = 0.5
    assert config_sha256(tweaked) != before


def _run_one(workspace, engines, tmp_path, call_id="OPS001"):  # noqa: ANN001
    import time
    state = StateDB(workspace.paths.state_db)
    rec = RunRecorder(workspace, load_rubric().sha256, "run")
    call = CallInput(call_id=call_id, audio_path=_speech_wav(tmp_path / f"{call_id}.wav"),
                     banker_id="B1")
    t0 = time.time()
    result = process_call(call, engines, state)
    rec.record(result, time.time() - t0, t0, state)
    manifest_path = rec.write(workspace.paths.output_dir)
    return manifest_path, result


def test_run_manifest_captures_the_batch(workspace, engines, tmp_path) -> None:  # noqa: ANN001
    manifest_path, result = _run_one(workspace, engines, tmp_path)
    assert manifest_path and manifest_path.is_file()
    manifest = RunManifest.model_validate_json(manifest_path.read_text(encoding="utf-8"))
    assert manifest.command == "run"
    assert manifest.counts.get(result.status) == 1
    assert manifest.fingerprint["code_version"] and manifest.fingerprint["config_sha256"]
    assert "gpu_hours" in manifest.cost
    # per-stage timing is derived from the state DB the core already writes
    assert manifest.calls[0].stage_seconds, "expected per-stage timing"
    assert "ingestion" in manifest.calls[0].stage_seconds
    # the per-call run link exists
    assert (workspace.paths.output_dir / "runs" / "by-call" / f"{result.call_id}.json").is_file()


def test_verify_reports_reproducible_then_what_changed(workspace, engines, tmp_path) -> None:  # noqa: ANN001
    _run_one(workspace, engines, tmp_path)
    sha = load_rubric().sha256

    ok = verify_call(workspace.paths.output_dir, "OPS001", workspace, sha)
    assert ok.reproducible and ok.run_id and not ok.changed

    # a rubric change is detected and named
    changed = verify_call(workspace.paths.output_dir, "OPS001", workspace, "DIFFERENT_RUBRIC")
    assert not changed.reproducible
    assert any("rubric" in c["field"] for c in changed.changed)

    # a config change is detected too
    tweaked = workspace.model_copy(deep=True)
    tweaked.judge.temperature = 0.9
    cfg_changed = verify_call(workspace.paths.output_dir, "OPS001", tweaked, sha)
    assert not cfg_changed.reproducible
    assert any("configuration" in c["field"] for c in cfg_changed.changed)


def test_verify_without_a_run_record_cannot_lie(workspace, engines, tmp_path) -> None:  # noqa: ANN001
    _run_one(workspace, engines, tmp_path)
    missing = verify_call(workspace.paths.output_dir, "NEVER_RAN", workspace, load_rubric().sha256)
    assert missing.run_id is None and not missing.reproducible and missing.reason


def test_a_resumed_run_does_not_invent_stage_timings(tmp_path) -> None:
    """A resumed call keeps the previous run's completed_at for every stage it
    skips. Differencing those against this run's start put the whole gap
    between the runs on one stage and gave skipped stages a duration."""
    import sqlite3

    from callqa.ops.runrecord import _stage_seconds

    db = tmp_path / "state.db"
    with sqlite3.connect(db) as c:
        c.execute("CREATE TABLE stages (call_id TEXT, stage TEXT, status TEXT, completed_at REAL)")
        day = 86400.0
        # run 1, three days ago: finished ingestion..asr, then died
        for i, st in enumerate(["ingestion", "audio", "asr"]):
            c.execute("INSERT INTO stages VALUES ('C1', ?, 'done', ?)", (st, 1000.0 + i))
        # run 2, today: resumed and did only the rest
        start = 1000.0 + 3 * day
        for i, st in enumerate(["speakers", "redaction"]):
            c.execute("INSERT INTO stages VALUES ('C1', ?, 'done', ?)", (st, start + 2 + 2 * i))

    out = _stage_seconds(db, "C1", start)
    assert set(out) == {"speakers", "redaction"}            # skipped stages: no entry
    assert out == {"speakers": 2.0, "redaction": 2.0}      # no three-day "stage"
