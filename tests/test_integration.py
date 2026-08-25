"""Integration tests: single-call E2E, locking, watch driver, full run,
resume behavior, and the no-PII-leak guarantee (per spec section 15)."""

from __future__ import annotations

import dataclasses
import shutil
import threading
import time
from pathlib import Path

import yaml

from callqa.cli import main as cli_main
from callqa.cli import watch_loop
from callqa.models import CallInput
from callqa.pipeline import process_call
from callqa.state import StateDB

RAW_ID = "123456782"  # seeded into mock dialog fixture 0
ALL_STAGES = ["ingestion", "audio", "asr", "speakers", "redaction", "features", "judge", "report"]


def write_config_file(tmp_path: Path, config) -> Path:  # noqa: ANN001
    cfg = {
        "paths": {
            "input_dir": str(config.paths.input_dir),
            "output_dir": str(config.paths.output_dir),
            "state_db": str(config.paths.state_db),
            "models_dir": str(config.paths.models_dir),
        },
        "run": {"mock": True},
        "audio": {"vad": "energy"},
        "asr": {"engine": "mock"},
        "judge": {"engine": "mock", "model": "mock"},
    }
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return path


def sample_call(workspace, call_id: str = "CALL001") -> CallInput:  # noqa: ANN001
    return CallInput(
        call_id=call_id,
        audio_path=workspace.paths.input_dir / "calls" / f"{call_id}.wav",
        banker_id="B001",
        banker_channel="L",
    )


# -- single-call E2E (the primary acceptance test) ---------------------------

def test_process_cli_single_call(tmp_path: Path, workspace) -> None:  # noqa: ANN001
    config_file = write_config_file(tmp_path, workspace)
    audio = workspace.paths.input_dir / "calls" / "CALL001.wav"
    exit_code = cli_main(
        ["process", "--audio", str(audio), "--config", str(config_file), "--mock"]
    )
    assert exit_code == 0
    out = workspace.paths.output_dir
    for artifact in [
        out / "ingestion" / "CALL001.json",
        out / "audio" / "CALL001.json",
        out / "transcripts" / "CALL001.json",
        out / "transcripts" / "CALL001.dialog.json",
        out / "redacted" / "CALL001.json",
        out / "features" / "CALL001.json",
        out / "scores" / "CALL001.json",
        out / "reports" / "calls" / "CALL001.html",
        out / "results" / "CALL001.json",
    ]:
        assert artifact.exists(), f"missing artifact: {artifact}"
    report = (out / "reports" / "calls" / "CALL001.html").read_text(encoding="utf-8")
    assert 'dir="rtl"' in report
    assert RAW_ID not in report
    # The raw transcripts directory carries its PII warning.
    assert (out / "transcripts" / "README.txt").exists()
    # State DB marks every stage done.
    state = StateDB(workspace.paths.state_db)
    assert state.completed_stages("CALL001") == ALL_STAGES


def test_process_lock_refused(workspace, engines) -> None:  # noqa: ANN001
    state = StateDB(workspace.paths.state_db)
    state.acquire_lock("CALL002")
    try:
        result = process_call(sample_call(workspace, "CALL002"), engines, state)
        assert result.status == "failed"
        assert "already being processed" in (result.error or "")
    finally:
        state.release_lock("CALL002")
    # After release, the same call processes fine.
    result = process_call(sample_call(workspace, "CALL002"), engines, state)
    assert result.status == "success"


def test_needs_human_review_exit_path(workspace, engines) -> None:  # noqa: ANN001
    class AlwaysBadJudge:
        name = "bad"

        def complete(self, request):  # noqa: ANN001
            return "{not valid json"

    bad_engines = dataclasses.replace(engines, judge=AlwaysBadJudge())
    result = process_call(sample_call(workspace, "CALL003"), bad_engines)
    assert result.status == "needs_human_review"
    assert result.exit_code == 1
    # judge/report stages did not complete
    assert "judge" not in result.stages_completed


# -- resume ------------------------------------------------------------------

def test_resume_reruns_only_missing_stage(workspace, engines) -> None:  # noqa: ANN001
    call = sample_call(workspace, "CALL004")
    result = process_call(call, engines)
    assert result.status == "success"
    out = workspace.paths.output_dir
    features_path = out / "features" / "CALL004.json"
    scores_path = out / "scores" / "CALL004.json"
    scores_mtime = scores_path.stat().st_mtime_ns
    transcript_mtime = (out / "transcripts" / "CALL004.json").stat().st_mtime_ns

    features_path.unlink()
    result = process_call(call, engines)
    assert result.status == "success"
    assert features_path.exists()  # recomputed
    # Other stages were skipped: artifacts untouched.
    assert scores_path.stat().st_mtime_ns == scores_mtime
    assert (out / "transcripts" / "CALL004.json").stat().st_mtime_ns == transcript_mtime


# -- watch driver ------------------------------------------------------------

def test_watch_picks_up_stable_file(tmp_path: Path, workspace, engines) -> None:  # noqa: ANN001
    # Isolated input dir so moving files does not disturb the shared sample set.
    input_dir = tmp_path / "watch_input"
    (input_dir / "calls").mkdir(parents=True)
    shutil.copy(workspace.paths.input_dir / "metadata.csv", input_dir / "metadata.csv")

    config = workspace.model_copy(deep=True)
    config.paths.input_dir = input_dir
    config.watch.poll_seconds = 0.1
    config.watch.stable_seconds = 0.3
    engines = dataclasses.replace(engines, config=config)

    stop = threading.Event()
    results = []

    def run_watch() -> None:
        results.extend(watch_loop(config, engines, stop_event=stop))

    thread = threading.Thread(target=run_watch, daemon=True)
    thread.start()
    try:
        # Drop the file in two chunks to simulate an in-progress recording copy.
        source = (workspace.paths.input_dir / "calls" / "CALL005.wav").read_bytes()
        target = input_dir / "calls" / "CALL005.wav"
        with target.open("wb") as fh:
            fh.write(source[: len(source) // 2])
            fh.flush()
        time.sleep(0.25)  # watcher sees the growing file, must NOT process it
        assert not (input_dir / "processed" / "CALL005.wav").exists()
        with target.open("ab") as fh:
            fh.write(source[len(source) // 2:])
        # Wait for stabilization + processing.
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline and not results:
            time.sleep(0.1)
    finally:
        stop.set()
        thread.join(timeout=30)

    assert len(results) == 1
    assert results[0].call_id == "CALL005"
    assert results[0].status == "success"
    assert (input_dir / "processed" / "CALL005.wav").exists()
    assert not target.exists()


# -- full run + reports + no-leak sweep --------------------------------------

def test_full_run_report_calibrate_no_leaks(tmp_path: Path, workspace) -> None:  # noqa: ANN001
    config_file = write_config_file(tmp_path, workspace)
    assert cli_main(["run", "--config", str(config_file), "--mock"]) == 0
    assert cli_main(["report", "--config", str(config_file), "--mock"]) == 0
    assert cli_main(["calibrate", "--config", str(config_file), "--mock"]) == 0
    assert cli_main(["validate-inputs", "--config", str(config_file), "--mock"]) == 0

    out = workspace.paths.output_dir
    state = StateDB(workspace.paths.state_db)
    for call_id in ["CALL001", "CALL002", "CALL003", "CALL004", "CALL005", "CALL006"]:
        assert (out / "reports" / "calls" / f"{call_id}.html").exists()
        assert state.completed_stages(call_id) == ALL_STAGES
    assert (out / "reports" / "index.html").exists()
    assert (out / "reports" / "calibration.html").exists()
    assert (out / "reports" / "calibration.json").exists()
    banker_reports = list((out / "reports" / "bankers").glob("*.html"))
    assert len(banker_reports) == 3

    # No raw PII digits anywhere outside the raw transcripts directory.
    for path in out.rglob("*"):
        if not path.is_file() or path.suffix not in (".html", ".json"):
            continue
        if path.is_relative_to(out / "transcripts"):
            continue  # raw transcripts are the one guarded exception
        content = path.read_text(encoding="utf-8", errors="ignore")
        assert RAW_ID not in content, f"raw Israeli ID leaked into {path}"
        assert "052-1234567" not in content, f"raw phone leaked into {path}"
