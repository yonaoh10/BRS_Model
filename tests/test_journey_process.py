"""`journey process` and `journey estimate`: order, deadline, lock, resume."""

from __future__ import annotations

import importlib.util
import os
import sys
from datetime import datetime
from pathlib import Path

import pytest

from callqa.config import load_config
from callqa.journey.engine import (
    DatasetLock,
    Deadline,
    ProcessLocked,
    estimate,
    ordered_calls,
    process_dataset,
)
from callqa.journey.store import dataset_dir, load_content, load_dataset, resolve_dataset_id

REPO_ROOT = Path(__file__).resolve().parent.parent


def _demo():
    spec = importlib.util.spec_from_file_location(
        "generate_journey_demo", REPO_ROOT / "scripts" / "generate_journey_demo.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def batch(tmp_path, monkeypatch):
    ws = tmp_path / "jp"
    _demo().generate(ws, stories=10, seed=11, audio=True, seconds=1.0)
    for key, sub in (("OUTPUT_DIR", "output"), ("INPUT_DIR", "input")):
        monkeypatch.setenv(f"CALLQA_PATHS__{key}", str(ws / sub))
    monkeypatch.setenv("CALLQA_PATHS__STATE_DB", str(ws / "state.db"))
    from callqa.cli import main
    assert main(["journey", "import", "--xlsx", str(ws / "input" / "handoff.xlsx"),
                 "--audio", str(ws / "input" / "recordings.zip"),
                 "--atlas", str(ws / "input" / "atlas")]) == 0
    config = load_config(None, {"paths": {"output_dir": str(ws / "output"),
                                          "input_dir": str(ws / "input"),
                                          "state_db": str(ws / "state.db")},
                                "run": {"mock": True}})
    # three calls are not transcribed yet
    redacted = sorted((ws / "output" / "redacted").glob("*.json"))
    for p in redacted[:3]:
        p.unlink()
    return config, [p.stem for p in redacted[:3]]


def test_deadline_parsing():
    now = datetime(2026, 9, 24, 22, 0)
    assert Deadline.from_args("07:00", None, now).at == datetime(2026, 9, 25, 7, 0)
    assert Deadline.from_args("23:30", None, now).at == datetime(2026, 9, 24, 23, 30)
    assert Deadline.from_args("07:00", 2, now).at == datetime(2026, 9, 25, 0, 0)
    assert Deadline.from_args(None, None, now).at is None
    assert not Deadline().passed()


def test_returns_first_puts_the_busiest_stories_first(batch):
    config, _ = batch
    ds = load_dataset(config)
    calls = ordered_calls(ds, "returns-first")
    by_call = {i.call_id: i.story_key for i in ds.interactions if i.call_id}
    n_contacts = {s.story_key: sum(1 for i in ds.interactions if i.story_key == s.story_key)
                  for s in ds.stories}
    order = [n_contacts[by_call[c]] for c, _f, _d in calls]
    assert order == sorted(order, reverse=True)
    assert len({c for c, _f, _d in calls}) == len(calls)
    assert all(files for _c, files, _d in calls)


def test_process_transcribes_what_is_missing_then_reads_and_reports(batch):
    config, missing = batch
    s = process_dataset(config, mock=True)
    assert s.transcribed == 3 and not s.failed and s.content_ran
    assert s.report is not None and s.report.exists()
    for cid in missing:
        assert (config.paths.output_dir / "redacted" / f"{cid}.json").exists()
    # a second run finds everything done
    s2 = process_dataset(config, mock=True, report=False)
    assert s2.transcribed == 0 and s2.calls_done_before == s2.calls_total


def test_a_passed_deadline_stops_before_any_work(batch):
    config, missing = batch
    s = process_dataset(config, mock=True, max_hours=1e-9)
    assert s.stopped_by_deadline and s.transcribed == 0 and not s.content_ran
    assert not (config.paths.output_dir / "redacted" / f"{missing[0]}.json").exists()


def test_skip_transcribe_reads_what_is_there(batch):
    config, _ = batch
    s = process_dataset(config, mock=True, skip_transcribe=True, report=False)
    assert s.transcribed == 0 and s.content_ran
    assert load_content(config, resolve_dataset_id(config, None)).cards


def test_one_run_per_dataset(batch):
    config, _ = batch
    folder = dataset_dir(config, resolve_dataset_id(config, None))
    with DatasetLock(folder):
        with pytest.raises(ProcessLocked):
            process_dataset(config, mock=True)
    # a lock left by a process that is gone is taken over
    (folder / "process.lock").write_text("999999999", encoding="utf-8")
    with DatasetLock(folder):
        assert (folder / "process.lock").read_text(encoding="utf-8") == str(os.getpid())
    assert not (folder / "process.lock").exists()


def test_estimate_measures_on_the_calls_left(batch):
    config, _ = batch
    est = estimate(config, sample_calls=2, mock=True)
    assert est.sampled_calls == 2 and est.calls_left == 1
    assert est.sec_per_audio_sec is not None and est.audio_hours_left is not None
    assert est.requests_left > 0 and est.lines()


def test_cli_process(batch, capsys):
    from callqa.cli import main
    assert main(["journey", "process", "--mock", "--limit-stories", "3"]) == 0
    assert "report:" in capsys.readouterr().out
