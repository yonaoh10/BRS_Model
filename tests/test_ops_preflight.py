"""Preflight: fail before the batch starts, not in the middle of it."""

from __future__ import annotations

import json
from pathlib import Path

from callqa.config import load_config
from callqa.ops.preflight import (
    Check,
    _engine_checks,
    _reachable,
    _verify_local_model,
    run_preflight,
)
from callqa.ops.provenance import dir_sha256


def _fake_model(models_dir: Path) -> tuple[Path, str]:
    model_dir = models_dir / "ivrit-whisper-large-v3-turbo-ct2"
    model_dir.mkdir(parents=True)
    (model_dir / "model.bin").write_bytes(b"fake weights v1")
    (model_dir / "config.json").write_text("{}", encoding="utf-8")
    sha = dir_sha256(model_dir)
    (models_dir / "MODELS_MANIFEST.json").write_text(json.dumps({
        "asr": {"role": "asr", "model_id": "ivrit-ai/whisper", "local_path": str(model_dir),
                # the real total, as download_models.py records it - the value
                # used to be a hand-written 15 for 17 bytes on disk, which the
                # size check (rightly) called an incomplete copy
                "size_bytes": sum(f.stat().st_size for f in model_dir.rglob("*") if f.is_file()),
                "sha256": sha, "downloaded_at": "2026-01-01T00:00:00+00:00"},
    }), encoding="utf-8")
    return model_dir, sha


def _cfg(tmp_path: Path):  # noqa: ANN202
    return load_config(None, {
        "paths": {"models_dir": str(tmp_path / "models"),
                  "output_dir": str(tmp_path / "output"),
                  "input_dir": str(tmp_path / "input"),
                  "state_db": str(tmp_path / "s.db")},
        "run": {"mock": False},
        "asr": {"engine": "faster_whisper"},
        "judge": {"engine": "mock", "model": "mock"},
    })


def test_mock_preflight_needs_nothing(workspace) -> None:  # noqa: ANN001
    (workspace.paths.input_dir / "calls").mkdir(parents=True, exist_ok=True)
    checks = run_preflight(workspace)
    assert all(c.ok for c in checks), [c.detail for c in checks if not c.ok]


def test_present_model_passes_fast_and_deep(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    model_dir, _ = _fake_model(tmp_path / "models")
    fast = _verify_local_model(cfg, "asr", model_dir, deep=False)
    deep = _verify_local_model(cfg, "asr", model_dir, deep=True)
    assert fast.ok and deep.ok and "verified" in deep.detail


def test_a_missing_model_fails_critically(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    (tmp_path / "models").mkdir()
    checks = _engine_checks(cfg, deep=False)
    asr = next(c for c in checks if c.name == "model:asr")
    assert not asr.ok and asr.critical


def test_tampered_weights_fail_the_deep_check(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    model_dir, _ = _fake_model(tmp_path / "models")
    (model_dir / "model.bin").write_bytes(b"fake weights v2 - RETRAINED")   # same name, new bytes
    deep = _verify_local_model(cfg, "asr", model_dir, deep=True)
    assert not deep.ok and "MISMATCH" in deep.detail


def test_reachability_reports_an_unreachable_endpoint() -> None:
    class _Dead:
        def check_connectivity(self):  # noqa: ANN001
            raise ConnectionError("refused")

    class _Live:
        def check_connectivity(self):  # noqa: ANN001
            return True

    assert not _reachable("judge", _Dead).ok
    assert _reachable("judge", _Live).ok


def test_disk_check_is_present_in_a_full_preflight(workspace) -> None:  # noqa: ANN001
    (workspace.paths.input_dir / "calls").mkdir(parents=True, exist_ok=True)
    names = {c.name for c in run_preflight(workspace)}
    assert {"configuration", "inputs", "disk space"} <= names


def test_check_dataclass_shape() -> None:
    c = Check("x", True, True, "ok")
    assert c.name == "x" and c.ok and c.critical


def test_preflight_finds_an_enabled_ner_model_missing_before_the_batch(tmp_path, monkeypatch):
    """The redactor refuses to start when redaction.ner is on and the model is
    absent; preflight must say so at second zero, not the batch at call one."""
    from callqa.config import Config
    from callqa.ops.preflight import _ner_check

    config = Config()
    config.paths.models_dir = tmp_path / "models"
    config.redaction.ner = True
    check = _ner_check(config)
    assert not check.ok and check.critical
    assert "download_models.py --ner" in check.detail


def test_preflight_looks_for_diarization_where_pyannote_loads_it(tmp_path, monkeypatch):
    """Not models/: pyannote loads through the Hugging Face cache of whoever
    ran the download, which on an air-gapped machine must be copied across."""
    from callqa.config import Config
    from callqa.ops.preflight import _diarization_check

    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path / "hub"))
    config = Config()
    missing = _diarization_check(config)
    assert not missing.ok and not missing.critical       # stereo calls still run
    (tmp_path / "hub" / "models--pyannote--speaker-diarization-community-1").mkdir(parents=True)
    assert _diarization_check(config).ok


def test_a_truncated_model_copy_fails_the_fast_check(tmp_path: Path) -> None:
    """Moving 20 GB onto an air-gapped machine is where truncation happens, and
    a truncated directory still exists and is non-empty - which was all the
    default check looked at. It passed; only --deep, re-hashing tens of GB,
    would have caught it."""
    cfg = _cfg(tmp_path)
    model_dir, _ = _fake_model(tmp_path / "models")
    (model_dir / "model.bin").write_bytes(b"fake")          # the copy stopped early
    fast = _verify_local_model(cfg, "asr", model_dir, deep=False)
    assert not fast.ok and fast.critical and "size MISMATCH" in fast.detail


def test_preflight_says_why_the_inputs_are_not_ready(tmp_path: Path) -> None:
    """On a fresh extract metadata.csv does not exist yet, and preflight said
    only "invalid: 1 problem(s)" - indistinguishable from a malformed file."""
    from callqa.ops.preflight import _inputs_check

    check, n = _inputs_check(_cfg(tmp_path))
    assert not check.ok and n == 0
    assert "not found" in check.detail
