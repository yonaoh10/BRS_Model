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
                "size_bytes": 15, "sha256": sha, "downloaded_at": "2026-01-01T00:00:00+00:00"},
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
