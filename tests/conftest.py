"""Shared pytest fixtures: an isolated workspace with generated sample data."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="session")
def sample_data_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Generate the synthetic sample set once per test session."""
    input_dir = tmp_path_factory.mktemp("input")
    subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "generate_sample_data.py"),
            "--input-dir",
            str(input_dir),
        ],
        check=True,
        capture_output=True,
    )
    return input_dir


@pytest.fixture()
def workspace(tmp_path: Path, sample_data_dir: Path):
    """Mock-mode Config with isolated input/output/state paths."""
    from callqa.config import load_config

    config = load_config(
        None,
        {
            "paths": {
                "input_dir": str(sample_data_dir),
                "output_dir": str(tmp_path / "output"),
                "state_db": str(tmp_path / "state.db"),
                "models_dir": str(tmp_path / "models"),
            },
            "run": {"mock": True},
            "audio": {"vad": "energy"},
        },
    )
    return config


@pytest.fixture()
def engines(workspace):
    from callqa.engines import build_engines

    return build_engines(workspace)


@pytest.fixture(autouse=True)
def _no_callqa_env_leakage(monkeypatch: pytest.MonkeyPatch) -> None:
    """Start every test without CALLQA_* overrides in the environment.

    cli.main() calls load_dotenv(), which exports the operator's .env into
    os.environ for the rest of the process - including every later test.
    After `runpod_cli.py up` writes CALLQA_ASR__BASE_URL into .env (its
    documented job), any CLI-invoking test silently armed that override and
    tests asserting on missing endpoints failed. Tests that need such a
    variable set it themselves via monkeypatch, which runs after this.
    """
    import os

    for key in [k for k in os.environ if k.startswith("CALLQA_")]:
        monkeypatch.delenv(key)
