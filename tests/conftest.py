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
