"""Unit tests: config loading and validation failures."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from callqa.config import load_config


def test_defaults_load_without_file() -> None:
    config = load_config(None)
    assert config.asr.language == "he"
    assert config.judge.temperature == 0.0


def test_repo_config_file_loads() -> None:
    config = load_config("config/config.yaml")
    assert config.asr.engine == "faster_whisper"
    assert config.judge.base_url.startswith("http://127.0.0.1")
    # {models_dir} placeholder is interpolated.
    assert "{models_dir}" not in config.asr.model_dir


def test_language_autodetect_rejected() -> None:
    with pytest.raises(ValidationError, match="never autodetect"):
        load_config(None, {"asr": {"language": "auto"}})


def test_invalid_engine_rejected() -> None:
    with pytest.raises(ValidationError):
        load_config(None, {"asr": {"engine": "whisper_cpp"}})


def test_invalid_watch_poll_rejected() -> None:
    with pytest.raises(ValidationError):
        load_config(None, {"watch": {"poll_seconds": 0}})


def test_invalid_banker_channel_rejected() -> None:
    with pytest.raises(ValidationError):
        load_config(None, {"speakers": {"banker_channel": "left"}})


def test_missing_file_raises() -> None:
    with pytest.raises(FileNotFoundError):
        load_config("no/such/config.yaml")


def test_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CALLQA_JUDGE__BASE_URL", "http://localhost:9999/v1")
    monkeypatch.setenv("CALLQA_RUN__MOCK", "true")
    config = load_config(None)
    assert config.judge.base_url == "http://localhost:9999/v1"
    assert config.run.mock is True


def test_cli_override_beats_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("CALLQA_PATHS__OUTPUT_DIR", str(tmp_path / "env_out"))
    config = load_config(None, {"paths": {"output_dir": str(tmp_path / "cli_out")}})
    assert config.paths.output_dir == tmp_path / "cli_out"
