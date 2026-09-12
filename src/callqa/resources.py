"""Locating the rubric, the recommendations and the config file.

These live in `config/` next to the code rather than inside the package,
because the bank is expected to edit them. That makes them findable only
relative to something, and "relative to the current working directory" meant
that running the CLI from anywhere but the repository root failed with a bare
FileNotFoundError naming a path the operator never typed.
"""

from __future__ import annotations

import os
from pathlib import Path

ENV_CONFIG_DIR = "CALLQA_CONFIG_DIR"


def config_dirs() -> list[Path]:
    """Every directory a configuration file may live in, most specific first."""
    candidates: list[Path] = []
    override = os.environ.get(ENV_CONFIG_DIR)
    if override:
        candidates.append(Path(override))
    candidates.append(Path.cwd() / "config")
    # Walk up from this module: src/callqa/resources.py -> repo root/config.
    here = Path(__file__).resolve()
    for parent in here.parents:
        candidate = parent / "config"
        if candidate.is_dir():
            candidates.append(candidate)
            break
    seen: set[Path] = set()
    return [c for c in candidates if not (c in seen or seen.add(c))]


def find_config(name: str, explicit: str | Path | None = None) -> Path:
    """Resolve one configuration file, or say exactly where it was looked for."""
    if explicit is not None:
        path = Path(explicit)
        if path.is_file():
            return path
        if path.is_absolute() or path.parent != Path("."):
            raise FileNotFoundError(f"{name}: no such file: {path}")
        name = path.name
    for directory in config_dirs():
        candidate = directory / name
        if candidate.is_file():
            return candidate
    searched = "\n  ".join(str(d) for d in config_dirs())
    raise FileNotFoundError(
        f"could not find {name}. Looked in:\n  {searched}\n"
        f"Run the CLI from the project directory, or set {ENV_CONFIG_DIR}."
    )
