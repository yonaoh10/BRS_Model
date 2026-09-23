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
    # Last resort: the defaults shipped INSIDE the package. In an installed
    # (non-editable) wheel there is no repo-root config/ above site-packages, so
    # without this every command fails on the first load; with it the wheel is
    # self-contained and an operator can still override via a cwd config/ or
    # CALLQA_CONFIG_DIR (both take precedence above).
    candidates.append(here.parent / "config_defaults")
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


def load_yaml(path: Path) -> object:
    """Parse one of the bank-editable YAML files, with errors that say what to do.

    These files are Hebrew and meant to be edited on Windows, where an editor
    may save them as UTF-16 ("Unicode") or add a byte-order mark, and where a
    path pasted between double quotes turns its backslashes into escape
    sequences ("C:\\Users\\..." is \\U, the start of an 8-digit escape). Each of
    those failed with a bare decoder or scanner traceback naming neither the
    file nor the cause.
    """
    import yaml

    raw = path.read_bytes()
    try:
        if raw[:2] in (b"\xff\xfe", b"\xfe\xff"):
            text = raw.decode("utf-16")
        else:
            text = raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        raise ValueError(f"{path} is not saved as UTF-8. Open it in Notepad and save it "
                         "again with Encoding: UTF-8.") from None
    try:
        return yaml.safe_load(text)
    except yaml.YAMLError as exc:
        hint = ""
        if "escape" in str(exc):
            hint = (" A Windows path inside double quotes is read as escape sequences:"
                    " use forward slashes (C:/Users/...) or single quotes ('C:\\Users\\...').")
        raise ValueError(f"{path} is not valid YAML: {exc}.{hint}") from None
