"""Load `.env` from the project directory, without a dependency.

Only the handful of secrets and per-machine settings live there: the RunPod
key, the Hugging Face token, the endpoints of a running GPU box. A variable
that is already set in the environment is never overridden, so a value
exported in the shell, or injected by a scheduler, always wins over the file.

The file is looked for next to the code (the project root) and in the current
directory, in that order. Anything else - ~/.env, parent directories - is
deliberately ignored: a secrets file this pipeline picks up should be one the
operator put there on purpose.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

_LOADED: set[Path] = set()


def project_root() -> Path:
    """The repository directory: src/callqa/dotenv.py -> repo root."""
    return Path(__file__).resolve().parents[2]


def candidate_files() -> list[Path]:
    seen: list[Path] = []
    for directory in (project_root(), Path.cwd()):
        path = directory / ".env"
        if path not in seen:
            seen.append(path)
    return seen


def parse_env_file(text: str) -> dict[str, str]:
    """KEY=value lines; `#` comments; optional `export`; simple quoting."""
    values: dict[str, str] = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[len("export "):].lstrip()
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip()
        if not key or not key.replace("_", "").isalnum():
            continue
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        elif " #" in value:
            value = value.split(" #", 1)[0].rstrip()
        values[key] = value
    return values


def load_dotenv(explicit: Path | None = None) -> list[str]:
    """Load the first `.env` found. Returns the names that were actually set."""
    paths = [explicit] if explicit else candidate_files()
    for path in paths:
        if path is None or not path.is_file() or path.resolve() in _LOADED:
            continue
        try:
            values = parse_env_file(path.read_text(encoding="utf-8"))
        except OSError as exc:
            logger.warning("could not read %s: %s", path, exc)
            continue
        _LOADED.add(path.resolve())
        applied = []
        for key, value in values.items():
            if key in os.environ or not value:
                continue
            os.environ[key] = value
            applied.append(key)
        if applied:
            logger.debug("loaded %d value(s) from %s: %s", len(applied), path,
                         ", ".join(applied))
            # A .env in an attacker-controlled cwd can point the judge/ASR at a
            # host of its choosing (validate_endpoint permits any https host),
            # and redacted transcripts then egress there. Surface which endpoint
            # was set and to which HOST so a redirected destination is visible -
            # but log only the host, not the full URL, which would otherwise be
            # dumped into the terminal scrollback on every command.
            for key in applied:
                if key.endswith("__BASE_URL"):
                    host = urlparse(os.environ[key]).netloc or "(unparseable)"
                    logger.warning("egress endpoint %s set from %s -> %s",
                                   key, path, host)
        return applied
    return []


def write_env_values(values: dict[str, str], path: Path | None = None) -> Path:
    """Update or append KEY=value lines in `.env`, keeping everything else.

    Used by the cloud CLI to hand the endpoints of a freshly started GPU box
    to the pipeline, instead of printing shell exports for the operator to
    paste by hand.
    """
    target = path or (project_root() / ".env")
    lines: list[str] = []
    if target.exists():
        lines = target.read_text(encoding="utf-8").splitlines()
    remaining = dict(values)
    for i, line in enumerate(lines):
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key = stripped.split("=", 1)[0].removeprefix("export ").strip()
        if key in remaining:
            lines[i] = f"{key}={remaining.pop(key)}"
    for key, value in remaining.items():
        lines.append(f"{key}={value}")
    fd = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    try:
        target.chmod(0o600)
    except OSError:  # pragma: no cover
        pass
    for key, value in values.items():
        os.environ[key] = value
    return target
