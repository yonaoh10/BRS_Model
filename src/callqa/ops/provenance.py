"""The environment fingerprint: everything that determines an output.

A ScoreCard already records the prompt hash, prompt version, rubric hash, model
name and engine. What it cannot record without changing the core is the code
version, the effective configuration, and the model WEIGHTS (a name is not a
fingerprint). This module assembles all of it into one comparable record, used
by the run manifest (to stamp what produced a batch) and by `callqa verify` (to
detect what has changed since).
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

from callqa import __version__
from callqa.config import Config
from callqa.judge.prompts import PROMPT_VERSION

# repo root when this runs from a source checkout (src/callqa/ops/…), so a git
# sha is available in development; on a tarball/air-gapped deploy there is no
# .git and we fall back to the package version.
_REPO_ROOT = Path(__file__).resolve().parents[3]


def git_sha() -> str:
    """The commit SHA (with a -dirty suffix if the tree is modified), or 'none'
    when this is not a git checkout — the normal air-gapped deployment case."""
    try:
        head = subprocess.run(
            ["git", "-C", str(_REPO_ROOT), "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=5)
        if head.returncode != 0:
            return "none"
        dirty = subprocess.run(
            ["git", "-C", str(_REPO_ROOT), "status", "--porcelain"],
            capture_output=True, text=True, timeout=5).stdout.strip()
        return head.stdout.strip() + ("-dirty" if dirty else "")
    except (OSError, subprocess.SubprocessError):  # pragma: no cover - defensive
        return "none"


def config_sha256(config: Config) -> str:
    """Content hash of the EFFECTIVE config, so a changed setting is detectable."""
    payload = json.dumps(config.model_dump(mode="json"), sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def read_model_manifest(models_dir: Path) -> list[dict]:
    """The model inventory written by scripts/download_models.py, if present.

    Returns a list of {role, model_id, sha256, size_bytes}. `sha256` is "" for
    manifests written before weight hashing existed — verify treats that as
    "unknown", not "matched".
    """
    manifest = models_dir / "MODELS_MANIFEST.json"
    try:
        data = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    out: list[dict] = []
    for role, entry in sorted((data or {}).items()):
        if not isinstance(entry, dict):
            continue
        out.append({
            "role": role,
            "model_id": entry.get("model_id", ""),
            "sha256": entry.get("sha256", ""),
            "size_bytes": entry.get("size_bytes", 0),
        })
    return out


def fingerprint(config: Config, rubric_sha256: str) -> dict:
    """One record combining code, config, rubric, prompt and model identities."""
    return {
        "code_version": __version__,
        "git_sha": git_sha(),
        "python": sys.version.split()[0],
        "config_sha256": config_sha256(config),
        "rubric_sha256": rubric_sha256,
        "prompt_version": PROMPT_VERSION,
        "models": read_model_manifest(config.paths.models_dir),
    }
