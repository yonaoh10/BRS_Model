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
from callqa.portable import run_text

# repo root when this runs from a source checkout (src/callqa/ops/…), so a git
# sha is available in development; on a tarball/air-gapped deploy there is no
# .git and we fall back to the package version.
_REPO_ROOT = Path(__file__).resolve().parents[3]


def git_sha() -> str:
    """The commit SHA (with a -dirty suffix if the tree is modified), or 'none'
    when this is not a git checkout — the normal air-gapped deployment case."""
    # The .git check is not redundant with the rev-parse below. `git -C <dir>`
    # walks UP until it finds a repository, so an extracted release unpacked
    # anywhere inside somebody else's checkout would stamp THAT repository's
    # commit onto every run manifest: a provenance record that is confidently,
    # silently wrong. No .git of our own means no sha.
    if not (_REPO_ROOT / ".git").exists():
        return "none"
    try:
        head = run_text(["git", "-C", str(_REPO_ROOT), "rev-parse", "HEAD"], timeout=5)
        if head.returncode != 0:
            return "none"
        dirty = run_text(["git", "-C", str(_REPO_ROOT), "status", "--porcelain"],
                         timeout=5).stdout.strip()
        return head.stdout.strip() + ("-dirty" if dirty else "")
    except (OSError, subprocess.SubprocessError):  # pragma: no cover - defensive
        return "none"


def dir_sha256(path: Path) -> str:
    """A deterministic content hash of every file under `path`.

    Merkle-style: the hash of sorted (relative-path, file-content-hash) pairs, so
    it is stable across machines and independent of filesystem walk order. Used
    to fingerprint model weights once at download and verify them at preflight.
    """
    root = Path(path)
    top = hashlib.sha256()
    for f in sorted(root.rglob("*")):
        if not f.is_file():
            continue
        fh = hashlib.sha256()
        with f.open("rb") as fp:
            for chunk in iter(lambda fp=fp: fp.read(1 << 20), b""):
                fh.update(chunk)
        top.update(f.relative_to(root).as_posix().encode("utf-8"))
        top.update(b"\0")
        top.update(fh.hexdigest().encode("ascii"))
        top.update(b"\n")
    return top.hexdigest()


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
