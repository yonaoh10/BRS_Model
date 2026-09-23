"""`callqa preflight`: verify the environment BEFORE a batch touches a call, so
a missing model or a full disk stops the run at second zero, not at call 400.

Checks the configuration, the presence and hashes of the models the current
config will use, the reachability of any remote endpoint, free disk, and the
input files — reusing the loaders and connectivity checks already in the core.
"""

from __future__ import annotations

import logging
import shutil
from dataclasses import dataclass
from pathlib import Path

from callqa.config import Config
from callqa.ops.provenance import dir_sha256, read_model_manifest

logger = logging.getLogger(__name__)

# Rough per-call output footprint (redacted audio ~2 MB/min + transcripts +
# scores + report). Deliberately generous; the floor below dominates for small
# batches. Used only to size the disk check.
PER_CALL_BYTES = 20 * 1024 * 1024
DISK_FLOOR_BYTES = 1 * 1024 * 1024 * 1024      # never proceed with < 1 GiB free


@dataclass
class Check:
    name: str
    ok: bool
    critical: bool
    detail: str


def _existing_ancestor(path: Path) -> Path:
    p = path.resolve()
    while not p.exists() and p != p.parent:
        p = p.parent
    return p


def _disk_check(config: Config, n_calls: int) -> Check:
    free = shutil.disk_usage(_existing_ancestor(config.paths.output_dir)).free
    need = max(DISK_FLOOR_BYTES, n_calls * PER_CALL_BYTES)
    return Check("disk space", free >= need, True,
                 f"{free // (1024**3)} GiB free; need ~{need // (1024**2)} MiB "
                 f"for {n_calls} call(s)")


def _inputs_check(config: Config) -> tuple[Check, int]:
    from callqa.ingestion import load_metadata

    v = load_metadata(config.paths.input_dir / "metadata.csv",
                      config.paths.input_dir / "calls")
    if not v.ok:
        return Check("inputs", False, True,
                     f"metadata.csv invalid: {len(v.problems)} problem(s)"), 0
    return Check("inputs", True, True, f"{len(v.rows)} call(s) in metadata.csv"), len(v.rows)


def _verify_local_model(config: Config, role: str, model_dir: Path, deep: bool) -> Check:
    if not model_dir.exists() or not any(model_dir.iterdir()):
        return Check(f"model:{role}", False, True, f"missing or empty: {model_dir}")
    entry = next((m for m in read_model_manifest(config.paths.models_dir)
                  if m["role"] == role), None)
    if entry is None:
        return Check(f"model:{role}", True, False,
                     f"present at {model_dir} (no manifest entry to verify against)")
    if deep and entry.get("sha256"):
        actual = dir_sha256(model_dir)
        if actual != entry["sha256"]:
            return Check(f"model:{role}", False, True,
                         "weight hash MISMATCH — not the model that produced earlier results")
        return Check(f"model:{role}", True, True, "present; weight hash verified")
    return Check(f"model:{role}", True, True, "present (fast check; use --deep to re-hash)")


def _reachable(name: str, build) -> Check:  # noqa: ANN001
    try:
        build().check_connectivity()
        return Check(name, True, True, "reachable")
    except Exception as exc:  # noqa: BLE001 - any failure means "not ready"
        return Check(name, False, True, f"unreachable: {type(exc).__name__}: {exc}")


def _engine_checks(config: Config, deep: bool) -> list[Check]:
    if config.run.mock or (config.asr.engine == "mock" and config.judge.engine == "mock"):
        return [Check("engines", True, False,
                      "mock engines — no models or endpoints required")]
    checks: list[Check] = []
    if config.asr.engine == "faster_whisper":
        checks.append(_verify_local_model(config, "asr", Path(config.asr.model_dir), deep))
    if config.judge.engine == "vllm":
        from callqa.judge.vllm_judge import VLLMJudge
        checks.append(_reachable("judge endpoint", lambda: VLLMJudge(config.judge)))
    return checks


def run_preflight(config: Config, deep: bool = False) -> list[Check]:
    """All checks. `deep` re-hashes local model weights (slow)."""
    checks = [Check("configuration", True, True, "loaded and valid")]
    inputs, n_calls = _inputs_check(config)
    checks.append(inputs)
    checks.extend(_engine_checks(config, deep))
    checks.append(_disk_check(config, n_calls))
    return checks
