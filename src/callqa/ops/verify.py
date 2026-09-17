"""`callqa verify <call_id>`: is a stored result still reproducible, and if not,
exactly which input changed.

Compares the fingerprint captured when the call was produced (from its run link)
against the CURRENT environment fingerprint. This does not alter the core's
resume behaviour — it only reports staleness the operator then acts on.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from callqa.config import Config
from callqa.ops.models import CallRunLink
from callqa.ops.provenance import fingerprint

# Scalar fingerprint fields compared verbatim, with the human label used in output.
_SCALAR_FIELDS = [
    ("code_version", "code version"),
    ("git_sha", "code (git) revision"),
    ("config_sha256", "configuration"),
    ("rubric_sha256", "rubric"),
    ("prompt_version", "judge prompt version"),
]


@dataclass
class VerifyResult:
    call_id: str
    reproducible: bool
    changed: list[dict] = field(default_factory=list)
    run_id: str | None = None
    reason: str = ""


def _model_changes(was: list[dict], now: list[dict]) -> list[dict]:
    wmap = {m.get("role"): m for m in was}
    nmap = {m.get("role"): m for m in now}
    changes: list[dict] = []
    for role in sorted(set(wmap) | set(nmap)):
        w, n = wmap.get(role), nmap.get(role)
        if w is None or n is None:
            changes.append({"field": f"model[{role}]",
                            "was": (w or {}).get("model_id", "(absent)"),
                            "now": (n or {}).get("model_id", "(absent)")})
            continue
        if w.get("sha256") and n.get("sha256"):
            if w["sha256"] != n["sha256"]:
                changes.append({"field": f"model[{role}] weights",
                                "was": w["sha256"][:12], "now": n["sha256"][:12]})
        elif w.get("model_id") != n.get("model_id"):
            changes.append({"field": f"model[{role}]",
                            "was": w.get("model_id"), "now": n.get("model_id")})
    return changes


def verify_call(output_dir: Path, call_id: str, config: Config,
                rubric_sha256: str) -> VerifyResult:
    link_path = output_dir / "runs" / "by-call" / f"{call_id}.json"
    try:
        link = CallRunLink.model_validate_json(link_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return VerifyResult(
            call_id, reproducible=False, run_id=None,
            reason=("no run record for this call — it was produced before run "
                    "manifests existed, or under a different output dir. Re-run "
                    "it to capture provenance."))
    stored = link.fingerprint
    current = fingerprint(config, rubric_sha256)
    changed: list[dict] = []
    for key, label in _SCALAR_FIELDS:
        if stored.get(key) != current.get(key):
            changed.append({"field": label, "was": stored.get(key), "now": current.get(key)})
    changed += _model_changes(stored.get("models", []), current.get("models", []))
    return VerifyResult(call_id, reproducible=not changed, changed=changed, run_id=link.run_id)
