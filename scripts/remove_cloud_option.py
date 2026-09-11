#!/usr/bin/env python3
"""Remove the dev-phase cloud option completely.

Run this once the bank's own GPU servers are ready. It deletes every
cloud-only file AND unwires the three code references, so the repository is
left exactly as if the cloud option had never existed - then proves it by
running ruff and the test suite.

    python scripts/remove_cloud_option.py            # show what would change
    python scripts/remove_cloud_option.py --apply    # do it
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

PATHS_TO_DELETE = [
    "cloud",
    "config/config.cloud.yaml",
    "src/callqa/asr/remote_engine.py",
    "tests/test_remote_asr.py",
    "scripts/remove_cloud_option.py",  # this script removes itself last
]

# (file, exact text to remove, replacement)
EDITS: list[tuple[str, str, str]] = [
    (
        "src/callqa/engines.py",
        """    elif config.asr.engine == "remote":
        # DEV-ONLY cloud path - deleted when the project moves on-prem.
        from callqa.asr.remote_engine import RemoteASREngine

        asr = RemoteASREngine(config.asr)
        asr.check_connectivity()
""",
        "",
    ),
    (
        "src/callqa/config.py",
        """    # faster_whisper = in-process (bank server). remote = HTTP client to a
    # cloud-hosted ASR server (DEV ONLY; see cloud/README.md). mock = fake.
    engine: Literal["faster_whisper", "remote", "mock"] = "faster_whisper\"""",
        '    engine: Literal["faster_whisper", "mock"] = "faster_whisper"',
    ),
    (
        "src/callqa/config.py",
        """    # --- remote engine only (dev phase); ignored by faster_whisper/mock ---
    base_url: str = ""
    api_key: str | None = None
    timeout_sec: float = 900.0
""",
        "",
    ),
    (
        "src/callqa/config.py",
        """
    @model_validator(mode="after")
    def _remote_needs_base_url(self) -> ASRConfig:
        if self.engine == "remote" and not self.base_url:
            raise ValueError("asr.engine='remote' requires asr.base_url (see cloud/README.md)")
        return self
""",
        "",
    ),
]


def check(apply: bool) -> int:
    problems: list[str] = []
    plan: list[str] = []

    for rel in PATHS_TO_DELETE:
        path = REPO_ROOT / rel
        if path.exists():
            plan.append(f"delete   {rel}")
        else:
            plan.append(f"skip     {rel} (already gone)")

    for rel, old, _new in EDITS:
        path = REPO_ROOT / rel
        text = path.read_text(encoding="utf-8") if path.exists() else ""
        if old in text:
            plan.append(f"unwire   {rel}")
        else:
            problems.append(
                f"{rel}: expected cloud-specific block not found - it was edited by hand. "
                "Remove it manually, then re-run."
            )

    print("Plan:")
    for line in plan:
        print(f"  {line}")
    if problems:
        print("\nPROBLEMS:")
        for p in problems:
            print(f"  {p}")
        return 1
    if not apply:
        print("\nDry run. Re-run with --apply to make these changes.")
        return 0

    # 1. unwire code references first (the files stay)
    for rel, old, new in EDITS:
        path = REPO_ROOT / rel
        path.write_text(path.read_text(encoding="utf-8").replace(old, new, 1), encoding="utf-8")
        print(f"unwired  {rel}")

    # 2. delete cloud-only files (this script last, so failures leave it usable)
    self_path = Path(__file__).resolve()
    for rel in PATHS_TO_DELETE:
        path = REPO_ROOT / rel
        if path.resolve() == self_path or not path.exists():
            continue
        shutil.rmtree(path) if path.is_dir() else path.unlink()
        print(f"deleted  {rel}")

    # 3. prove the pipeline is intact
    print("\nVerifying...")
    for cmd in (["ruff", "check", "src", "tests", "scripts"],
                [sys.executable, "-m", "pytest", "-q"]):
        result = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True)
        tail = (result.stdout + result.stderr).strip().splitlines()[-3:]
        print(f"  $ {' '.join(cmd[:3])} -> exit {result.returncode}")
        for line in tail:
            print(f"    {line}")
        if result.returncode != 0:
            print("\nVerification FAILED - inspect the output above before committing.")
            return 1

    print(f"\nDone. Finally remove this script: git rm {self_path.relative_to(REPO_ROOT)}")
    print("Also drop the 'cloud option' section from README.md and the")
    print("cloud-* targets from the Makefile, then commit.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--apply", action="store_true", help="actually make the changes")
    return check(parser.parse_args().apply)


if __name__ == "__main__":
    raise SystemExit(main())
