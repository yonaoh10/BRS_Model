"""The documents must describe the software that actually exists.

The engineer who deploys this has no other source of truth, so a path that
moved or a flag that was renamed is not a cosmetic problem - it is the
difference between a working deployment and an afternoon of guessing. Three
kinds of claim are checked mechanically, because all three have drifted before:

  - every repo file a document points at exists
  - every ``callqa <subcommand>`` a document shows is a real subcommand
  - every ``CALLQA_*`` variable a document names is one something actually reads

What this cannot check is whether the prose is true; that still needs a human.
It is deliberately narrow, because a checker that cries wolf gets deleted: only
inline-code spans are examined, only strings carrying a known file extension
count as paths, and runtime artifacts are excluded by name.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent

# Where a document may reasonably name a file from: documents refer to
# "cli.py" or "judge/runner.py" meaning inside the package, and to
# "config.yaml" meaning inside config/.
_ROOTS = ("", "src/callqa", "scripts", "tests", "dashboard", "config", "docs", "eval")

# Created while the system runs, so absent from a fresh checkout. Naming one is
# correct documentation, not a broken reference.
_RUNTIME_ARTIFACTS = {
    "metadata.csv", "human_ratings.csv", "MODELS_MANIFEST.json",
    "index.html", "calibration.html", "calibration.json", "licences.md",
}

_CODE_SPAN_RE = re.compile(r"`([^`\n]+)`")
_PATH_RE = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9_./-]*\."
                      r"(?:py|sh|md|yaml|yml|json|html|j2|csv|css|txt)$")
_CALLQA_CMD_RE = re.compile(r"\bcallqa\s+([a-z][a-z-]+)")
_ENV_RE = re.compile(r"\b(CALLQA_[A-Z0-9_]+)\b")

# Spelled out in prose as an example of the naming scheme, not a real variable.
_DOCUMENTED_PLACEHOLDERS = {"CALLQA_SECTION__FIELD"}


def _tracked_docs() -> list[Path]:
    """Only documents that ship. An untracked working note is not a deliverable
    and must not be able to fail the build.

    Falls back to globbing when git cannot answer, which is not a corner case:
    a release is unpacked from an archive and has no .git at all, and the
    deployment instructions tell the engineer to run this suite there. Asking
    git unconditionally made the whole suite fail to COLLECT on exactly the
    machine it was written to reassure.
    """
    try:
        listed = subprocess.run(["git", "-C", str(REPO), "ls-files", "*.md"],
                                capture_output=True, text=True, check=True).stdout.split()
        docs = [REPO / rel for rel in listed if (REPO / rel).exists()]
    except (OSError, subprocess.SubprocessError):
        docs = []
    if not docs:
        docs = sorted([*REPO.glob("*.md"), *REPO.glob("docs/*.md"),
                       *REPO.glob("dashboard/*.md")])
    assert docs, "no documentation found to check"
    return docs


DOCS = _tracked_docs()


# Documented as deletable in one command for the bank hand-off. A document may
# name a file inside one; once the directory is gone, so is the reference, and
# that is the supported state rather than a broken link.
_OPTIONAL_DIRS = ("dashboard",)


def _resolves(candidate: str) -> bool:
    # Written by a run, so absent from a fresh checkout by design.
    if candidate.startswith(("data/", "models/", "wheels/", "logs/")):
        return True
    top = candidate.split("/", 1)[0]
    if top in _OPTIONAL_DIRS and not (REPO / top).exists():
        return True
    if Path(candidate).name in _RUNTIME_ARTIFACTS:
        return True
    return any((REPO / root / candidate).exists() for root in _ROOTS)


def _subcommands() -> set[str]:
    from callqa.cli import build_parser

    out: set[str] = set()
    for action in build_parser()._actions:              # noqa: SLF001 - argparse has no public API
        if getattr(action, "choices", None):
            out.update(str(c) for c in action.choices)
    return out


def _readable_env_names() -> set[str]:
    """Every CALLQA_ variable something in the tree actually reads: the ones the
    config model derives from its own sections, plus any named literally in the
    scripts and the dashboard (start_vllm.sh reads CALLQA_JUDGE_API_KEY, which
    is a shell variable and not a config override)."""
    from callqa.config import Config

    names = set(_DOCUMENTED_PLACEHOLDERS)
    for section, field in Config.model_fields.items():
        sub = getattr(field.annotation, "model_fields", None)
        if sub is None:
            continue
        names.update(f"CALLQA_{section.upper()}__{key.upper()}" for key in sub)
    for pattern in ("scripts/*", "dashboard/*.py", "src/callqa/**/*.py"):
        for path in REPO.glob(pattern):
            if path.is_file():
                names.update(_ENV_RE.findall(path.read_text(encoding="utf-8",
                                                            errors="replace")))
    return names


def _ids(path: Path) -> str:
    return str(path.relative_to(REPO))


@pytest.mark.parametrize("doc", DOCS, ids=_ids)
def test_every_repo_file_a_document_points_at_exists(doc: Path) -> None:
    text = doc.read_text(encoding="utf-8")
    missing = sorted({
        span for span in _CODE_SPAN_RE.findall(text)
        if _PATH_RE.match(span) and not _resolves(span)
    })
    assert not missing, f"{_ids(doc)} points at files that do not exist: {missing}"


@pytest.mark.parametrize("doc", DOCS, ids=_ids)
def test_every_callqa_subcommand_a_document_shows_is_real(doc: Path) -> None:
    real = _subcommands()
    text = doc.read_text(encoding="utf-8")
    unknown = sorted({
        word
        for span in _CODE_SPAN_RE.findall(text)          # inside backticks only:
        for word in _CALLQA_CMD_RE.findall(span)         # prose says "callqa is..."
        if word not in real
    })
    assert not unknown, (
        f"{_ids(doc)} shows callqa subcommands that do not exist: {unknown} "
        f"(real: {sorted(real)})")


@pytest.mark.parametrize("doc", DOCS, ids=_ids)
def test_every_env_var_a_document_names_is_read_somewhere(doc: Path) -> None:
    real = _readable_env_names()
    text = doc.read_text(encoding="utf-8")
    unknown = sorted({name for name in _ENV_RE.findall(text) if name not in real})
    assert not unknown, (
        f"{_ids(doc)} names environment variables nothing reads: {unknown}")


def test_the_env_example_only_names_variables_the_code_reads() -> None:
    """`.env.example` is the operator's map of what can be configured. A key
    left there after the feature was removed sends them hunting for a setting
    that cannot do anything - which is how RUNPOD_API_KEY outlived the cloud."""
    from callqa.dotenv import parse_env_file

    real = _readable_env_names() | {"HF_TOKEN"}
    keys = set(parse_env_file((REPO / ".env.example").read_text(encoding="utf-8")))
    assert keys <= real, f"stale keys in .env.example: {sorted(keys - real)}"
