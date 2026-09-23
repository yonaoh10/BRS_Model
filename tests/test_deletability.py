"""The deliverable must stay `rm -rf dashboard/`-able.

The core (src/callqa) must never import the optional add-on directory, or
removing it would break the pipeline. This scans every import in the core and
fails if any reaches into dashboard/. It also pins the version so the release
number is not left stale.

`cloud/` is still named here although the directory is gone: the guard is what
keeps a future add-on from being wired into the core the way the cloud option
never was.
"""

from __future__ import annotations

import ast
from pathlib import Path

import callqa

_SRC = Path(__file__).resolve().parent.parent / "src" / "callqa"
_FORBIDDEN = {"dashboard", "cloud"}


def _imported_top_levels(tree: ast.AST):
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield alias.name.split(".")[0]
        elif isinstance(node, ast.ImportFrom):
            # ignore relative imports (node.level > 0); they can't reach a sibling top-level pkg
            if node.level == 0 and node.module:
                yield node.module.split(".")[0]


def test_core_never_imports_dashboard_or_cloud() -> None:
    offenders: list[str] = []
    for py in _SRC.rglob("*.py"):
        tree = ast.parse(py.read_text(encoding="utf-8"))
        for top in _imported_top_levels(tree):
            if top in _FORBIDDEN:
                offenders.append(f"{py.relative_to(_SRC)} imports {top}")
    assert not offenders, (
        "the core must be deletable-independent of dashboard/ and cloud/:\n"
        + "\n".join(offenders))


def test_version_is_released() -> None:
    assert callqa.__version__ == "1.1.0"
