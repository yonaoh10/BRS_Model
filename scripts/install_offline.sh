#!/usr/bin/env bash
# Install dependencies from the local wheels/ bundle - no network needed.
# Run on the target server after transferring the repository + wheels/.
#
# Usage: ./scripts/install_offline.sh [--server]
#   (default installs core deps only; --server also installs GPU engines)
#
# Installs into a virtual environment at ./.venv, created if missing. Set
# CALLQA_VENV to use a different location, or CALLQA_VENV=none to install into
# whatever interpreter is already active (e.g. one you manage yourself).
set -euo pipefail

cd "$(dirname "$0")/.."

if [[ ! -d wheels ]]; then
    echo "ERROR: wheels/ not found. Build it first with scripts/build_offline_bundle.sh" >&2
    exit 1
fi

# An empty wheels/ passed the directory check and then printed "Done" having
# installed nothing, which only surfaced when the first call failed.
shopt -s nullglob
wheel_files=(wheels/*.whl wheels/*.tar.gz)
if (( ${#wheel_files[@]} == 0 )); then
    echo "ERROR: wheels/ is empty. Run scripts/build_offline_bundle.sh on a machine" >&2
    echo "       with network access and copy the whole directory across." >&2
    exit 1
fi
echo "Found ${#wheel_files[@]} packages in wheels/"

# A virtual environment, not the system interpreter, and `python3 -m pip`, not
# a bare `pip`. The previous version called `pip` and `python` directly, and on
# exactly the servers this is for that fails at the first line: RHEL and Rocky
# ship `python3`/`pip3` and no bare `pip`, and Debian 12, Ubuntu 24.04 and RHEL 9
# mark the system interpreter "externally managed" (PEP 668), so `pip install`
# into it is refused outright.
VENV="${CALLQA_VENV:-.venv}"
if [[ "$VENV" != "none" ]]; then
    if [[ ! -x "$VENV/bin/python" ]]; then
        echo "Creating a virtual environment at $VENV ..."
        python3 -m venv "$VENV"
    fi
    PY="$VENV/bin/python"
else
    PY="$(command -v python3 || command -v python)"
fi
echo "Using interpreter: $PY ($("$PY" --version 2>&1))"

# The wheels are built for one Python version (build_offline_bundle.sh uses the
# interpreter it runs under). A mismatch fails on the first compiled package
# with "No matching distribution", which does not say why - so say it here.
bundled="$(find wheels -name '*-cp3*-*.whl' | head -1 | sed -E 's/.*-cp(3)([0-9]+)-.*/\1.\2/')"
here="$("$PY" -c 'import sys; print(f"{sys.version_info[0]}.{sys.version_info[1]}")')"
if [[ -n "$bundled" && "$bundled" != "$here" ]]; then
    echo "ERROR: wheels/ was built for Python $bundled, but this interpreter is $here." >&2
    echo "       Rebuild the bundle with PY_VERSION=$here, or install Python $bundled." >&2
    exit 1
fi

PIP=("$PY" -m pip install --no-index --find-links wheels/)

echo "Installing core dependencies from wheels/ ..."
"${PIP[@]}" -r requirements.txt

if [[ "${1:-}" == "--server" ]]; then
    echo "Installing server engine dependencies from wheels/ ..."
    "${PIP[@]}" -r requirements-server.txt
fi

echo "Installing callqa package ..."
# --no-build-isolation builds with the interpreter's own setuptools, which a
# 3.12+ venv does not ship. Install it from the bundle first if it is missing.
if ! "$PY" -c "import setuptools" >/dev/null 2>&1; then
    echo "  setuptools is missing; installing it from wheels/ ..."
    "${PIP[@]}" setuptools wheel
fi
"${PIP[@]}" --no-build-isolation -e .

echo "Verifying the installation ..."
"$PY" -m callqa --help >/dev/null
"$PY" -c "from callqa.reporting.common import jinja_env; jinja_env().get_template('call_report.html.j2')"
echo
echo "Done. Activate the environment with:  source $VENV/bin/activate"
echo "Then:  python -m callqa preflight"
