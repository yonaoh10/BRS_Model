#!/usr/bin/env bash
# Install dependencies from the local wheels/ bundle - no network needed.
# Run on the bank server after transferring the repo + wheels/.
#
# Usage: ./scripts/install_offline.sh [--server]
#   (default installs core deps only; --server also installs GPU engines)
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

echo "Installing core dependencies from wheels/ ..."
pip install --no-index --find-links wheels/ -r requirements.txt

if [[ "${1:-}" == "--server" ]]; then
    echo "Installing server engine dependencies from wheels/ ..."
    pip install --no-index --find-links wheels/ -r requirements-server.txt
fi

echo "Installing callqa package ..."
# --no-build-isolation builds with the interpreter's own setuptools, which a
# 3.12+ venv does not ship. Install it from the bundle first if it is missing.
if ! python -c "import setuptools" >/dev/null 2>&1; then
    echo "  setuptools is missing; installing it from wheels/ ..."
    pip install --no-index --find-links wheels/ setuptools wheel
fi
pip install --no-index --find-links wheels/ --no-build-isolation -e .

echo "Verifying the installation ..."
python -m callqa --help >/dev/null
python -c "from callqa.reporting.common import jinja_env; jinja_env().get_template('call_report.html.j2')"
echo "Done. Next: python -m callqa validate-inputs"
