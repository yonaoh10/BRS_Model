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

echo "Installing core dependencies from wheels/ ..."
pip install --no-index --find-links wheels/ -r requirements.txt

if [[ "${1:-}" == "--server" ]]; then
    echo "Installing server engine dependencies from wheels/ ..."
    pip install --no-index --find-links wheels/ -r requirements-server.txt
fi

echo "Installing callqa package ..."
pip install --no-index --no-build-isolation -e .

echo "Done. Verify with: python -m callqa --help"
