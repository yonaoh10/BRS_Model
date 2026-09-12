#!/usr/bin/env bash
# Build the offline wheels bundle - run on an INTERNET-CONNECTED machine.
# Downloads every pinned dependency (core + server engines) into wheels/,
# ready to be transferred to the air-gapped bank server together with the repo.
#
# Usage: ./scripts/build_offline_bundle.sh [--core-only]
set -euo pipefail

cd "$(dirname "$0")/.."
mkdir -p wheels

echo "Downloading core dependencies into wheels/ ..."
pip download --only-binary=:all: --platform manylinux2014_x86_64 --python-version 3.11 -r requirements.txt -d wheels/

if [[ "${1:-}" != "--core-only" ]]; then
    echo "Downloading server engine dependencies into wheels/ ..."
    echo "(GPU wheels are large; pass --core-only to skip)"
    pip download --only-binary=:all: --platform manylinux2014_x86_64 --python-version 3.11 -r requirements-server.txt -d wheels/
fi

echo "Bundle ready: $(du -sh wheels | cut -f1) in wheels/"
echo "Transfer the repo + wheels/ to the server, then run scripts/install_offline.sh"

# --no-build-isolation on the target needs these present locally.
pip download --dest wheels setuptools wheel
