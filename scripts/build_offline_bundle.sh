#!/usr/bin/env bash
# Build the offline wheels bundle - run on an INTERNET-CONNECTED machine.
# Downloads every pinned dependency into wheels/, ready to be carried to an
# air-gapped server together with the repository.
#
# Usage: ./scripts/build_offline_bundle.sh [--core-only]
#
# The core bundle is small and is all the pipeline needs to run in mock mode
# and to pass its test suite. --core-only skips the GPU engine wheels, which
# are several gigabytes.
set -euo pipefail

cd "$(dirname "$0")/.."
mkdir -p wheels

# Target interpreter for the wheels. Change PY_VERSION to match the server.
PY_VERSION="${PY_VERSION:-3.11}"

# SEVERAL platform tags, not one, and this is the whole reason the script used
# to download nothing at all.
#
# A single `--platform manylinux2014_x86_64` cannot resolve the pin set: modern
# numpy publishes only manylinux_2_27/2_28 wheels and is simply invisible under
# the older tag, while PyYAML 6.0.1 publishes only a manylinux2014 wheel and is
# invisible under the newer one. Either tag alone fails on some pin, so the
# first ERROR aborted the run under `set -e` and wheels/ stayed empty - and the
# whole air-gapped install path depends on this step.
#
# pip accepts a wheel matching ANY listed platform, so listing the tags the
# ecosystem actually uses resolves all of them. `any` covers pure-Python wheels.
PLATFORMS=(
    --platform manylinux2014_x86_64
    --platform manylinux_2_28_x86_64
    --platform any
)

download() {
    pip download --only-binary=:all: "${PLATFORMS[@]}" \
        --python-version "$PY_VERSION" -r "$1" -d wheels/
}

echo "Downloading core dependencies into wheels/ (python $PY_VERSION) ..."
download requirements.txt

if [[ "${1:-}" != "--core-only" ]]; then
    echo "Downloading server engine dependencies into wheels/ ..."
    echo "(GPU wheels are large; pass --core-only to skip)"
    download requirements-server.txt
fi

# --no-build-isolation on the target needs these present locally.
pip download --only-binary=:all: "${PLATFORMS[@]}" \
    --python-version "$PY_VERSION" -d wheels/ setuptools wheel

echo
echo "Bundle ready: $(du -sh wheels | cut -f1) across $(find wheels -name '*.whl' | wc -l) wheels."
echo "Transfer the repository + wheels/ to the server, then run scripts/install_offline.sh"
