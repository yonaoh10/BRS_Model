#!/usr/bin/env bash
# The first thing to run on a new machine. Works with or without Docker.
#
#   ./scripts/first_run.sh            # mock pipeline on synthetic calls, then the dashboard
#   ./scripts/first_run.sh --docker   # the same, inside the container
#
# Proves the whole chain - ingestion, VAD, speaker separation, redaction,
# features, judge, reports, calibration - on this machine before any model
# is downloaded or any GPU is rented. Takes seconds.
set -euo pipefail
cd "$(dirname "$0")/.."

if [[ ! -f .env ]]; then
    cp .env.example .env
    echo "created .env from .env.example - paste your keys into it when you have them"
fi

if [[ "${1:-}" == "--docker" ]]; then
    if ! docker info >/dev/null 2>&1; then
        echo "ERROR: docker is installed but its daemon is not running (or not reachable)." >&2
        echo "       Start Docker Desktop / the docker service and try again," >&2
        echo "       or run without --docker to use a local Python instead." >&2
        exit 2
    fi
    exec docker compose up --build
fi

if ! command -v ffmpeg >/dev/null; then
    echo "ERROR: ffmpeg is not installed (apt install ffmpeg / brew install ffmpeg / winget install ffmpeg)." >&2
    exit 2
fi
if ! python -c "import callqa" >/dev/null 2>&1; then
    echo "installing the package ..."
    pip install -q -r requirements.txt && pip install -q -e .
fi

[[ -f data/input/metadata.csv ]] || python scripts/generate_sample_data.py
python -m callqa run --mock
python -m callqa report --mock
python -m callqa calibrate --mock || true      # FAIL is expected on 6 calls
echo
echo "Reports: data/output/reports/index.html"
echo "Starting the dashboard (Ctrl+C to stop) ..."
exec python dashboard/server.py
