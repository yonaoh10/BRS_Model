#!/usr/bin/env bash
# The first thing to run on a new machine.
#
#   ./scripts/first_run.sh
#
# Proves the whole chain - ingestion, VAD, speaker separation, redaction,
# features, judge, reports, calibration - on this machine in mock mode, before
# any model is downloaded and without a GPU. Takes seconds, needs no network.
#
# If this passes, the software is installed correctly and the only thing
# standing between you and real calls is the models (docs/DEPLOYMENT.md).
set -euo pipefail
cd "$(dirname "$0")/.."

if ! command -v python >/dev/null && command -v python3 >/dev/null; then
    PY=python3
else
    PY=python
fi

if ! "$PY" -c "import callqa" >/dev/null 2>&1; then
    echo "installing the package ..."
    "$PY" -m pip install -q -r requirements.txt && "$PY" -m pip install -q -e .
fi

# ffmpeg is NOT needed for this run: the synthetic calls are plain WAV and the
# pipeline reads those with the standard library. It IS needed for real
# recordings in any other container (m4a, mp3, stereo phone captures).
if ! command -v ffmpeg >/dev/null; then
    echo "note: ffmpeg is not installed. Mock mode does not need it, but real"
    echo "      recordings do (apt install ffmpeg / brew install ffmpeg)."
    echo
fi

[[ -f data/input/metadata.csv ]] || "$PY" scripts/generate_sample_data.py
"$PY" -m callqa run --mock
"$PY" -m callqa report --mock
"$PY" -m callqa calibrate --mock || true      # FAIL is expected on 6 calls
echo
echo "Reports: data/output/reports/index.html"
echo "Starting the dashboard (Ctrl+C to stop) ..."
exec "$PY" dashboard/server.py
