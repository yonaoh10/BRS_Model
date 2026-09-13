#!/usr/bin/env bash
# Container entrypoint: seed a first run when the input directory is empty,
# then hand over to whatever was asked for.
#
#   run --mock            (default) the whole pipeline on synthetic calls
#   run / watch / report  the real commands, same flags as `python -m callqa`
#   dashboard             the operator console on port 8765
#   shell                 an interactive shell inside the container
set -euo pipefail
cd /app

if [[ ! -f data/input/metadata.csv ]]; then
    echo "data/input/ is empty - generating the synthetic sample set for a first run."
    python scripts/generate_sample_data.py
fi

case "${1:-}" in
    dashboard)
        shift
        exec python dashboard/server.py --bind 0.0.0.0 --no-browser "$@"
        ;;
    shell)
        exec bash
        ;;
    *)
        exec python -m callqa "$@"
        ;;
esac
