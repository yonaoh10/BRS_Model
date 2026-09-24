"""The first thing to run on a new machine - Windows, Linux or macOS.

    Windows:        .venv\\Scripts\\python scripts\\first_run.py
    Linux / macOS:  .venv/bin/python scripts/first_run.py

Proves the whole chain - ingestion, VAD, speaker separation, redaction,
features, judge, reports, calibration - on this machine in mock mode, before
any model is downloaded and without a GPU. Takes seconds, needs no network.

If this passes, the software is installed correctly and the only thing
standing between you and real calls is the models (docs/DEPLOYMENT.md).

It works in its own folder, data/demo/, never in data/input and data/output:
run again after real recordings are in place, a mock run there would have
marked them processed - with mock transcripts and scores - and the next real
run would have skipped them.

    --no-dashboard   stop after the reports instead of opening the dashboard
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEMO = ROOT / "data" / "demo"

# The demo's own folders, as config overrides for every step it runs.
DEMO_ENV = {
    "CALLQA_PATHS__INPUT_DIR": str(DEMO / "input"),
    "CALLQA_PATHS__OUTPUT_DIR": str(DEMO / "output"),
    "CALLQA_PATHS__STATE_DB": str(DEMO / "callqa_state.db"),
}


def _step(*args: str) -> int:
    print(f"\n> python {' '.join(args)}", flush=True)
    return subprocess.call([sys.executable, *args], cwd=ROOT,
                           env=dict(os.environ, PYTHONUTF8="1", **DEMO_ENV))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--no-dashboard", action="store_true",
                        help="do not start the dashboard at the end")
    args = parser.parse_args(argv)

    try:
        import callqa  # noqa: F401
        from callqa.portable import configure_stdio, find_executable
    except ImportError:
        print("callqa is not installed in this interpreter. Run scripts/install.py "
              "first, then run this script with the interpreter it created "
              r"(.venv\Scripts\python on Windows, .venv/bin/python elsewhere).",
              file=sys.stderr)
        return 1
    configure_stdio()

    # ffmpeg is NOT needed for this run: the synthetic calls are plain WAV and
    # the pipeline reads those with the standard library. It IS needed for real
    # recordings in any other container (mp3, m4a, ...).
    if find_executable("ffmpeg") is None:
        print("note: ffmpeg was not found. This run does not need it; real MP3/M4A "
              "recordings do. See 'ffmpeg' in README.md (no admin rights needed).")

    if not (DEMO / "input" / "metadata.csv").is_file():
        if _step("scripts/generate_sample_data.py", "--input-dir", str(DEMO / "input")) != 0:
            return 1
    if _step("-m", "callqa", "run", "--mock") != 0:
        print("\nFIRST RUN FAILED at the step above.", file=sys.stderr)
        return 1
    # Calibration FAILs on six synthetic calls by design (too few to measure
    # agreement); it is run to prove it works, not for its verdict - and before
    # the reports, so the management report shows the calibration status.
    _step("-m", "callqa", "calibrate", "--mock")
    if _step("-m", "callqa", "report", "--mock") != 0:
        print("\nFIRST RUN FAILED at the step above.", file=sys.stderr)
        return 1

    reports = DEMO / "output" / "reports"
    print(f"\nFirst run OK. Reports: {reports / 'index.html'}")
    print(f"Management report: {reports / 'executive.html'}")
    print("To see the management report on 1,000 calls: "
          "scripts/generate_batch_demo.py (see README).")
    dashboard = ROOT / "dashboard" / "server.py"
    if args.no_dashboard or not dashboard.is_file():
        return 0
    print("Starting the dashboard (Ctrl+C to stop) ...", flush=True)
    try:
        return subprocess.call([sys.executable, str(dashboard), "--output-dir",
                                str(DEMO / "output")], cwd=ROOT)
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
