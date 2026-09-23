"""Build the offline wheels bundle - run on a machine WITH internet access.

    Windows:        py -3.12 scripts\\build_offline_bundle.py
    Linux:          python3.12 scripts/build_offline_bundle.py

    --core-only     skip the model engines (several GB): enough for mock mode
                    and the test suite
    --dest DIR      where to put the wheels (default: wheels/)

Downloads every pinned dependency as a pre-built wheel into wheels/. Copy the
project folder together with wheels/ to the machine without internet and run
scripts/install.py there: it sees wheels/ and installs from it, offline.

BUILD IT ON THE SAME OPERATING SYSTEM AND PYTHON VERSION AS THE TARGET - a
Windows machine with Python 3.12 for a Windows VDI with Python 3.12. pip
decides which dependencies a package needs by asking the machine it runs on
("is this Windows?", "is this Python 3.12?"), not the target: a Windows bundle
built on Linux silently lacks the Windows-only packages (colorama, needed by
pytest and tqdm) and tries to fetch Linux-only ones (triton, the nvidia-*
libraries torch wants on Linux), and fails. Only pre-built wheels are taken,
so nothing is compiled and no compiler is needed on either side.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# SEVERAL platform tags on Linux, and this is load-bearing. A single manylinux
# tag cannot resolve the pin set: modern numpy publishes only manylinux_2_27/
# 2_28 wheels and PyYAML 6.0.1 only manylinux2014. pip accepts a wheel matching
# ANY listed tag; `any` covers the pure-Python wheels.
PLATFORMS = {
    "linux": ["manylinux2014_x86_64", "manylinux_2_28_x86_64", "any"],
    "win32": ["win_amd64", "any"],
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--core-only", action="store_true",
                        help="skip requirements-server.txt (the model engines)")
    parser.add_argument("--dest", type=Path, default=ROOT / "wheels")
    args = parser.parse_args(argv)

    if sys.platform not in PLATFORMS:
        print(f"ERROR: run this on the target's operating system (Windows or Linux); "
              f"this is {sys.platform}.", file=sys.stderr)
        return 1
    if sys.version_info[:2] not in ((3, 11), (3, 12)):
        print("ERROR: run this with the Python version the target will use (3.11 or "
              "3.12), e.g. `py -3.12 scripts\\build_offline_bundle.py`.", file=sys.stderr)
        return 1
    version = f"{sys.version_info[0]}.{sys.version_info[1]}"

    args.dest.mkdir(parents=True, exist_ok=True)
    platform_args = [a for tag in PLATFORMS[sys.platform] for a in ("--platform", tag)]
    requirements = ["requirements.txt"] + ([] if args.core_only else ["requirements-server.txt"])
    for req in requirements:
        print(f"Downloading {req} for {sys.platform}, Python {version} ...", flush=True)
        code = subprocess.call(
            [sys.executable, "-m", "pip", "download", "--only-binary=:all:", *platform_args,
             "--python-version", version, "-r", str(ROOT / req), "-d", str(args.dest)],
            env=dict(os.environ, PYTHONUTF8="1", PIP_DISABLE_PIP_VERSION_CHECK="1"))
        if code != 0:
            print(f"\nERROR: downloading {req} failed (see pip's message above).",
                  file=sys.stderr)
            return 1

    wheels = sorted(args.dest.glob("*.whl"))
    size = sum(w.stat().st_size for w in wheels)
    print(f"\nBundle ready: {len(wheels)} wheels, {size / 2**20:,.0f} MiB in {args.dest}")
    print("Copy the project folder with this wheels/ folder inside it to the target, "
          "then run scripts/install.py there (add --server for the model engines).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
