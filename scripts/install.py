"""Install callqa into a private virtual environment - on Windows, Linux or macOS.

    Windows:        py -3.12 scripts\\install.py
    Linux / macOS:  python3 scripts/install.py

    --server        also install the model engines (ASR, diarization, NER)
    --wheels DIR    install only from DIR, with no network (the offline bundle
                    made by scripts/build_offline_bundle.py). Used automatically
                    when a non-empty wheels/ folder sits next to this project.
    --online        ignore wheels/ and install from the package index
    --venv PATH     where to create the environment (default: .venv)

No administrator rights, no bash, no PowerShell script and no `make` are
needed: this is the one command that has to work on a bank VDI desktop. It
creates .venv inside the project, installs the pinned dependencies into it,
makes the project's own code importable, and proves the result by importing
it with the new interpreter. Nothing outside the project folder is touched.

Afterwards, run everything with the environment's interpreter - no
"activation" needed, so PowerShell's script execution policy does not matter:

    Windows:        .venv\\Scripts\\python -m callqa preflight
    Linux / macOS:  .venv/bin/python -m callqa preflight
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SUPPORTED = ((3, 11), (3, 12))
IS_WINDOWS = os.name == "nt"


def _say(message: str) -> None:
    print(message, flush=True)


def _fail(message: str) -> int:
    print(f"\nERROR: {message}", file=sys.stderr, flush=True)
    return 1


def venv_python(venv: Path) -> Path:
    return venv / ("Scripts/python.exe" if IS_WINDOWS else "bin/python")


def _run(cmd: list[str]) -> int:
    # PYTHONUTF8 for every child: pip and its build steps otherwise write with
    # the ANSI code page on Windows, and a project path with a character that
    # page lacks raised UnicodeEncodeError halfway through the install.
    env = dict(os.environ, PYTHONUTF8="1", PIP_DISABLE_PIP_VERSION_CHECK="1")
    return subprocess.call(cmd, env=env)


def _bundle_problem(wheels: Path, version: tuple[int, int]) -> str | None:
    """Why this wheels/ folder cannot install here, or None."""
    files = list(wheels.glob("*.whl")) + list(wheels.glob("*.tar.gz"))
    if not files:
        return (f"{wheels} is empty. Build it with scripts/build_offline_bundle.py on a "
                "machine with internet access and copy the whole folder across.")
    tags = {m.group(1) for f in files if (m := re.search(r"-cp(3\d+)-", f.name))}
    here = f"{version[0]}{version[1]}"
    if tags and here not in tags:
        built = ", ".join(sorted(f"3.{t[1:]}" for t in tags))
        return (f"{wheels} was built for Python {built}, but this is Python "
                f"{version[0]}.{version[1]}. Rebuild it with Python "
                f"{version[0]}.{version[1]}, or install Python {built}.")
    compiled = [f.name for f in files if "-cp3" in f.name]
    windows = any("win_amd64" in n for n in compiled)
    linux = any("linux" in n for n in compiled)
    if IS_WINDOWS and linux and not windows:
        return (f"{wheels} holds Linux wheels. Build the bundle ON a Windows "
                "machine: py -3.12 scripts\\build_offline_bundle.py")
    if not IS_WINDOWS and windows and not linux:
        return (f"{wheels} holds Windows wheels. Build the bundle ON a Linux "
                "machine: python3 scripts/build_offline_bundle.py")
    return None


def link_project(python: Path) -> Path:
    """Make `src/` importable by `python`, whatever characters its path holds.

    Not `pip install -e .`: that writes the project's ABSOLUTE path into a
    .pth file in the ANSI code page, and Python 3.11/3.12 read .pth files back
    in that code page. With a Hebrew folder name on an English Windows the
    install either failed outright or produced an interpreter that could not
    start. A path RELATIVE to site-packages is plain ASCII, so it survives any
    code page; the environment lives inside the project, so it stays valid.
    """
    site = subprocess.check_output(
        [str(python), "-c", "import sysconfig; print(sysconfig.get_paths()['purelib'])"],
        env=dict(os.environ, PYTHONUTF8="1"), encoding="utf-8").strip()
    site_dir = Path(site)
    try:
        entry = os.path.relpath(ROOT / "src", site_dir)
    except ValueError:                    # another drive: only absolute works
        entry = str(ROOT / "src")
    pth = site_dir / "callqa-src.pth"
    pth.write_text(entry + "\n", encoding="utf-8")
    return pth


ENGINE_CHECK = ("import ctranslate2, onnxruntime, torch, faster_whisper; "
                "print('engines load: torch', torch.__version__, "
                "'ctranslate2', ctranslate2.__version__)")


def _venv_works(python: Path, version: tuple[int, int]) -> bool:
    try:
        out = subprocess.run([str(python), "-c", "import sys; print(sys.version_info[:2])"],
                             capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.SubprocessError):
        return False
    return out.returncode == 0 and out.stdout.strip() == str(version)


def _make_project_private() -> None:
    """Owner-only project folder, on Windows, before anything is added to it.

    Unpacked anywhere but the user's profile (C:\\BRS_Model-main, a D: data
    disk) the folder inherits "Users: read" and "Authenticated Users: modify":
    every other user of the machine could read .env's keys and the call data,
    or put their own ffmpeg.exe into tools/ for this program to run.
    """
    sys.path.insert(0, str(ROOT / "src"))
    try:
        from callqa.portable import make_private_dir, private_to_owner
    except ImportError:          # pragma: no cover - the source tree is incomplete
        return
    make_private_dir(ROOT)
    if private_to_owner(ROOT):
        _say(f"The project folder is private to you (and SYSTEM/Administrators): {ROOT}")
    else:
        _say(f"WARNING: could not make {ROOT} private to you. Keep it under your user "
             "folder (C:\\Users\\<name>), which is private already.")


def verify(python: Path) -> int:
    check = ("import callqa, callqa.cli; "
             "from callqa.reporting.common import jinja_env; "
             "jinja_env().get_template('call_report.html.j2'); "
             "print('callqa', callqa.__version__, 'from', callqa.__file__)")
    return _run([str(python), "-c", check])


def _utf8_stdio() -> None:
    # A path with Hebrew printed to a redirected stream (a log file, CI) is a
    # UnicodeEncodeError under the Windows ANSI code page. Same fix as
    # callqa.portable.configure_stdio, which cannot be imported before install.
    for stream in (sys.stdout, sys.stderr):
        if (getattr(stream, "encoding", "") or "").lower().replace("-", "") != "utf8":
            try:
                stream.reconfigure(encoding="utf-8", errors="backslashreplace")
            except (AttributeError, ValueError, OSError):
                pass


def main(argv: list[str] | None = None) -> int:
    _utf8_stdio()
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--server", action="store_true",
                        help="also install the model engines (requirements-server.txt)")
    parser.add_argument("--wheels", type=Path, default=None,
                        help="install offline from this folder of wheels")
    parser.add_argument("--online", action="store_true",
                        help="ignore wheels/ and use the package index")
    parser.add_argument("--venv", type=Path, default=ROOT / ".venv",
                        help="virtual environment location (default: .venv)")
    args = parser.parse_args(argv)

    version = sys.version_info[:2]
    if version not in SUPPORTED:
        return _fail(
            f"this is Python {version[0]}.{version[1]}; callqa needs 3.11 or 3.12 "
            "(its pinned wheels exist for those). On Windows install 3.12 from "
            "python.org (\"Install for current user\" needs no admin rights) and run "
            "`py -3.12 scripts\\install.py`.")

    wheels = args.wheels
    if wheels is None and not args.online and (ROOT / "wheels").is_dir():
        wheels = ROOT / "wheels"
    if wheels is not None:
        problem = _bundle_problem(wheels, version)
        if problem:
            return _fail(problem)

    if IS_WINDOWS:
        _make_project_private()

    venv = args.venv if args.venv.is_absolute() else Path.cwd() / args.venv
    python = venv_python(venv)
    if python.exists() and not _venv_works(python, version):
        # Copied from another machine or user (its pyvenv.cfg points at a
        # Python that is not here), or built by another Python version.
        _say(f"The environment at {venv} does not work here; rebuilding it ...")
        if _run([sys.executable, "-m", "venv", "--clear", str(venv)]) != 0:
            return _fail(f"could not rebuild {venv}; delete that folder and run this again.")
    if not python.exists():
        _say(f"Creating a virtual environment at {venv} ...")
        code = _run([sys.executable, "-m", "venv", str(venv)])
        if code != 0 or not python.exists():
            return _fail(f"could not create a virtual environment at {venv}. On Debian/"
                         "Ubuntu install the python3-venv package.")
    _say(f"Interpreter: {python}")

    pip = [str(python), "-m", "pip", "install"]
    if wheels is not None:
        _say(f"Installing OFFLINE from {wheels} (no network) ...")
        pip += ["--no-index", "--find-links", str(wheels)]
    else:
        _say("Installing from the package index (set by pip.ini / pip.conf, or PyPI) ...")

    # ONE resolution over both files: resolved one after the other, a pin the
    # engines disagree with was silently replaced by the second install.
    requirements = [ROOT / "requirements.txt"]
    if args.server:
        requirements.append(ROOT / "requirements-server.txt")
    _say("  " + " + ".join(r.name for r in requirements))
    if _run([*pip, *(a for r in requirements for a in ("-r", str(r)))]) != 0:
        return _fail("installing the requirements failed (see the pip output above).")

    pth = link_project(python)
    _say(f"Linked the project into the environment ({pth.name}).")
    if verify(python) != 0:
        return _fail("the environment was built but callqa does not import in it.")
    if args.server:
        check = subprocess.run([str(python), "-c", ENGINE_CHECK], capture_output=True,
                               encoding="utf-8", errors="replace",
                               env=dict(os.environ, PYTHONUTF8="1"))
        print(check.stdout, end="")
        if check.returncode != 0:
            print(check.stderr[-2000:], file=sys.stderr)
            hint = ""
            if IS_WINDOWS and ("DLL" in check.stderr or "msvcp" in check.stderr.lower()):
                hint = (" A DLL failed to load: on Windows that is almost always a missing "
                        "Microsoft Visual C++ 2015-2022 Redistributable (x64), which torch "
                        "and ctranslate2 need and Python does not ship. Installing it needs "
                        "admin rights: ask IT for vc_redist.x64.exe, then run this again.")
            return _fail("the model engines are installed but do not load (see above)." + hint)

    run = r".venv\Scripts\python" if IS_WINDOWS else ".venv/bin/python"
    if venv != ROOT / ".venv":
        run = str(python)
    _say("\nDone. From the project folder, run:")
    _say(f"  {run} scripts{os.sep}first_run.py      (a full test run, no models needed)")
    _say(f"  {run} -m callqa preflight")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
