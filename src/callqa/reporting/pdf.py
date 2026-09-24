"""A report as PDF, printed by the browser already on the machine.

The HTML reports are interactive, but a mail gateway may strip their styles
and a desktop may open them in Internet Explorer mode; a PDF looks the same
everywhere. Edge (on every Windows desktop) or Chrome prints it headless, with
the report's own print layout - nothing is installed and nothing goes online.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from pathlib import Path

_WINDOWS = [
    r"%ProgramFiles(x86)%\Microsoft\Edge\Application\msedge.exe",
    r"%ProgramFiles%\Microsoft\Edge\Application\msedge.exe",
    r"%ProgramFiles%\Google\Chrome\Application\chrome.exe",
    r"%ProgramFiles(x86)%\Google\Chrome\Application\chrome.exe",
    r"%LocalAppData%\Google\Chrome\Application\chrome.exe",
]
_NAMES = ["msedge", "microsoft-edge", "google-chrome", "chrome", "chromium", "chromium-browser"]


class PDFError(RuntimeError):
    pass


def find_browser() -> str | None:
    """CALLQA_BROWSER, else Edge, else Chrome/Chromium."""
    explicit = os.environ.get("CALLQA_BROWSER")
    if explicit:
        return explicit if Path(explicit).exists() else None
    for raw in _WINDOWS:
        path = os.path.expandvars(raw)
        if "%" not in path and Path(path).exists():
            return path
    for name in _NAMES:
        found = shutil.which(name)
        if found:
            return found
    return None


def print_pdf(html: Path, pdf: Path, *, browser: str | None = None,
              timeout: float = 180.0) -> Path:
    exe = browser or find_browser()
    if exe is None:
        raise PDFError("no Edge or Chrome found to print the PDF (set CALLQA_BROWSER to its path)")
    pdf.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="callqa-pdf-") as profile:
        cmd = [exe, "--headless=new", "--disable-gpu", "--no-first-run",
               "--disable-extensions", f"--user-data-dir={profile}",
               "--no-pdf-header-footer", "--run-all-compositor-stages-before-draw",
               "--virtual-time-budget=5000", f"--print-to-pdf={pdf}",
               html.resolve().as_uri()]
        posix = os.name == "posix"
        if posix and hasattr(os, "geteuid") and os.geteuid() == 0:
            # Chromium will not start as root (containers, CI) with its sandbox;
            # a bank desktop user is never root, so there it keeps the sandbox
            cmd.insert(1, "--no-sandbox")
        proc = _run(cmd, timeout)
        if not _printed(pdf) and posix and "--no-sandbox" not in cmd:
            # a Linux browser whose sandbox helper is not set up (Edge for
            # Linux on a CI runner) aborts; the page is a local file this
            # program wrote, so it is printed once more without the sandbox
            proc = _run([cmd[0], "--no-sandbox", *cmd[1:]], timeout)
    if not _printed(pdf):
        tail = (proc.stderr or b"").decode("utf-8", "replace")[-500:]
        raise PDFError(f"the browser printed no PDF (exit {proc.returncode}): {tail}")
    return pdf


def _printed(pdf: Path) -> bool:
    return pdf.exists() and pdf.stat().st_size >= 1000


def _run(cmd: list[str], timeout: float) -> subprocess.CompletedProcess:
    try:
        return subprocess.run(cmd, capture_output=True, timeout=timeout, check=False)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise PDFError(f"the browser could not print the PDF: {exc}") from exc
