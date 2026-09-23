"""The few operating-system calls whose POSIX spelling is wrong on Windows.

The bank's desktops are Microsoft VDI: Windows, no administrator rights, no WSL
guaranteed. Every call below was, until this module existed, written the POSIX
way somewhere in the pipeline, and on Windows each one either did not exist
(`os.uname`), did something destructive (`os.kill(pid, 0)` is CTRL_C_EVENT or
TerminateProcess there, never a liveness probe), or silently did nothing that
mattered (`chmod 0o600` only toggles the read-only flag, so raw customer audio
was readable by every other user of a multi-session host).

Standard library only; the Windows branches use ctypes against kernel32 and
advapi32, which every Windows has, and `icacls`, which every Windows has too.
"""

from __future__ import annotations

import logging
import os
import shutil
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

logger = logging.getLogger(__name__)

IS_WINDOWS = os.name == "nt"


# -- identity ---------------------------------------------------------------

def hostname() -> str:
    """This machine's name, on every OS (`os.uname` does not exist on Windows)."""
    return socket.gethostname()


# -- processes --------------------------------------------------------------

if IS_WINDOWS:  # pragma: no cover - exercised by the Windows CI job
    import ctypes
    from ctypes import wintypes

    _kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    _kernel32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    _kernel32.OpenProcess.restype = wintypes.HANDLE
    _kernel32.GetExitCodeProcess.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
    _kernel32.GetExitCodeProcess.restype = wintypes.BOOL
    _kernel32.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
    _kernel32.WaitForSingleObject.restype = wintypes.DWORD
    _kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
    _kernel32.CloseHandle.restype = wintypes.BOOL
    _kernel32.GetCurrentProcess.restype = wintypes.HANDLE
    _kernel32.LocalFree.argtypes = [ctypes.c_void_p]
    _kernel32.LocalFree.restype = ctypes.c_void_p

    _PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
    _SYNCHRONIZE = 0x00100000
    _STILL_ACTIVE = 259
    _ERROR_ACCESS_DENIED = 5
    _INFINITE = 0xFFFFFFFF


def pid_alive(pid: int) -> bool:
    """Is a process with this pid running on this machine? Never signals it.

    When the answer cannot be known, it is "alive": the caller uses a False to
    take over another process's work, and doing that to a live process is the
    worse mistake.
    """
    if pid <= 0:
        return False
    if not IS_WINDOWS:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            return True      # exists, owned by someone else
        return True
    handle = _kernel32.OpenProcess(_PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
    if not handle:
        # Access denied means it exists and belongs to another user; anything
        # else (invalid parameter) means there is no such process.
        return ctypes.get_last_error() == _ERROR_ACCESS_DENIED
    try:
        code = wintypes.DWORD()
        if not _kernel32.GetExitCodeProcess(handle, ctypes.byref(code)):
            return True
        return code.value == _STILL_ACTIVE
    finally:
        _kernel32.CloseHandle(handle)


def exit_when_process_ends(pid: int) -> None:
    """Start a daemon thread that ends THIS process when process `pid` ends.

    A child whose parent is killed keeps running on every OS. POSIX shows it
    by reparenting (getppid changes); Windows never reparents, so there the
    parent's process handle is waited on instead - which also cannot be fooled
    by the pid being reused.
    """
    if IS_WINDOWS:
        handle = _kernel32.OpenProcess(_SYNCHRONIZE, False, pid)
        if not handle:
            if ctypes.get_last_error() != _ERROR_ACCESS_DENIED:
                os._exit(0)          # already gone
            return                   # cannot watch it; do not guess

        def _wait() -> None:
            _kernel32.WaitForSingleObject(handle, _INFINITE)
            os._exit(0)
    else:
        def _wait() -> None:
            while os.getppid() == pid:
                time.sleep(2.0)
            os._exit(0)
    threading.Thread(target=_wait, name="parent-watch", daemon=True).start()


# -- files ------------------------------------------------------------------

def replace(src: str | os.PathLike, dst: str | os.PathLike) -> None:
    """`os.replace`, surviving the moment an antivirus holds a new file open.

    On Windows a rename onto (or of) a file that another process has open fails
    with PermissionError, and real-time scanners open every file just written.
    The lock lasts milliseconds; a short bounded retry is the standard answer.
    POSIX renames never hit this, so there it is a plain os.replace.
    """
    if not IS_WINDOWS:
        os.replace(src, dst)
        return
    delay = 0.05
    for attempt in range(10):
        try:
            os.replace(src, dst)
            return
        except PermissionError:
            if attempt == 9:
                raise
            time.sleep(delay)
            delay = min(delay * 2, 1.0)


_PRIVATE_DONE: set[str] = set()
_PRIVATE_LOCK = threading.Lock()


def _current_user_sid() -> str | None:  # pragma: no cover - Windows only
    advapi32 = ctypes.WinDLL("advapi32", use_last_error=True)
    advapi32.OpenProcessToken.argtypes = [wintypes.HANDLE, wintypes.DWORD,
                                          ctypes.POINTER(wintypes.HANDLE)]
    advapi32.OpenProcessToken.restype = wintypes.BOOL
    advapi32.GetTokenInformation.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p,
                                             wintypes.DWORD, ctypes.POINTER(wintypes.DWORD)]
    advapi32.GetTokenInformation.restype = wintypes.BOOL
    advapi32.ConvertSidToStringSidW.argtypes = [ctypes.c_void_p,
                                                ctypes.POINTER(wintypes.LPWSTR)]
    advapi32.ConvertSidToStringSidW.restype = wintypes.BOOL
    token_query, token_user = 0x0008, 1
    token = wintypes.HANDLE()
    if not advapi32.OpenProcessToken(_kernel32.GetCurrentProcess(), token_query,
                                     ctypes.byref(token)):
        return None
    try:
        size = wintypes.DWORD()
        advapi32.GetTokenInformation(token, token_user, None, 0, ctypes.byref(size))
        buf = ctypes.create_string_buffer(size.value)
        if not advapi32.GetTokenInformation(token, token_user, buf, size, ctypes.byref(size)):
            return None
        psid = ctypes.cast(buf, ctypes.POINTER(ctypes.c_void_p))[0]
        text = wintypes.LPWSTR()
        if not advapi32.ConvertSidToStringSidW(psid, ctypes.byref(text)):
            return None
        try:
            return text.value
        finally:
            _kernel32.LocalFree(ctypes.cast(text, ctypes.c_void_p))
    finally:
        _kernel32.CloseHandle(token)


def make_private_dir(path: Path) -> None:
    """Create `path` if needed and make it readable by its owner only.

    POSIX: mode 0700. Windows: chmod cannot express that, and a folder made
    under C:\\ inherits "Users: read" - on a multi-session VDI host that is
    every other employee logged in to the same machine. There the folder's
    inherited permissions are replaced by exactly three entries: the current
    user, SYSTEM and Administrators, inherited by everything created inside
    it later. The owner of a folder may set its permissions without admin
    rights. Identities are given as SIDs, so a Hebrew-language Windows (where
    the group names are translated) behaves the same.

    Best effort, like the chmod it replaces: a filesystem that refuses (a
    network share, FAT) is logged, never fatal.
    """
    path.mkdir(parents=True, exist_ok=True)
    if not IS_WINDOWS:
        try:
            path.chmod(0o700)
        except OSError:  # pragma: no cover - unusual filesystems
            logger.debug("could not restrict permissions on %s", path)
        return
    key = str(path.resolve()).lower()
    with _PRIVATE_LOCK:
        if key in _PRIVATE_DONE:
            return
        _PRIVATE_DONE.add(key)
    sid = _current_user_sid()
    if sid is None:
        logger.warning("could not determine the current user's SID; %s keeps its "
                       "inherited permissions", path)
        return
    cmd = ["icacls", str(path), "/inheritance:r", "/grant:r",
           f"*{sid}:(OI)(CI)F", "*S-1-5-18:(OI)(CI)F", "*S-1-5-32-544:(OI)(CI)F", "/Q"]
    try:
        proc = run_text(cmd, timeout=60)
    except (OSError, subprocess.SubprocessError) as exc:
        logger.warning("could not restrict permissions on %s: %s", path, exc)
        return
    if proc.returncode != 0:
        logger.warning("could not restrict permissions on %s: %s", path,
                       (proc.stdout + proc.stderr).strip()[:300])


def private_to_owner(path: Path) -> bool:
    """Does anyone other than the owner, SYSTEM and Administrators have access?
    False when it cannot be determined. For tests and preflight."""
    if not IS_WINDOWS:
        return (path.stat().st_mode & 0o077) == 0
    sid = _current_user_sid()
    try:
        proc = run_text(["icacls", str(path)], timeout=60)
    except (OSError, subprocess.SubprocessError):
        return False
    if proc.returncode != 0 or sid is None:
        return False
    # icacls prints names, not SIDs, and names are translated; so compare the
    # NUMBER of entries instead: after make_private_dir there are exactly three.
    entries = [line for line in proc.stdout.splitlines() if ":(" in line]
    return len(entries) == 3


# -- external programs ------------------------------------------------------

def project_root() -> Path:
    """The extracted project directory (src/callqa/portable.py -> root)."""
    return Path(__file__).resolve().parents[2]


def find_executable(name: str) -> str | None:
    """Where is `name` (ffmpeg, ffprobe)? None if nowhere.

    Installing a program on a VDI desktop usually needs admin rights and
    editing PATH is easy to get wrong, so besides PATH this looks where a user
    without rights can simply unzip it: the folder named by CALLQA_FFMPEG_DIR,
    or a `tools` folder inside the project (any sub-folder of it, since the
    Windows builds of ffmpeg unzip to e.g. tools/ffmpeg-...-win64-lgpl/bin).
    """
    exe = f"{name}.exe" if IS_WINDOWS else name
    candidates: list[Path] = []
    configured = os.environ.get("CALLQA_FFMPEG_DIR")
    if configured:
        base = Path(configured)
        candidates += [base / exe, base / "bin" / exe]
    tools = project_root() / "tools"
    if tools.is_dir():
        candidates += [tools / exe, tools / "bin" / exe]
        candidates += sorted(tools.glob(f"*/{exe}")) + sorted(tools.glob(f"*/bin/{exe}"))
    for candidate in candidates:
        if candidate.is_file():
            return str(candidate)
    return shutil.which(name)


def run_text(cmd: list[str], timeout: float | None = None,
             **kwargs) -> subprocess.CompletedProcess:  # noqa: ANN003
    """subprocess.run capturing text, decoded as UTF-8 whatever the OS.

    `text=True` alone decodes with the ANSI code page on Windows (cp1252 or
    cp1255), and ffmpeg, ffprobe and git all write UTF-8: a Hebrew file name in
    an error message was a UnicodeDecodeError instead of the message.
    """
    return subprocess.run(cmd, capture_output=True, encoding="utf-8", errors="replace",
                          timeout=timeout, **kwargs)


# -- console ----------------------------------------------------------------

def configure_stdio() -> None:
    """Make printing Hebrew safe wherever stdout/stderr go.

    A Windows console is written as Unicode already. But redirected to a file
    or a pipe - a Scheduled Task, `> log.txt`, CI - Python encodes with the
    ANSI code page, and cp1252 cannot encode Hebrew: the first Hebrew message
    raised UnicodeEncodeError and took the command down with it. Redirected
    streams are switched to UTF-8 (what every other tool writes), and nothing
    unencodable can raise.
    """
    for stream in (sys.stdout, sys.stderr):
        encoding = (getattr(stream, "encoding", None) or "").lower().replace("-", "")
        if encoding in ("utf8", "utf8sig"):
            continue
        try:
            stream.reconfigure(encoding="utf-8", errors="backslashreplace")
        except (AttributeError, ValueError, OSError):  # not a TextIOWrapper
            pass
