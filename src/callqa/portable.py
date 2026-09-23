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
import stat
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
        is_parent = os.getppid() == pid

        def _wait() -> None:
            # A dead parent shows as reparenting at once; any other process as
            # its pid disappearing.
            while pid_alive(pid) and (not is_parent or os.getppid() == pid):
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


def remove_file(path: Path) -> None:
    """Delete a file, the way Windows needs it deleted.

    On Windows a file with the read-only attribute cannot be deleted (a
    recording copied from a read-only share or a DMS export keeps it), and one
    an antivirus is scanning cannot be deleted for a moment. Both are retried;
    anything still failing raises, so the caller knows the data is still there.
    """
    attempts = 6 if IS_WINDOWS else 1
    for attempt in range(attempts):
        try:
            path.unlink()
            return
        except FileNotFoundError:
            return
        except PermissionError:
            if attempt == attempts - 1:
                raise
            try:
                os.chmod(path, stat.S_IWRITE)
            except OSError:
                pass
            time.sleep(0.2 * (attempt + 1))


def remove_tree(path: Path) -> bool:
    """shutil.rmtree that clears read-only flags and waits out brief locks.
    True when the tree is gone."""
    def _retry(func, target, _exc) -> None:  # noqa: ANN001
        for attempt in range(5):
            try:
                os.chmod(target, stat.S_IWRITE)
                func(target)
                return
            except FileNotFoundError:
                return
            except OSError:
                time.sleep(0.2 * (attempt + 1))

    if sys.version_info >= (3, 12):
        shutil.rmtree(path, onexc=_retry)
    else:  # pragma: no cover - 3.11
        shutil.rmtree(path, onerror=_retry)
    return not path.exists()


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
    resolved = path.resolve()
    if resolved == Path(resolved.anchor):
        logger.warning("not changing the permissions of a drive root (%s)", resolved)
        return
    key = str(resolved).lower()
    with _PRIVATE_LOCK:
        if key in _PRIVATE_DONE:
            return
        _PRIVATE_DONE.add(key)
    sid = _current_user_sid()
    if sid is None:
        logger.warning("could not determine the current user's SID; %s keeps its "
                       "inherited permissions", path)
        return
    cmd = [_icacls(), str(path), "/inheritance:r", "/grant:r",
           f"*{sid}:(OI)(CI)F", "*S-1-5-18:(OI)(CI)F", "*S-1-5-32-544:(OI)(CI)F", "/Q"]
    try:
        proc = run_text(cmd, timeout=60, encoding="oem")
    except (OSError, subprocess.SubprocessError) as exc:
        logger.warning("could not restrict permissions on %s: %s", path, exc)
        return
    if proc.returncode != 0:
        logger.warning("could not restrict permissions on %s: %s", path,
                       (proc.stdout + proc.stderr).strip()[:300])


def make_private_root(path: Path) -> None:
    """A folder the pipeline writes raw material under, made owner-only on
    Windows so everything created inside inherits that. On POSIX the files
    themselves are written 0600, so the folder is only created."""
    if IS_WINDOWS:
        make_private_dir(path)
    else:
        path.mkdir(parents=True, exist_ok=True)


def _icacls() -> str:
    # The system copy by absolute path: a bare "icacls" is resolved through the
    # current directory and PATH, where anyone able to write there could put
    # their own.
    return os.path.join(os.environ.get("SystemRoot", r"C:\Windows"), "System32", "icacls.exe")


def location_risk(path: Path) -> str | None:
    """Why `path` is a bad place for raw call data on this machine, or None.

    Windows only. OneDrive's Known Folder Move silently syncs Desktop and
    Documents to the cloud - raw transcripts included - and SQLite's locking
    is unreliable on a network share.
    """
    if not IS_WINDOWS:
        return None
    try:
        resolved = path.resolve()
    except OSError:
        return None
    text = str(resolved)
    if text.startswith("\\\\"):
        return "a network share (UNC path)"
    for var in ("OneDrive", "OneDriveCommercial", "OneDriveConsumer"):
        root = os.environ.get(var)
        if root and resolved.is_relative_to(Path(root)):
            return "a OneDrive folder, which syncs its contents to the cloud"
    drive = resolved.anchor
    if drive:
        _kernel32.GetDriveTypeW.argtypes = [wintypes.LPCWSTR]
        _kernel32.GetDriveTypeW.restype = wintypes.UINT
        if _kernel32.GetDriveTypeW(drive) == 4:          # DRIVE_REMOTE
            return f"a network drive ({drive})"
    return None


_OWNER_ONLY_SIDS = {"S-1-5-18", "S-1-5-32-544"}          # SYSTEM, Administrators


def _allowed_sids(path: Path) -> list[str] | None:  # pragma: no cover - Windows only
    """The SIDs the folder's DACL ALLOWS, read with the Win32 security API.

    Not icacls's output: that is a display format, translated into the
    machine's language and printed in the console code page.
    """
    advapi32 = ctypes.WinDLL("advapi32", use_last_error=True)
    advapi32.GetNamedSecurityInfoW.argtypes = [
        wintypes.LPCWSTR, ctypes.c_int, wintypes.DWORD,
        ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(ctypes.c_void_p),
        ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(ctypes.c_void_p),
        ctypes.POINTER(ctypes.c_void_p)]
    advapi32.GetNamedSecurityInfoW.restype = wintypes.DWORD
    advapi32.GetAce.argtypes = [ctypes.c_void_p, wintypes.DWORD, ctypes.POINTER(ctypes.c_void_p)]
    advapi32.GetAce.restype = wintypes.BOOL
    advapi32.ConvertSidToStringSidW.argtypes = [ctypes.c_void_p,
                                                ctypes.POINTER(wintypes.LPWSTR)]
    advapi32.ConvertSidToStringSidW.restype = wintypes.BOOL
    se_file_object, dacl_security_information = 1, 0x4
    dacl, descriptor = ctypes.c_void_p(), ctypes.c_void_p()
    if advapi32.GetNamedSecurityInfoW(str(path), se_file_object, dacl_security_information,
                                      None, None, ctypes.byref(dacl), None,
                                      ctypes.byref(descriptor)) != 0:
        return None
    try:
        if not dacl.value:
            return None                      # a NULL DACL grants everyone everything
        ace_count = ctypes.cast(dacl, ctypes.POINTER(ctypes.c_uint16))[2]
        sids: list[str] = []
        for index in range(ace_count):
            ace = ctypes.c_void_p()
            if not advapi32.GetAce(dacl, index, ctypes.byref(ace)):
                return None
            if ctypes.cast(ace, ctypes.POINTER(ctypes.c_ubyte))[0] != 0:
                continue                     # only ACCESS_ALLOWED_ACE grants anything
            text = wintypes.LPWSTR()
            # ACCESS_ALLOWED_ACE: 4-byte header, 4-byte mask, then the SID.
            if not advapi32.ConvertSidToStringSidW(ace.value + 8, ctypes.byref(text)):
                return None
            sids.append(text.value)
            _kernel32.LocalFree(ctypes.cast(text, ctypes.c_void_p))
        return sids
    finally:
        _kernel32.LocalFree(descriptor)


def acl_summary(path: Path) -> str:
    """For diagnostics: who may access `path`, as SIDs (Windows) or a mode."""
    if not IS_WINDOWS:
        return oct(path.stat().st_mode & 0o777)
    return f"user={_current_user_sid()} allowed={_allowed_sids(path)}"


def private_to_owner(path: Path) -> bool:
    """Is `path` accessible only to its owner (and, on Windows, SYSTEM and
    Administrators)? False when it cannot be determined."""
    if not IS_WINDOWS:
        return (path.stat().st_mode & 0o077) == 0
    user = _current_user_sid()
    sids = _allowed_sids(path)
    if user is None or not sids:
        return False
    return user in sids and set(sids) <= _OWNER_ONLY_SIDS | {user}


def held_open_for_writing(path: Path) -> bool:
    """Is some other program still writing `path`? Windows only; else False.

    Explorer, robocopy and CopyFileEx set a copy's final SIZE before writing
    its data, so "the size stopped changing" - the watch driver's test - is
    true from the first moment of a slow copy, and the call was transcribed
    from a WAV whose tail was still zeros. Opening the file while refusing to
    share it with writers fails exactly while a writer holds it.
    """
    if not IS_WINDOWS:
        return False
    _kernel32.CreateFileW.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD,
                                      ctypes.c_void_p, wintypes.DWORD, wintypes.DWORD,
                                      wintypes.HANDLE]
    _kernel32.CreateFileW.restype = wintypes.HANDLE
    generic_read, file_share_read, open_existing = 0x80000000, 0x1, 3
    handle = _kernel32.CreateFileW(str(path), generic_read, file_share_read, None,
                                   open_existing, 0, None)
    if handle is None or handle == wintypes.HANDLE(-1).value:
        return ctypes.get_last_error() == 32            # ERROR_SHARING_VIOLATION
    _kernel32.CloseHandle(handle)
    return False


def move(src: Path, dst: Path) -> None:
    """Move a file: a same-volume rename, retried past a brief antivirus lock
    (portable.replace); a copy-and-delete only when the volumes differ."""
    try:
        replace(src, dst)
    except OSError as exc:
        if getattr(exc, "winerror", None) == 17 or exc.errno == 18:   # other volume
            shutil.move(str(src), str(dst))
        else:
            raise


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


def run_text(cmd: list[str], timeout: float | None = None, encoding: str = "utf-8",
             **kwargs) -> subprocess.CompletedProcess:  # noqa: ANN003
    """subprocess.run capturing text, decoded as UTF-8 whatever the OS.

    `text=True` alone decodes with the ANSI code page on Windows (cp1252 or
    cp1255), and ffmpeg, ffprobe and git all write UTF-8: a Hebrew file name in
    an error message was a UnicodeDecodeError instead of the message. Windows'
    own console tools (icacls) write the OEM code page instead: encoding="oem".
    Undecodable bytes are replaced, never raised.
    """
    return subprocess.run(cmd, capture_output=True, encoding=encoding, errors="replace",
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
