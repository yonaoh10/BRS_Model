"""The OS calls that differ between POSIX and Windows (callqa.portable).

These run on every CI platform, Windows included: each one exists because its
POSIX spelling crashed, killed a process, or silently did nothing on Windows.
"""

from __future__ import annotations

import io
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

from callqa import portable


def test_hostname_is_known() -> None:
    assert portable.hostname()


def test_pid_alive_answers_without_harming_the_process() -> None:
    """On Windows os.kill(pid, 0) is CTRL_C_EVENT/TerminateProcess; the probe
    that replaced it must leave a live process alive."""
    assert portable.pid_alive(os.getpid())
    assert not portable.pid_alive(0)
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    try:
        for _ in range(3):
            assert portable.pid_alive(child.pid)
        time.sleep(0.5)
        assert child.poll() is None, "the liveness probe must never end the process"
    finally:
        child.kill()
        child.wait(timeout=10)
    assert not portable.pid_alive(child.pid)


def test_a_worker_ends_when_the_process_it_watches_ends() -> None:
    watched = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(1.5)"])
    code = ("import sys, time; from callqa.portable import exit_when_process_ends; "
            "exit_when_process_ends(int(sys.argv[1])); time.sleep(60); sys.exit(3)")
    worker = subprocess.Popen([sys.executable, "-c", code, str(watched.pid)])
    try:
        watched.wait(timeout=20)          # reaped, so its pid is really gone
        assert worker.wait(timeout=20) == 0
    finally:
        if worker.poll() is None:
            worker.kill()
            worker.wait(timeout=10)


def test_replace_waits_out_a_briefly_open_target(tmp_path: Path) -> None:
    """Windows refuses to rename onto a file another process holds open - which
    an antivirus does to every new file for a moment."""
    src, dst = tmp_path / "new.txt", tmp_path / "out.txt"
    dst.write_text("old", encoding="utf-8")
    src.write_text("new", encoding="utf-8")
    holder = dst.open("rb")
    threading.Timer(0.3, holder.close).start()
    portable.replace(src, dst)
    holder.close()
    assert dst.read_text(encoding="utf-8") == "new"


def test_a_private_dir_is_private(tmp_path: Path) -> None:
    target = tmp_path / "raw" / "audio"
    portable.make_private_dir(target)
    assert target.is_dir()
    assert portable.private_to_owner(target), portable.acl_summary(target)
    # and what is created inside it later inherits that
    inner = target / "call.wav"
    inner.write_bytes(b"RIFF")
    if os.name == "nt":
        assert portable.private_to_owner(inner), portable.acl_summary(inner)


def test_ffmpeg_is_found_where_a_user_without_admin_can_put_it(tmp_path, monkeypatch) -> None:  # noqa: ANN001
    exe = "ffmpeg.exe" if os.name == "nt" else "ffmpeg"
    (tmp_path / "bin").mkdir()
    (tmp_path / "bin" / exe).write_bytes(b"")
    monkeypatch.setenv("CALLQA_FFMPEG_DIR", str(tmp_path))
    assert portable.find_executable("ffmpeg") == str(tmp_path / "bin" / exe)


def test_run_text_decodes_utf8_whatever_the_code_page() -> None:
    proc = portable.run_text([sys.executable, "-c",
                              "import sys; sys.stdout.buffer.write('שלום'.encode('utf-8'))"])
    assert proc.stdout == "שלום"


def test_hebrew_to_a_redirected_ansi_stream_does_not_raise(monkeypatch) -> None:  # noqa: ANN001
    """A Scheduled Task redirects stdout to a file, and Python then encodes it
    with the ANSI code page: cp1252 has no Hebrew."""
    raw = io.BytesIO()
    stream = io.TextIOWrapper(raw, encoding="cp1252")
    monkeypatch.setattr(sys, "stdout", stream)
    portable.configure_stdio()
    print("שיחה נכשלה")
    sys.stdout.flush()
    assert raw.getvalue().decode("utf-8").strip() == "שיחה נכשלה"


def test_a_dotenv_saved_by_windows_tools_is_read(tmp_path: Path, monkeypatch) -> None:  # noqa: ANN001
    """Notepad's BOM, PowerShell 5.1's UTF-16 and an "ANSI" save all used to
    lose the judge key - the BOM variant silently."""
    from callqa import dotenv

    for i, data in enumerate([
        "\ufeffCALLQA_TEST_KEY_A=one\n".encode(),
        "CALLQA_TEST_KEY_B=two\r\n".encode("utf-16"),
        "# הערה\nCALLQA_TEST_KEY_C=three\n".encode("cp1255"),
    ]):
        env = tmp_path / f"{i}.env"
        env.write_bytes(data)
        monkeypatch.setattr(dotenv, "_LOADED", set())
        dotenv.load_dotenv(env)
    assert os.environ.pop("CALLQA_TEST_KEY_A") == "one"
    assert os.environ.pop("CALLQA_TEST_KEY_B") == "two"
    assert os.environ.pop("CALLQA_TEST_KEY_C") == "three"


def test_yaml_errors_name_the_file_and_the_windows_cause(tmp_path: Path) -> None:
    import pytest

    from callqa.resources import load_yaml

    bad = tmp_path / "config.yaml"
    bad.write_text('asr:\n  model_dir: "C:\\Users\\x\\models"\n', encoding="utf-8")
    with pytest.raises(ValueError, match="single quotes"):
        load_yaml(bad)
    utf16 = tmp_path / "rubric.yaml"
    utf16.write_bytes("name: 'רובריקה'\n".encode("utf-16"))
    assert load_yaml(utf16) == {"name": "רובריקה"}
