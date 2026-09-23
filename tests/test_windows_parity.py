"""The same inputs give the same results on Windows and Linux, and the
Windows-only safeguards do what they say.

Each hash and ordering here once differed between the two: a model hashed on
the Linux download machine failed verification on a Windows desktop, a rubric
checked out with CRLF put Windows results in a separate cohort, and fast
stages sorted by name when Windows' 15.6 ms clock gave them one timestamp.
"""

from __future__ import annotations

import os
import stat
import sys
import time
from pathlib import Path, PureWindowsPath

import pytest

REPO = Path(__file__).resolve().parents[1]
windows_only = pytest.mark.skipif(sys.platform != "win32", reason="Windows behaviour")


def test_a_model_folder_hashes_the_same_on_every_os(tmp_path: Path) -> None:
    """The literal was computed on Linux; the Windows CI job must agree."""
    from callqa.ops.provenance import dir_sha256

    (tmp_path / "README.md").write_bytes(b"readme\n")
    (tmp_path / "config.json").write_bytes(b"{}\n")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "Tok.json").write_bytes(b"tok\n")
    assert dir_sha256(tmp_path) == \
        "9152a73f61e39a14900cfd9349de0304aae57fa5645bd87f36bd2961e596c690"


def test_the_rubric_hash_ignores_crlf_and_a_byte_order_mark(tmp_path: Path) -> None:
    from callqa.rubric import load_rubric

    original = REPO / "config" / "rubric.yaml"
    windows_copy = tmp_path / "rubric.yaml"
    windows_copy.write_bytes(b"\xef\xbb\xbf" + original.read_bytes().replace(b"\n", b"\r\n"))
    assert load_rubric(windows_copy).sha256 == load_rubric(original).sha256


def test_the_config_hash_ignores_path_separators() -> None:
    """A Windows config holds WindowsPath("data\\output"); it must hash like
    the Linux PosixPath("data/output")."""
    from callqa.ops.provenance import _portable

    windows = {"paths": {"output_dir": PureWindowsPath("data\\output")}}
    posix = {"paths": {"output_dir": Path("data/output")}}
    assert _portable(windows) == _portable(posix) == {"paths": {"output_dir": "data/output"}}


def test_stages_done_in_one_clock_tick_keep_their_order(tmp_path: Path, monkeypatch) -> None:  # noqa: ANN001
    from callqa import state as state_mod

    db = state_mod.StateDB(tmp_path / "s.db")
    monkeypatch.setattr(state_mod.time, "time", lambda: 1000.0)
    for stage in ("speakers", "redaction", "features"):
        db.mark_stage_done("C1", stage, None)
    assert db.completed_stages("C1") == ["speakers", "redaction", "features"]


def test_banker_report_names_never_collide_on_windows() -> None:
    from callqa.ingestion import is_windows_reserved
    from callqa.reporting.banker_report import _banker_slugs

    slugs = _banker_slugs(["B001", "b001", "NUL"])
    names = [s.casefold() for s in slugs.values()]
    assert len(set(names)) == 3
    assert not any(is_windows_reserved(s) for s in slugs.values())


# -- Windows-only safeguards ------------------------------------------------

@windows_only
def test_retention_destroys_a_read_only_recording(tmp_path: Path) -> None:
    """A recording copied from a read-only share keeps the attribute, and
    Windows refuses to delete it."""
    from callqa.ops.retention import apply_retention

    raw = tmp_path / "out" / "transcripts" / "C1.json"
    raw.parent.mkdir(parents=True)
    raw.write_text("{}", encoding="utf-8")
    os.chmod(raw, stat.S_IREAD)
    old = time.time() - 200 * 86400
    os.utime(raw, (old, old))
    destroyed = apply_retention(tmp_path / "out", raw_days=90)
    assert len(destroyed) == 1 and not raw.exists()


@windows_only
def test_a_onedrive_folder_is_recognised(tmp_path: Path, monkeypatch) -> None:  # noqa: ANN001
    from callqa.portable import location_risk

    monkeypatch.setenv("OneDrive", str(tmp_path))
    assert "OneDrive" in (location_risk(tmp_path / "BRS_Model-main" / "data") or "")
    monkeypatch.delenv("OneDrive")
    assert location_risk(tmp_path) is None


@windows_only
def test_a_reused_pid_does_not_keep_a_dead_lock_alive() -> None:
    from callqa.portable import started_after

    assert started_after(os.getpid(), time.time() + 60) is False     # not in the future
    assert started_after(os.getpid(), 0.0) is True                   # started after 1970


def test_verbose_logging_hides_torchaudio_ffmpeg_probe_tracebacks() -> None:
    import logging

    from callqa.cli import _setup_logging

    _setup_logging(verbose=True)
    assert not logging.getLogger("torio._extension.utils").isEnabledFor(logging.DEBUG)
