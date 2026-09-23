"""`callqa preflight`: verify the environment BEFORE a batch touches a call, so
a missing model or a full disk stops the run at second zero, not at call 400.

Checks the configuration, the presence and hashes of the models the current
config will use, the reachability of any remote endpoint, free disk, and the
input files — reusing the loaders and connectivity checks already in the core.
"""

from __future__ import annotations

import logging
import shutil
from dataclasses import dataclass
from pathlib import Path

from callqa.config import Config
from callqa.ops.provenance import dir_sha256, read_model_manifest
from callqa.portable import IS_WINDOWS, location_risk, private_to_owner

logger = logging.getLogger(__name__)

# Rough per-call output footprint (redacted audio ~2 MB/min + transcripts +
# scores + report). Deliberately generous; the floor below dominates for small
# batches. Used only to size the disk check.
PER_CALL_BYTES = 20 * 1024 * 1024
DISK_FLOOR_BYTES = 1 * 1024 * 1024 * 1024      # never proceed with < 1 GiB free


@dataclass
class Check:
    name: str
    ok: bool
    critical: bool
    detail: str


def _existing_ancestor(path: Path) -> Path:
    p = path.resolve()
    while not p.exists() and p != p.parent:
        p = p.parent
    return p


def _disk_check(config: Config, n_calls: int) -> Check:
    free = shutil.disk_usage(_existing_ancestor(config.paths.output_dir)).free
    need = max(DISK_FLOOR_BYTES, n_calls * PER_CALL_BYTES)
    return Check("disk space", free >= need, True,
                 f"{free // (1024**3)} GiB free; need ~{need // (1024**2)} MiB "
                 f"for {n_calls} call(s)")


def _inputs_check(config: Config) -> tuple[Check, int]:
    from callqa.ingestion import load_metadata

    v = load_metadata(config.paths.input_dir / "metadata.csv",
                      config.paths.input_dir / "calls")
    if not v.ok:
        # Say WHICH problem. "invalid: 1 problem(s)" was printed for a
        # metadata.csv that simply did not exist yet - on a fresh extract, the
        # first thing an operator sees - with nothing saying so.
        from callqa.redaction import sanitize_error

        first = v.problems[0]
        more = f" (+{len(v.problems) - 1} more; run validate-inputs)" if len(v.problems) > 1 else ""
        return Check("inputs", False, True,
                     f"metadata.csv: {sanitize_error(first.problem, limit=200)}{more}"), 0
    return Check("inputs", True, True, f"{len(v.rows)} call(s) in metadata.csv"), len(v.rows)


def _verify_local_model(config: Config, role: str, model_dir: Path, deep: bool) -> Check:
    if not model_dir.exists() or not any(model_dir.iterdir()):
        return Check(f"model:{role}", False, True, f"missing or empty: {model_dir}")
    entry = next((m for m in read_model_manifest(config.paths.models_dir)
                  if m["role"] == role), None)
    if entry is None:
        return Check(f"model:{role}", True, False,
                     f"present at {model_dir} (no manifest entry to verify against)")
    if deep and entry.get("sha256"):
        actual = dir_sha256(model_dir)
        if actual != entry["sha256"]:
            return Check(f"model:{role}", False, True,
                         "weight hash MISMATCH — not the model that produced earlier results")
        return Check(f"model:{role}", True, True, "present; weight hash verified")
    # The fast path compares total SIZE with what the download recorded. A
    # directory that merely exists and is non-empty is exactly what a truncated
    # copy looks like - and moving 20 GB onto an air-gapped machine by disk or
    # share is where truncation happens. Summing sizes is instant; hashing is
    # what --deep is for.
    expected = entry.get("size_bytes")
    if expected:
        actual_size = sum(f.stat().st_size for f in model_dir.rglob("*") if f.is_file())
        if actual_size != expected:
            return Check(f"model:{role}", False, True,
                         f"size MISMATCH at {model_dir}: {actual_size:,} bytes on disk, "
                         f"{expected:,} recorded at download. Incomplete copy? Re-transfer "
                         "it, then run preflight --deep.")
    return Check(f"model:{role}", True, True,
                 "present, size matches the download (fast check; --deep re-hashes)")


def _hf_hub_cache() -> Path:
    """Where huggingface_hub keeps its cache, resolved the way it resolves it,
    without importing it (it is deliberately not a runtime dependency)."""
    import os

    def expand(value: str) -> Path:
        # huggingface_hub expands both: HF_HOME=%LOCALAPPDATA%\hf in .env works
        # at run time, so preflight has to read it the same way.
        return Path(os.path.expandvars(os.path.expanduser(value)))

    for name in ("HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE"):
        if os.environ.get(name):
            return expand(os.environ[name])
    home = os.environ.get("HF_HOME") or os.path.join(
        os.environ.get("XDG_CACHE_HOME", os.path.expanduser("~/.cache")), "huggingface")
    return expand(home) / "hub"


def _diarization_check(config: Config) -> Check:
    """Can the diarization model be loaded on THIS machine, offline?

    Not critical - a genuinely stereo recording never needs it - but a mono or
    dual-mono one cannot be processed without it, and that failure otherwise
    surfaces on the first such call in the middle of a batch. The runtime
    dependency is not models/: pyannote loads through the Hugging Face cache of
    whoever ran the download, and on an air-gapped machine that cache has to
    have been copied across and HF_HOME pointed at it.
    """
    model = config.speakers.diarization_model
    if Path(model).exists():
        return Check("model:diarization", True, False, f"local pipeline at {model}")
    cached = _hf_hub_cache() / ("models--" + model.replace("/", "--"))
    if cached.exists():
        # The folder can exist with nothing usable in it. The cache keeps its
        # files as symlinks into blobs/, and a copy onto Windows made without
        # the right to create symlinks (no admin, no Developer Mode) leaves
        # them missing or empty - and the first mono call is where that showed.
        files = [f for f in (cached / "snapshots").glob("*/*") if not f.is_dir()]
        if files and not any(f.is_file() and f.stat().st_size > 0 for f in files):
            return Check(
                "model:diarization", False, False,
                f"the Hugging Face cache at {cached} has no readable files: it was copied "
                "without following its symlinks. Copy it again with links resolved "
                "(e.g. zip it on the source machine, or `cp -rL`).")
        return Check("model:diarization", True, False, f"in the Hugging Face cache ({cached})")
    return Check(
        "model:diarization", False, False,
        f"'{model}' is not in the Hugging Face cache at {_hf_hub_cache()}. Mono "
        "recordings will fail. Run scripts/download_models.py --diarization as the "
        "user that runs the pipeline, or copy that cache here and set HF_HOME."
    )


def _ner_check(config: Config) -> Check:
    """redaction.ner is a privacy control; if it is on, its model must be here.
    The redactor refuses to start without it, so find out now, not at call 1."""
    model_dir = config.paths.models_dir / "dictabert-ner"
    if model_dir.exists() and any(model_dir.iterdir()):
        return Check("model:ner", True, True, f"present at {model_dir}")
    return Check("model:ner", False, True,
                 f"redaction.ner is on but {model_dir} is missing. Run "
                 "scripts/download_models.py --ner, or set redaction.ner: false.")


def _decoder_check() -> Check:
    """What will read recordings that are not plain PCM WAV."""
    from callqa.audio import have_pyav
    from callqa.portable import find_executable

    ffmpeg = find_executable("ffmpeg")
    if ffmpeg:
        return Check("audio decoder", True, False, f"ffmpeg at {ffmpeg}")
    if have_pyav():
        return Check("audio decoder", True, False,
                     "PyAV (installed with the engines) - mp3, m4a and telephony WAV work")
    return Check("audio decoder", False, False,
                 "neither ffmpeg nor PyAV: only plain PCM WAV can be read. Unzip ffmpeg "
                 "into tools/, or install requirements-server.txt.")


def _vc_runtime_check() -> Check:  # pragma: no cover - Windows only
    """torch and ctranslate2 need msvcp140.dll, which Python does not ship and
    installing needs admin rights - so it is found out here, not at call 1."""
    import ctypes

    try:
        ctypes.WinDLL("msvcp140.dll")
        return Check("Visual C++ runtime", True, True, "msvcp140.dll present")
    except OSError:
        return Check("Visual C++ runtime", False, True,
                     "msvcp140.dll is missing: the model engines cannot load. Ask IT to "
                     "install the Microsoft Visual C++ 2015-2022 Redistributable (x64).")


def _reachable(name: str, build) -> Check:  # noqa: ANN001
    try:
        build().check_connectivity()
        return Check(name, True, True, "reachable")
    except Exception as exc:  # noqa: BLE001 - any failure means "not ready"
        return Check(name, False, True, f"unreachable: {type(exc).__name__}: {exc}")


def _engine_checks(config: Config, deep: bool) -> list[Check]:
    if config.run.mock or (config.asr.engine == "mock" and config.judge.engine == "mock"):
        return [Check("engines", True, False,
                      "mock engines — no models or endpoints required")]
    checks: list[Check] = [_decoder_check()]
    if IS_WINDOWS:
        checks.append(_vc_runtime_check())
    if config.asr.engine == "faster_whisper":
        checks.append(_verify_local_model(config, "asr", Path(config.asr.model_dir), deep))
    if config.speakers.mode != "stereo":
        checks.append(_diarization_check(config))
    if config.redaction.ner:
        checks.append(_ner_check(config))
    if config.judge.engine == "vllm":
        from callqa.judge.vllm_judge import VLLMJudge
        checks.append(_reachable("judge endpoint", lambda: VLLMJudge(config.judge)))
    return checks


def _location_checks(config: Config) -> list[Check]:
    """Where the raw data will live (Windows).

    OneDrive's Known Folder Move syncs Desktop and Documents to the cloud, and
    that is where an unzipped download lands by habit; SQLite's locking is
    unreliable on a network share. The output and the state database may be
    in neither. An input folder on a share can be a deliberate drop folder, so
    that one only warns - unless it is OneDrive.
    """
    checks: list[Check] = []
    places = (("output", config.paths.output_dir, True),
              ("state database", config.paths.state_db.parent, True),
              ("input", config.paths.input_dir, False))
    for label, path, critical in places:
        risk = location_risk(_existing_ancestor(path))
        if risk:
            checks.append(Check(
                f"location:{label}", False, critical or "OneDrive" in risk,
                f"{path} is on {risk}. Keep the project in a local folder under your "
                "user profile, e.g. C:\\Users\\<name>\\BRS_Model-main."))
    output = config.paths.output_dir
    if IS_WINDOWS and output.exists():
        private = private_to_owner(output)
        checks.append(Check(
            "privacy", private, True,
            "the output folder is readable by this user, SYSTEM and Administrators only"
            if private else
            f"{output} can be read by other users of this machine. Run any pipeline "
            "command once (it restricts the folder), or restrict it with icacls."))
    return checks


def run_preflight(config: Config, deep: bool = False) -> list[Check]:
    """All checks. `deep` re-hashes local model weights (slow)."""
    checks = [Check("configuration", True, True, "loaded and valid")]
    checks.extend(_location_checks(config))
    inputs, n_calls = _inputs_check(config)
    checks.append(inputs)
    checks.extend(_engine_checks(config, deep))
    checks.append(_disk_check(config, n_calls))
    return checks
