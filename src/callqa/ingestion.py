"""Ingestion: metadata.csv validation + audio probing (stage 1)."""

from __future__ import annotations

import csv
import hashlib
import hmac
import io
import json
import logging
import os
import re
import secrets
import subprocess
import unicodedata
import wave
from dataclasses import dataclass, field
from pathlib import Path

from callqa.models import CallInput, CallMeta
from callqa.portable import find_executable, run_text

logger = logging.getLogger(__name__)

REQUIRED_COLUMNS = ["call_id", "banker_id", "file_name"]
OPTIONAL_COLUMNS = ["call_date", "call_type", "banker_channel", "banker_name", "segment"]
# .wav/.mp3 are what the bank's recorders produce. The rest are what a phone
# produces, which is what a staged test call arrives as; they are accepted only
# when ffmpeg is present, since the dependency-free fallback reads WAV only.
AUDIO_EXTENSIONS = {".wav", ".mp3"}
FFMPEG_AUDIO_EXTENSIONS = {".m4a", ".aac", ".mp4", ".ogg", ".opus", ".flac", ".wma", ".amr"}


# A call_id becomes a filename, a URL path segment, an HTML attribute and a
# log line. Anything outside this set has been a path-traversal or an
# injection vector rather than an identifier.
CALL_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
_UNSAFE_CALL_ID_CHARS = re.compile(r"[^A-Za-z0-9._-]+")
# A recording named after the customer is common and puts an identifier in
# every artifact path, log line and report title.
_LONG_DIGIT_RUN = re.compile(r"\d{7,}")


# Names Windows reserves for devices, in any case and with any extension:
# "CON.json" is the console, not a file, so a call called CON could never be
# written. Rejected in metadata and prefixed when derived from a file name.
WINDOWS_RESERVED_NAMES = frozenset(
    {"con", "prn", "aux", "nul", "conin$", "conout$"}
    | {f"{dev}{n}" for dev in ("com", "lpt") for n in range(1, 10)}
)


def is_windows_reserved(name: str) -> bool:
    return name.split(".", 1)[0].strip().casefold() in WINDOWS_RESERVED_NAMES


# Separators people put inside phone and ID numbers: "050-123-4567".
_NUMBER_SEPARATORS = re.compile(r"[\s\-_.()+/]")


def _looks_like_an_identifier(text: str) -> bool:
    """Seven or more digits once the separators are gone: a phone, national ID
    or account number, however the recorder punctuated it."""
    return bool(_LONG_DIGIT_RUN.search(_NUMBER_SEPARATORS.sub("", text)))


_ID_KEY: bytes | None = None


def _install_key() -> bytes:
    """A random per-installation key for naming recordings by digest.

    A plain SHA-256 of a ten-digit phone number is reversed by trying all ten
    billion numbers; keyed with a secret that never leaves this machine it
    cannot be. Created once, beside the pipeline's data, readable by its owner.
    """
    global _ID_KEY
    if _ID_KEY is None:
        from callqa.portable import make_private_root, project_root

        folder = project_root() / "data"
        make_private_root(folder)
        path = folder / ".callqa-id-key"
        try:
            fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0),
                         0o600)
            with os.fdopen(fd, "wb") as fh:
                fh.write(secrets.token_bytes(32))
        except FileExistsError:
            pass
        _ID_KEY = path.read_bytes()
    return _ID_KEY


def _digest(text: str) -> str:
    return hmac.new(_install_key(), text.encode("utf-8"), hashlib.sha256).hexdigest()


def shown_name(path: Path) -> str:
    """A recording's name as it may appear in a log, a message or an artifact.

    Recorders name files after the caller ("050-1234567.wav", a customer's
    name in Hebrew), and those names used to reach every log line, error and
    the ingestion record. Such a name is shown as the call id derived from it;
    a plain safe name ("C0001.wav") is shown as itself.
    """
    stem = path.stem
    if CALL_ID_RE.match(stem) and not _looks_like_an_identifier(stem):
        return path.name
    return sanitize_call_id(stem, warn=False) + path.suffix.lower()


def sanitize_call_id(raw: str, *, warn: bool = True) -> str:
    """Turn an arbitrary filename stem into a safe call_id.

    Two DIFFERENT names must never produce the same id: the id keys every
    stage's state, so the second recording was "processed" by reusing the
    first one's transcript and scores. Replacing the unsafe characters alone
    did exactly that to Hebrew file names - "שיחה.wav" and "הקלטה.wav" both
    became "call", "שיחה 1" and "הקלטה 1" both "1". Whenever the name had to
    change, a short hash of the ORIGINAL name is appended, so distinct names
    stay distinct and the same name always gives the same id.
    """
    original = (raw or "").strip()
    if original and _looks_like_an_identifier(original):
        # A number in the name is the customer's, not the call's: none of it
        # survives into the id, which appears in every path, log and report.
        return f"call-{_digest(original)[:12]}"
    cleaned = _UNSAFE_CALL_ID_CHARS.sub("_", original).strip("._-")
    if not cleaned:
        cleaned = "call"
    if not cleaned[0].isalnum():
        cleaned = "c" + cleaned
    if cleaned != original and original:
        cleaned = f"{cleaned[:55]}-{_digest(original)[:8]}"
    if is_windows_reserved(cleaned):
        cleaned = "c_" + cleaned
    cleaned = cleaned[:64]
    if warn:
        if cleaned != original:
            # The ORIGINAL name is not logged: a recording named after the
            # customer is exactly the case this path exists for.
            logger.warning("a recording's name is not filename-safe; using call_id %r",
                           cleaned)
        if _LONG_DIGIT_RUN.search(cleaned):
            logger.warning(
                "call_id %r contains a long digit run. If recordings are named "
                "after the customer, that identifier ends up in every artifact "
                "path, log line and report title.", cleaned,
            )
    return cleaned


class IngestionError(ValueError):
    """Raised when inputs fail validation; message contains a problem table."""


@dataclass
class MetadataProblem:
    row: int  # 1-based data row number (0 = header/file level)
    column: str
    problem: str


@dataclass
class MetadataValidation:
    rows: dict[str, dict[str, str]] = field(default_factory=dict)  # call_id -> row
    # A call recorded in several files: call_id -> its file names in segment
    # order. Only calls with more than one file appear here.
    segments: dict[str, list[str]] = field(default_factory=dict)
    problems: list[MetadataProblem] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.problems

    def problem_table(self) -> str:
        """The problems, safe to log.

        A problem quotes the offending value so the operator can find it - and
        the offending value is a call id or a recording's file name, which is
        exactly where a customer's number turns up when recordings are named
        after the caller. Every other error surface in the package already went
        through sanitize_error; this one printed the value verbatim to the log.
        The row and column still say where to look.
        """
        from callqa.redaction import sanitize_error

        lines = [f"{'row':>4}  {'column':<16} problem", f"{'-'*4}  {'-'*16} {'-'*40}"]
        for p in self.problems:
            lines.append(f"{p.row:>4}  {p.column:<16} {sanitize_error(p.problem, limit=300)}")
        return "\n".join(lines)


def _read_text_any_encoding(path: Path) -> str:
    """Read a CSV the bank exported from whatever tool it uses.

    Excel on a Hebrew Windows machine writes cp1255 or UTF-16, and the loader
    died on both with a UnicodeDecodeError before it could report anything.
    """
    raw = path.read_bytes()
    # UTF-16 is tried only on its byte-order mark: without one it decodes
    # almost any even-length byte string into plausible-looking nonsense, and
    # a cp1255 file would come back as unreadable CJK rather than Hebrew.
    if raw[:2] in (b"\xff\xfe", b"\xfe\xff"):
        return raw.decode("utf-16")
    for encoding in ("utf-8-sig", "cp1255"):
        try:
            return raw.decode(encoding)
        except (UnicodeDecodeError, UnicodeError):
            continue
    # Neither encoding reads the WHOLE file: an Excel round trip left the old
    # rows in UTF-8 and saved the new ones in cp1255. Decoding the file as one
    # (iso-8859-8 "succeeds" on anything) turned the UTF-8 rows into garbage,
    # banker names included - and a garbled banker name is not redacted. So
    # each line is decoded on its own.
    lines = []
    for line in raw.removeprefix(b"\xef\xbb\xbf").split(b"\n"):
        try:
            lines.append(line.decode("utf-8"))
        except UnicodeDecodeError:
            lines.append(line.decode("cp1255", errors="replace"))
    return "\n".join(lines)


# Invisible direction and formatting marks (Unicode category Cf) that arrive
# with text pasted from a Hebrew UI, Outlook or a web page. "CALL001.wav" with
# a right-to-left mark in front of it looks identical and is a different name.
def _visible(value: str) -> str:
    return "".join(ch for ch in value if unicodedata.category(ch) != "Cf").strip()


def _csv_reader(text: str) -> csv.DictReader:
    """A DictReader with the delimiter the file actually uses.

    Excel's only Unicode export on older Office is "Unicode Text": UTF-16 and
    TAB-separated; a European regional format writes ';'. Parsed as commas,
    the whole header was one column and the error said the columns were missing.
    """
    header = text.lstrip("\ufeff").split("\n", 1)[0]
    delimiter = ","
    if "," not in header:
        delimiter = "\t" if "\t" in header else (";" if ";" in header else ",")
    return csv.DictReader(io.StringIO(text, newline=""), delimiter=delimiter)


def load_metadata(metadata_csv: Path, calls_dir: Path | None = None) -> MetadataValidation:
    """Load and strictly validate metadata.csv.

    Checks: required columns present, call_id/banker_id/file_name non-empty,
    call_id unique, banker_channel in {L, R} when given, and (if calls_dir is
    provided) referenced audio files exist.
    """
    result = MetadataValidation()
    if not metadata_csv.exists():
        result.problems.append(MetadataProblem(0, "-", f"metadata file not found: {metadata_csv}"))
        return result

    try:
        text = _read_text_any_encoding(metadata_csv)
    except (OSError, UnicodeDecodeError) as exc:
        result.problems.append(MetadataProblem(
            0, "-", f"could not read {metadata_csv.name}: {exc}"))
        return result
    reader = _csv_reader(text)
    seen_ids: set[str] = set()
    first_id: dict[str, str] = {}
    seen_segments: dict[str, dict[int, str]] = {}
    fieldnames = [_visible(c) for c in (reader.fieldnames or [])]
    missing = [c for c in REQUIRED_COLUMNS if c not in fieldnames]
    if missing:
        result.problems.append(
            MetadataProblem(0, ",".join(missing), "required column(s) missing")
        )
        return result
    for i, raw_row in enumerate(reader, start=1):
        extra = raw_row.pop(None, None)
        if extra:
            # csv.DictReader puts surplus fields under the None key, and
            # the value is a LIST - stripping it raised AttributeError and
            # killed the whole validation run.
            result.problems.append(MetadataProblem(
                i, "-", f"row has {len(extra)} more fields than the header"))
            continue
        row = {_visible(k or ""): _visible(v) if isinstance(v, str) else ""
               for k, v in raw_row.items()}
        if not any(row.values()):
            # Excel writes a cleared row as ",,,,,,". It is not a call,
            # and reporting it as three empty values made the whole file
            # invalid - after which watch/process ran EVERY call on
            # defaults (banker unknown, channel L).
            continue
        for col in REQUIRED_COLUMNS:
            if not row.get(col):
                result.problems.append(MetadataProblem(i, col, "empty value"))
        call_id = row.get("call_id", "")
        segment = row.get("segment", "")
        # Case-insensitively: on Windows "A100" and "a100" are the same
        # file, so two calls would overwrite each other's artifacts.
        if call_id and call_id.casefold() in seen_ids:
            first = first_id[call_id.casefold()]
            # A second row of one call is another recorded part of it only
            # when both rows say which part they are; anything else is the
            # mistake this check always caught.
            parts = seen_segments.get(first)
            if parts is not None and segment.isdigit() and int(segment) not in parts:
                file_name = row.get("file_name", "")
                if not file_name or Path(file_name).name != file_name:
                    result.problems.append(MetadataProblem(
                        i, "file_name", "must be a file name inside calls/, not a path"))
                    continue
                if calls_dir is not None and not (calls_dir / file_name).exists():
                    result.problems.append(
                        MetadataProblem(i, "file_name", f"audio file not found: {file_name}"))
                parts[int(segment)] = file_name
                continue
            hint = ("" if segment else
                    " (a call recorded in several files needs a 'segment' column: 1, 2, ...)")
            result.problems.append(MetadataProblem(
                i, "call_id", f"duplicate call_id '{call_id}'{hint}"))
            continue
        if call_id and is_windows_reserved(call_id):
            result.problems.append(MetadataProblem(
                i, "call_id", f"'{call_id}' is a name Windows reserves for a device; "
                "choose another call_id"))
            continue
        if call_id and not CALL_ID_RE.match(call_id):
            result.problems.append(MetadataProblem(
                i, "call_id",
                f"'{call_id}' is not a safe identifier: use letters, digits, "
                f". _ - only (max 64 characters)",
            ))
            continue
        channel = row.get("banker_channel", "")
        if channel and channel not in ("L", "R"):
            result.problems.append(
                MetadataProblem(i, "banker_channel", f"must be L or R, got '{channel}'")
            )
        file_name = row.get("file_name", "")
        if file_name and (Path(file_name).is_absolute() or Path(file_name).name != file_name):
            # An absolute path or one containing a separator reads a file
            # outside the recordings directory - and the pipeline then
            # writes its RAW transcript into the output tree.
            result.problems.append(MetadataProblem(
                i, "file_name",
                f"must be a file name inside calls/, not a path: {file_name!r}"))
            continue
        if calls_dir is not None and file_name:
            if not (calls_dir / file_name).exists():
                result.problems.append(
                    MetadataProblem(i, "file_name", f"audio file not found: {file_name}")
                )
        if call_id:
            result.rows[call_id] = row
            seen_ids.add(call_id.casefold())
            first_id[call_id.casefold()] = call_id
            if segment.isdigit():
                seen_segments[call_id] = {int(segment): row.get("file_name", "")}
    for call_id, parts in seen_segments.items():
        if len(parts) > 1:
            result.segments[call_id] = [parts[k] for k in sorted(parts)]
    return result


def call_input_from_metadata(row: dict[str, str], calls_dir: Path) -> CallInput:
    return CallInput(
        call_id=row["call_id"],
        audio_path=calls_dir / row["file_name"],
        banker_id=row["banker_id"],
        call_date=row.get("call_date") or None,
        call_type=row.get("call_type") or None,
        banker_channel=row.get("banker_channel") or None,  # type: ignore[arg-type]
        banker_name=row.get("banker_name") or None,
    )


def _probe_with_ffprobe(audio_path: Path) -> dict | None:
    try:
        proc = run_text(
            [
                find_executable("ffprobe") or "ffprobe", "-v", "error", "-select_streams", "a:0",
                "-show_entries", "stream=channels,sample_rate,codec_name,duration",
                "-show_entries", "format=duration",
                "-of", "json", str(audio_path),
            ],
            timeout=60,
        )
        if proc.returncode != 0:
            return None
        return json.loads(proc.stdout)
    except (OSError, subprocess.TimeoutExpired, json.JSONDecodeError):
        return None


def _have_pyav() -> bool:
    import importlib.util

    return importlib.util.find_spec("av") is not None


def _probe_with_pyav(audio_path: Path) -> dict | None:
    """The same answer ffprobe gives, from PyAV (installed with the model
    engines, FFmpeg included): mp3/m4a, and telephony WAV (G.711 mu-law/A-law),
    which the standard library's wave module refuses."""
    if not _have_pyav():
        return None
    try:
        import av

        with av.open(str(audio_path)) as container:
            stream = container.streams.audio[0]
            ctx = stream.codec_context
            channels = getattr(ctx, "channels", None) or len(ctx.layout.channels)
            duration = (float(stream.duration * stream.time_base) if stream.duration
                        else (container.duration or 0) / 1_000_000)
            return {"streams": [{"channels": channels, "sample_rate": ctx.sample_rate,
                                 "codec_name": ctx.name, "duration": duration}],
                    "format": {"duration": duration}}
    except Exception:  # noqa: BLE001 - "could not probe" is reported by the caller
        return None


def _probe_with_wave(audio_path: Path) -> dict | None:
    """Dependency-free fallback for .wav files (used when ffprobe is absent)."""
    if audio_path.suffix.lower() != ".wav":
        return None
    try:
        with wave.open(str(audio_path), "rb") as wf:
            rate = wf.getframerate()
            frames = wf.getnframes()
            return {
                "streams": [
                    {
                        "channels": wf.getnchannels(),
                        "sample_rate": str(rate),
                        "codec_name": "pcm_s16le",
                        "duration": str(frames / rate if rate else 0.0),
                    }
                ],
                "format": {"duration": str(frames / rate if rate else 0.0)},
            }
    except (wave.Error, OSError, EOFError):
        return None


def _as_float(value: object) -> float:
    """ffprobe reports an unknown duration as the string 'N/A'."""
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 0.0


def probe_audio(call: CallInput) -> CallMeta:
    """Probe the audio file (ffprobe, wave-module fallback) -> CallMeta."""
    if not call.audio_path.exists():
        raise IngestionError(f"audio file not found: {call.audio_path}")
    suffix = call.audio_path.suffix.lower()
    if suffix not in AUDIO_EXTENSIONS:
        if suffix not in FFMPEG_AUDIO_EXTENSIONS:
            raise IngestionError(
                f"unsupported audio extension '{suffix}' (expected .wav/.mp3)"
            )
        if not find_executable("ffmpeg") and not _have_pyav():
            raise IngestionError(
                f"'{suffix}' needs ffmpeg (unzip it into tools/) or the model engines "
                "(requirements-server.txt), neither of which is here"
            )
    info = (_probe_with_ffprobe(call.audio_path) or _probe_with_wave(call.audio_path)
            or _probe_with_pyav(call.audio_path))
    if info is None or not info.get("streams"):
        raise IngestionError(f"could not probe audio file: {shown_name(call.audio_path)}")
    stream = info["streams"][0]
    duration = _as_float(stream.get("duration")) or _as_float(
        info.get("format", {}).get("duration")
    )
    if duration <= 0:
        raise IngestionError(f"audio has zero duration: {shown_name(call.audio_path)}")
    meta = CallMeta(
        call_id=call.call_id,
        banker_id=call.banker_id,
        # Kept for the record, so kept safe: this file outlives the raw
        # transcript that retention deletes.
        file_name=shown_name(call.audio_path),
        duration_sec=round(duration, 3),
        channels=int(stream.get("channels") or 1),
        sample_rate=int(stream.get("sample_rate") or 0),
        codec=stream.get("codec_name"),
        call_date=call.call_date,
        call_type=call.call_type,
        banker_channel=call.banker_channel,
    )
    logger.info("ingestion done: call_id=%s channels=%d duration=%.1fs",
                meta.call_id, meta.channels, meta.duration_sec)
    return meta
