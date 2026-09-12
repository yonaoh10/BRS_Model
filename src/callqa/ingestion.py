"""Ingestion: metadata.csv validation + audio probing (stage 1)."""

from __future__ import annotations

import csv
import json
import logging
import shutil
import subprocess
import wave
from dataclasses import dataclass, field
from pathlib import Path

from callqa.models import CallInput, CallMeta

logger = logging.getLogger(__name__)

REQUIRED_COLUMNS = ["call_id", "banker_id", "file_name"]
OPTIONAL_COLUMNS = ["call_date", "call_type", "banker_channel", "banker_name"]
# .wav/.mp3 are what the bank's recorders produce. The rest are what a phone
# produces, which is what a staged test call arrives as; they are accepted only
# when ffmpeg is present, since the dependency-free fallback reads WAV only.
AUDIO_EXTENSIONS = {".wav", ".mp3"}
FFMPEG_AUDIO_EXTENSIONS = {".m4a", ".aac", ".mp4", ".ogg", ".opus", ".flac", ".wma", ".amr"}


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
    problems: list[MetadataProblem] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.problems

    def problem_table(self) -> str:
        lines = [f"{'row':>4}  {'column':<16} problem", f"{'-'*4}  {'-'*16} {'-'*40}"]
        for p in self.problems:
            lines.append(f"{p.row:>4}  {p.column:<16} {p.problem}")
        return "\n".join(lines)


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

    with metadata_csv.open(newline="", encoding="utf-8-sig") as fh:
        reader = csv.DictReader(fh)
        fieldnames = [c.strip() for c in (reader.fieldnames or [])]
        missing = [c for c in REQUIRED_COLUMNS if c not in fieldnames]
        if missing:
            result.problems.append(
                MetadataProblem(0, ",".join(missing), "required column(s) missing")
            )
            return result
        for i, raw_row in enumerate(reader, start=1):
            row = {(k or "").strip(): (v or "").strip() for k, v in raw_row.items()}
            for col in REQUIRED_COLUMNS:
                if not row.get(col):
                    result.problems.append(MetadataProblem(i, col, "empty value"))
            call_id = row.get("call_id", "")
            if call_id in result.rows:
                result.problems.append(MetadataProblem(i, "call_id", f"duplicate call_id '{call_id}'"))
                continue
            channel = row.get("banker_channel", "")
            if channel and channel not in ("L", "R"):
                result.problems.append(
                    MetadataProblem(i, "banker_channel", f"must be L or R, got '{channel}'")
                )
            if calls_dir is not None and row.get("file_name"):
                if not (calls_dir / row["file_name"]).exists():
                    result.problems.append(
                        MetadataProblem(i, "file_name", f"audio file not found: {row['file_name']}")
                    )
            if call_id:
                result.rows[call_id] = row
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
        proc = subprocess.run(
            [
                "ffprobe", "-v", "error", "-select_streams", "a:0",
                "-show_entries", "stream=channels,sample_rate,codec_name,duration",
                "-show_entries", "format=duration",
                "-of", "json", str(audio_path),
            ],
            capture_output=True, text=True, timeout=60,
        )
        if proc.returncode != 0:
            return None
        return json.loads(proc.stdout)
    except (OSError, subprocess.TimeoutExpired, json.JSONDecodeError):
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
        if not shutil.which("ffmpeg"):
            raise IngestionError(
                f"'{suffix}' needs ffmpeg, which is not installed; convert to .wav first"
            )
    info = _probe_with_ffprobe(call.audio_path) or _probe_with_wave(call.audio_path)
    if info is None or not info.get("streams"):
        raise IngestionError(f"could not probe audio file: {call.audio_path.name}")
    stream = info["streams"][0]
    duration = float(stream.get("duration") or info.get("format", {}).get("duration") or 0.0)
    if duration <= 0:
        raise IngestionError(f"audio has zero duration: {call.audio_path.name}")
    meta = CallMeta(
        call_id=call.call_id,
        banker_id=call.banker_id,
        file_name=call.audio_path.name,
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
