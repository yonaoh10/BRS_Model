"""Shared pieces of every importer: ids, audio sources, dataset ids."""

from __future__ import annotations

import hashlib
import re
import zipfile
from dataclasses import dataclass
from pathlib import Path

from callqa.ingestion import CALL_ID_RE, _digest, _looks_like_an_identifier, sanitize_call_id

AUDIO_SUFFIXES = (".nmf", ".wav", ".mp3")
HEX_CALL = re.compile(r"^[0-9a-fA-F]{12,32}$")
UM_ID = re.compile(r"^UM-?\d+$", re.I)
COR_ID = re.compile(r"^COR-?\d+$", re.I)
# NICE export names: 1_<call key>_<segment key>[.nmf]
SEGMENT_FILE = re.compile(r"^(\d+)_(\d+)_(\d+)(?:\.[A-Za-z0-9]+)?$")


def safe_ref(prefix: str, raw: str) -> str:
    """A filename-safe, stable reference for a source id. A source id that
    could be read as a number about a person is replaced by a keyed digest."""
    text = str(raw).strip()
    candidate = f"{prefix}-{text}".lower()
    if CALL_ID_RE.match(candidate) and not _looks_like_an_identifier(text):
        return candidate
    return f"{prefix}-{_digest(prefix + ':' + text)[:12]}"


def call_id_for(call_key: str) -> str:
    """The pipeline id of a call: the same function that names any recording."""
    return sanitize_call_id(call_key.lower(), warn=False)


def segment_parts(file_name: str) -> tuple[str, int] | None:
    """('7646321893530733870', 764632190...) from a NICE file name."""
    m = SEGMENT_FILE.match(Path(file_name).name)
    if not m:
        return None
    return m.group(2), int(m.group(3))


def shown_file(file_name: str) -> str:
    """How a recording's name may be printed: never its digits."""
    return "file-" + _digest("file:" + Path(file_name).name.lower())[:10]


def text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value).strip()


@dataclass
class AudioSource:
    """A folder or a ZIP of recordings, indexed by lower-case name (with and
    without the extension). Nothing is extracted."""

    path: Path
    is_zip: bool
    members: dict[str, str]        # key -> member name / relative path

    @classmethod
    def open(cls, path: str | Path) -> AudioSource:
        p = Path(path)
        members: dict[str, str] = {}
        if p.is_file() and zipfile.is_zipfile(p):
            with zipfile.ZipFile(p) as zf:
                names = [i.filename for i in zf.infolist() if not i.is_dir()]
            is_zip = True
        elif p.is_dir():
            names = [str(f.relative_to(p)) for f in p.rglob("*") if f.is_file()]
            is_zip = False
        else:
            raise FileNotFoundError(f"audio source not found: {p}")
        for name in names:
            base = Path(name.replace("\\", "/")).name
            if not base.lower().endswith(AUDIO_SUFFIXES):
                continue
            members.setdefault(base.lower(), name)
            members.setdefault(Path(base).stem.lower(), name)
        return cls(path=p, is_zip=is_zip, members=members)

    def find(self, file_name: str) -> str | None:
        base = Path(file_name.replace("\\", "/")).name.lower()
        return self.members.get(base) or self.members.get(Path(base).stem.lower())

    def files(self) -> list[str]:
        return sorted(set(self.members.values()))

    def read(self, member: str) -> bytes:
        if self.is_zip:
            with zipfile.ZipFile(self.path) as zf:
                return zf.read(member)
        return (self.path / member).read_bytes()


def dataset_id(created: str, parts: list[str]) -> str:
    digest = hashlib.sha256("\n".join(parts).encode("utf-8")).hexdigest()[:8]
    return f"ds-{created}-{digest}"
