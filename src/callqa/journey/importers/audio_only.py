"""Tier T0: recordings and nothing else.

A folder or ZIP of NICE recordings, with no list of contacts, is a legal
batch. Each distinct call key in the file names (`1_<call key>_<segment>`)
becomes one recorded call, its parts in segment order, and one story of a
single contact - because no account ties calls together, every call stands
alone: there are no returns, no promises kept or broken, no direction and
no abandonment, and the report says so in its sources table.

The only clock a bare recording carries is the file's modification time
(the NICE header holds relative times only), so a call is dated by the
earliest part's modification time and the dataset's manifest records
`time_basis: file_mtime`. Files whose names are not NICE segment names are
counted and left out: nothing says which call they are.
"""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime
from pathlib import Path

from callqa.journey.importers.build import Built, RawInteraction, RawSegment, build_dataset
from callqa.journey.importers.common import AudioSource, segment_parts, shown_file
from callqa.journey.models import ImportReport, Manifest

SOURCE = "journey-audio-only"
TIME_BASIS = "file_mtime"


def import_audio_only(audio: str | Path) -> Built:
    report = ImportReport(source=SOURCE)
    source = AudioSource.open(audio)
    by_call: dict[str, list[tuple[int, str]]] = defaultdict(list)
    unnamed: list[str] = []
    for member in source.files():
        parts = segment_parts(Path(member.replace("\\", "/")).name)
        if parts is None:
            unnamed.append(shown_file(member))
            continue
        by_call[parts[0]].append((parts[1], member))
    if unnamed:
        report.add("audio_unnamed", "warning",
                   "recordings whose names are not NICE segment names (1_<call>_<part>); "
                   "nothing says which call they belong to, so they were left out",
                   count=len(unnamed), examples=unnamed[:5])
    interactions: list[RawInteraction] = []
    segments: list[RawSegment] = []
    undated = 0
    for n, (call_key, parts) in enumerate(sorted(by_call.items()), start=1):
        parts.sort()
        times = [t for t in (source.modified_at(m) for _seg, m in parts) if t is not None]
        at = min(times) if times else None
        if at is None:
            undated += 1
            at = datetime(1970, 1, 1)
        for seq, (_seg, member) in enumerate(parts, start=1):
            segments.append(RawSegment(call_key=call_key, file_name=Path(member).name, seq=seq))
        # one story per call: the call key stands in for the account, so the
        # private map holds no account number at all for this batch
        interactions.append(RawInteraction(row=n, branch="", account=f"call:{call_key}", at=at,
                                           channel="call", source_id=call_key, recorded=True))
    if undated:
        report.add("audio_undated", "warning",
                   "calls none of whose parts carries a modification time; dated 1970-01-01 "
                   "and so ordered first", count=undated)
    if not interactions:
        report.add("no_interactions", "error",
                   "no recordings with NICE segment names were found in the audio source")
    built = build_dataset(SOURCE, interactions, segments, [], report, audio=source)
    built.dataset.contract_version = 2
    built.dataset.manifest = Manifest(contract_version=2, source=SOURCE, tier="T0",
                                      angles=["vendor"], time_basis=TIME_BASIS)
    return built
