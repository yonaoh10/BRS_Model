"""`journey process`: a whole batch, from recordings to the report, resumable.

    1 transcribe   every recorded call of the batch: its parts assembled into
                   one recording (journey/assemble.py), then the call pipeline
                   up to redaction (or through the judge, with --score)
    2 content      the reading tasks over what is transcribed (journey/content.py)
    3 report       the journey report

Stories go in priority order (returns-first: the stories with the most
returns first, so a night that ends early has covered what matters most).
Everything done is kept - the pipeline's state database for calls, the answer
cache for the reading tasks - so the next run continues where this one
stopped. A deadline (--until 07:00, --max-hours 6) is checked between calls;
on the cpu profile transcription and the language model never run at the same
time, so they do not compete for the same cores.

One run per dataset at a time: a lock file in the dataset's folder.
"""

from __future__ import annotations

import logging
import os
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path

from callqa.config import Config
from callqa.journey.models import JourneyDataset
from callqa.journey.store import dataset_dir, load_dataset, resolve_dataset_id

logger = logging.getLogger(__name__)

TRANSCRIBE_UNTIL = "redaction"      # enough for the reading tasks


class ProcessLocked(RuntimeError):
    pass


@dataclass
class Deadline:
    at: datetime | None = None

    @classmethod
    def from_args(cls, until: str | None, max_hours: float | None,
                  now: datetime | None = None) -> Deadline:
        now = now or datetime.now()
        limits = []
        if until:
            hh, mm = (int(x) for x in until.split(":", 1))
            t = now.replace(hour=hh, minute=mm, second=0, microsecond=0)
            if t <= now:
                t += timedelta(days=1)
            limits.append(t)
        if max_hours:
            limits.append(now + timedelta(hours=max_hours))
        return cls(min(limits) if limits else None)

    def passed(self, now: datetime | None = None) -> bool:
        return self.at is not None and (now or datetime.now()) >= self.at

    def left(self, now: datetime | None = None) -> float | None:
        return None if self.at is None else (self.at - (now or datetime.now())).total_seconds()


@dataclass
class ProcessSummary:
    calls_total: int = 0
    calls_done_before: int = 0
    transcribed: int = 0
    failed: list[str] = field(default_factory=list)
    stopped_by_deadline: bool = False
    content_ran: bool = False
    content_failed: int = 0
    report: Path | None = None
    seconds: dict[str, float] = field(default_factory=dict)


# -- lock ---------------------------------------------------------------------------

class DatasetLock:
    def __init__(self, folder: Path) -> None:
        self.path = folder / "process.lock"

    def __enter__(self) -> DatasetLock:
        from callqa.portable import pid_alive

        self.path.parent.mkdir(parents=True, exist_ok=True)
        for _ in range(2):
            try:
                fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            except FileExistsError:
                try:
                    pid = int(self.path.read_text(encoding="utf-8").strip() or "0")
                except (OSError, ValueError):
                    pid = 0
                if pid and pid_alive(pid):
                    raise ProcessLocked(f"another `journey process` (pid {pid}) is running "
                                        "on this dataset") from None
                self.path.unlink(missing_ok=True)       # left by a run that died
                continue
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                fh.write(str(os.getpid()))
            return self
        raise ProcessLocked("could not take the dataset lock")

    def __exit__(self, *exc: object) -> None:
        self.path.unlink(missing_ok=True)


# -- which calls, in which order ------------------------------------------------------

def ordered_calls(dataset: JourneyDataset, priority: str = "returns-first",
                  limit_stories: int | None = None) -> list[tuple[str, list[str], str | None]]:
    """(call_id, part file names, call date) of every recorded call, stories by
    priority, a story's calls in time order."""
    by_story: dict[str, list] = {}
    for i in dataset.interactions:
        by_story.setdefault(i.story_key, []).append(i)
    stories = sorted(dataset.stories, key=lambda s: s.story_no)
    if priority == "returns-first":
        stories.sort(key=lambda s: (-(len(by_story.get(s.story_key, [])) - 1), s.story_no))
    if limit_stories:
        stories = stories[:limit_stories]
    out, seen = [], set()
    for s in stories:
        for i in sorted(by_story.get(s.story_key, []), key=lambda i: i.at):
            if i.channel != "call" or not i.recorded or not i.call_key:
                continue
            audio = dataset.calls.get(i.call_key)
            if audio is None or not audio.segments or audio.call_id in seen:
                continue
            seen.add(audio.call_id)
            out.append((audio.call_id, [seg.file_name for seg in sorted(audio.segments,
                                                                       key=lambda x: x.seq)],
                        i.at.date().isoformat()))
    return out


def is_transcribed(config: Config, call_id: str) -> bool:
    return (config.paths.output_dir / "redacted" / f"{call_id}.json").exists()


# -- the run --------------------------------------------------------------------------

def transcribe_calls(config: Config, dataset: JourneyDataset,
                     calls: list[tuple[str, list[str], str | None]], deadline: Deadline, *,
                     score: bool = False, engines=None,  # noqa: ANN001
                     progress: Callable[[str], None] | None = None) -> ProcessSummary:
    from callqa.cli import assemble_input
    from callqa.engines import build_engines
    from callqa.journey.assemble import AssembleError
    from callqa.models import CallInput
    from callqa.pipeline import process_call
    from callqa.state import StateDB

    summary = ProcessSummary(calls_total=len(calls))
    todo = []
    for call_id, files, day in calls:
        if is_transcribed(config, call_id) and not score:
            summary.calls_done_before += 1
        else:
            todo.append((call_id, files, day))
    if not todo:
        return summary
    if dataset.audio_source is None:
        summary.failed.extend(c for c, _f, _d in todo)
        logger.error("this dataset was imported without --audio; nothing to transcribe")
        return summary
    engines = engines or build_engines(config)
    state = StateDB(config.paths.state_db)
    source = Path(dataset.audio_source)
    for n, (call_id, files, day) in enumerate(todo, start=1):
        if deadline.passed():
            summary.stopped_by_deadline = True
            break
        t0 = time.monotonic()
        try:
            call = CallInput(call_id=call_id, audio_path=source, call_date=day)
            call = assemble_input(call, files, source, config)
        except (AssembleError, OSError, ValueError) as exc:
            summary.failed.append(call_id)
            logger.error("call_id=%s could not be assembled: %s", call_id, exc)
            continue
        result = process_call(call, engines, state,
                              stop_after=None if score else TRANSCRIBE_UNTIL)
        if result.status == "failed":
            summary.failed.append(call_id)
        else:
            summary.transcribed += 1
        summary.seconds[call_id] = time.monotonic() - t0
        if progress:
            progress(f"  {n}/{len(todo)} calls ({summary.calls_done_before} done before)")
    return summary


def process_dataset(config: Config, dataset_id: str | None = None, *, profile: str | None = None,
                    until: str | None = None, max_hours: float | None = None,
                    priority: str = "returns-first", limit_stories: int | None = None,
                    score: bool = False, mock: bool = False, skip_transcribe: bool = False,
                    report: bool = True, engines=None,  # noqa: ANN001
                    progress: Callable[[str], None] | None = None) -> ProcessSummary:
    from callqa.journey.content import run_content

    ds_id = resolve_dataset_id(config, dataset_id)
    deadline = Deadline.from_args(until, max_hours)
    with DatasetLock(dataset_dir(config, ds_id)):
        dataset = load_dataset(config, ds_id)
        calls = ordered_calls(dataset, priority, limit_stories)
        t0 = time.monotonic()
        if skip_transcribe:
            summary = ProcessSummary(calls_total=len(calls), calls_done_before=sum(
                1 for c, _f, _d in calls if is_transcribed(config, c)))
        else:
            summary = transcribe_calls(config, dataset, calls, deadline, score=score,
                                       engines=engines, progress=progress)
        summary.seconds["transcribe"] = time.monotonic() - t0
        if deadline.passed():
            summary.stopped_by_deadline = True
            return summary
        t1 = time.monotonic()
        _layer, stats = run_content(
            config, ds_id, mock=mock, profile=profile, stop=deadline.passed,
            progress=(lambda d, t: progress(f"  {d}/{t} stories read")) if progress else None)
        summary.content_ran = True
        summary.content_failed = stats.failed
        if stats.skipped:
            summary.stopped_by_deadline = True
        summary.seconds["content"] = time.monotonic() - t1
        if report:
            from callqa.reporting.journey import build_journey_report
            summary.report = build_journey_report(config, ds_id).html
        return summary


# -- estimate -------------------------------------------------------------------------

NOMINAL_REQUEST_SEC = {"cpu": 30.0, "gpu": 5.0}     # used only when nothing could be measured


@dataclass
class Estimate:
    calls_left: int
    audio_hours_left: float | None
    sampled_calls: int
    sec_per_audio_sec: float | None
    requests_left: int
    sec_per_request: float | None
    request_measured: bool
    concurrency: int
    profile: str

    @property
    def transcribe_hours(self) -> float | None:
        if self.calls_left == 0:
            return 0.0
        if self.audio_hours_left is None or self.sec_per_audio_sec is None:
            return None
        return self.audio_hours_left * self.sec_per_audio_sec

    @property
    def content_hours(self) -> float | None:
        if self.sec_per_request is None:
            return None
        return self.requests_left * self.sec_per_request / self.concurrency / 3600

    def lines(self) -> list[str]:
        def h(x: float | None) -> str:
            return "unknown" if x is None else f"{x:.1f} h"
        audio = "unknown" if self.audio_hours_left is None else f"{self.audio_hours_left:.1f} h"
        out = [f"profile: {self.profile} (concurrency {self.concurrency})",
               f"transcribe: {self.calls_left:,} calls left, {audio} of audio"
               + (f"; measured {self.sec_per_audio_sec:.2f} s per audio second on "
                  f"{self.sampled_calls} call(s)" if self.sec_per_audio_sec is not None else ""),
               f"  -> about {h(self.transcribe_hours)}",
               f"read: up to {self.requests_left:,} model requests (cached answers cost nothing)"
               + (f" at {self.sec_per_request:.1f} s each ("
                  + ("measured" if self.request_measured else "a rough guess: nothing to measure")
                  + ")" if self.sec_per_request is not None else ""),
               f"  -> about {h(self.content_hours)}"]
        t, c = self.transcribe_hours, self.content_hours
        if t is not None and c is not None:
            out.append(f"total: about {t + c:.1f} h")
        return out


def _audio_seconds(source: Path, files: list[str]) -> float | None:
    from callqa.journey.importers.common import AudioSource
    from callqa.journey.nmf import NMFError, inspect_bytes

    try:
        src = AudioSource.open(source)
    except (OSError, ValueError):
        return None
    total = 0.0
    for name in files:
        member = src.find(name)
        if member is None:
            return None
        data = src.read(member)
        if str(member).lower().endswith(".nmf") or data[:4] != b"RIFF":
            try:
                d = inspect_bytes(data).duration_sec
            except NMFError:
                return None
            if d is None:
                return None
            total += d
        else:
            import io
            import wave
            try:
                with wave.open(io.BytesIO(data)) as wf:
                    total += wf.getnframes() / float(wf.getframerate())
            except (wave.Error, EOFError):
                return None
    return total


def estimate(config: Config, dataset_id: str | None = None, *, sample_calls: int = 2,
             profile: str | None = None, mock: bool = False, engines=None  # noqa: ANN001
             ) -> Estimate:
    """Measure, then predict: transcribe `sample_calls` of the calls still to
    do (the work is kept), time the reading of one story, and scale."""
    from callqa.journey.content import run_content
    from callqa.journey.llm.client import resolve_profile
    from callqa.journey.timeline import build_timelines

    ds_id = resolve_dataset_id(config, dataset_id)
    dataset = load_dataset(config, ds_id)
    prof = resolve_profile(config, profile)
    left = [c for c in ordered_calls(dataset) if not is_transcribed(config, c[0])]
    audio_left = 0.0 if not left else None
    rate = None
    sampled = 0
    if left and dataset.audio_source:
        per = [_audio_seconds(Path(dataset.audio_source), files) for _c, files, _d in left]
        audio_left = None if any(p is None for p in per) else sum(per) / 3600
        sample = left[:sample_calls]
        summary = transcribe_calls(config, dataset, sample, Deadline(), engines=engines)
        sampled = summary.transcribed
        spent = sum(summary.seconds.get(c, 0.0) for c, _f, _d in sample)
        sample_audio = [p for (c, _f, _d), p in zip(left, per, strict=False)
                        if c in summary.seconds and p]
        if sampled and sample_audio:
            rate = spent / sum(sample_audio)
        if audio_left is not None and sample_audio:
            audio_left = max(0.0, audio_left - sum(sample_audio) / 3600)
        left = left[sampled:]
    # requests: a card per contact with text, plus two per story that has one
    timelines = build_timelines(dataset)
    with_text = sum(1 for tl in timelines for c in tl.contacts if c.has_content)
    stories_with_text = sum(1 for tl in timelines if any(c.has_content for c in tl.contacts))
    requests = with_text + 2 * stories_with_text
    t0 = time.monotonic()
    _layer, stats = run_content(config, ds_id, mock=mock, profile=profile, limit_stories=1)
    spent = time.monotonic() - t0
    measured = stats.engine_calls > 0
    per_request = spent / stats.engine_calls if measured else NOMINAL_REQUEST_SEC[prof.name]
    return Estimate(calls_left=len(left), audio_hours_left=audio_left, sampled_calls=sampled,
                    sec_per_audio_sec=rate, requests_left=requests, sec_per_request=per_request,
                    request_measured=measured, concurrency=prof.concurrency, profile=prof.name)
