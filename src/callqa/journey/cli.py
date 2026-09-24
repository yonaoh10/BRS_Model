"""`callqa journey ...` - the customer-journey commands."""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import UTC, datetime
from pathlib import Path

from callqa.models import EXIT_FAILED, EXIT_SUCCESS

logger = logging.getLogger("callqa.journey")

COUNT_LABELS = [
    ("stories", "stories (accounts)"),
    ("interactions", "contacts"),
    ("returns", "returns (contacts after the first)"),
    ("calls", "phone contacts"),
    ("recorded_calls", "  recorded"),
    ("unrecorded_calls", "  not recorded"),
    ("audio_files_mapped", "audio files mapped to calls"),
    ("multi_file_calls", "calls recorded in more than one file"),
    ("correspondences", "written correspondences"),
    ("messages", "messages"),
    ("atlas_sessions", "Atlas banker sessions"),
    ("atlas_ops", "Atlas operations"),
]


def _config(args: argparse.Namespace):
    from callqa.cli import _load_config
    return _load_config(args)


def print_report(report, out=None) -> None:
    out = out or sys.stdout
    counts = report.counts.model_dump()
    width = max(len(label) for _k, label in COUNT_LABELS)
    for key, label in COUNT_LABELS:
        value = counts.get(key, 0)
        if key.startswith("atlas") and not value:
            continue
        print(f"  {label.ljust(width)}  {value:>7,}", file=out)
    for column_file, cols in report.dropped_columns.items():
        print(f"  dropped columns in {column_file}: {', '.join(cols)}", file=out)
    for issue in report.issues:
        tag = {"error": "ERROR", "warning": "warning", "info": "note"}[issue.severity]
        extra = f" (e.g. {', '.join(issue.examples)})" if issue.examples else ""
        print(f"  [{tag}] {issue.message}: {issue.count:,}{extra}", file=out)


def cmd_journey_import(args: argparse.Namespace) -> int:
    from callqa.journey.importers.atlas import attach_atlas
    from callqa.journey.pseudo import write_private_map
    from callqa.journey.store import private_dir, save_dataset
    from callqa.journey.xlsx import XlsxError

    config = _config(args)
    redact = config.journey.redact_messages
    try:
        if args.xlsx:
            from callqa.journey.importers.workbook import import_workbook
            built = import_workbook(args.xlsx, audio=args.audio, redact_messages=redact)
        else:
            from callqa.journey.importers.contract import import_contract
            built = import_contract(args.contract, audio=args.audio, redact_messages=redact)
        if args.atlas:
            attach_atlas(built.dataset, args.atlas,
                         join_tolerance_sec=config.journey.atlas.join_tolerance_sec)
    except (XlsxError, FileNotFoundError, OSError) as exc:
        print(f"import failed: {exc}", file=sys.stderr)
        return EXIT_FAILED
    dataset = built.dataset
    print(f"source: {dataset.source}")
    print_report(dataset.report)
    if not dataset.report.ok:
        print("import has errors; nothing was written", file=sys.stderr)
        return EXIT_FAILED
    if args.dry_run:
        print("dry run: nothing was written")
        return EXIT_SUCCESS
    folder = save_dataset(config, dataset)
    write_private_map(private_dir(config, dataset.dataset_id), built.private_rows)
    print(f"dataset {dataset.dataset_id} written to {folder}")
    return EXIT_SUCCESS


def cmd_journey_reveal(args: argparse.Namespace) -> int:
    """The account behind a story number: printed to this console only, and
    every look is written to the dataset's private log."""
    from callqa.journey.pseudo import read_private_map
    from callqa.journey.store import private_dir, resolve_dataset_id

    config = _config(args)
    ds = resolve_dataset_id(config, args.dataset)
    folder = private_dir(config, ds)
    try:
        mapping = read_private_map(folder)
    except FileNotFoundError:
        print("this dataset has no private map", file=sys.stderr)
        return EXIT_FAILED
    if args.story_no not in mapping:
        print(f"no story {args.story_no} in {ds}", file=sys.stderr)
        return EXIT_FAILED
    _key, branch, account = mapping[args.story_no]
    import getpass
    stamp = datetime.now(UTC).isoformat(timespec="seconds")
    with (folder / "reveal.log").open("a", encoding="utf-8") as log:
        log.write(f"{stamp}\t{getpass.getuser()}\tstory {args.story_no}\n")
    print(f"story {args.story_no:03d}: branch {branch}, account {account}")
    return EXIT_SUCCESS


def cmd_journey_report(args: argparse.Namespace) -> int:
    """The journey report of a dataset: HTML, the stories and the returns as
    CSV, and the numbers as JSON, under <output_dir>/reports/."""
    from callqa.reporting.journey import build_journey_report

    config = _config(args)
    try:
        report = build_journey_report(config, args.dataset, with_text=not args.no_quotes,
                                      name=args.name, title=args.title)
    except (FileNotFoundError, ValueError) as exc:
        print(f"journey report failed: {exc}", file=sys.stderr)
        return EXIT_FAILED
    a = report.analysis
    print(f"{len(a.stories):,} stories, {sum(s.contacts for s in a.stories):,} contacts")
    print(f"report: {report.html}")
    return EXIT_SUCCESS


def _nmf_members(target: Path) -> list[tuple[str, bytes]]:
    """(printable name, bytes) of every .nmf in a file, folder or ZIP."""
    from callqa.journey.importers.common import AudioSource, shown_file

    if target.is_file() and target.suffix.lower() == ".nmf":
        return [(shown_file(target.name), target.read_bytes())]
    source = AudioSource.open(target)
    return [(shown_file(m), source.read(m)) for m in source.files()
            if m.lower().endswith(".nmf")]


def cmd_nmf_info(args: argparse.Namespace) -> int:
    """Structure of NICE recordings - never their audio. File names are shown
    as digests (recorder names are long digit runs)."""
    from callqa.journey.nmf import NMFError, decode_file, inspect_bytes

    members = _nmf_members(Path(args.target))
    if args.limit:
        members = members[: args.limit]
    if not members:
        print("no .nmf files found", file=sys.stderr)
        return EXIT_FAILED
    bad = 0
    for name, data in members:
        print(f"== {name}")
        try:
            info = inspect_bytes(data)
        except NMFError as exc:
            bad += 1
            print(f"  unreadable: {exc}")
            continue
        for line in info.lines():
            print("  " + line)
        if args.probe_decode:
            try:
                streams, _ = decode_file(data)
                for s in streams:
                    import numpy as np
                    rms = float(np.sqrt((s.pcm.astype(np.float64) ** 2).mean())) if len(s.pcm) else 0
                    print(f"  decoded stream {s.stream}: {len(s.pcm) / 8000:.1f} s, "
                          f"level {rms:.0f}, alignment {s.alignment}")
            except NMFError as exc:
                bad += 1
                print(f"  decode failed: {exc}")
    print(f"{len(members)} file(s), {bad} with problems")
    return EXIT_FAILED if bad else EXIT_SUCCESS


def cmd_nmf_convert(args: argparse.Namespace) -> int:
    """NICE recordings to WAV for a listening check, into a folder only the
    current user can read (the voices are customers')."""
    import wave

    import numpy as np

    from callqa.journey.nmf import NMFError, decode_file
    from callqa.portable import make_private_dir

    out = Path(args.out)
    make_private_dir(out)
    bad = 0
    for name, data in _nmf_members(Path(args.target))[: args.limit or None]:
        try:
            streams, _ = decode_file(data)
        except NMFError as exc:
            bad += 1
            print(f"{name}: {exc}", file=sys.stderr)
            continue
        channels = [s.pcm for s in streams[:2]]
        n = max(len(c) for c in channels)
        pcm = np.zeros((n, len(channels)), dtype=np.int16)
        for i, c in enumerate(channels):
            pcm[: len(c), i] = c
        with wave.open(str(out / f"{name}.wav"), "wb") as wf:
            wf.setnchannels(len(channels))
            wf.setsampwidth(2)
            wf.setframerate(8000)
            wf.writeframes(pcm.tobytes())
        print(f"{name}.wav: {n / 8000:.1f} s, {len(channels)} channel(s)")
    return EXIT_FAILED if bad else EXIT_SUCCESS


def register_top_level(sub: argparse._SubParsersAction, add_common) -> None:
    p = sub.add_parser("nmf-info", help="structure of NICE .nmf recordings (never their audio)")
    p.add_argument("target", help="an .nmf file, a folder, or a ZIP")
    p.add_argument("--probe-decode", action="store_true",
                   help="also decode, and print duration and level per stream")
    p.add_argument("--limit", type=int, default=0, help="only the first N files")
    add_common(p)
    p.set_defaults(func=cmd_nmf_info)

    p = sub.add_parser("nmf-convert", help="NICE .nmf recordings to WAV, for a listening check")
    p.add_argument("target", help="an .nmf file, a folder, or a ZIP")
    p.add_argument("--out", required=True, help="output folder (made private to this user)")
    p.add_argument("--limit", type=int, default=0, help="only the first N files")
    add_common(p)
    p.set_defaults(func=cmd_nmf_convert)


def register(sub: argparse._SubParsersAction, add_common) -> None:
    register_top_level(sub, add_common)
    journey = sub.add_parser("journey", help="customer journeys: repeat contacts across calls, "
                                             "messages and banker actions")
    jsub = journey.add_subparsers(dest="journey_command", required=True)

    p = jsub.add_parser("import", help="read a batch into a journey dataset")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--xlsx", type=Path, help="the bank's handoff workbook (three sheets)")
    src.add_argument("--contract", type=Path,
                     help="a folder in the journey contract (interactions.csv, ...)")
    p.add_argument("--audio", type=Path, default=None, help="a folder or ZIP of the recordings")
    p.add_argument("--atlas", type=Path, default=None,
                   help="the Atlas tables: a folder of CSV exports or one workbook")
    p.add_argument("--dry-run", action="store_true", help="count and check only; write nothing")
    add_common(p)
    p.set_defaults(func=cmd_journey_import)

    p = jsub.add_parser("report", help="the journey report (HTML, CSV, JSON)")
    p.add_argument("--dataset", default=None, help="dataset id (default: the latest import)")
    p.add_argument("--name", default=None, help="file name: journey-<name>.html")
    p.add_argument("--title", default=None, help="the report's title")
    p.add_argument("--no-quotes", action="store_true",
                   help="leave out every quote and reasoning (for wide distribution)")
    add_common(p)
    p.set_defaults(func=cmd_journey_report)

    p = jsub.add_parser("reveal", help="which account a story number is (logged)")
    p.add_argument("story_no", type=int)
    p.add_argument("--dataset", default=None, help="dataset id (default: the latest import)")
    add_common(p)
    p.set_defaults(func=cmd_journey_reveal)
