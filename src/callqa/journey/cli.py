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


def register(sub: argparse._SubParsersAction, add_common) -> None:
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

    p = jsub.add_parser("reveal", help="which account a story number is (logged)")
    p.add_argument("story_no", type=int)
    p.add_argument("--dataset", default=None, help="dataset id (default: the latest import)")
    add_common(p)
    p.set_defaults(func=cmd_journey_reveal)
