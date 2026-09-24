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
            from callqa.journey.sessions import rules_from_config
            from callqa.journey.vocab import load_atlas_codes
            attach_atlas(built.dataset, args.atlas,
                         join_tolerance_sec=config.journey.atlas.join_tolerance_sec,
                         rules=rules_from_config(config.journey.atlas),
                         codes=load_atlas_codes(config.journey.atlas.codes))
    except (XlsxError, FileNotFoundError, OSError, ValueError) as exc:
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


def cmd_journey_atlas_check(args: argparse.Namespace) -> int:
    """The banker sessions of a dataset, measured as the bank's Atlas project
    measures them (ATL_R01/R02): the completeness checks, the sessions by kind
    and unit class, and the page of numbers. With --expect, every figure is
    compared with the published one, and a difference fails the command."""
    from callqa.journey.session_analysis import compare_expected, rows_by_category
    from callqa.journey.sessions import SESSION_KIND_HE, UNIT_CLASS_HE
    from callqa.journey.store import resolve_dataset_id
    from callqa.reporting.journey.render import _analyse, session_analysis_of
    from callqa.resources import load_yaml

    config = _config(args)
    try:
        dataset, _content, tax, units, analysis, facts = _analyse(
            config, resolve_dataset_id(config, args.dataset))
    except (FileNotFoundError, ValueError) as exc:
        print(f"no dataset: {exc}", file=sys.stderr)
        return EXIT_FAILED
    if not dataset.atlas_sessions:
        print("this dataset has no Atlas sessions (import with --atlas)", file=sys.stderr)
        return EXIT_FAILED
    a = session_analysis_of(config, dataset, analysis, facts, tax, units)
    print(f"dataset {dataset.dataset_id}")
    for c in a.checks:
        print(f"  [{'ok' if c['ok'] else '!!'}] {c['check']}: {c['value']}")
    print("sessions of covered stories by kind: " + " / ".join(
        f"{SESSION_KIND_HE[k]} {v:,}" for k, v in a.kinds.items()))
    print("sessions of covered stories by unit: " + " / ".join(
        f"{UNIT_CLASS_HE[k]} {v:,}" for k, v in a.classes.items()))
    print("log rows by code category: " + " / ".join(
        f"{SESSION_KIND_HE[k]} {v:,}" for k, v in rows_by_category(dataset).items()))
    for h in a.head:
        print(f"  {h.item:>2}. {h.finding_he}: {h.value_txt}")
    if not args.expect:
        return EXIT_SUCCESS
    try:
        expected = load_yaml(args.expect) or {}
    except (OSError, ValueError) as exc:
        print(f"cannot read {args.expect}: {exc}", file=sys.stderr)
        return EXIT_FAILED
    diffs = compare_expected(a, dataset, expected)
    if diffs:
        print(f"{len(diffs)} figure(s) differ from {args.expect.name}:")
        for d in diffs:
            print(f"  - {d}")
        return EXIT_FAILED
    print(f"every figure matches {args.expect.name}")
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


def cmd_journey_content(args: argparse.Namespace) -> int:
    """Read the batch's recorded calls and correspondences: cards, why each
    return happened, and each story's headline and status (content.json)."""
    from callqa.journey.content import run_content

    config = _config(args)

    def progress(done: int, total: int) -> None:
        if done == total or done % 10 == 0:
            print(f"  {done}/{total} stories", flush=True)

    from callqa.journey.engine import DatasetLock, ProcessLocked
    from callqa.journey.store import dataset_dir, resolve_dataset_id

    try:
        ds_id = resolve_dataset_id(config, args.dataset)
        with DatasetLock(dataset_dir(config, ds_id)):
            layer, stats = run_content(config, ds_id, mock=args.mock, profile=args.profile,
                                       limit_stories=args.limit_stories, progress=progress)
    except (FileNotFoundError, ProcessLocked) as exc:
        print(f"journey content failed: {exc}", file=sys.stderr)
        return EXIT_FAILED
    except Exception as exc:  # noqa: BLE001 - e.g. the model server is not reachable
        from callqa.redaction import sanitize_error
        print(f"journey content failed: {sanitize_error(exc)}", file=sys.stderr)
        return EXIT_FAILED
    print(f"engine {layer.engine} ({layer.model}): {stats.stories:,} stories, "
          f"{stats.cards:,} cards from {stats.contacts_with_text:,} contacts with text")
    print(f"  not transcribed yet: {stats.no_text:,} · failed: {stats.failed:,} · "
          f"retries: {stats.retries:,} · quotes dropped: {stats.dropped_quotes:,} · "
          f"from cache: {stats.cache_hits:,}")
    if stats.not_saved:
        print("  not saved: content.json holds a complete reading by another engine or model; "
              "run without --limit-stories to replace it")
    return EXIT_SUCCESS if not stats.failed else EXIT_FAILED


def cmd_journey_process(args: argparse.Namespace) -> int:
    """The whole batch: transcribe what is not transcribed, read, report.
    Resumable, and stoppable at a deadline."""
    from callqa.journey.engine import ProcessLocked, process_dataset

    config = _config(args)
    try:
        s = process_dataset(config, args.dataset, profile=args.profile, until=args.until,
                            max_hours=args.max_hours, priority=args.priority,
                            limit_stories=args.limit_stories, score=args.score,
                            mock=args.mock, skip_transcribe=args.skip_transcribe,
                            progress=lambda line: print(line, flush=True))
    except (ProcessLocked, FileNotFoundError, ValueError) as exc:
        print(f"journey process: {exc}", file=sys.stderr)
        return EXIT_FAILED
    print(f"calls: {s.calls_total:,} recorded; {s.calls_done_before:,} were done, "
          f"{s.transcribed:,} transcribed now, {len(s.failed):,} failed")
    if s.stopped_by_deadline:
        print("stopped at the deadline; run the same command again to continue")
    if s.report:
        print(f"report: {s.report}")
    return EXIT_FAILED if (s.failed or s.content_failed) else EXIT_SUCCESS


def cmd_journey_estimate(args: argparse.Namespace) -> int:
    """How long `journey process` will take here, measured on this machine."""
    from callqa.journey.engine import ProcessLocked, estimate

    config = _config(args)
    try:
        est = estimate(config, args.dataset, sample_calls=args.sample_calls,
                       profile=args.profile, mock=args.mock)
    except (FileNotFoundError, ProcessLocked) as exc:
        print(f"journey estimate: {exc}", file=sys.stderr)
        return EXIT_FAILED
    for line in est.lines():
        print(line)
    return EXIT_SUCCESS


def cmd_journey_label_sample(args: argparse.Namespace) -> int:
    """A blind labelling form over a stratified sample of returns."""
    from callqa.journey.content import contact_view
    from callqa.journey.evaluate import sample_returns, write_label_form
    from callqa.journey.store import dataset_dir, load_content, load_dataset, resolve_dataset_id
    from callqa.journey.timeline import build_timelines
    from callqa.journey.vocab import load_taxonomy

    config = _config(args)
    try:
        ds_id = resolve_dataset_id(config, args.dataset)
    except FileNotFoundError as exc:
        print(f"journey label-sample: {exc}", file=sys.stderr)
        return EXIT_FAILED
    dataset = load_dataset(config, ds_id)
    content = load_content(config, ds_id)
    if content is None:
        print("run `callqa journey content` first: the sample is drawn by the model's "
              "categories", file=sys.stderr)
        return EXIT_FAILED
    sample = sample_returns(dataset, content, args.n, args.seed)
    wanted = {iid for _s, _c, iid in sample}
    views: dict[str, list[dict]] = {}
    for tl in build_timelines(dataset):
        for n, c in enumerate(tl.contacts):
            nxt = tl.contacts[n + 1].interaction.interaction_id if n + 1 < len(tl.contacts) else None
            if c.interaction.interaction_id in wanted or nxt in wanted:
                v = contact_view(config.paths.output_dir, c, config.journey.uncertain_word_prob)
                if v is not None:
                    views[c.interaction.interaction_id] = [
                        {"no": ln.no, "who": ln.who, "text": ln.text, "uncertain": ln.uncertain}
                        for ln in v.lines]
    out = Path(args.out) if args.out else dataset_dir(config, ds_id) / "labels" / "label_form.html"
    write_label_form(out, dataset, content, load_taxonomy(config.journey.taxonomy), sample, views)
    print(f"{len(sample)} returns to label: {out}")
    return EXIT_SUCCESS


def cmd_journey_eval(args: argparse.Namespace) -> int:
    """Human labels against the content layer; optionally a regression gate."""
    import json as _json

    from callqa.journey.evaluate import compare_baseline, evaluate, read_labels, save_eval
    from callqa.journey.store import dataset_dir, load_content, load_dataset, resolve_dataset_id
    from callqa.journey.vocab import load_taxonomy

    config = _config(args)
    tax = load_taxonomy(config.journey.taxonomy)
    try:
        ds_id = resolve_dataset_id(config, args.dataset)
        labels = read_labels(Path(args.labels), tax)
    except (FileNotFoundError, ValueError) as exc:
        print(f"journey eval: {exc}", file=sys.stderr)
        return EXIT_FAILED
    content = load_content(config, ds_id)
    if content is None:
        print("run `callqa journey content` first", file=sys.stderr)
        return EXIT_FAILED
    result = evaluate(load_dataset(config, ds_id), content, labels, tax)
    for line in result.lines():
        print(line)
    save_eval(dataset_dir(config, ds_id) / "eval.json", result)
    if args.write_baseline:
        Path(args.write_baseline).write_text(_json.dumps(result.to_json(), ensure_ascii=False,
                                                         indent=2), encoding="utf-8")
        print(f"baseline written: {args.write_baseline}")
    if args.baseline:
        problems = compare_baseline(result, _json.loads(Path(args.baseline).read_text(
            encoding="utf-8")))
        for p in problems:
            print(f"REGRESSION: {p}", file=sys.stderr)
        if problems:
            return EXIT_FAILED
        print("no regression against the baseline")
    return EXIT_SUCCESS


def cmd_journey_report(args: argparse.Namespace) -> int:
    """The journey report of a dataset: HTML, the stories and the returns as
    CSV, and the numbers as JSON, under <output_dir>/reports/."""
    from callqa.reporting.journey import build_contact_report, build_journey_report

    config = _config(args)
    try:
        if args.contact:
            story_no, _, n = args.contact.partition(":")
            if not (story_no.strip().isdigit() and n.strip().isdigit()):
                raise ValueError("--contact takes STORY:N, e.g. 7:3")
            html = build_contact_report(config, args.dataset, int(story_no), int(n),
                                        with_text=not args.no_quotes)
            print(f"report: {html}")
        else:
            report = build_journey_report(config, args.dataset, with_text=not args.no_quotes,
                                          name=args.name, title=args.title, level=args.level)
            html = report.html
            a = report.analysis
            print(f"{len(a.stories):,} stories, {sum(s.contacts for s in a.stories):,} contacts")
            print(f"report: {html}")
    except (FileNotFoundError, ValueError) as exc:
        print(f"journey report failed: {exc}", file=sys.stderr)
        return EXIT_FAILED
    if args.pdf:
        from callqa.reporting.pdf import PDFError, print_pdf
        try:
            pdf = print_pdf(html, html.with_suffix(".pdf"))
        except PDFError as exc:
            print(f"PDF: {exc}", file=sys.stderr)
            return EXIT_FAILED
        print(f"PDF: {pdf}")
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

    p = jsub.add_parser("content", help="read the calls and messages: cards, reasons, stories")
    p.add_argument("--dataset", default=None, help="dataset id (default: the latest import)")
    p.add_argument("--profile", choices=["cpu", "gpu"], default=None,
                   help="override journey.llm.profile for this run")
    p.add_argument("--limit-stories", type=int, default=None,
                   help="only the first N stories (a trial run)")
    add_common(p)
    p.set_defaults(func=cmd_journey_content)

    p = jsub.add_parser("process", help="transcribe, read and report a whole batch (resumable)")
    p.add_argument("--dataset", default=None, help="dataset id (default: the latest import)")
    p.add_argument("--profile", choices=["cpu", "gpu"], default=None)
    p.add_argument("--until", default=None, metavar="HH:MM",
                   help="stop cleanly at this time of day (e.g. 07:00)")
    p.add_argument("--max-hours", type=float, default=None, help="stop cleanly after N hours")
    p.add_argument("--priority", choices=["returns-first", "order"], default="returns-first",
                   help="which stories first (default: the ones with the most returns)")
    p.add_argument("--limit-stories", type=int, default=None)
    p.add_argument("--score", action="store_true",
                   help="also score each call on the rubric (for quality inside journeys)")
    p.add_argument("--skip-transcribe", action="store_true",
                   help="only read and report what is already transcribed")
    add_common(p)
    p.set_defaults(func=cmd_journey_process)

    p = jsub.add_parser("estimate", help="measure, then predict how long `process` will take")
    p.add_argument("--dataset", default=None)
    p.add_argument("--profile", choices=["cpu", "gpu"], default=None)
    p.add_argument("--sample-calls", type=int, default=2,
                   help="calls to transcribe as the sample (the work is kept)")
    add_common(p)
    p.set_defaults(func=cmd_journey_estimate)

    p = jsub.add_parser("report", help="the journey report (HTML, CSV, JSON)")
    p.add_argument("--dataset", default=None, help="dataset id (default: the latest import)")
    p.add_argument("--name", default=None, help="file name: journey-<name>.html")
    p.add_argument("--title", default=None, help="the report's title")
    p.add_argument("--no-quotes", action="store_true",
                   help="leave out every quote and reasoning (for wide distribution)")
    p.add_argument("--pdf", action="store_true",
                   help="also print a PDF next to it, with the Edge or Chrome on this machine")
    p.add_argument("--level", choices=("all", "call", "session"), default="all",
                   help="which analysis levels the report carries beside the stories: "
                        "single contacts (call), banker sessions (session), or both (default)")
    p.add_argument("--contact", default=None, metavar="STORY:N",
                   help="a page for one contact only - e.g. 7:3 = story 7, its third contact")
    add_common(p)
    p.set_defaults(func=cmd_journey_report)

    p = jsub.add_parser("atlas-check",
                        help="the banker sessions measured as the bank's Atlas project does")
    p.add_argument("--dataset", default=None, help="dataset id (default: the latest import)")
    p.add_argument("--expect", type=Path, default=None,
                   help="published figures to compare with (eval/atlas_r02_expected.yaml)")
    add_common(p)
    p.set_defaults(func=cmd_journey_atlas_check)

    p = jsub.add_parser("label-sample", help="a blind labelling form over a sample of returns")
    p.add_argument("--dataset", default=None)
    p.add_argument("--n", type=int, default=60, help="returns in the sample (default 60)")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--out", default=None, help="where to write the form (default: the dataset)")
    add_common(p)
    p.set_defaults(func=cmd_journey_label_sample)

    p = jsub.add_parser("eval", help="the content layer against human labels")
    p.add_argument("--labels", required=True, help="labels.csv (story_no,contact_no,category,...)")
    p.add_argument("--dataset", default=None)
    p.add_argument("--baseline", default=None, help="fail on a drop against this result")
    p.add_argument("--write-baseline", default=None, help="save this result as a baseline")
    add_common(p)
    p.set_defaults(func=cmd_journey_eval)

    p = jsub.add_parser("reveal", help="which account a story number is (logged)")
    p.add_argument("story_no", type=int)
    p.add_argument("--dataset", default=None, help="dataset id (default: the latest import)")
    add_common(p)
    p.set_defaults(func=cmd_journey_reveal)
