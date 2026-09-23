"""CLI entrypoints: process / watch / run / report / calibrate / validate-inputs.

The drivers here are thin: all per-call logic lives in pipeline.process_call.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, wait
from pathlib import Path

from callqa.config import Config, load_config
from callqa.dotenv import load_dotenv
from callqa.ingestion import shown_name
from callqa.models import (
    EXIT_FAILED,
    EXIT_NEEDS_HUMAN_REVIEW,
    EXIT_SUCCESS,
    CallInput,
    CallResult,
)
from callqa.portable import (
    IS_WINDOWS,
    configure_stdio,
    disable_console_quick_edit,
    held_open_for_writing,
    make_private_root,
    move,
    project_root,
)
from callqa.resources import find_config

logger = logging.getLogger("callqa")

# robocopy marks a file it is still copying with a 1980-01-01 timestamp.
_ROBOCOPY_IN_PROGRESS = 315619200.0     # 1980-01-02T00:00:00Z

# Everything probe_audio can read. Watching only .wav/.mp3 left an .m4a sitting
# in the drop directory forever: not processed, not quarantined, not logged.
WATCHED_EXTENSIONS = frozenset({".wav", ".mp3", ".m4a", ".aac", ".mp4", ".ogg",
                                ".opus", ".flac", ".wma", ".amr"})


def _started_in_system_folder() -> bool:
    """Task Scheduler starts a task in C:\\Windows\\System32 unless told
    otherwise, and the data paths are relative to the current folder."""
    system_root = os.environ.get("SystemRoot")
    if not system_root:
        return False
    try:
        return Path.cwd().resolve().is_relative_to(Path(system_root).resolve())
    except OSError:
        return False


def _setup_logging(verbose: bool = False, log_file: str | None = None) -> None:
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stderr)]
    if log_file:
        # UTF-8, whatever redirection the caller would have used: a Scheduled
        # Task's output has nowhere to go, and PowerShell's `*>` re-encodes
        # Hebrew through the console code page into mojibake.
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=handlers,
    )


def _add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", default=None, help="path to config.yaml")
    parser.add_argument("--mock", action="store_true", help="run with deterministic mock engines")
    parser.add_argument("--force", action="store_true", help="rerun completed stages")
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("--log-file", default=None,
                        help="also write the log to this file (UTF-8); for scheduled runs")


def _load_config(args: argparse.Namespace) -> Config:
    overrides: dict = {"run": {}}
    if args.mock:
        overrides["run"]["mock"] = True
    if getattr(args, "force", False):
        overrides["run"]["force"] = True
    if getattr(args, "max_workers", None):
        overrides["run"]["max_workers"] = args.max_workers
    config_path = args.config
    if config_path is None:
        try:
            config_path = str(find_config("config.yaml"))
        except FileNotFoundError:
            logger.warning(
                "no config.yaml found; running on built-in defaults. Pass "
                "--config, or run from the project directory."
            )
    return load_config(config_path, overrides)


def _build_engines(config: Config):  # noqa: ANN202
    from callqa.engines import build_engines

    _secure_data_dirs(config)
    return build_engines(config)


def _secure_data_dirs(config: Config) -> None:
    """Make the folders that will hold raw audio and transcripts owner-only
    BEFORE anything is written into them (Windows; POSIX files are 0600).

    The output folder is the pipeline's own. The state database's folder and
    the input folder are included only when they are inside the project: an
    input folder elsewhere may be a drop share another system writes into, and
    its permissions are not this program's to change.
    """
    make_private_root(config.paths.output_dir)
    root = project_root()
    for folder in (config.paths.state_db.parent, config.paths.input_dir):
        try:
            inside = folder.resolve().is_relative_to(root)
        except OSError:
            inside = False
        if inside:
            make_private_root(folder)


def _metadata_lookup(config: Config) -> dict[str, dict[str, str]]:
    from callqa.ingestion import load_metadata

    metadata_csv = config.paths.input_dir / "metadata.csv"
    if not metadata_csv.exists():
        return {}
    validation = load_metadata(metadata_csv)
    if not validation.ok:
        logger.error("metadata.csv has problems; rows are NOT being used:\n%s",
                     validation.problem_table())
        return {}
    return validation.rows


def _distinct_from_case_twins(call_id: str, stem: str, config: Config) -> str:
    """An id derived from a file name must not differ from an earlier call's
    only in letter case: on Windows (and macOS) "CALL7" and "call7" name the
    same artifact files, so a second recording would overwrite the first
    one's transcript, scores and report. Such an id gets a digest suffix."""
    from callqa.state import StateDB

    if not config.paths.state_db.exists():
        return call_id
    twins = StateDB(config.paths.state_db).ids_differing_only_in_case(call_id)
    if not twins:
        return call_id
    from callqa.ingestion import _digest

    return f"{call_id[:50]}-{_digest('case:' + stem)[:8]}"


def _call_input(
    audio_path: Path, config: Config, args: argparse.Namespace | None = None
) -> CallInput:
    """Build a CallInput for one file: CLI args > metadata.csv > defaults.

    The row is found by FILE NAME, then by call_id. Matching on the filename
    stem alone meant that any recorder whose naming differs from the bank's
    call ids silently lost every row: unknown banker, default channel (which
    can invert the two speakers) and no banker name to redact - while
    validate-inputs still reported the file as fine.
    """
    from callqa.ingestion import call_input_from_metadata, sanitize_call_id

    rows = _metadata_lookup(config)
    # Sanitize the operator-supplied id: it becomes artifact filenames, and a
    # value like "../../x" would otherwise escape output_dir (model_copy below
    # bypasses CallInput's field validator, so the model guard does not catch it).
    explicit_id = getattr(args, "call_id", None)
    if explicit_id:
        explicit_id = sanitize_call_id(explicit_id)
    row = None
    if explicit_id:
        row = rows.get(explicit_id)
    # normcase: on Windows "REC001.WAV" on disk IS the "REC001.wav" in the
    # sheet, and an exact comparison ran the call on defaults (banker unknown,
    # channel L) while validate-inputs, asking the filesystem, said all was well.
    if row is None:
        name = os.path.normcase(audio_path.name)
        row = next((r for r in rows.values()
                    if os.path.normcase(r.get("file_name", "")) == name), None)
    if row is None:
        stem = os.path.normcase(audio_path.stem)
        row = next((r for cid, r in rows.items() if os.path.normcase(cid) == stem), None)
    call_id = explicit_id or (row or {}).get("call_id") or _distinct_from_case_twins(
        sanitize_call_id(audio_path.stem), audio_path.stem, config)
    if row is not None:
        call = call_input_from_metadata(row, audio_path.parent)
        call = call.model_copy(update={"audio_path": audio_path, "call_id": call_id})
    else:
        if rows:
            logger.warning(
                "%s has no row in metadata.csv; processing with defaults "
                "(banker unknown, channel L, banker name not redacted)", shown_name(audio_path),
            )
        call = CallInput(call_id=call_id, audio_path=audio_path)
    if args is not None:
        updates = {}
        if getattr(args, "banker_id", None):
            updates["banker_id"] = args.banker_id
        if getattr(args, "banker_channel", None):
            updates["banker_channel"] = args.banker_channel
        if updates:
            call = call.model_copy(update=updates)
    return call


# -- commands ----------------------------------------------------------------

def cmd_process(args: argparse.Namespace) -> int:
    """One call through the full pipeline; exit code 0/1/2."""
    config = _load_config(args)
    # expanduser: cmd.exe and PowerShell do not expand "~" for a program.
    audio_path = Path(args.audio).expanduser()
    if not audio_path.exists():
        logger.error("audio file not found: %s", audio_path)
        return EXIT_FAILED
    engines = _build_engines(config)
    call = _call_input(audio_path, config, args)
    from callqa.state import StateDB
    state = StateDB(config.paths.state_db)
    recorder = _run_recorder(config, "process")
    result = _record_call(recorder, call, engines, state)
    if recorder is not None:
        recorder.write(config.paths.output_dir)
    if result.status == "success":
        logger.info("report: %s", result.report_path)
    else:
        logger.error("call %s: status=%s error=%s", result.call_id, result.status, result.error)
    return result.exit_code


def _run_recorder(config: Config, command: str):  # noqa: ANN202
    """A best-effort run recorder. Never let provenance capture break a batch."""
    try:
        from callqa.ops.runrecord import RunRecorder
        from callqa.rubric import load_rubric

        return RunRecorder(config, load_rubric().sha256, command)
    except Exception as exc:  # noqa: BLE001
        logger.debug("run-record disabled: %s", exc)
        return None


def _record_call(recorder, call: CallInput, engines, state) -> CallResult:  # noqa: ANN001
    """process_call, timed, with the outcome recorded on the run manifest."""
    from callqa.pipeline import process_call

    t0 = time.time()
    result = process_call(call, engines, state)
    if recorder is not None:
        recorder.record(result, time.time() - t0, t0, state)
    return result


def cmd_run(args: argparse.Namespace) -> int:
    """PoC driver: iterate an existing directory of calls through process_call."""
    from callqa.ingestion import load_metadata

    config = _load_config(args)
    calls_dir = config.paths.input_dir / "calls"
    metadata_csv = config.paths.input_dir / "metadata.csv"
    validation = load_metadata(metadata_csv, calls_dir)
    if not validation.ok:
        logger.error("metadata validation failed:\n%s", validation.problem_table())
        return EXIT_FAILED
    if not validation.rows:
        logger.error("no calls found in %s", metadata_csv)
        return EXIT_FAILED

    engines = _build_engines(config)
    calls = [_call_input(calls_dir / row["file_name"], config) for row in validation.rows.values()]

    # One StateDB shared across the pool: it is thread-safe (a connection per
    # operation), and building one per worker made every worker re-assert the
    # WAL pragma at once, which raced into "database is locked".
    from callqa.state import StateDB
    state = StateDB(config.paths.state_db)

    recorder = _run_recorder(config, "run")
    results: list[CallResult] = []
    max_workers = config.run.max_workers
    disable_console_quick_edit()
    if max_workers > 1:
        # Timed waits, not pool.map: on Windows an untimed wait cannot be
        # interrupted, so Ctrl+C did nothing until every call had finished.
        pool = ThreadPoolExecutor(max_workers=max_workers)
        futures = [pool.submit(_record_call, recorder, c, engines, state) for c in calls]
        try:
            pending = set(futures)
            while pending:
                _done, pending = wait(pending, timeout=1.0)
        except KeyboardInterrupt:
            pool.shutdown(wait=False, cancel_futures=True)
            logger.error("interrupted: calls not yet started were cancelled; "
                         "rerun to resume")
            raise
        pool.shutdown()
        results = [f.result() for f in futures]
    else:
        results = [_record_call(recorder, c, engines, state) for c in calls]
    if recorder is not None:
        path = recorder.write(config.paths.output_dir)
        if path:
            logger.info("run record: %s", path)

    ok = sum(1 for r in results if r.status == "success")
    review = sum(1 for r in results if r.status == "needs_human_review")
    failed = sum(1 for r in results if r.status == "failed")
    logger.info("run complete: %d success, %d needs_human_review, %d failed", ok, review, failed)
    for r in results:
        if r.status != "success":
            logger.warning("  %s: %s (%s)", r.call_id, r.status, r.error)
    if failed:
        return EXIT_FAILED
    if review:
        return 1
    return 0


def _size_settled(path: Path, size: int, pause: float = 1.0) -> bool:
    """Re-stat after a moment: a file still being copied keeps growing."""
    time.sleep(pause)
    try:
        return path.stat().st_size == size
    except OSError:
        return False


def watch_loop(
    config: Config,
    engines,  # noqa: ANN001
    stop_event: threading.Event | None = None,
    max_cycles: int | None = None,
) -> list[CallResult]:
    """Production ingestion driver: poll input dir, process stable files
    sequentially, move them to processed/ or failed/."""
    from callqa.state import StateDB

    calls_dir = config.paths.input_dir / "calls"
    processed_dir = config.paths.input_dir / "processed"
    failed_dir = config.paths.input_dir / "failed"
    calls_dir.mkdir(parents=True, exist_ok=True)
    state = StateDB(config.paths.state_db)
    recorder = _run_recorder(config, "watch")

    sizes: dict[Path, tuple[int, float]] = {}  # path -> (size, unchanged_since)
    results: list[CallResult] = []
    skipped: set[Path] = set()
    cycles = 0
    while (stop_event is None or not stop_event.is_set()) and (
        max_cycles is None or cycles < max_cycles
    ):
        cycles += 1
        now = time.monotonic()
        for path in sorted(calls_dir.glob("*")):
            if not path.is_file():
                continue
            if path.suffix.lower() not in WATCHED_EXTENSIONS:
                if path not in skipped:
                    skipped.add(path)
                    logger.warning("ignoring %s: %s is not an audio extension this "
                                   "pipeline reads", shown_name(path), path.suffix or "(none)")
                continue
            try:
                stat = path.stat()
            except OSError:
                continue
            size = stat.st_size
            prev = sizes.get(path)
            if prev is None:
                # First sighting: date it from the file's own mtime (mapped onto
                # the monotonic clock) rather than now, so a recording that was
                # already finished before this process started counts as stable
                # immediately. Without this a single-cycle run (`watch --once`,
                # e.g. from cron) could never process anything.
                age = max(0.0, time.time() - stat.st_mtime)
                sizes[path] = (size, now - age)
                prev = sizes[path]
                if age >= config.watch.stable_seconds and not _size_settled(path, size):
                    # `cp -p` and `rsync -t` give a file the SOURCE's mtime, so
                    # a copy that is still running can look hours old. The
                    # mtime shortcut is only safe if the size also holds still.
                    logger.info("%s is still being written; waiting", shown_name(path))
                    sizes[path] = (size, now)
                    continue
            elif prev[0] != size:
                sizes[path] = (size, now)
                continue
            if now - prev[1] < config.watch.stable_seconds:
                continue
            if held_open_for_writing(path) or stat.st_mtime < _ROBOCOPY_IN_PROGRESS:
                # robocopy stamps a file 1980-01-01 until its copy completes.
                logger.info("%s is still being written; waiting", shown_name(path))
                continue
            # Stable: process it (sequential, one call at a time).
            call = _call_input(path, config)
            result = _record_call(recorder, call, engines, state)
            if recorder is not None:
                # Rewrite the manifest after each call so a long-running watch
                # that is killed still leaves an accurate record on disk.
                recorder.write(config.paths.output_dir)
            results.append(result)
            sizes.pop(path, None)
            if result.error and result.error.startswith("locked:"):
                # Another process is handling this recording. Leave it where it
                # is; quarantining a healthy call into failed/ loses it, since
                # nothing ever rescans that directory.
                logger.info("%s is locked by another process; leaving it in place",
                            shown_name(path))
                continue
            if config.watch.move_processed:
                target_dir = processed_dir if result.status != "failed" else failed_dir
                target_dir.mkdir(parents=True, exist_ok=True)
                target = target_dir / path.name
                if target.exists():
                    # Never over an earlier recording of the same name (a
                    # recorder that reuses names, or one differing only in
                    # case on Windows): that deleted the earlier original.
                    target = target_dir / f"{path.stem}.{int(time.time())}{path.suffix}"
                try:
                    move(path, target)
                except OSError as exc:
                    # Not str(exc): on Windows it repeats both full paths.
                    logger.error("could not move %s: %s", shown_name(path),
                                 type(exc).__name__)
        if max_cycles is not None and cycles >= max_cycles:
            break
        if stop_event is not None:
            if stop_event.wait(config.watch.poll_seconds):
                break
        else:
            time.sleep(config.watch.poll_seconds)
    return results


def cmd_watch(args: argparse.Namespace) -> int:
    config = _load_config(args)
    engines = _build_engines(config)
    disable_console_quick_edit()
    logger.info(
        "watching %s (poll=%.0fs, stable=%.0fs)",
        config.paths.input_dir / "calls",
        config.watch.poll_seconds,
        config.watch.stable_seconds,
    )
    try:
        results = watch_loop(config, engines, max_cycles=1 if args.once else None)
    except KeyboardInterrupt:
        logger.info("watch stopped")
        return EXIT_SUCCESS
    # A scheduler only sees the exit code, so a batch that contained a failure
    # must not report success.
    if any(r.status == "failed" for r in results):
        return EXIT_FAILED
    if any(r.status == "needs_human_review" for r in results):
        return EXIT_NEEDS_HUMAN_REVIEW
    return EXIT_SUCCESS


def cmd_report(args: argparse.Namespace) -> int:
    from callqa.reporting.banker_report import generate_banker_reports
    from callqa.rubric import load_rubric

    config = _load_config(args)
    rubric = load_rubric()
    try:
        written = generate_banker_reports(
            config.paths.output_dir, rubric, config.reporting.group_comparison
        )
    except FileNotFoundError as exc:
        logger.error("%s", exc)
        return EXIT_FAILED
    for path in written:
        logger.info("report written: %s", path)
    return 0


def cmd_calibrate(args: argparse.Namespace) -> int:
    from callqa.aggregation import load_scorecards
    from callqa.calibration import CalibrationError, calibrate, load_human_ratings
    from callqa.reporting.calibration_report import write_calibration_reports
    from callqa.rubric import load_rubric

    config = _load_config(args)
    rubric = load_rubric()
    cards = load_scorecards(config.paths.output_dir)
    try:
        ratings = load_human_ratings(
            config.paths.input_dir / "human_ratings.csv", [d.id for d in rubric.dimensions]
        )
        result = calibrate(cards, ratings, rubric)
    except CalibrationError as exc:
        logger.error("calibration failed: %s", exc)
        return EXIT_FAILED
    html_path, json_path = write_calibration_reports(result, rubric, config.paths.output_dir)
    logger.info("calibration reports: %s, %s", html_path, json_path)
    logger.info(
        "overall QWK=%s -> %s",
        result.overall_qwk,
        "PASS" if result.overall_pass else "FAIL",
    )
    return 0


def cmd_validate_inputs(args: argparse.Namespace) -> int:
    from callqa.ingestion import load_metadata

    config = _load_config(args)
    calls_dir = config.paths.input_dir / "calls"
    validation = load_metadata(config.paths.input_dir / "metadata.csv", calls_dir)
    if not validation.ok:
        print("Input validation FAILED:\n" + validation.problem_table(), file=sys.stderr)
        return EXIT_FAILED
    print(f"Input validation OK: {len(validation.rows)} calls in metadata.csv")
    ratings_csv = config.paths.input_dir / "human_ratings.csv"
    if ratings_csv.exists():
        from callqa.calibration import CalibrationError, load_human_ratings
        from callqa.rubric import load_rubric

        try:
            ratings = load_human_ratings(
                ratings_csv, [d.id for d in load_rubric().dimensions]
            )
            print(f"human_ratings.csv OK: {len(ratings)} rated calls")
        except CalibrationError as exc:
            print(f"human_ratings.csv INVALID: {exc}", file=sys.stderr)
            return EXIT_FAILED
    return 0


def cmd_retention(args: argparse.Namespace) -> int:
    """List (or destroy) RAW PII-bearing artifacts older than the retention window."""
    from callqa.ops.retention import apply_retention, find_expired

    config = _load_config(args)
    raw_days = config.retention.raw_days
    out = config.paths.output_dir
    if args.apply:
        destroyed = apply_retention(out, raw_days, input_dir=config.paths.input_dir)
        total = sum(d["bytes"] for d in destroyed)
        print(f"destroyed {len(destroyed)} raw artifact(s) older than {raw_days} days "
              f"({total // (1024*1024)} MiB); log: {out / 'retention' / 'log.jsonl'}")
        # "Destroyed N" alone read as success when some could not be deleted
        # (read-only or locked on Windows) and are still on disk, raw PII and all.
        left = find_expired(out, raw_days, input_dir=config.paths.input_dir)
        if left:
            print(f"{len(left)} expired raw artifact(s) could NOT be destroyed and are "
                  "still on disk; see the log above and retry", file=sys.stderr)
            return EXIT_FAILED
        return EXIT_SUCCESS
    expired = find_expired(out, raw_days, input_dir=config.paths.input_dir)
    if not expired:
        print(f"nothing older than {raw_days} days")
        return EXIT_SUCCESS
    total = sum(e.bytes for e in expired)
    print(f"{len(expired)} raw artifact(s) older than {raw_days} days "
          f"({total // (1024*1024)} MiB) — run with --apply to destroy:")
    for e in expired[:50]:
        print(f"  {shown_name(e.path)}  ({e.age_days}d, {e.bytes // 1024} KiB)")
    if len(expired) > 50:
        print(f"  ... and {len(expired) - 50} more")
    return EXIT_SUCCESS


def cmd_review_queue(args: argparse.Namespace) -> int:
    """List calls held for human review that have no recorded verdict yet."""
    from callqa.ops.review import pending_reviews

    config = _load_config(args)
    ratings = config.paths.input_dir / "human_ratings.csv"
    pending = pending_reviews(config.paths.output_dir, ratings, getattr(args, "rater", None))
    if not pending:
        print("no calls awaiting review")
        return EXIT_SUCCESS
    print(f"{len(pending)} call(s) awaiting review:")
    for cid in pending:
        print(f"  {cid}")
    return EXIT_SUCCESS


def cmd_review(args: argparse.Namespace) -> int:
    """Record a reviewer's per-dimension verdict into the calibration set."""
    from callqa.ops.review import ReviewError, record_review
    from callqa.rubric import load_rubric

    config = _load_config(args)
    dim_ids = [d.id for d in load_rubric().dimensions]
    scores: dict[str, int] = {}
    for pair in (args.scores or "").split(","):
        pair = pair.strip()
        if not pair:
            continue
        if "=" not in pair:
            print(f"bad --scores entry '{pair}' (want dim=value)", file=sys.stderr)
            return EXIT_FAILED
        dim, _, value = pair.partition("=")
        try:
            scores[dim.strip()] = int(value)
        except ValueError:
            print(f"bad score for {dim}: {value!r}", file=sys.stderr)
            return EXIT_FAILED
    ratings = config.paths.input_dir / "human_ratings.csv"
    try:
        record_review(ratings, args.call_id, args.rater, scores, dim_ids)
    except PermissionError:
        print(f"review not recorded: {ratings.name} is open in another program (Excel?). "
              "Close it and run this again.", file=sys.stderr)
        return EXIT_FAILED
    except ReviewError as exc:
        print(f"review rejected: {exc}", file=sys.stderr)
        return EXIT_FAILED
    print(f"recorded {args.rater}'s review of {args.call_id} -> {ratings}")
    print("run `callqa calibrate` to fold it into the agreement measurement")
    return EXIT_SUCCESS


def cmd_drift(args: argparse.Namespace) -> int:
    """Compare the current output window to a known-good baseline and flag drift."""
    import json

    from callqa.ops.drift import (
        DRIFT_DIR,
        DriftBaselineError,
        append_history,
        build_baseline,
        check_drift,
    )

    config = _load_config(args)
    baseline_path = config.paths.output_dir / DRIFT_DIR / "baseline.json"

    if args.set_baseline:
        try:
            baseline = build_baseline(config.paths.output_dir, force=args.force)
        except DriftBaselineError as exc:
            print(str(exc), file=sys.stderr)
            return EXIT_FAILED
        baseline_path.parent.mkdir(parents=True, exist_ok=True)
        baseline_path.write_text(json.dumps(baseline, ensure_ascii=False, indent=2),
                                 encoding="utf-8")
        print(f"drift baseline set from {baseline['n_calls']} calls: {baseline_path}")
        return EXIT_SUCCESS

    if not baseline_path.exists():
        print("no drift baseline — run `callqa drift --set-baseline` on a known-good "
              "period first", file=sys.stderr)
        return EXIT_FAILED
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    results = check_drift(config.paths.output_dir, baseline)
    append_history(config.paths.output_dir, results)
    flagged = False
    for r in results:
        mark = {"ok": "OK  ", "warn": "WARN", "flag": "FLAG", "skip": "--  "}[r.status]
        print(f"[{mark}] {r.name}: {r.detail}")
        if r.status == "flag":
            flagged = True
    if flagged:
        print("drift DETECTED — investigate before trusting new scores", file=sys.stderr)
        return EXIT_FAILED
    print("no drift flagged")
    return EXIT_SUCCESS


def cmd_eval(args: argparse.Namespace) -> int:
    """Evaluate the whole system against the golden set; optionally gate on a baseline."""
    from callqa.eval.harness import DEFAULT_GOLDEN, EvalReport, compare, evaluate
    from callqa.state import atomic_write_model

    config = _load_config(args)
    golden = Path(args.golden_dir) if args.golden_dir else None
    report = evaluate(config, golden)

    stamp = report.created_at.replace(":", "").replace("-", "")
    report_path = config.paths.output_dir / "eval" / f"{stamp}.json"
    atomic_write_model(report_path, report)
    print(f"eval over {report.n_calls} calls ({report.seconds_total}s):")
    print(f"  WER {report.wer_mean}  CER {report.cer_mean}  "
          f"role-acc {report.role_accuracy_mean}")
    print(f"  redaction recall {report.redaction_recall_mean}  "
          f"precision {report.redaction_precision_mean}  F1 {report.redaction_f1_mean}")
    print(f"  redaction applied to the written transcripts {report.redaction_applied_recall_mean}")
    print(f"  judge QWK {report.judge_qwk}")
    print(f"  report: {report_path}")
    if report.golden_set == "synthetic":
        # Printed every time, not once in a doc nobody opens: these numbers are
        # quoted in status reports, and "WER 0.0" reads as an accuracy claim.
        print("\n  NOTE: synthetic golden set. WER/CER/role/QWK are regression "
              "sentinels,\n        not accuracy - the references are the mock "
              "pipeline's own output.\n        Redaction recall/precision ARE "
              "real (gold identifiers are hand-labelled).")

    if args.set_baseline:
        # A baseline is a claim that some exact version of the system produced
        # these numbers. Blessing one from a modified working tree records a
        # commit that never contained the code that was measured, so the claim
        # cannot be checked by anybody, ever. The previous baseline shipped
        # with git_sha "...-dirty" for exactly this reason.
        sha = report.fingerprint.get("git_sha", "none")
        if sha.endswith("-dirty") and not args.force:
            print("refusing to bless a baseline from a modified working tree: the "
                  f"fingerprint would record {sha}, which is not a version anyone "
                  "can check out.\nCommit first, or pass --force if you know why "
                  "you want an unverifiable baseline.", file=sys.stderr)
            return EXIT_FAILED
        baseline_path = DEFAULT_GOLDEN.parent / "baseline.json"
        atomic_write_model(baseline_path, report)
        print(f"baseline updated: {baseline_path}")
        return EXIT_SUCCESS
    if args.baseline:
        baseline = EvalReport.model_validate_json(
            Path(args.baseline).read_text(encoding="utf-8"))
        regressions = compare(report, baseline)
        if regressions:
            print("REGRESSION vs baseline:", file=sys.stderr)
            for r in regressions:
                print(f"  - {r['metric']}: {r['was']} -> {r['now']} ({r['delta']:+})",
                      file=sys.stderr)
            return EXIT_FAILED
        print("no regressions vs baseline")
    return EXIT_SUCCESS


def cmd_preflight(args: argparse.Namespace) -> int:
    """Verify models, endpoints, disk and inputs before a batch starts."""
    from callqa.ops.preflight import run_preflight

    config = _load_config(args)
    checks = run_preflight(config, deep=getattr(args, "deep", False))
    failed_critical = False
    for c in checks:
        mark = "OK  " if c.ok else ("FAIL" if c.critical else "WARN")
        print(f"[{mark}] {c.name}: {c.detail}")
        if not c.ok and c.critical:
            failed_critical = True
    if failed_critical:
        print("preflight FAILED — do not start the batch", file=sys.stderr)
        return EXIT_FAILED
    print("preflight OK")
    return EXIT_SUCCESS


def cmd_verify(args: argparse.Namespace) -> int:
    """Is a stored result still reproducible, and if not, what changed?"""
    from callqa.ops.verify import verify_call
    from callqa.rubric import load_rubric

    config = _load_config(args)
    res = verify_call(config.paths.output_dir, args.call_id, config, load_rubric().sha256)
    if res.run_id is None:
        print(f"{args.call_id}: {res.reason}", file=sys.stderr)
        return EXIT_FAILED
    if res.reproducible:
        print(f"{args.call_id}: reproducible (produced by run {res.run_id})")
        return EXIT_SUCCESS
    print(f"{args.call_id}: NOT reproducible (produced by run {res.run_id}) — "
          f"inputs changed since:", file=sys.stderr)
    for change in res.changed:
        print(f"  - {change['field']}: {change['was']} -> {change['now']}", file=sys.stderr)
    return EXIT_FAILED


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="callqa", description="Hebrew Call-QA pipeline")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("process", help="process ONE call end-to-end (exit 0/1/2)")
    p.add_argument("--audio", required=True, help="path to the recording (.wav/.mp3)")
    p.add_argument("--call-id", default=None)
    p.add_argument("--banker-id", default=None)
    p.add_argument("--banker-channel", choices=["L", "R"], default=None)
    _add_common_args(p)
    p.set_defaults(func=cmd_process)

    p = sub.add_parser("watch", help="production ingestion: poll input dir, process new files")
    p.add_argument("--once", action="store_true", help="run one poll cycle and exit")
    _add_common_args(p)
    p.set_defaults(func=cmd_watch)

    p = sub.add_parser("run", help="PoC driver: process every call in metadata.csv")
    p.add_argument("--max-workers", type=int, default=None)
    _add_common_args(p)
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("report", help="generate per-banker reports + index")
    _add_common_args(p)
    p.set_defaults(func=cmd_report)

    p = sub.add_parser("calibrate", help="calibrate judge vs human ratings (QWK)")
    _add_common_args(p)
    p.set_defaults(func=cmd_calibrate)

    p = sub.add_parser("validate-inputs", help="validate metadata.csv and human_ratings.csv")
    _add_common_args(p)
    p.set_defaults(func=cmd_validate_inputs)

    p = sub.add_parser("verify", help="is a stored call's score still reproducible?")
    p.add_argument("call_id", help="the call to check")
    _add_common_args(p)
    p.set_defaults(func=cmd_verify)

    p = sub.add_parser("preflight", help="verify models, endpoints, disk and inputs before a batch")
    p.add_argument("--deep", action="store_true", help="re-hash local model weights (slow)")
    _add_common_args(p)
    p.set_defaults(func=cmd_preflight)

    p = sub.add_parser("retention", help="retire raw PII-bearing artifacts past the window")
    p.add_argument("--apply", action="store_true", help="destroy them (default: list only)")
    _add_common_args(p)
    p.set_defaults(func=cmd_retention)

    p = sub.add_parser("review-queue", help="list calls held for human review with no verdict")
    p.add_argument("--rater", default=None, help="only calls this rater hasn't reviewed")
    _add_common_args(p)
    p.set_defaults(func=cmd_review_queue)

    p = sub.add_parser("review", help="record a reviewer's verdict into the calibration set")
    p.add_argument("call_id")
    p.add_argument("--rater", required=True, help="reviewer id")
    p.add_argument("--scores", required=True,
                   help="comma-separated dim=score, e.g. "
                        "identification=3,compliance=4,empathy=5,...")
    _add_common_args(p)
    p.set_defaults(func=cmd_review)

    p = sub.add_parser("drift", help="flag drift in scores/review-rate/quality vs a baseline")
    p.add_argument("--set-baseline", action="store_true",
                   help="capture the current output as the known-good baseline")
    _add_common_args(p)
    p.set_defaults(func=cmd_drift)

    p = sub.add_parser("eval", help="evaluate the whole system against the golden set")
    p.add_argument("--golden-dir", default=None,
                   help="a real golden set (calls/ + metadata.csv + references.json); "
                        "default is the synthetic in-repo set")
    p.add_argument("--baseline", default=None, help="fail on a regression vs this eval report")
    p.add_argument("--set-baseline", action="store_true", help="save this run as eval/baseline.json")
    _add_common_args(p)
    p.set_defaults(func=cmd_eval)

    return parser


def main(argv: list[str] | None = None) -> int:
    # First of all: a Hebrew message printed to a redirected stream on Windows
    # (a Scheduled Task, `> log.txt`) otherwise raises UnicodeEncodeError.
    configure_stdio()
    # The runtime never downloads and never reports home - enforced, not just
    # documented. pyannote.audio 4 sends usage telemetry to otel.pyannote.ai
    # on every pipeline load unless told not to (HF_HUB_OFFLINE does not stop
    # it), and huggingface_hub revalidates a cached model online by default.
    # setdefault: an operator who deliberately sets one keeps their value.
    for name, value in (("PYANNOTE_METRICS_ENABLED", "0"), ("HF_HUB_OFFLINE", "1"),
                        ("TRANSFORMERS_OFFLINE", "1"), ("HF_HUB_DISABLE_TELEMETRY", "1")):
        os.environ.setdefault(name, value)
    if IS_WINDOWS and _started_in_system_folder():
        print("callqa was started in the Windows system folder, so its data folders "
              "would be created there. Run it from the project folder - in Task "
              "Scheduler, set 'Start in' to the project folder.", file=sys.stderr)
        return EXIT_FAILED
    # Before the parser: config overrides come from the environment, and the
    # operator's `.env` is where the endpoint of a GPU box or a judge key lives.
    load_dotenv()
    args = build_parser().parse_args(argv)
    _setup_logging(getattr(args, "verbose", False), getattr(args, "log_file", None))
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
