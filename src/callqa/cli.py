"""CLI entrypoints: process / watch / run / report / calibrate / validate-inputs.

The drivers here are thin: all per-call logic lives in pipeline.process_call.
"""

from __future__ import annotations

import argparse
import logging
import shutil
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from callqa.config import Config, load_config
from callqa.models import (
    EXIT_FAILED,
    EXIT_NEEDS_HUMAN_REVIEW,
    EXIT_SUCCESS,
    CallInput,
    CallResult,
)
from callqa.resources import find_config

logger = logging.getLogger("callqa")

# Everything probe_audio can read. Watching only .wav/.mp3 left an .m4a sitting
# in the drop directory forever: not processed, not quarantined, not logged.
WATCHED_EXTENSIONS = frozenset({".wav", ".mp3", ".m4a", ".aac", ".mp4", ".ogg",
                                ".opus", ".flac", ".wma", ".amr"})

# Everything probe_audio can read. Watching only .wav/.mp3 left an .m4a sitting
# in the drop directory forever, unprocessed and unmentioned.
WATCHED_EXTENSIONS = frozenset({".wav", ".mp3", ".m4a", ".aac", ".mp4", ".ogg",
                                ".opus", ".flac", ".wma", ".amr"})


def _setup_logging(verbose: bool = False) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )


def _add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", default=None, help="path to config.yaml")
    parser.add_argument("--mock", action="store_true", help="run with deterministic mock engines")
    parser.add_argument("--force", action="store_true", help="rerun completed stages")
    parser.add_argument("-v", "--verbose", action="store_true")


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

    return build_engines(config)


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
    explicit_id = getattr(args, "call_id", None)
    row = None
    if explicit_id:
        row = rows.get(explicit_id)
    if row is None:
        row = next((r for r in rows.values() if r.get("file_name") == audio_path.name), None)
    if row is None:
        row = rows.get(audio_path.stem)
    call_id = explicit_id or (row or {}).get("call_id") or sanitize_call_id(audio_path.stem)
    if row is not None:
        call = call_input_from_metadata(row, audio_path.parent)
        call = call.model_copy(update={"audio_path": audio_path, "call_id": call_id})
    else:
        if rows:
            logger.warning(
                "%s has no row in metadata.csv; processing with defaults "
                "(banker unknown, channel L, banker name not redacted)", audio_path.name,
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
    from callqa.pipeline import process_call

    config = _load_config(args)
    audio_path = Path(args.audio)
    if not audio_path.exists():
        logger.error("audio file not found: %s", audio_path)
        return EXIT_FAILED
    engines = _build_engines(config)
    call = _call_input(audio_path, config, args)
    result = process_call(call, engines)
    if result.status == "success":
        logger.info("report: %s", result.report_path)
    else:
        logger.error("call %s: status=%s error=%s", result.call_id, result.status, result.error)
    return result.exit_code


def _process_one(call: CallInput, engines) -> CallResult:  # noqa: ANN001
    from callqa.pipeline import process_call

    return process_call(call, engines)


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

    results: list[CallResult] = []
    max_workers = config.run.max_workers
    if max_workers > 1:
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            results = list(pool.map(lambda c: _process_one(c, engines), calls))
    else:
        results = [_process_one(c, engines) for c in calls]

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


def watch_loop(
    config: Config,
    engines,  # noqa: ANN001
    stop_event: threading.Event | None = None,
    max_cycles: int | None = None,
) -> list[CallResult]:
    """Production ingestion driver: poll input dir, process stable files
    sequentially, move them to processed/ or failed/."""
    calls_dir = config.paths.input_dir / "calls"
    processed_dir = config.paths.input_dir / "processed"
    failed_dir = config.paths.input_dir / "failed"
    calls_dir.mkdir(parents=True, exist_ok=True)

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
                                   "pipeline reads", path.name, path.suffix or "(none)")
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
            elif prev[0] != size:
                sizes[path] = (size, now)
                continue
            if now - prev[1] < config.watch.stable_seconds:
                continue
            # Stable: process it (sequential, one call at a time).
            call = _call_input(path, config)
            result = _process_one(call, engines)
            results.append(result)
            sizes.pop(path, None)
            if result.error and result.error.startswith("locked:"):
                # Another process is handling this recording. Leave it where it
                # is; quarantining a healthy call into failed/ loses it, since
                # nothing ever rescans that directory.
                logger.info("%s is locked by another process; leaving it in place",
                            path.name)
                continue
            if config.watch.move_processed:
                target_dir = processed_dir if result.status != "failed" else failed_dir
                target_dir.mkdir(parents=True, exist_ok=True)
                try:
                    shutil.move(str(path), str(target_dir / path.name))
                except OSError as exc:
                    logger.error("could not move %s: %s", path.name, exc)
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

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    _setup_logging(getattr(args, "verbose", False))
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
