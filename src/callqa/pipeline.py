"""process_call(): the production core.

Takes ONE call (recording + metadata) and produces the complete per-call
output: transcript, speaker attribution, redaction, features, judge scores,
and the per-call HTML report. Knows nothing about directories of calls,
batches, or schedules - drivers feed calls into this function one at a time.

Guarantees:
- Per-call atomicity: an SQLite lock per call_id prevents double processing.
- Every artifact is written atomically (tmp file + rename).
- Idempotent/resumable: completed stages are skipped on re-run (state in
  SQLite; --force reruns everything).
- Each call ends in an explicit status envelope; exceptions never escape.
"""

from __future__ import annotations

import logging
from pathlib import Path

from callqa.audio import prepare_audio
from callqa.engines import Engines
from callqa.features import compute_features
from callqa.ingestion import probe_audio
from callqa.judge.runner import NeedsHumanReviewError, run_judge
from callqa.models import (
    AudioArtifact,
    CallInput,
    CallMeta,
    CallResult,
    DialogTranscript,
    Features,
    RedactedTranscript,
    ScoreCard,
    Transcript,
    TranscriptBundle,
)
from callqa.reporting.call_report import render_call_report
from callqa.speakers.stereo import (
    assign_mono_roles,
    merge_stereo,
    speaker_segments_from_dialog,
)
from callqa.state import CallLockedError, StateDB, atomic_write_model, atomic_write_text

logger = logging.getLogger(__name__)

STAGES = [
    "ingestion",
    "audio",
    "asr",
    "speakers",
    "redaction",
    "features",
    "judge",
    "report",
]

RAW_TRANSCRIPTS_README = (
    "WARNING: files in this directory contain RAW, UNREDACTED transcripts\n"
    "including customer PII. Access must be restricted. Only the redacted\n"
    "transcripts (../redacted/) may be used for judging, reports, or logs.\n"
)


class _ArtifactStore:
    """Stage artifact paths + load/save with state tracking."""

    def __init__(self, output_dir: Path, state: StateDB, call_id: str, force: bool) -> None:
        self.output_dir = output_dir
        self.state = state
        self.call_id = call_id
        self.force = force

    def path(self, stage_dir: str, suffix: str = ".json") -> Path:
        return self.output_dir / stage_dir / f"{self.call_id}{suffix}"

    def is_done(self, stage: str) -> bool:
        return not self.force and self.state.is_stage_done(self.call_id, stage)

    def mark_done(self, stage: str, artifact_path: Path) -> None:
        self.state.mark_stage_done(self.call_id, stage, artifact_path)


def process_call(call: CallInput, engines: Engines, state: StateDB | None = None) -> CallResult:
    """The canonical single-call entrypoint. Never raises; returns a status envelope."""
    config = engines.config
    state = state or StateDB(config.paths.state_db)
    call_id = call.call_id
    try:
        state.acquire_lock(call_id)
    except CallLockedError as exc:
        return CallResult(call_id=call_id, status="failed", error=str(exc))

    stages_completed: list[str] = []
    try:
        if config.run.force:
            state.clear_call(call_id)
        store = _ArtifactStore(config.paths.output_dir, state, call_id, config.run.force)

        # -- stage 1: ingestion ------------------------------------------
        meta_path = store.path("ingestion")
        if store.is_done("ingestion"):
            meta = CallMeta.model_validate_json(meta_path.read_text(encoding="utf-8"))
        else:
            meta = probe_audio(call)
            atomic_write_model(meta_path, meta)
            store.mark_done("ingestion", meta_path)
        stages_completed.append("ingestion")

        # -- stage 2: audio ----------------------------------------------
        audio_path = store.path("audio")
        if store.is_done("audio"):
            audio_art = AudioArtifact.model_validate_json(audio_path.read_text(encoding="utf-8"))
        else:
            audio_art = prepare_audio(call, meta, config, config.paths.output_dir / "audio" / "wav")
            atomic_write_model(audio_path, audio_art)
            store.mark_done("audio", audio_path)
        stages_completed.append("audio")

        # -- stage 3: asr (RAW transcripts) ------------------------------
        transcripts_dir = config.paths.output_dir / "transcripts"
        readme = transcripts_dir / "README.txt"
        if not readme.exists():
            atomic_write_text(readme, RAW_TRANSCRIPTS_README)
        bundle_path = store.path("transcripts")
        if store.is_done("asr"):
            bundle = TranscriptBundle.model_validate_json(bundle_path.read_text(encoding="utf-8"))
        else:
            if audio_art.is_stereo:
                banker_t = engines.asr.transcribe(
                    Path(audio_art.banker_wav or ""), call_id=call_id, role="banker",
                    vad_segments=audio_art.banker_segments,
                )
                customer_t = engines.asr.transcribe(
                    Path(audio_art.customer_wav or ""), call_id=call_id, role="customer",
                    vad_segments=audio_art.customer_segments,
                )
                bundle = TranscriptBundle(call_id=call_id, banker=banker_t, customer=customer_t)
            else:
                mono_t = engines.asr.transcribe(
                    Path(audio_art.mono_wav or ""), call_id=call_id, role=None,
                    vad_segments=audio_art.mono_segments,
                )
                bundle = TranscriptBundle(call_id=call_id, mono=mono_t)
            atomic_write_model(bundle_path, bundle)
            store.mark_done("asr", bundle_path)
        stages_completed.append("asr")

        # -- stage 4: speakers (RAW dialog, stored under transcripts/) ---
        dialog_path = store.path("transcripts", suffix=".dialog.json")
        if store.is_done("speakers"):
            dialog = DialogTranscript.model_validate_json(dialog_path.read_text(encoding="utf-8"))
        else:
            if audio_art.is_stereo:
                assert bundle.banker is not None and bundle.customer is not None
                dialog = merge_stereo(call_id, bundle.banker, bundle.customer)
            else:
                assert bundle.mono is not None
                if engines.mono_diarizer is None:
                    raise RuntimeError("mono recording but no diarizer configured")
                diarized = engines.mono_diarizer.diarize(Path(audio_art.mono_wav or ""), call_id)
                dialog = assign_mono_roles(call_id, bundle.mono, diarized)
            atomic_write_model(dialog_path, dialog)
            store.mark_done("speakers", dialog_path)
        stages_completed.append("speakers")

        # -- stage 5: redaction ------------------------------------------
        redacted_path = store.path("redacted")
        if store.is_done("redaction"):
            redacted = RedactedTranscript.model_validate_json(
                redacted_path.read_text(encoding="utf-8")
            )
        else:
            extra_names = [call.banker_name] if call.banker_name else []
            redacted = engines.redactor.redact_dialog(dialog, extra_names)
            atomic_write_model(redacted_path, redacted)
            store.mark_done("redaction", redacted_path)
        stages_completed.append("redaction")

        # -- stage 6: features -------------------------------------------
        features_path = store.path("features")
        if store.is_done("features"):
            features = Features.model_validate_json(features_path.read_text(encoding="utf-8"))
        else:
            if audio_art.is_stereo:
                banker_segments = audio_art.banker_segments
                customer_segments = audio_art.customer_segments
            else:
                banker_segments, customer_segments = speaker_segments_from_dialog(dialog)
            features = compute_features(
                call_id, banker_segments, customer_segments, dialog, meta.duration_sec
            )
            atomic_write_model(features_path, features)
            store.mark_done("features", features_path)
        stages_completed.append("features")

        # -- stage 7: judge ----------------------------------------------
        score_path = store.path("scores")
        if store.is_done("judge"):
            scorecard = ScoreCard.model_validate_json(score_path.read_text(encoding="utf-8"))
        else:
            try:
                scorecard = run_judge(
                    engines.judge, config.judge, engines.rubric,
                    call_id, call.banker_id, redacted, features,
                )
            except NeedsHumanReviewError as exc:
                logger.warning("call needs human review: call_id=%s", call_id)
                result = CallResult(
                    call_id=call_id,
                    status="needs_human_review",
                    error=str(exc),
                    stages_completed=stages_completed,
                )
                _write_result(config.paths.output_dir, result)
                return result
            atomic_write_model(score_path, scorecard)
            store.mark_done("judge", score_path)
        stages_completed.append("judge")

        # -- stage 8: per-call report ------------------------------------
        report_path = config.paths.output_dir / "reports" / "calls" / f"{call_id}.html"
        if not store.is_done("report") or not report_path.exists():
            html = render_call_report(engines.rubric, meta, scorecard, features, redacted)
            atomic_write_text(report_path, html)
            store.mark_done("report", report_path)
        stages_completed.append("report")

        result = CallResult(
            call_id=call_id,
            status="success",
            stages_completed=stages_completed,
            report_path=str(report_path),
            scorecard_path=str(score_path),
        )
        _write_result(config.paths.output_dir, result)
        logger.info("call done: call_id=%s status=%s", call_id, result.status)
        return result
    except Exception as exc:  # noqa: BLE001 - one bad call never stops a driver
        logger.exception("call failed: call_id=%s stage=%s", call_id,
                         STAGES[len(stages_completed)] if len(stages_completed) < len(STAGES) else "?")
        result = CallResult(
            call_id=call_id,
            status="failed",
            error=f"{type(exc).__name__}: {exc}",
            stages_completed=stages_completed,
        )
        try:
            _write_result(config.paths.output_dir, result)
        except OSError:
            pass
        return result
    finally:
        state.release_lock(call_id)


def _write_result(output_dir: Path, result: CallResult) -> None:
    atomic_write_model(output_dir / "results" / f"{result.call_id}.json", result)
