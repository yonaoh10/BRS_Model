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

import json
import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import TypeVar

from pydantic import BaseModel

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
    TranscriptBundle,
)
from callqa.redaction import sanitize_error
from callqa.reporting.call_report import render_call_report
from callqa.speakers.stereo import (
    assign_mono_roles,
    merge_stereo,
    speaker_segments_from_dialog,
)
from callqa.state import CallLockedError, StateDB, atomic_write_model, atomic_write_text

logger = logging.getLogger(__name__)

# Below these a "call" is a misfire, a voicemail beep or a dropped connection.
MIN_CALL_SECONDS = 20.0
MIN_SPEECH_SECONDS = 10.0

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


_T = TypeVar("_T", bound=BaseModel)


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

    def load(self, stage: str, path: Path, model: type[_T]) -> _T | None:
        """Read a completed stage's artifact, or None to recompute it.

        A stage marked done whose artifact is missing, truncated, unparseable
        or belongs to a DIFFERENT call is not a resumable state. Treating it as
        one made a corrupt file a permanent failure that only --force could
        clear, and let an artifact copied under the wrong name publish another
        call's score under this call's id.
        """
        if not self.is_done(stage):
            return None
        try:
            artifact = model.model_validate_json(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            logger.warning("call_id=%s stage=%s: artifact unusable (%s); recomputing",
                           self.call_id, stage, type(exc).__name__)
            return None
        if getattr(artifact, "call_id", self.call_id) != self.call_id:
            logger.error(
                "call_id=%s stage=%s: artifact belongs to call %r; recomputing",
                self.call_id, stage, getattr(artifact, "call_id", None),
            )
            return None
        return artifact


class _EarlyDiarization:
    """Pyannote diarization racing the ASR stage, in a separate process.

    Diarization reads only the audio, so on a mono recording it can run while
    ASR is still transcribing. Measured on the 6-core dev machine: 172.7 s of
    ASR + 228.5 s of diarization sequentially (~401 s) become ~347 s of
    combined wall - contention stretches both, so the win is the ~55 s gap,
    not the full ASR time.

    A separate PROCESS, not a thread: ctranslate2 and torch each bundle their
    own libiomp5, and one process holding both aborts at random on Intel
    macOS. Everything here is best-effort - any failure of the worker just
    means the speakers stage diarizes in-process exactly as it did before.
    """

    def __init__(self, proc: subprocess.Popen, out_path: Path, log_path: Path) -> None:
        self._proc = proc
        self._out_path = out_path
        self._log_path = log_path

    @classmethod
    def start(cls, engines: Engines, audio_art: AudioArtifact,
              call_id: str) -> _EarlyDiarization | None:
        config = engines.config
        if not config.speakers.parallel_diarization or audio_art.is_stereo:
            return None
        if getattr(engines.mono_diarizer, "name", None) != "pyannote":
            return None                    # mock diarizer is instant; nothing to hide
        wav = audio_art.mono_wav
        if not wav or not Path(wav).is_file():
            return None
        out_path = Path(f"{wav}.diar.json")
        log_path = Path(f"{wav}.diar.log")
        out_path.unlink(missing_ok=True)
        env = dict(os.environ)
        # Deliberately NOT capping threads: the perf QA measured an explicit
        # 3+3 split at 394 s combined wall vs ~347 s when both engines keep
        # their defaults and let the scheduler arbitrate. The operator's own
        # OMP_NUM_THREADS, if set, is inherited like everything else.
        try:
            with open(log_path, "w", encoding="utf-8") as log:
                proc = subprocess.Popen(
                    [sys.executable, "-m", "callqa.speakers.diar_worker",
                     str(wav), call_id, str(out_path)],
                    stdin=subprocess.PIPE, stdout=log, stderr=log, env=env,
                )
            assert proc.stdin is not None
            proc.stdin.write(config.speakers.model_dump_json().encode("utf-8"))
            proc.stdin.close()
        except OSError as exc:
            logger.warning("call_id=%s: could not start diarization worker (%s); "
                           "will diarize in-process", call_id, exc)
            return None
        logger.info("call_id=%s: diarization running alongside ASR (pid %d)",
                    call_id, proc.pid)
        return cls(proc, out_path, log_path)

    def collect(self, call_id: str, duration_sec: float) -> list | None:
        """The worker's segments, or None to fall back to in-process work."""
        from callqa.speakers.diarization import DiarizedSegment

        timeout = max(900.0, duration_sec * 10.0)
        try:
            code = self._proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            logger.warning("call_id=%s: diarization worker exceeded %.0fs; killing it",
                           call_id, timeout)
            self.cancel()
            return None
        try:
            if code != 0:
                raise OSError(f"worker exited {code}: {self._worker_log_tail()}")
            segments = [
                DiarizedSegment(label=s["label"], start=float(s["start"]),
                                end=float(s["end"]))
                for s in json.loads(self._out_path.read_text(encoding="utf-8"))
            ]
        except (OSError, ValueError, KeyError, TypeError) as exc:
            logger.warning("call_id=%s: diarization worker unusable (%s); "
                           "diarizing in-process", call_id, exc)
            return None
        finally:
            self._cleanup_files()
        return segments

    def cancel(self) -> None:
        if self._proc.poll() is None:
            self._proc.kill()
            self._proc.wait(timeout=10)
        self._cleanup_files()

    def _worker_log_tail(self) -> str:
        try:
            return self._log_path.read_text(encoding="utf-8", errors="replace")[-300:]
        except OSError:
            return "<no worker log>"

    def _cleanup_files(self) -> None:
        self._out_path.unlink(missing_ok=True)
        self._log_path.unlink(missing_ok=True)


def _report_is_intact(path: Path) -> bool:
    """A report that was truncated mid-write is not a finished report."""
    try:
        return path.is_file() and path.read_text(encoding="utf-8").rstrip().endswith("</html>")
    except (OSError, UnicodeDecodeError):
        return False


def process_call(call: CallInput, engines: Engines, state: StateDB | None = None) -> CallResult:
    """The canonical single-call entrypoint. Never raises; returns a status envelope."""
    config = engines.config
    state = state or StateDB(config.paths.state_db)
    call_id = call.call_id
    try:
        state.acquire_lock(call_id)
    except CallLockedError as exc:
        # Prefixed so a driver can tell "somebody else has this call" apart
        # from "this call is broken" and not quarantine a healthy recording.
        return CallResult(call_id=call_id, status="failed", error=f"locked: {exc}")

    stages_completed: list[str] = []
    review_reasons: list[str] = []
    early_diar: _EarlyDiarization | None = None
    try:
        if config.run.force:
            state.clear_call(call_id)
        store = _ArtifactStore(config.paths.output_dir, state, call_id, config.run.force)

        # -- stage 1: ingestion ------------------------------------------
        meta_path = store.path("ingestion")
        meta = store.load("ingestion", meta_path, CallMeta)
        if meta is None:
            meta = probe_audio(call)
            atomic_write_model(meta_path, meta)
            store.mark_done("ingestion", meta_path)
        stages_completed.append("ingestion")

        # -- stage 2: audio ----------------------------------------------
        audio_path = store.path("audio")
        audio_art = store.load("audio", audio_path, AudioArtifact)
        if audio_art is None:
            audio_art = prepare_audio(call, meta, config, config.paths.output_dir / "audio" / "wav")
            atomic_write_model(audio_path, audio_art)
            store.mark_done("audio", audio_path)
        speech_sec = sum(
            seg.end - seg.start
            for seg in (audio_art.banker_segments + audio_art.customer_segments
                        + audio_art.mono_segments)
        )
        if meta.duration_sec < MIN_CALL_SECONDS:
            review_reasons.append(
                f"recording is only {meta.duration_sec:.1f}s long"
            )
        if speech_sec < MIN_SPEECH_SECONDS:
            # Silence scores as well as anything else - the judge will happily
            # fill in eight dimensions from an empty transcript - so a call
            # with nothing in it must not arrive as a finished score.
            review_reasons.append(
                f"only {speech_sec:.1f}s of speech was detected in this recording"
            )
        stages_completed.append("audio")

        # -- stage 3: asr (RAW transcripts) ------------------------------
        transcripts_dir = config.paths.output_dir / "transcripts"
        readme = transcripts_dir / "README.txt"
        if not readme.exists():
            atomic_write_text(readme, RAW_TRANSCRIPTS_README)
        bundle_path = store.path("transcripts")
        bundle = store.load("asr", bundle_path, TranscriptBundle)
        if bundle is None and not store.is_done("speakers"):
            # ASR is about to spend minutes on this recording; a mono call's
            # diarization needs none of it, so it runs alongside.
            early_diar = _EarlyDiarization.start(engines, audio_art, call_id)
        if bundle is None:
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
        dialog = store.load("speakers", dialog_path, DialogTranscript)
        if dialog is None:
            if audio_art.is_stereo:
                assert bundle.banker is not None and bundle.customer is not None
                dialog = merge_stereo(call_id, bundle.banker, bundle.customer)
            else:
                assert bundle.mono is not None
                if engines.mono_diarizer is None:
                    raise RuntimeError("mono recording but no diarizer configured")
                diarized = None
                if early_diar is not None:
                    diarized = early_diar.collect(call_id, meta.duration_sec)
                    early_diar = None
                if diarized is None:
                    diarized = engines.mono_diarizer.diarize(
                        Path(audio_art.mono_wav or ""), call_id)
                dialog = assign_mono_roles(call_id, bundle.mono, diarized)
            atomic_write_model(dialog_path, dialog)
            store.mark_done("speakers", dialog_path)
        stages_completed.append("speakers")

        # Who-is-who was inferred, not observed. If the evidence was close to
        # even, the whole report could be inverted, so the call is reported and
        # held for a human rather than published as fact.
        if (dialog.attribution_mode == "mono_diarized"
                and dialog.role_confidence < config.speakers.min_role_confidence):
            review_reasons.append(
                f"speaker roles inferred with low confidence "
                f"({dialog.role_confidence:.2f} < {config.speakers.min_role_confidence:.2f})"
            )

        # -- stage 5: redaction ------------------------------------------
        redacted_path = store.path("redacted")
        redacted = store.load("redaction", redacted_path, RedactedTranscript)
        if redacted is None:
            extra_names = [call.banker_name] if call.banker_name else []
            redacted = engines.redactor.redact_dialog(dialog, extra_names)
            atomic_write_model(redacted_path, redacted)
            store.mark_done("redaction", redacted_path)
        if not redacted.enabled:
            review_reasons.append(
                "redaction was DISABLED for this call; the transcript, the judge "
                "prompt and this report contain raw customer identifiers"
            )
        stages_completed.append("redaction")

        # -- stage 6: features -------------------------------------------
        features_path = store.path("features")
        features = store.load("features", features_path, Features)
        if features is None:
            if audio_art.is_stereo:
                banker_segments = audio_art.banker_segments
                customer_segments = audio_art.customer_segments
            else:
                banker_segments, customer_segments = speaker_segments_from_dialog(dialog)
            features = compute_features(
                call_id, banker_segments, customer_segments, dialog, meta.duration_sec,
                overlap_metrics_available=audio_art.is_stereo,
            )
            atomic_write_model(features_path, features)
            store.mark_done("features", features_path)
        stages_completed.append("features")

        # -- stage 7: judge ----------------------------------------------
        score_path = store.path("scores")
        scorecard = store.load("judge", score_path, ScoreCard)
        if scorecard is None:
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
                    error=sanitize_error(exc),
                    stages_completed=stages_completed,
                )
                _write_result(config.paths.output_dir, result)
                return result
            atomic_write_model(score_path, scorecard)
            store.mark_done("judge", score_path)
        stages_completed.append("judge")

        # -- stage 8: per-call report ------------------------------------
        report_path = config.paths.output_dir / "reports" / "calls" / f"{call_id}.html"
        if not store.is_done("report") or not _report_is_intact(report_path):
            html = render_call_report(engines.rubric, meta, scorecard, features, redacted,
                                      dialog=dialog, review_reasons=review_reasons,
                                      min_role_confidence=config.speakers.min_role_confidence)
            atomic_write_text(report_path, html)
            store.mark_done("report", report_path)
        stages_completed.append("report")

        result = CallResult(
            call_id=call_id,
            status="needs_human_review" if review_reasons else "success",
            error="; ".join(review_reasons) or None,
            stages_completed=stages_completed,
            report_path=str(report_path),
            scorecard_path=str(score_path),
        )
        _write_result(config.paths.output_dir, result)
        logger.info("call done: call_id=%s status=%s", call_id, result.status)
        return result
    except Exception as exc:  # noqa: BLE001 - one bad call never stops a driver
        stage = STAGES[len(stages_completed)] if len(stages_completed) < len(STAGES) else "?"
        # Not logger.exception: a traceback raised before redaction carries the
        # transcript text that put it there, and the log is not a place raw
        # customer data may reach.
        logger.error("call failed: call_id=%s stage=%s error=%s",
                     call_id, stage, sanitize_error(exc))
        result = CallResult(
            call_id=call_id,
            status="failed",
            error=f"{type(exc).__name__}: {sanitize_error(exc)}",
            stages_completed=stages_completed,
        )
        try:
            _write_result(config.paths.output_dir, result)
        except OSError:
            pass
        return result
    finally:
        if early_diar is not None:
            # An ASR failure must not orphan a diarization process.
            early_diar.cancel()
        state.release_lock(call_id)


def _write_result(output_dir: Path, result: CallResult) -> None:
    atomic_write_model(output_dir / "results" / f"{result.call_id}.json", result)
