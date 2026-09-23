"""Audio preparation (stage 2): conversion to 16 kHz WAV, stereo channel
split, and VAD speech-segment detection.

Uses ffmpeg when available; falls back to a pure-Python path for WAV inputs.
VAD engines: 'energy' (dependency-free, default in mock) and 'silero'
(lazy import; model ships inside the silero-vad wheel - no download at runtime).
"""

from __future__ import annotations

import logging
import shutil
import subprocess
import wave
from pathlib import Path

import numpy as np

from callqa.config import Config
from callqa.models import AudioArtifact, CallInput, CallMeta, VADSegment

logger = logging.getLogger(__name__)


class AudioError(RuntimeError):
    pass


def _have_ffmpeg() -> bool:
    return shutil.which("ffmpeg") is not None


def _run_ffmpeg(args: list[str]) -> None:
    proc = subprocess.run(["ffmpeg", "-y", "-v", "error", *args], capture_output=True, text=True)
    if proc.returncode != 0:
        raise AudioError(f"ffmpeg failed: {proc.stderr.strip()[:500]}")


def _read_wav(path: Path) -> tuple[np.ndarray, int]:
    """Read a PCM WAV into float32 [-1, 1], shape (n_samples, n_channels)."""
    with wave.open(str(path), "rb") as wf:
        n_channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        rate = wf.getframerate()
        raw = wf.readframes(wf.getnframes())
    if sampwidth == 2:
        data = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    elif sampwidth == 1:
        data = (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    elif sampwidth == 4:
        data = np.frombuffer(raw, dtype=np.int32).astype(np.float32) / 2147483648.0
    else:
        raise AudioError(f"unsupported WAV sample width: {sampwidth}")
    return data.reshape(-1, n_channels), rate


def _write_wav(path: Path, samples: np.ndarray, rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    clipped = np.clip(samples, -1.0, 1.0)
    pcm = (clipped * 32767.0).astype(np.int16)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(rate)
        wf.writeframes(pcm.tobytes())


def _resample_linear(samples: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
    if src_rate == dst_rate:
        return samples
    n_dst = int(round(len(samples) * dst_rate / src_rate))
    src_t = np.arange(len(samples)) / src_rate
    dst_t = np.arange(n_dst) / dst_rate
    return np.interp(dst_t, src_t, samples).astype(np.float32)


def _restrict(path: Path) -> None:
    """Customer voice at rest. Every other artifact is written 0600; these
    were left at the process umask, i.e. world-readable."""
    try:
        path.chmod(0o600)
    except OSError:  # pragma: no cover - unusual filesystems
        logger.debug("could not restrict permissions on %s", path)


def _extract_channel(
    src: Path, dst: Path, channel: int | None, target_rate: int
) -> None:
    """Extract one channel (or downmix if channel is None) to 16 kHz mono WAV."""
    if _have_ffmpeg():
        if channel is None:
            _run_ffmpeg(["-i", str(src), "-ac", "1", "-ar", str(target_rate), str(dst)])
        else:
            pan = f"pan=mono|c0=c{channel}"
            _run_ffmpeg(["-i", str(src), "-af", pan, "-ar", str(target_rate), str(dst)])
        _restrict(dst)
        return
    # Pure-Python fallback (WAV inputs only).
    if src.suffix.lower() != ".wav":
        raise AudioError("ffmpeg is required for non-WAV inputs")
    data, rate = _read_wav(src)
    mono = data.mean(axis=1) if channel is None else data[:, min(channel, data.shape[1] - 1)]
    _write_wav(dst, _resample_linear(mono, rate, target_rate), target_rate)
    _restrict(dst)


# -- channel probing ---------------------------------------------------------

# A file can declare two channels and still carry only one recording. Both
# cases below break the channel-split path silently: duplicated channels put
# BOTH speakers on both sides (talk_ratio pinned near 0.5, phantom
# interruptions everywhere), and a dead channel leaves one party with no
# speech at all. Neither is visible in the metadata, only in the samples.
# A single recording saved as stereo is not bit-identical on both channels:
# lossy encoding, a gain trim, a one-sample delay or a DC offset all leave a
# residual, and a phase-inverted copy leaves a very large one. Correlation
# survives all of those, so it decides, and the residual test is kept only as
# a fast path for the exactly-identical case.
DUPLICATE_CORRELATION = 0.98     # |r| this high means one recording, not two
SILENT_CHANNEL_RATIO = 0.01      # channel RMS below 1% of the louder one
PROBE_SECONDS = 240              # total sampled, spread across the whole file
PROBE_WINDOWS = 6                # ... in this many windows


# What a decoder raises on a file it cannot read: wave.Error and EOFError from a
# truncated or malformed header, OSError from an unreadable file, ValueError when
# the frame count and channel count disagree, AudioError for a width we reject.
# On the PROBE path every one of them means the same thing - "this file cannot be
# sampled" - and probe_channels must answer dual_mono, never propagate.
_UNDECODABLE = (wave.Error, EOFError, OSError, ValueError, AudioError)


def _decode_stereo_window(
    src: Path, start: float, seconds: float, rate: int = 8000
) -> np.ndarray | None:
    """Decode one window of a file to (n, 2) float32. None if not stereo.

    Returns None - never raises - when the file cannot be decoded at all. The
    caller treats that as "could not sample the channels" and falls back to the
    diarization path, which infers the speakers and records that it inferred.
    Raising here would abort a call that the safe path could still process, and
    on a machine without ffmpeg it aborted on the stdlib WAV reader alone.
    """
    if _have_ffmpeg():
        try:
            proc = subprocess.run(
                ["ffmpeg", "-v", "error", "-ss", f"{start:.3f}", "-t", f"{seconds:.3f}",
                 "-i", str(src), "-ac", "2", "-ar", str(rate), "-f", "s16le", "-"],
                capture_output=True,
            )
        except OSError as exc:
            # `shutil.which` found something; exec failed anyway. A broken
            # symlink, a binary for the wrong architecture, a PATH entry on a
            # filesystem mounted noexec, or an ffmpeg removed between the check
            # and the call. "Present" and "runnable" are different questions,
            # and the fail-closed promise has to survive the gap between them.
            logger.warning("ffmpeg could not be run for channel probing (%s); "
                           "treating %s as one recording", exc, src.name)
            return None
        if proc.returncode != 0 or not proc.stdout:
            return None
        data = np.frombuffer(proc.stdout, dtype=np.int16).astype(np.float32) / 32768.0
        if data.size < 2:
            return None
        return data[: (data.size // 2) * 2].reshape(-1, 2)
    if src.suffix.lower() != ".wav":
        return None
    try:
        data, src_rate = _read_wav(src)
    except _UNDECODABLE as exc:
        logger.warning("could not decode %s for channel probing (%s: %s)",
                       src.name, type(exc).__name__, exc)
        return None
    if data.shape[1] < 2:
        return None
    lo = int(start * src_rate)
    return data[lo:lo + int(seconds * src_rate), :2]


def _decode_stereo(src: Path, duration: float = 0.0) -> np.ndarray | None:
    """Sample the file in windows spread across its whole length.

    Looking only at the opening misreads a call that is dual-mono while the
    agent is alone on the line and genuinely two-channel once the customer
    joins, and the mirror case hands one party several minutes of silence.
    """
    if duration <= PROBE_SECONDS:
        return _decode_stereo_window(src, 0.0, max(duration, PROBE_SECONDS))
    window = PROBE_SECONDS / PROBE_WINDOWS
    step = (duration - window) / max(1, PROBE_WINDOWS - 1)
    chunks = [
        chunk for i in range(PROBE_WINDOWS)
        if (chunk := _decode_stereo_window(src, i * step, window)) is not None
        and len(chunk)
    ]
    if not chunks:
        return None
    return np.concatenate(chunks, axis=0)


def probe_channels(src: Path, duration: float = 0.0) -> str:
    """Classify what a two-channel file actually contains.

    Returns "stereo" (two genuinely different recordings), "dual_mono" (the
    same recording on both channels, however it was re-encoded or trimmed) or
    "single_channel" (one side effectively silent).

    A file that cannot be decoded is reported as dual_mono, not stereo: taking
    the channel split on a file nothing could read would attribute the call by
    guesswork, while the diarization path infers the speakers and says that it
    did.
    """
    samples = _decode_stereo(src, duration)
    if samples is None or len(samples) == 0:
        logger.warning("could not sample the channels of %s; treating it as one "
                       "recording", src.name)
        return "dual_mono"
    left, right = samples[:, 0], samples[:, 1]
    left = left - left.mean()                  # a DC offset is not a speaker
    right = right - right.mean()
    rms_l, rms_r = float(np.sqrt((left ** 2).mean())), float(np.sqrt((right ** 2).mean()))
    loud = max(rms_l, rms_r)
    if loud < 1e-6:
        return "dual_mono"       # silence throughout; nothing to split
    if min(rms_l, rms_r) < SILENT_CHANNEL_RATIO * loud:
        return "single_channel"
    correlation = float(np.dot(left, right) / (len(left) * rms_l * rms_r))
    if abs(correlation) >= DUPLICATE_CORRELATION:
        # abs(): a phase-inverted copy is still one recording, and it is the
        # case a plain difference test is worst at.
        return "dual_mono"
    return "stereo"


# -- VAD ---------------------------------------------------------------------

# About -46 dBFS: below this, a 16-bit phone recording is room tone.
ABSOLUTE_SPEECH_RMS = 0.005
# Speech alternates between sound and silence, so its quietest frames are far
# below its loudest. A signal whose floor is close to its peak is a tone, hold
# music or a clipped line - loud throughout, and not a conversation. The
# threshold is deliberately high: someone who never pauses still falls well
# under it.
FLAT_LEVEL_RATIO = 0.8


def energy_vad(
    wav_path: Path,
    min_speech_ms: int = 250,
    frame_ms: int = 20,
    merge_gap_ms: int = 150,
) -> list[VADSegment]:
    """Dependency-free energy-based VAD over a 16 kHz mono WAV."""
    data, rate = _read_wav(wav_path)
    mono = data[:, 0]
    frame_len = max(1, int(rate * frame_ms / 1000))
    n_frames = len(mono) // frame_len
    if n_frames == 0:
        return []
    frames = mono[: n_frames * frame_len].reshape(n_frames, frame_len)
    rms = np.sqrt((frames ** 2).mean(axis=1))
    floor = float(np.percentile(rms, 10))
    peak = float(np.percentile(rms, 95))
    # A purely relative threshold has no idea what silence is: a recording of
    # nothing but room tone gets one scaled to room tone and reports the whole
    # file as speech. An absolute floor and a required dynamic range give it
    # the missing reference.
    threshold = max(ABSOLUTE_SPEECH_RMS, floor + 0.15 * (peak - floor))
    if peak < ABSOLUTE_SPEECH_RMS or floor > FLAT_LEVEL_RATIO * peak:
        # Either nothing here is loud enough to be speech, or the level never
        # changes (hold music, a tone, clipping) - both mean "no speech found"
        # rather than "speech throughout".
        return []
    speech = rms > threshold

    segments: list[VADSegment] = []
    start: int | None = None
    for i, is_speech in enumerate(speech):
        if is_speech and start is None:
            start = i
        elif not is_speech and start is not None:
            segments.append(VADSegment(start=start * frame_ms / 1000, end=i * frame_ms / 1000))
            start = None
    if start is not None:
        segments.append(VADSegment(start=start * frame_ms / 1000, end=n_frames * frame_ms / 1000))

    # Merge segments separated by tiny gaps, then drop too-short segments.
    merged: list[VADSegment] = []
    for seg in segments:
        if merged and (seg.start - merged[-1].end) * 1000 <= merge_gap_ms:
            merged[-1] = VADSegment(start=merged[-1].start, end=seg.end)
        else:
            merged.append(seg)
    min_len = min_speech_ms / 1000
    return [s for s in merged if s.duration >= min_len]


def silero_vad(wav_path: Path, min_speech_ms: int = 250) -> list[VADSegment]:
    """Silero VAD (lazy import; weights ship inside the silero-vad wheel).

    The file is read here rather than through silero's own read_audio: that
    helper goes via torchaudio, which in current versions refuses to decode
    anything without torchcodec installed. By this point the input is already
    16 kHz mono PCM, so the wave module is both sufficient and one less
    dependency to pin.
    """
    import torch
    from silero_vad import get_speech_timestamps, load_silero_vad

    model = _get_silero_model(load_silero_vad)
    samples, _ = _read_wav(wav_path)
    audio = torch.from_numpy(samples[:, 0].copy())
    stamps = get_speech_timestamps(
        audio, model, return_seconds=True, min_speech_duration_ms=min_speech_ms
    )
    return [VADSegment(start=float(s["start"]), end=float(s["end"])) for s in stamps]


_SILERO_MODEL = None


def _get_silero_model(loader):  # noqa: ANN001 - loader typed by lazy import
    global _SILERO_MODEL
    if _SILERO_MODEL is None:
        _SILERO_MODEL = loader()
    return _SILERO_MODEL


def run_vad(wav_path: Path, config: Config) -> tuple[str, list[VADSegment]]:
    engine = config.audio.vad
    if engine == "silero" and not config.run.mock:
        try:
            return "silero", silero_vad(wav_path, config.audio.min_speech_ms)
        except ImportError:
            logger.warning("silero-vad not installed; falling back to energy VAD")
        except Exception as exc:  # noqa: BLE001
            # A VAD that cannot load must not fail the call: energy VAD is a
            # real fallback, and losing the call is worse than losing accuracy.
            logger.warning("silero VAD failed (%s); falling back to energy VAD", exc)
    return "energy", energy_vad(wav_path, config.audio.min_speech_ms)


# -- stage entrypoint --------------------------------------------------------

def resolve_banker_channel(call: CallInput, config: Config) -> int:
    """Return the 0-based channel index for the banker (0=L, 1=R)."""
    setting = config.speakers.banker_channel
    if setting == "from_metadata":
        channel = call.banker_channel or "L"
        if call.banker_channel is None:
            logger.warning(
                "call_id=%s: banker_channel not in metadata; defaulting to L", call.call_id
            )
    else:
        channel = setting
    return 0 if channel == "L" else 1


def prepare_audio(
    call: CallInput, meta: CallMeta, config: Config, wav_dir: Path
) -> AudioArtifact:
    """Convert / split audio to 16 kHz mono WAV(s) and run VAD per channel."""
    rate = config.audio.target_sample_rate
    wav_dir.mkdir(parents=True, exist_ok=True)
    try:
        wav_dir.chmod(0o700)
    except OSError:  # pragma: no cover
        pass

    if meta.channels > 2:
        # Classified from a downmix but extracted with a two-channel pan, so
        # the extra channels would vanish without a word.
        logger.warning("call_id=%s: file has %d channels; only the first two are "
                       "used", call.call_id, meta.channels)
    # `speakers.mode` was a documented setting that no code read: an operator
    # who set it to force one path got the automatic behaviour anyway, silently.
    # It is honoured here, where the decision is actually made.
    #   auto   - probe the channels and decide (the default, and what to use)
    #   stereo - trust the two channels; for a recorder known to keep the
    #            parties apart on a file the probe misreads
    #   mono   - always diarize; for a recorder whose "stereo" is two
    #            microphones in one room, where a channel split is meaningless
    if config.speakers.mode == "mono":
        layout = "mono"
        logger.info("call_id=%s: speakers.mode=mono; diarizing regardless of the "
                    "file's channels", call.call_id)
    elif config.speakers.mode == "stereo":
        if meta.channels < 2:
            raise AudioError(
                f"speakers.mode='stereo' but {call.audio_path.name} has "
                f"{meta.channels} channel(s). Set speakers.mode to auto or mono."
            )
        layout = "stereo"
        logger.info("call_id=%s: speakers.mode=stereo; splitting channels without "
                    "probing them", call.call_id)
    else:
        layout = (probe_channels(call.audio_path, meta.duration_sec)
                  if meta.channels >= 2 else "mono")
    if layout != "stereo":
        # Two declared channels that carry one recording: the split would
        # produce two copies of the same conversation, so downmix and let the
        # diarizer separate the speakers instead.
        if meta.channels >= 2:
            logger.warning(
                "call_id=%s: file declares %d channels but they are %s; "
                "treating it as a mono recording (speaker attribution will need "
                "diarization)", call.call_id, meta.channels, layout,
            )

    if layout == "stereo":
        banker_idx = resolve_banker_channel(call, config)
        customer_idx = 1 - banker_idx
        banker_wav = wav_dir / f"{call.call_id}.banker.wav"
        customer_wav = wav_dir / f"{call.call_id}.customer.wav"
        _extract_channel(call.audio_path, banker_wav, banker_idx, rate)
        _extract_channel(call.audio_path, customer_wav, customer_idx, rate)
        vad_engine, banker_segments = run_vad(banker_wav, config)
        _, customer_segments = run_vad(customer_wav, config)
        artifact = AudioArtifact(
            call_id=call.call_id,
            is_stereo=True,
            banker_wav=str(banker_wav),
            customer_wav=str(customer_wav),
            sample_rate=rate,
            vad_engine=vad_engine,
            channel_layout="stereo",
            banker_segments=banker_segments,
            customer_segments=customer_segments,
        )
    else:
        mono_wav = wav_dir / f"{call.call_id}.mono.wav"
        _extract_channel(call.audio_path, mono_wav, None, rate)
        vad_engine, mono_segments = run_vad(mono_wav, config)
        artifact = AudioArtifact(
            call_id=call.call_id,
            is_stereo=False,
            mono_wav=str(mono_wav),
            sample_rate=rate,
            vad_engine=vad_engine,
            channel_layout=layout,
            mono_segments=mono_segments,
        )
    logger.info(
        "audio done: call_id=%s layout=%s vad=%s segments=%d",
        call.call_id,
        artifact.channel_layout,
        vad_engine,
        len(artifact.banker_segments) + len(artifact.customer_segments) + len(artifact.mono_segments),
    )
    return artifact
