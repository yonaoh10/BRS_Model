"""Measure the silero VAD path exactly as callqa.audio.silero_vad runs it."""
import resource
import sys
import time

sys.path.insert(0, "/Users/yonatanohayon/Desktop/BRS_Model-main/src")

WAV = "/private/tmp/claude-501/-Users-yonatanohayon-Desktop-BRS-Model-main/e125e1e3-6108-476d-ab24-71bc406394de/scratchpad/qa_perf/REAL002.mono.wav"

r0 = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
from callqa.audio import silero_vad  # noqa: E402

t0 = time.perf_counter()
segs = silero_vad(__import__("pathlib").Path(WAV))
t = time.perf_counter() - t0
r1 = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
print("silero_vad: %.2fs, %d segments, peak_rss=%.1f MB (baseline before torch import %.1f MB)"
      % (t, len(segs), r1 / 1e6, r0 / 1e6))
