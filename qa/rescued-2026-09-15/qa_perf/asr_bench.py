"""ASR benchmark: one variant per process. Usage:
   asr_bench.py <name> <beam> <vad:0|1> <cpu_threads:int|0=default> <clip:0|1> <out.json>
"""
import json
import resource
import sys
import time

NAME, BEAM, VAD, THREADS, CLIP, OUT = (
    sys.argv[1], int(sys.argv[2]), sys.argv[3] == "1", int(sys.argv[4]),
    sys.argv[5] == "1", sys.argv[6],
)
WAV = "/private/tmp/claude-501/-Users-yonatanohayon-Desktop-BRS-Model-main/e125e1e3-6108-476d-ab24-71bc406394de/scratchpad/qa_perf/REAL002.mono.wav"
MODEL = "/Users/yonatanohayon/Desktop/BRS_Model-main/models/ivrit-whisper-large-v3-turbo-ct2"
VADJSON = "/Users/yonatanohayon/Desktop/BRS_Model-main/data/output/audio/REAL002.json"

from faster_whisper import WhisperModel  # noqa: E402

t0 = time.perf_counter()
kw = {}
if THREADS:
    kw["cpu_threads"] = THREADS
model = WhisperModel(MODEL, device="cpu", compute_type="int8", local_files_only=True, **kw)
t_load = time.perf_counter() - t0

tkw = dict(language="he", word_timestamps=True, beam_size=BEAM, vad_filter=VAD)
if CLIP:
    segs = json.load(open(VADJSON))["mono_segments"]
    clip = []
    for s in segs:
        clip += [s["start"], s["end"]]
    tkw["clip_timestamps"] = clip

t1 = time.perf_counter()
segments_iter, info = model.transcribe(WAV, **tkw)
out_segs = []
for seg in segments_iter:
    out_segs.append({
        "start": round(seg.start, 2), "end": round(seg.end, 2),
        "text": seg.text.strip(), "avg_logprob": round(seg.avg_logprob, 3),
        "n_words": len(seg.words or []),
    })
t_trans = time.perf_counter() - t1

rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss  # bytes on macOS
res = {
    "name": NAME, "beam": BEAM, "vad_filter": VAD, "cpu_threads": THREADS,
    "clip": CLIP, "load_sec": round(t_load, 2), "transcribe_sec": round(t_trans, 2),
    "duration_after_vad": getattr(info, "duration_after_vad", None),
    "n_segments": len(out_segs),
    "n_words": sum(s["n_words"] for s in out_segs),
    "peak_rss_mb": round(rss / 1e6, 1),
    "text": " ".join(s["text"] for s in out_segs),
    "segments": out_segs,
}
json.dump(res, open(OUT, "w"), ensure_ascii=False, indent=1)
print(NAME, "load", res["load_sec"], "transcribe", res["transcribe_sec"],
      "words", res["n_words"], "rss_mb", res["peak_rss_mb"])
