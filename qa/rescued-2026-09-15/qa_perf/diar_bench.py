"""Diarization benchmark. Usage:
   diar_bench.py <name> <num_speakers:int|0> <mode:full|vadcrop> <out.json>
Times pyannote 3.1 with a step hook (segmentation / embeddings / rest).
"""
import json
import resource
import sys
import time

NAME, NSPK, MODE, OUT = sys.argv[1], int(sys.argv[2]), sys.argv[3], sys.argv[4]
WAV = "/private/tmp/claude-501/-Users-yonatanohayon-Desktop-BRS-Model-main/e125e1e3-6108-476d-ab24-71bc406394de/scratchpad/qa_perf/REAL002.mono.wav"
VADJSON = "/Users/yonatanohayon/Desktop/BRS_Model-main/data/output/audio/REAL002.json"

import torch  # noqa: E402
from pyannote.audio import Pipeline  # noqa: E402

torch.set_num_threads(torch.get_num_threads())  # default

t0 = time.perf_counter()
pipe = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1")
t_load = time.perf_counter() - t0

step_times = {}


def hook(step_name, step_artifact=None, file=None, total=None, completed=None):
    now = time.perf_counter()
    rec = step_times.setdefault(step_name, {"first": now, "last": now, "calls": 0})
    rec["last"] = now
    rec["calls"] += 1


inp = WAV
mapping = None
if MODE == "vadcrop":
    import wave

    import numpy as np
    with wave.open(WAV, "rb") as wf:
        rate = wf.getframerate()
        raw = wf.readframes(wf.getnframes())
    data = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    segs = json.load(open(VADJSON))["mono_segments"]
    # concatenate speech regions with 0.2 s padding each side, merged
    pad = 0.2
    regions = []
    for s in segs:
        a, b = max(0.0, s["start"] - pad), min(len(data) / rate, s["end"] + pad)
        if regions and a <= regions[-1][1]:
            regions[-1][1] = max(regions[-1][1], b)
        else:
            regions.append([a, b])
    pieces = [data[int(a * rate):int(b * rate)] for a, b in regions]
    cat = np.concatenate(pieces)
    mapping = regions
    inp = {"waveform": torch.from_numpy(cat).unsqueeze(0), "sample_rate": rate}
    print("vadcrop: %.1fs of %.1fs kept in %d regions"
          % (len(cat) / rate, len(data) / rate, len(regions)))

kw = {"hook": hook}
if NSPK:
    kw["num_speakers"] = NSPK

t1 = time.perf_counter()
ann = pipe(inp, **kw)
t_diar = time.perf_counter() - t1

turns = [(round(t.start, 2), round(t.end, 2), lab)
         for t, _, lab in ann.itertracks(yield_label=True)]
steps = {k: {"span_sec": round(v["last"] - v["first"], 2), "calls": v["calls"],
             "t_first": round(v["first"] - t1, 2)}
         for k, v in step_times.items()}
rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
res = {
    "name": NAME, "num_speakers_arg": NSPK, "mode": MODE,
    "load_sec": round(t_load, 2), "diarize_sec": round(t_diar, 2),
    "steps": steps, "n_turns": len(turns),
    "n_speakers_found": len({t[2] for t in turns}),
    "speech_attributed_sec": round(sum(b - a for a, b, _ in turns), 1),
    "peak_rss_mb": round(rss / 1e6, 1),
    "turns": turns, "vadcrop_regions": mapping,
}
json.dump(res, open(OUT, "w"), ensure_ascii=False, indent=1)
print(NAME, "load", res["load_sec"], "diarize", res["diarize_sec"], "steps", steps,
      "speakers", res["n_speakers_found"], "rss_mb", res["peak_rss_mb"])
