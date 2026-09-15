"""cProfile the baseline ASR transcribe to see where time goes inside faster-whisper."""
import cProfile
import io
import pstats

from faster_whisper import WhisperModel

WAV = "/private/tmp/claude-501/-Users-yonatanohayon-Desktop-BRS-Model-main/e125e1e3-6108-476d-ab24-71bc406394de/scratchpad/qa_perf/REAL002.mono.wav"
MODEL = "/Users/yonatanohayon/Desktop/BRS_Model-main/models/ivrit-whisper-large-v3-turbo-ct2"

model = WhisperModel(MODEL, device="cpu", compute_type="int8", local_files_only=True)

def run():
    it, _ = model.transcribe(WAV, language="he", word_timestamps=True,
                             beam_size=5, vad_filter=True)
    return sum(1 for _ in it)

pr = cProfile.Profile()
pr.enable()
n = run()
pr.disable()
s = io.StringIO()
ps = pstats.Stats(pr, stream=s).sort_stats("cumulative")
ps.print_stats(30)
print(s.getvalue()[:6000])
print("segments:", n)
