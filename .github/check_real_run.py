"""Assertions for the Windows real-engines workflow (not shipped logic).

`callqa run` exits 0 with a call in needs_human_review, and a judge that never
answered ALSO ends in needs_human_review - so the exit code alone cannot tell
"the real stack works on Windows" from "everything timed out politely". This
reads the run's own log and artifacts and fails loudly unless each engine
demonstrably did its job.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

log = Path(sys.argv[1]).read_text(encoding="utf-8", errors="replace")
call_id = sys.argv[2]
out = Path("data/output")
problems: list[str] = []


def need(condition: bool, message: str) -> None:
    if not condition:
        problems.append(message)


result = json.loads((out / "results" / f"{call_id}.json").read_text(encoding="utf-8"))
print("status:", result["status"], "| error:", result.get("error"))
need(result["status"] in ("success", "needs_human_review"),
     f"the call {result['status']}: {result.get('error')}")
need("faster-whisper model loaded" in log, "the real ASR model was never loaded")
need("no usable CUDA GPU" in log,
     "the CPU fallback for asr.compute_type was not exercised")
need("judge call failed" not in log,
     "the judge endpoint failed at the transport level (see run.log)")
need("judge done" in log or "judge validation failed" in log,
     "no judge response was ever received and parsed")

redacted = json.loads((out / "redacted" / f"{call_id}.json").read_text(encoding="utf-8"))
words = sum(len(t.get("text", "").split()) for t in redacted["turns"])
speakers = {t.get("speaker") for t in redacted["turns"]}
print(f"redacted transcript: {len(redacted['turns'])} turns, {words} words, "
      f"speakers {sorted(map(str, speakers))}")
need(words >= 10, "the ASR produced (almost) no words from real speech")
need({"banker", "customer"} <= speakers, "the stereo split did not yield both speakers")
need(redacted.get("enabled") is True, "redaction did not run")

if result["status"] == "success":
    report = out / "reports" / "calls" / f"{call_id}.html"
    need(report.is_file() and report.read_text(encoding="utf-8").rstrip().endswith("</html>"),
         "no complete HTML report for the call")
else:
    # A call held for review has no scorecard, so no score report. With the
    # 0.5B test judge that is the expected outcome: its quotes rarely survive
    # evidence verification. What must hold is that it ANSWERED, in the
    # schema, and that the answer was checked - "judge validation failed".
    need("judge validation failed" in log,
         "the call was held for review, but not because a judge answer was checked")

if problems:
    print("\nREAL-ENGINE RUN ON WINDOWS FAILED:")
    for p in problems:
        print("  -", p)
    sys.exit(1)
print("\nreal engines on Windows: OK")
