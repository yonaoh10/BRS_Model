# QA Round 3 — summary and disposition (2026-09-16)

Four adversarial read-only reviews ran against everything new since `6625eab`
(number-word normaliser, parallel diarization, dashboard/report transcript,
network-volume cloud). Each agent's full findings are in the sibling files
(`redaction.md`, `judge.md`, `dashboard.md`, `pipeline.md`). This is the
disposition — what was fixed, with the commit and regression test, and what was
consciously accepted.

## Fixed (with a regression test each)

| ID | Sev | Area | Fix |
|----|-----|------|-----|
| redaction F1 | HIGH | amount exception leaked account/ID numbers near a currency word | 9+ digit runs and explicit-identifier-phrase runs override the amount exception |
| redaction F3 | HIGH | a combining mark (niqqud) inside a digit run leaked the identifier | normalize_for_detection drops Cf/Mn/Me before detection |
| redaction F2 | MED | `_repeated_fragments` over-masked amounts/years sharing digits | suffix (read-back tail) match + amount check |
| judge HIGH-1 | HIGH | 0.90 snap accepted meaning-inverted quotes (flipped fee, dropped negation) | snap must preserve numbers and negations verbatim, else -> human review |
| judge MED-2 | MED | un-schema'd prose could carry markup | `_scrub` strips HTML tag-like runs |
| judge MED-3 | MED | timestamp regex/schema digit-count mismatch | aligned to 1-4 minute digits |
| judge LOW-6 | LOW | `_chat` returned None on content:null | raises instead (contract is -> str) |
| pipeline F1 | HIGH | `release_lock` deleted any owner's lock -> double-processing | delete only our own pid+hostname |
| pipeline F2 | HIGH | pod created/billing before state saved -> strand | volume + pod ids persisted the instant they exist |
| pipeline F15 | HIGH | wheel omitted config/*.yaml -> every command failed | ships `config_defaults/*.yaml`; find_config fallback; drift test |
| pipeline F7 | MED | losing one stage left downstream stale | recomputing a stage forces every later stage to recompute |
| pipeline F8 | MED | SIGKILL of parent orphaned the diar worker (held GPU) | worker runs a parent-death watchdog |
| pipeline F9 | LOW | empty worker output consumed as real | falls back to in-process |
| pipeline F14 | INFO | duplicate WATCHED_EXTENSIONS | removed |
| config F12 | LOW | typo'd config SECTION silently ignored | Config is StrictModel; env parser ignores non-section vars |
| runpod F3 | MED | `down` claimed success without confirming stop | re-reads status, warns if still RUNNING |
| runpod F4 | MED | out-of-band pod removal deadlocked the CLI | 404 treated as gone; state cleared; `up` recreates |
| runpod F13 | LOW | hostile-cwd .env egress redirect was silent | egress endpoints from a .env are logged loudly |
| dashboard MED-1 | MED | /api/transcript had no size guard | caps turns + per-turn length, truncated flag |
| dashboard MED-2 | MED | red total next to green gate badge confused reviewers | added a legend: score band is separate from the gate |
| judge MED-4 | MED | hardcoded 600s judge timeout | configurable `JudgeConfig.request_timeout_sec` |
| pipeline F11 | LOW | ct2/torch order was silent-until-segfault | logs a warning if torch is already resident when the engine is built |

## Accepted, with reason (not fixed)

- **redaction F4** (LOW, O(n²) in identifier count): a soft-DoS only on a
  pathological all-identifiers transcript (measured 4.8 s at 8000 ids / 216k
  chars); a real 2 h call is ~2.5 s. The suffix-match fix (F2) also shrank the
  candidate set. Not worth complicating the hot path.
- **redaction F1 residual** (the bare "החשבון 481902 שקלים" 6-8 digit case): a
  number in that length range with only bare account context and a trailing
  currency word is treated as an amount by design — masking it would blind the
  compliance/clarity scoring on balances/fees, which are discussed constantly.
  The explicit-identifier-phrase escape ("מספר החשבון", "מסתיים ב", 9+ digits)
  covers the unambiguous account-number cases. This is the documented
  privacy/utility line, not an oversight.
- **judge MED-5** (the "27B clears the bar" claim is n=1): correct — it is an
  assessment, not a code defect. The landed gemma re-run of REAL002 adds a
  second data point; broad reliability still needs the QWK calibration set.
- **pipeline F5** (no proactive billing poller): by design the meter-stop
  (`down`) and the warning (`status`) are manual. The network-volume migration
  makes a forgotten pod cheaper to recover from, and `status` now always prints
  the standing storage cost. A background biller-watchdog is out of scope for a
  removable dev-only tool.
- **pipeline F10** (darwin eager CT2 load fires on resumed runs too): darwin
  dev-machine only, ~1 GB of wasted RSS on a resume; the bank's Linux target
  never takes the branch. Not a correctness issue; fixing it would require the
  engine to know at construction whether the ASR stage will run.
- **dashboard MED-3 / MED-4 and the LOW cosmetics** (wrong evidence quotes;
  role-confidence vs signal table; `<mark>` nesting on a quote equal to a
  wrapper substring; mask-prefix legend): the wrong evidence quotes are a
  MOCK-judge artifact and are replaced by the real gemma scorecard; the
  role-confidence normalisation (|margin| / total signal weight) is the
  deliberate, documented choice from HANDOVER §5; the `<mark>` nesting is a
  rendering-only cosmetic (the agent confirmed it is NOT an XSS), on a quote a
  real judge never emits. Left as-is.

## Not reproduced / confirmed sound

Each agent's "Checked and sound" section records what was attacked and held:
the index-map mapping (80k-string property fuzz, zero shift/drop/duplicate),
the end-to-end raw-PII grep (clean across every artifact, log and endpoint),
cross-speaker snap protection, mask-token verbatim + digit-reconstruction
rejection, bounded retries always landing a failing judge in
needs_human_review, resume correctness, atomic writes, and the dashboard's
round-1 controls (token/Origin/Host/traversal/CSP).
