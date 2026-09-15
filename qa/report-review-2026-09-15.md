# Report & dashboard review — a bank reviewer's read (2026-09-15)

The fourth overnight QA agent (report/dashboard quality from a bank
reviewer's point of view) never produced anything; this is that review,
done against the rendered REAL002 report (real ASR + diarization) and the
CALL003 mock report, plus the live dashboard.

## What was fixed as a result (this commit)

1. **Interruption counts on a mono call read as measurements.** REAL002's
   features carry `overlap_metrics_available: false` - exclusive
   diarization cannot see overlap - yet the report table printed
   "קטיעות של הבנקאי את הלקוח: 0". A reviewer reads that as "this banker
   never interrupts", which is precisely the "zero presented as a
   measurement" failure the features stage was designed to avoid; the flag
   existed and the template ignored it. The rows now say
   "לא ניתן למדוד בהקלטה חד-ערוצית" on mono and keep the numbers on
   stereo. Regression tests cover both.

2. **The report had no transcript.** The static HTML report is the bank
   deliverable (CLI + static HTML - no dashboard on the bank's side), and
   a reviewer deciding whether "אמפתיה 2/5" is fair had exactly one quote
   per dimension to go on. The report now carries the full REDACTED
   transcript in a collapsible section: speaker labels, mm:ss timestamps,
   mask tokens visible (seeing what was removed is the point), and the
   judge's evidence quotes highlighted where they occur with the dimension
   named on hover. Redaction-disabled artifacts contribute no transcript.
   This mirrors the dashboard drawer feature landed the same day.

## Observed and deliberately not changed

- The mock banner, review-hold banner and attribution banner (added in
  2a603d9) read correctly on both reports; the score formula and the gate
  rule are stated where the numbers are.
- The evidence quote shown for "סגירה והצעד הבא" on REAL002 is the
  opening greeting - a mock-judge artifact, covered by the mock banner;
  the real-judge run replaces it.
- The stored REAL002 report still shows the pre-normaliser redaction
  (the `415 / 926 -9265` ID fragments are visible in the transcript
  section). The artifact regenerates - with them masked - on the next
  forced run; the normaliser tests already pin the behavior.
- Mixed digits-in-RTL display ("9265- 926") is the browser's bidi
  ordering of the ASR's own hyphenation; adding dir attributes per-number
  would touch every text path for cosmetic gain. Left alone.
- Dashboard: transcript drawer, state endpoint and reports all pass the
  visual QA harness at 0 findings (light/dark x desktop/phone). The two
  HANDOVER questions that belong to the product owner remain open: is
  #ea580c the bank's real brand orange, and does the Overview lead with
  the right three things.
