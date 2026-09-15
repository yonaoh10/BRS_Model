# Judge & rubric calibration QA — conclusions

The overnight agent left three experiment scripts and died before writing
any findings. This document is the missing half: the scripts were run
(2026-09-15, post-crash session) against the current code and REAL002's real
redacted transcript, and every observation below comes from that output.

## exp1 — scoring arithmetic (PASS, exact)

Hand-computed totals match `weighted_total` on every case: linear map
(1→0, 3→50, 5→100), per-dimension weights summing to 1.0, weighted sum.
Gate behavior is exactly as specified: a gate dimension at 2 caps the total
at 59.0 (verified for both gates individually, both at once, and combined
with otherwise-perfect scores); a gate at 3 does NOT trip (all-5s +
identification=3 → 92.5, uncapped). REAL002's stored mock total (56.2)
reproduces by hand (56.25): weights and rounding are faithful.

Calibration observation: all-3s ("meets expectations" across the board)
lands at exactly 50.0 — the midpoint, not a passing-feeling 70. Anyone
presenting these totals to bankers should know the scale is anchored to
1=0 / 3=50 / 5=100; the per-call report already spells this out.

## exp3 — evidence verification (PASS, boundary confirmed)

- The 0.90 similarity snap threshold behaves exactly per theory on a
  60-char quote: ≤6 substituted characters snaps to the real span, ≥8 is
  rejected. The rejection message and the snap coexist correctly.
- Cross-SPEAKER splices are rejected in both directions; cross-TURN
  same-speaker quotes (2 and 3 turns) verify — the same-speaker-block
  merge did its job.
- Quote supply after block-merging is healthy: 10 of 51 raw turns are
  individually below MIN_QUOTE_CHARS=8, but only 1 of 23 same-speaker
  blocks is unquotable. Every rubric dimension's natural evidence window
  on REAL002 has multiple quotable blocks (identification: 9, closure: 6).
- Mask tokens: a quote containing `<חשבון:████>` verifies verbatim; a
  near-quote that OMITS the mask snaps to the real masked span; a quote
  where the judge replaced the mask with digits (reconstructing masked
  content) is rejected. All three are the desired outcomes — the last one
  doubles as an anti-leak guard.

## exp4 — misattribution and relevance (one real limitation)

- A banker quote attributed to the customer is rejected; a phrase said in
  variants by both parties is checked against the attributed speaker's own
  turns. No false accepts found.
- **The one structural limitation: relevance is not verified.** A judge
  that quotes the same verbatim greeting as "evidence" for all eight
  dimensions passes verification — the verifier checks existence,
  speaker and timestamp, and cannot check that the quote supports the
  reasoning. Score trustworthiness therefore still rests on judge model
  quality; this is exactly why dictalm2.0-7B (which failed even the
  verbatim bar) is below the floor, and why the gemma-3-27b run (82.5,
  zero retries) matters. If this ever needs closing, it is a semantic
  check (e.g. a second cheap judge pass over quote↔reasoning pairs), not
  a regex.

## Redaction cross-check (fixed since the scripts were written)

exp4's dump of unmasked digit runs in REAL002's stored redacted artifact
shows the cross-turn ID fragments (`עד 415` / `926 -9265`) in the clear.
That artifact predates the Hebrew number-word normaliser and the
dictation-context branch (commit "Fold Hebrew number-words to digits...");
re-running redaction on the same dialog now masks them (verified in that
commit's tests and on the real transcript). The remaining visible numbers
are amounts (10/30/120,000 שקל — deliberately unmasked) and the 3-digit
branch number, which is below every masking threshold by design.

## Verdict

Arithmetic exact, gate exact, evidence verification does what it claims at
its published thresholds, no false accepts found in the adversarial cases.
The open calibration risk is judge-model quality (relevance of evidence and
score judgment), not the verification machinery — which is what the QWK
calibration against human ratings (still requiring ≥20 rated calls) is for.
