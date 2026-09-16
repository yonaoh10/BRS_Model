# QA Round 3 — Redaction adversarial review (2026-09-16)

Reviewer: adversarial security review (read-only).
Target: Hebrew number-word normaliser + index-map in `src/callqa/redaction.py`.

Status: COMPLETE. 4 findings (2 HIGH, 1 MEDIUM, 1 LOW). Index-map mapping and end-to-end
grep both came out clean. See RANKING at the bottom.

---

## Findings (unranked, appended as found)

### F1 — HIGH — Amount exception leaves a 9-digit account/ID number unmasked when a currency word follows within 14 chars
- File: `src/callqa/redaction.py:310-312` (`_looks_like_amount`) and `_classify_run` ordering at `:350-364`.
- The amount check (`_looks_like_amount`, returns None) runs BEFORE the ACCOUNT_LIKE
  (`:352`) and the dictation-context ID/account branch (`:361-364`). Any number that is
  not checksum-valid (so it misses the earlier ID/phone/card gates) is therefore
  suppressed the moment a currency word from `AMOUNT_AFTER` appears in the 14-char
  window after it.
- Repro (mock/regex path):
  - `מספר החשבון 761534892 שקל`  -> OUT: `מספר החשבון 761534892 שקל` (UNMASKED). 761534892 is a full 9-digit account/ID, id/account context present, but leaks because "שקל" follows.
  - `תעודת זהות 481902123 שקל`   -> UNMASKED (id-context 9-digit number, checksum-invalid, leaks).
  - `החשבון שלי 481902 שקלים`    -> UNMASKED.
  - `החשבון 481902 אלף`          -> UNMASKED ("אלף"/thousand is in AMOUNT_AFTER, so it doubles as an account-number suppressor).
- Why it matters: bank calls discuss account numbers and amounts in the same breath.
  Any account/customer number spoken adjacent to a currency word (or "אלף"/"מיליון")
  is left in the clear in every downstream artifact. Checksum-valid Israeli IDs/phones/
  cards are safe (their gates precede the amount check); non-checksum account numbers
  and the dictation-context ID branch are not.

### F2 — MEDIUM — `_repeated_fragments` over-masks amounts/years that share 4-5 digits with a masked identifier
- File: `src/callqa/redaction.py:400-420` (`_repeated_fragments`).
- The read-back rule masks ANY 4-5 digit run whose digits are a substring of a masked
  >=6-digit identifier. It consults no amount check, no context, no currency window.
- Repro: `חשבון 761534 ומחיר 1534 שקל`
  -> OUT: `חשבון <חשבון:████> ומחיר <חשבון:████> שקל`
  The price "1534 שקל" ("מחיר"=price, "שקל" follows) is masked purely because "1534"
  is a substring of the masked "761534". The brief states amounts must never be masked.
- Also masks incidental years/percentages/dates that happen to be a digit-substring of
  any masked identifier in the same dialog (e.g. an ID containing "2015" would cause a
  spoken year 2015 to be masked). Blinds the compliance/clarity scoring the judge reads.
- Direction: false positive (over-mask). Lower severity than a leak, but it silently
  removes quotable numeric evidence.

### F3 — HIGH — A combining mark (or any non-INVISIBLE format/mark char) inside a digit run splits it and leaks the identifier
- File: `src/callqa/redaction.py:124-127` (`INVISIBLE` set) and `:130-149` (`normalize_for_detection`).
- `normalize_for_detection` only strips a hardcoded `INVISIBLE` frozenset (bidi/zero-width/
  soft-hyphen). It does NOT strip Unicode combining marks (category Mn, e.g. Hebrew niqqud
  U+05B0-U+05BF, or U+0301). Such a char is neither invisible-stripped nor a digit nor a
  `SEPARATORS` member, so it breaks a digit run into two shorter runs that each fall below
  the masking threshold.
- Repro: `תעודת זהות 12345́6782` (U+0301 between "12345" and "6782")
  -> OUT: `תעודת זהות 12345́6782` (UNMASKED). The full valid ID 123456782 leaks.
  Same with an emoji inside the run (`12345😀6782`), though emoji mid-number is implausible;
  Hebrew niqqud between digits is realistic ASR/keyboard output.
- Why it matters: a single combining codepoint anywhere inside a dictated/typed identifier
  defeats detection entirely — the same class of leak the INVISIBLE handling was built to
  stop, just one Unicode category wider. Fix direction: strip `unicodedata.category(ch)`
  in {'Cf','Mn','Me'} (or treat them as separators), not a hardcoded list.
- Note: the index-map MAPPING is sound for these chars (bidi/zwsp at edges/inside map
  correctly, arabic-indic folds and masks); this is a DETECTION gap, not a mapping bug.


---

### F4 — LOW — `_repeated_fragments` makes `find_pii` O(n^2) in the number of identifiers
- File: `src/callqa/redaction.py:414-415`.
- For every digit run it scans all existing spans (`any(... for s in spans)`), giving
  O(runs x spans). Measured `redact_text` on a synthetic transcript of repeated IDs:
  1000 ids/27k chars = 0.10s; 2000/54k = 0.39s; 4000/108k = 1.22s; 8000/216k = 4.76s
  (doubling input ~4x time = quadratic). A normal call has hundreds of identifiers so
  this is fine today (2h-style 144k-char transcript = 2.5s; 100k alternating digit/sep = 0.21s),
  but a pathological or very long/high-PII transcript is a soft DoS. Not a correctness/leak issue.

---

## Checked and sound

- INDEX-MAP MAPPING (target #1): property-fuzzed 80,000 random strings mixing Hebrew
  digit-words, real digits, bidi/zero-width invisibles, Arabic-Indic & full-width digits,
  combining marks, emoji, punctuation, whitespace and newlines. Every `find_pii` span was
  in-bounds and ordered, the unmasked remainder was verbatim original, and `redact_text`
  output equalled original-minus-masked-ranges in all cases. Specifically verified: folded
  run at start, at end, adjacent folded runs, folded run immediately followed by real digits,
  real digits followed by folded run. No off-by-one / shift / drop / duplication in the
  `fold_spans` -> `normalize_for_detection` index-map composition.
  (scratchpad/fuzz_indexmap.py, fuzz2.py)
- Invisible chars (bidi RLM/zero-width) at edges and inside a run: stripped in normalisation,
  mapping stays correct, identifier still masked (`תעודת זהות ‏123456782‏` -> masked, RLM left
  outside the mask). Arabic-Indic digit ID folds and masks correctly.
- Checksum-backed identifiers are NOT subject to the amount exception (their gates precede it):
  a valid Israeli ID / phone / card followed by "שקל" is still masked. Only non-checksum
  ACCOUNT_LIKE / dictation-ID paths are exposed (see F1).
- Card number split across 3 turns (with an empty middle turn) is detected over the joined
  document and masked once in the starting turn; the continuation tail is removed, not leaked.
- Dictated identifiers in Hebrew words work: valid ID words + id-context -> ISRAELI_ID;
  phone words + phone-context -> PHONE. vav-prefix (ו-) folds inside a run; the ה- article
  does not; commas act as dictation pauses. Runs < MIN_DIGIT_WORD_RUN (4) and ordinary uses
  of "אחת"/"אחד"/"שתי" stay unmasked.
- Amount-in-words stays unmasked: `מאה עשרים אלף שקל`, `מאתיים שקל` are untouched (quantity
  words never fold; only 0-9 words fold). Digit amount `481902 שקלים` stays unmasked.
- `_repeated_fragments` read-back masking works across turns: a last-4 read-back of a masked
  identifier is masked with it (true positive). (Its over-mask failure mode is F2.)
- `sanitize_error`: truncation at the 500-char limit cannot expose raw PII — redaction runs
  before truncation, and truncation can only clip a short mask token, never reveal digits.
  Verified with inputs crafted to place a mask at the boundary.
- END-TO-END GREP (target #3, the CRITICAL check): grepped every artifact under data/output
  (redacted, results, scores, reports, features, ingestion, audio) plus *.log files and the
  live dashboard `/api/state` and `/api/transcript` responses (via tests/test_dashboard_server.py,
  20 passed) for RAW_ID=123456782 and RAW_PHONE=052-1234567. NO raw value in any consumer-facing
  artifact, log, or endpoint. Raw values appear ONLY in data/output/transcripts/ (CALL001/CALL004),
  which is the pre-redaction ASR stage, by design (pipeline.py:69 RAW_TRANSCRIPTS_README warns it
  is raw and access-restricted; only ../redacted/ feeds judge/reports/logs/dashboard). CALL001 and
  REAL002 redact their ID+phone+account numbers correctly with no surviving long digit run.

---

## RANKING (most severe first)

1. **F1 — HIGH** — Amount exception leaves non-checksum account/ID numbers unmasked when a
   currency word (incl. "אלף"/"מיליון") falls within 14 chars after them. Real leak in bank
   phrasing where numbers and money are spoken together; could surface a raw account number in
   shipped redacted artifacts (would then be CRITICAL for that call).
2. **F3 — HIGH** — A combining mark (Hebrew niqqud, category Mn) or other non-INVISIBLE format
   char inside a digit run splits it below threshold and leaks the whole identifier;
   normalisation only strips a hardcoded INVISIBLE list, not Unicode Cf/Mn/Me.
3. **F2 — MEDIUM** — `_repeated_fragments` over-masks amounts/years/dates that share 4-5 digits
   with a masked identifier (no amount/context check), blinding compliance/clarity scoring.
4. **F4 — LOW** — `_repeated_fragments` is O(n^2) in identifier count; soft DoS on pathological
   transcripts, fine for normal calls.
