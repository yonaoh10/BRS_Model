# QA Round 3 — Adversarial Security Review (2026-09-16)

**Reviewer:** adversarial security review (read-only)
**Scope:** dashboard/server.py, dashboard/prototype.html, src/callqa/reporting/ (call_report.py + templates/*.j2)
**New attack surface:** /api/transcript/<call_id>; static per-call report now embeds full transcript

_Findings appended incrementally the moment they are confirmed._

## Method
- Ran the real pipeline in mock mode on the seeded fixture (`scripts/generate_sample_data.py`
  + `callqa run/report/calibrate --mock`) → 6 calls, output under scratchpad `work/output`.
  Raw fixture PII: RAW_ID=`123456782`, RAW_PHONE=`052-1234567`.
- Stood up the live server exactly as `tests/test_dashboard_server.py` does
  (`Handler.token="test-token-value"`, `ThreadingHTTPServer(("127.0.0.1",0), Handler)`),
  drove it with `urllib` from `.venv/bin/python`. Scratch scripts in scratchpad, not the repo.
- Confirmed raw PII lives in `output/transcripts/*.json` (raw) but not in `output/redacted/*.json`.

**Headline: no CRITICAL, no HIGH. The two new doors (the /api/transcript endpoint and the
transcript embedded in the static report) are both sound on auth, traversal, raw-PII and XSS.
4 MEDIUM and 4 LOW findings, all robustness / reviewer-clarity, none a data breach.**

---

## MEDIUM — /api/transcript has no response-size guard; a large redacted artifact yields an unbounded body
- **file:line** dashboard/server.py:287-314 (`redacted = _load_json(...)`; body built and
  `json.dumps(body, ...)` with no length cap), reached via `_load_json` at :63-67 which does
  `path.read_text()` (whole file into memory).
- **repro** Wrote `redacted/BIG1.json` = 400 turns × 50 000 chars each, then
  `GET /api/transcript/BIG1?t=...` → HTTP 200 with a **40,030,063-byte** response body. There is
  no `Content-Length` ceiling, no turn count/char cap, and no streaming: the file is read whole,
  a second full copy is built as the `turns` list, and a third as the serialized JSON. `/api/state`
  by contrast stays tiny (9 168 bytes for 6 calls, `carries_transcripts=False` — verified there is
  no `turns`/`transcript` key on any call), so the split-endpoint DoS goal for /api/state holds.
- **why it matters** `ThreadingHTTPServer` serves each request on its own thread, so N concurrent
  fetches of a pathological call hold N× the artifact in memory and can OOM the console. Realistic
  exploitability is low (localhost, token-gated, single user, and the artifact must first be
  written to disk by the pipeline itself — it is not attacker-injected over the wire), so this is a
  robustness gap rather than a breach, but the answer to "is there any size guard?" is: none, on
  either the read or the serialize path. The browser drawer would also lock up rendering it.

## MEDIUM — Report: red total score with a green "gates OK" badge and no legend for the total colour
- **file:line** src/callqa/reporting/templates/call_report.html.j2:94 (`color: {{ total_color }}`)
  + reporting/common.py:52-59 (`score_color`: ratio<0.6 → `#c0392b`, the same red as `.badge-fail`).
- **repro** REAL002 (`data/output/reports/calls/REAL002.html:140`) shows the total `56.2` rendered
  in `#c0392b` (fail-red) while line 144 shows `<span class="badge badge-ok">כל ממדי השער תקינים</span>`
  (all gate dimensions OK) and there is no gate-cap. Both gate dims scored 3/5.
- **why it matters** Red is the pipeline's "fail" colour and is reused verbatim for `badge-fail`,
  the gate-cap warning, and low scores. A reviewer who has never seen the system sees a big red
  number and a "gates fine" badge together and cannot tell whether the call failed the gate or
  merely scored low. Nothing on the page states that the total's colour is a 3-band traffic light
  (<60 red) independent of the gate. Add a one-line legend or decouple the colours.

## MEDIUM — Report: cited evidence quotes are semantically unrelated to their dimension, and the same wrong quotes are highlighted in the transcript
- **file:line** call_report.html.j2:124-129 (evidence loop) and :180-184 (the `<mark>` highlight
  fed by `_transcript_turns`/`_evidence_segments`, call_report.py:46-107).
- **repro** In REAL002: dimension "סגירה והצעד הבא" (closure) cites `[00:08] "שלום, הגעת למוקד הבנק."`
  — the call's OPENING line (report line 230-231), and that opening is then highlighted in the
  transcript as closure evidence (line 297-298). Dimension "זיהוי ואימות לקוח" (identification)
  cites `[01:06] "רגע אחד, אני מושך את פירוט החיובים."` (pulling up the charges — line 160-161),
  not the actual identification exchange at `[00:24]`/`[00:48]`. Empathy cites a plan-switch line.
- **why it matters** This is a mock-judge artifact and the mock banner (line 65-72) does say the
  quotes are not real. But the mechanism — snap-to-transcript highlighting — makes a *wrong* quote
  look authoritative (it is boxed, coloured, and titled "צוטט כראיה בממד: …"). A reviewer who trusts
  the highlight in a mock report will carry that trust into a real one; conversely, seeing obvious
  nonsense highlighted trains them to distrust the evidence feature entirely. The report should make
  the mock↔evidence link explicit ("in a mock run the highlighted spans are placeholder, not cited").

## MEDIUM — Report: role-confidence "65%" cannot be reconciled with the near-unanimous signal table it claims to summarise, and every objective metric silently depends on it
- **file:line** call_report.html.j2:50-55 (`רמת ודאות {{ role_confidence*100 }}%` … "על סמך מאזן
  הסימנים בטבלה שלהלן") + the signals table :62-75 + the metrics table :133-155.
- **repro** REAL002 lines 82-85 state `רמת ודאות 65%` with the only reference point being
  `מתחת ל-34% … בדיקה אנושית` (below 34% → human review). The table immediately below (lines 95-129)
  shows 6 of 7 signals voting "בנקאי" and only one ("מסירת פרטי זיהוי") voting "לקוח" — visually
  near-unanimous. The "מדדים אובייקטיביים" table (talk ratio 67%, 9 questions, WPM, etc.) is
  presented as hard fact with its inference caveat sitting in a different card far above.
- **why it matters** A bank reviewer cannot tell whether 65% is "good" or "barely acceptable", and
  the number appears to contradict the table it is derived from (why 65% when 6/7 agree?). The 34%
  floor is so low that "passed" conveys little. Because this call is mono, the entire objective-metrics
  table rides on this one inferred split, yet the table carries no inline caveat. Frame the confidence
  qualitatively and repeat the "inferred, verify against transcript" caveat on the metrics table.

## LOW — Transcript evidence-highlight can emit malformed nested `<mark>` markup (no XSS)
- **file:line** dashboard/prototype.html:892-906 (`loadTranscript`: `html = html.split(q).join('<mark …>' + q + '</mark>')`).
- **repro** Served `redacted/INJ1.json` with turn text `hello </script><img src=x onerror=alert(1)> <mark>x</mark> …`
  and a scorecard whose evidence quotes were `<img src=x onerror=alert(1)>` and the bare word `mark`.
  Faithfully re-ran the page's `esc()`+split/join logic: output contained
  `<<mark class="ev" …>mark</mark> class="ev" …>` — the second quote (`mark`) matched *inside the
  literal `<mark` wrapper injected by the first quote*, so the split/join wrapped our own markup and
  produced broken nested tags.
- **why it matters** It is only a rendering-corruption bug, **not** XSS: I verified the final string
  contains no raw `<script`, no raw `<img `, and no unescaped `onerror=`/`onmouseover=` attribute —
  every attacker byte (`< > " '`) is `esc()`-ed before the substring match, and the only literal HTML
  inserted is the page's own trusted `<mark>` template (its `title` interpolates `esc(byId[...])`).
  So a stored-XSS into the token-holding console is NOT possible here. But an evidence quote equal to
  a substring of the wrapper (`mark`, `ev`, `class`) mangles the drawer. Match on span offsets (as the
  static report's `_evidence_segments` already does) instead of `split().join()` on rendered HTML.

## LOW — Report mask-token legend explains only `<████>`, not the typed prefix, and the type labels are inconsistent
- **file:line** call_report.html.j2:170-172 (legend: `הסימונים <████> מציינים פרטים שהוסרו`).
- **repro** REAL002 legend shows the generic `<████>` form, but the actual tokens in the transcript
  carry a type prefix: `<ת"ז:████>` (line 354), `<חשבון:████>` (368, 382, 504, 511), `<טלפון:████>`
  (375). The prefix is never explained. Worse, the types are inconsistent: at `[00:33]` the banker
  reads back the customer's *ID* but it is masked `<חשבון:████>` (account), and the *card* last-4 at
  `[01:59]`/`[02:05]` is also masked `<חשבון:████>`.
- **why it matters** A reviewer must guess that ת"ז=ID / חשבון=account / טלפון=phone, and then sees an
  ID and a card number both labelled "account", which invites doubt about redaction accuracy on a
  report whose whole selling point is trustworthy PII handling. Explain the prefixes in the legend.

## LOW — Report asserts the total but never shows its derivation, and the summary tone fights the red score
- **file:line** call_report.html.j2:94-111 (score card) + :113-131 (per-dimension list).
- **repro** REAL002 shows `56.2` (red) with the formula in prose ("1=0, 3=50, 5=100, weighted by
  rubric weights", line 142-143) and the summary "שיחה עניינית; הבנקאי טיפל … בהתאם לנהלים" (line 149,
  businesslike, handled per procedure). The eight dimension scores are listed but there is no
  per-dimension contribution/weight-times-points breakdown, so a reviewer cannot reconstruct 56.2.
- **why it matters** The single most consequential number on the page (it can feed employee review) is
  asserted, not shown, and its red colour is contradicted by a neutral-positive summary. A small
  "contribution to total" column would make the score auditable and settle the tone mismatch.

## LOW — Report leans on visibly-garbled speaker attribution for its objective metrics
- **file:line** call_report.html.j2:133-155 (metrics) built on the mono role split; warning at :35-61.
- **repro** REAL002 transcript labels the banker as naming himself both "סער" (`[00:06]`, line 290)
  and "דני" (`[00:10]`, line 305), and attributes a run of one-word interjections ("אה?", "כן כן, נו",
  "שלום.") to the banker. The report *does* flag this (mono → inferred, 65% confidence, "verify against
  transcript") — this is honest, not hidden.
- **why it matters** Because the honesty and the metrics live in separate cards, a reviewer skimming the
  "objective metrics" table takes "banker talk ratio 67% / 9 questions" as measured fact while the
  transcript right below shows the attribution is shaky. Noted as LOW precisely because the report is
  upfront about it; the fix is proximity (inline caveat), not new disclosure.

---

## Checked and sound

**/api/transcript auth & DNS-rebinding (server.py:199-215, 246-247)** — `_authorized()` runs at the
top of `do_GET` for every route including `/api/transcript/*`. Verified live: no token → 403,
`?t=wrong` → 403, foreign `Origin: https://evil.example` → 403, spoofed `Host: evil.example` → 403,
correct token → 200. Same controls as `/api/state`.

**/api/transcript path traversal (server.py:283-290)** — `unquote` is applied once, then
`CALL_ID_RE.fullmatch` (`^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$`, ingestion.py:32) rejects any separator.
All of these returned 404 with zero raw leakage: `..%2f..%2fresults%2fCALL001`,
`..%2f..%2ftranscripts%2fCALL001`, `..%5c…` (backslash), `..%252f…` (double-encoded — single unquote
leaves a literal `%`, which the charset rejects), `%2fetc%2fpasswd`, `..%2f…%2fetc%2fpasswd`,
`CALL001%2e%2e`, `CALL001/../../../transcripts/CALL001`, a 200-char id, `CALL001%00.json` (NUL), and
`CALL001.dialog` (a real file, but it lives in `transcripts/` not `redacted/`, so 404). The endpoint
is hard-wired to `output_dir/"redacted"/f"{call_id}.json"` — there is no route to `transcripts/`,
`results/`, `scores/` (read only for evidence, same id), or the filesystem.

**Raw-PII through the endpoint (server.py:287-314)** — For all six calls,
`GET /api/transcript/CALL00n` returned 200 with the mask token `████` present and neither `123456782`
nor `052-1234567` anywhere in the JSON. The raw values *are* present in `output/transcripts/CALL001.json`
and `CALL004.json`, confirming the test is not vacuous.

**Disabled-redaction artifact (server.py:302-312)** — Wrote `redacted/RAWLEAK1.json` with
`enabled:false` carrying `123456782`/`052-1234567` in a turn. Endpoint returned
`{"enabled": false, "turns": [], "evidence": []}` — the fact, never the text. No raw leak.

**Static report XSS (reporting/common.py:19-28, call_report.html.j2)** — `jinja_env()` uses
`select_autoescape(["html","j2"])`; the template is `call_report.html.j2` (ends `.j2`) → autoescape ON.
Confirmed in the rendered REAL002: `<` in mask tokens is escaped (`&lt;ת&#34;ז:████&gt;`, line 354),
the recommendation's quotes are `&#34;` (line 254). `_evidence_segments` (call_report.py:46-77) splits
on integer offsets and emits `(segment, dim)` tuples; segments and the `title` are both autoescaped, so
the `<mark>` wrapping cannot be driven out of its markup by attacker text in a quote or turn. The
static report's transcript is empty when `redacted.enabled` is false (call_report.py:88-89).

**Live-drawer XSS (prototype.html:734-735, 878-915)** — both `t.text` and `e.quote` pass through
`esc()` (escapes `& < > " '`) before the substring match and before insertion; the injected `<mark>`
is the page's own literal with an `esc()`-ed `title`. A crafted turn containing `</script>`,
`<img onerror=…>`, a `<mark>` literal, a mask look-alike and a `" onmouseover=` fragment rendered with
no executable HTML (see the LOW nesting finding for the one benign corruption case).

**/api/state size (server.py:89-113, 261-274)** — no per-call `turns`/`transcript` key; 9 168 bytes for
6 calls. Transcripts are correctly split to their own endpoint.

**Round-1 controls, re-verified live:** token required on every route (`/api/state` no-token → 403);
foreign Origin → 403; spoofed Host → 403; `/reports/../../../etc/passwd` → 404 and
`/reports/..%2f..%2f..%2fetc%2fpasswd` → 404; sibling `reports_backup/secret.html` via
`/reports/../reports_backup/secret.html` → 404 (server.py:322-329 uses `is_relative_to`, not a string
prefix); `/reports/index.html` → 200. Token never logged: `log_message` (server.py:237-242) turned
`GET /api/transcript/CALL001?t=SUPERSECRETTOKEN123 …` into `?t=<redacted>` in the emitted log line.
Security headers present on responses: `Content-Security-Policy` (default-src 'self', frame-ancestors
'none', base-uri 'none'), `X-Frame-Options: DENY`, `X-Content-Type-Options: nosniff`,
`Referrer-Policy: no-referrer`, `Cache-Control: no-store`. `--bind` warning present (server.py:365-367).

**Token comparison** — `hmac.compare_digest` (server.py:215), constant-time.
