# Operator dashboard — design decisions

**Status: working, under review.** `server.py` serves `prototype.html` with
live data read from `data/output/`. Opened as a plain file the same page falls
back to embedded sample figures, so it doubles as a standalone prototype.

> **Scope note.** The original build spec puts a web application out of scope
> ("static HTML reports only", §3). This dashboard is built anyway because the
> operator asked for one, but it lives entirely in `dashboard/` so the bank
> deliverable can stay CLI + static HTML, exactly like the `cloud/` option.
> Deleting this directory removes it.

## The one rule: orange is brand chrome, never status

This is measured, not preference. Using the data-viz palette validator:

| pair | separation | verdict |
|---|---|---|
| brand `#ea580c` vs warning `#ca8a04` | ΔE **2.9** (deuteranopia) | indistinguishable |
| brand `#ea580c` vs danger `#dc2626` | ΔE **8.7** (normal vision) | below the 15 floor |

So orange appears only in the masthead, the primary button, the active nav
item, the focus ring, and data bars. Every status carries an **icon plus a
Hebrew word**, never colour alone.

## Contrast: computed for every token

The attractive oranges fail WCAG, which is the trap this design avoids:

| step | on white | white text on it |
|---|---|---|
| `#f97316` | 2.80:1 | fails |
| `#ea580c` | 3.56:1 | **fails** — the classic bright-orange button is inaccessible |
| `#c2410c` | 5.18:1 | passes |
| `#9a3412` | 7.31:1 | passes |

Hence: primary button `#c2410c`, headings `#9a3412`, accents/borders `#ea580c`
(UI-only, ≥3:1).

Status trio validated **as a set** (lightness band, chroma floor, CVD
separation, normal-vision floor all pass): success `#047857`, warning
`#ca8a04`, danger `#dc2626`. The amber takes **dark** ink (`#451a03`, 5.10:1)
because white on amber is 2.94:1. Darkening the amber instead was tried and
rejected: it collapses colourblind separation from red.

Muted ink is `#6f6862`, not the prettier `#78716c` — the rendered audit caught
the latter at 4.36:1 against the sunken fill. Tokens must be checked against
the tightest surface they land on, not just the card.

## Layout

RTL. A masthead and a **state strip both pinned to the top**: the strip carries
the GPU status, elapsed time and money spent, because the real risk is not the
hourly rate but forgetting the machine is running. A regression test asserts
the cost meter never leaves the viewport.

Navigation is a right-hand rail with five destinations. Overview holds a single
alert, three tiles, one chart and a table — deliberately no second chart, no
sparklines, no metric soup.

The dimension bars map `(score-1)/4`, matching the pipeline's own scoring
formula, so a 1 reads empty and a 5 reads full. Mapping `score/5` compressed
every real difference into the top half of the track.

## Typography

Heebo for the interface (a Hebrew face with a real weight range) and IBM Plex
Mono for call ids, timestamps and prompt hashes — the audit trail is this
system's instrument, so it gets an instrument face. Hebrew runs at 1.65 line
height: its letters have no ascenders to open up the line, so Latin leading
reads as a wall.

Latin and numeric runs inside Hebrew are wrapped in `.ltr` / `.mono`
(`unicode-bidi: isolate`) or the bidi algorithm reverses call ids and prices.

## QA

`python dashboard/qa_dashboard.py` renders the page in headless Chromium across
light/dark × desktop/phone plus the detail drawer, and fails on:

- text contrast below WCAG AA, computed against the **actually painted**
  background rather than the declared one
- horizontal overflow at phone width
- tap targets under 24px
- Latin/digit runs in RTL without bidi isolation
- focus states with no visible ring
- the cost meter scrolling out of view

It has already caught six real defects: the muted-ink contrast failure, a
551px-wide overflow at 390px (grid items default to `min-width:auto`), the
cost meter scrolling away, unisolated Latin inside a Hebrew label, and two
scoping bugs where helpers were swallowed into `renderAll()` so the page threw
a silent ReferenceError and quietly kept showing its sample data.

That last pair is the instructive one: the page *looked* correct, because the
fallback made it look correct. The harness only found it once it failed on any
page error and asserted that the detail drawer actually opens.

An independent adversarial review then found three more in shipped code:

- **Path traversal.** The `/reports/` guard was `str(target).startswith(str(root))`.
  A string prefix also accepts a *sibling* directory — `reports_backup`,
  `reports-old` — so files outside the reports tree were reachable. Now
  `Path.is_relative_to`, with a test that creates such a sibling.
- **Focus ring failed WCAG 1.4.11.** `rgb(234 88 12 / 0.35)` composites to
  1.55:1 on the card and 1.60:1 on the dark card; 3:1 is required. The first
  audit only asserted a ring *existed*. It now rejects translucent rings, and
  the ring is opaque (`#c2410c` light, `#fb923c` dark).
- **Unescaped interpolation.** The page built HTML from ASR-derived text with
  `innerHTML` in 13 places, in a document that holds the session token and can
  start GPUs. All data is escaped now.

## Running it

```bash
python dashboard/server.py          # prints a URL with a one-time token
python dashboard/qa_dashboard.py    # the visual QA harness
```

It is read-only: it renders what the pipeline already wrote. Actions that
spend money or mutate state are gated behind `--allow-actions` and are not
implemented yet.

## Removing it

`rm -rf dashboard/ tests/test_dashboard_server.py` and nothing else changes.
The server is intentionally not registered as a `callqa` subcommand, so the
package the bank receives has no reference to it.

## Independent review

A parallel research pass (five research lenses, three independent design
proposals, three judges, a synthesis and an adversarial critic) was run against
this work rather than before it. It converged on the same six destinations and
the same stdlib decision, explicitly rejecting FastAPI + uvicorn + sse-starlette
(ten wheels replacing a server that already works), htmx (65KB, and adopting it
means rewriting the working render path), Alpine (needs `unsafe-eval`) and
Chart.js (208KB, canvas-only, invisible to CSS and screen readers, manual RTL
axis work). Its verdict was BUILD_WITH_FIXES, and its three verified defects are
fixed above.

Useful things it established that are worth keeping in view:

- A dashboard is a **single-screen** medium (Few): Overview must answer "is the
  machine burning money, is anything running, did anything fail" with no
  scrolling at 1366×768. Everything else is a drill-down.
- **One filled orange button per screen.** Read-only screens (Calls, Bankers)
  should have none at all — that absence is what makes the orange button on
  Run, Cloud and Calibration mean something.
- Carbon's productive type set is **14px base with fixed, non-fluid headings**
  for operational UI; Hebrew takes the looser line-height of each pair.

## Still to build

Ordered by what actually blocks an operator, per the critique:

1. **The cloud round trip is incomplete.** Starting a pod generates per-pod
   secrets and prints shell exports the operator is expected to run by hand.
   `RUNPOD_API_KEY` also has no path into a double-clicked launcher. Until this
   is wired, the cloud buttons would fail with a raw English exception.
2. **No safe first run.** `--mock` is exactly what a non-expert needs on day
   one, and there is no way to enter it from the UI. The empty state should
   offer a demo run.
3. **Validation messages are English.** The intake screen promises plain
   Hebrew ("row 7, banker_channel, must be L or R"); `ingestion.py` emits
   English strings. Translating in the dashboard needs structured problem
   codes rather than free text, so this is a small core change.
4. **No re-run after a rubric or prompt change.** Scorecards carry
   `prompt_sha256` precisely so stale calls can be found; nothing surfaces
   "N calls were scored with an older prompt".
5. **`watch` has no surface.** If an operator starts it from a terminal, the
   dashboard cannot see it and the two will contend for the same locks.
6. Progress that streams during a batch, rather than a snapshot per page load.
7. A designed empty state for day one.
