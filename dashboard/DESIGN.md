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

## Still to build

- Progress that streams while a batch runs, rather than a snapshot per load
- The screen for starting and stopping the cloud machine from here, which is
  why `--allow-actions` exists but does nothing yet
- A designed empty state for day one, before any call exists
