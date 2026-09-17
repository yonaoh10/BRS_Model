# Reviewer dashboard — design decisions

**Status: rebuilt 2026-09-17.** `server.py` serves `prototype.html` (one
self-contained HTML file) with live data read from `data/output/`. Every pixel
maps to real pipeline output; there is **no embedded sample data**, so opened as
a bare file the page shows a "connect to the server" empty state rather than
faking figures.

> **Scope.** The bank deliverable stays CLI + static HTML (spec §3). This
> dashboard is an add-on that lives entirely in `dashboard/`; `rm -rf dashboard/
> tests/test_dashboard_server.py` removes it with nothing else changed.

## Audience: the bank quality reviewer

An earlier version tried to be both a cloud **operator** console (a pinned
GPU/cost "state strip", model-availability pills) and a **reviewer** console.
The operator half could never be truthful — the dashboard is offline (strict
CSP, reads only `data/output/`) and cannot know the real pod/cost/model state —
so it was hardcoded demo values, and the page read as stitched-together. It is
now purely the reviewer's tool: which calls need attention, the score
distribution, per-call detail with the redacted transcript, banker comparison,
and judge calibration. The cloud lifecycle belongs to `cloud/runpod_cli.py`, not
here. Provenance a reviewer *does* need — model, engine, prompt version, run
date — lives quietly in the sidebar footer, read from the real `audit{}`.

## Visual identity: restraint, not decoration

A calm instrument aesthetic (the Linear/Stripe/Vercel tradition), earned by
rigor rather than ornament:

- **App shell.** A right-anchored sidebar (RTL) holds the brand, navigation and
  provenance, and anchors the layout so content fills the width in composed
  grids instead of floating in a narrow centred column. This is what fixes the
  "content in a sea of empty space" failure of the first attempt.
- **Colour is rare.** The canvas is near-monochrome cool slate. **Data bars are
  a muted slate (`--bar`), not colour** — magnitude reads without noise. The
  indigo accent (`--brand #4f46e5`) is reserved for interaction only: active
  nav, focus ring, links, the one primary button per screen. Status colour
  (green/amber/red) appears only in small badges, always with an icon and a
  Hebrew word, never colour alone.
- **Data viz, refined.** Thin (5–6px) bars; the banker comparison is a
  **dot-plot** on a shared 0–100 axis with the group-median line, not fat bars;
  the score distribution is a compact histogram with a screen-reader table
  mirror; calibration is a hero verdict with a QWK-vs-threshold gauge.
- **Type & space.** System fonts only (the CSP blocks web fonts — the old page's
  Google-Fonts link silently failed when served). Tabular-lining numerals on
  every figure so columns align; mono for ids/timestamps/hashes. An 8px spatial
  grid; hairline borders; a single soft shadow reserved for the floating drawer.

## Accessibility is enforced, not asserted

`python dashboard/qa_dashboard.py` renders the **live** page across light+dark ×
desktop(1280×900)+phone(390×844) plus the open detail drawer, and fails on:
text contrast below WCAG AA computed against the *actually painted* background;
horizontal overflow at phone width; tap targets under 24px; Latin/digit runs in
RTL without bidi isolation; focus states with no opaque (≥0.95α, ≥3:1) ring;
any JS error; and any view left under 40 chars. It must stay at **0 findings**.

It renders the live server (not the file) because the redesign ships no fallback
data — a file with no server correctly shows only the empty state, which has no
table or chart to audit. It caught the two real defects in this rebuild: a
muted-ink colour (`--ink-3`) at 3.1:1 that had to darken to pass AA on the
sidebar's off-white fill, and an un-isolated `0.70` in the calibration gauge
label. The retired **cost-meter "never scrolls away" check** went with the
state strip it guarded.

## Playback without a raw-audio door

The detail drawer plays the recording while the transcript follows along, the
spoken line highlighted, with the judge's evidence and the silenced spans marked
on the timeline. The recording has the customer reading identifiers aloud, so a
play button would be a second raw-PII exit the redaction stage never covered.
It is closed the same way the transcript door is: the browser only ever reaches
a REDACTED recording. The pipeline writes a copy silenced wherever the transcript
was masked (`callqa.audio_redaction`, produced in the redaction stage, stored
under `data/output/redacted_audio/`); `/api/audio/<id>` serves only that
directory — the raw WAVs under `audio/wav/` have no route at all — with the same
token, Origin/Host and containment guards as `/reports/`, plus HTTP Range so the
browser can seek. The audio is only as trustworthy as the text mask it mirrors:
silenced exactly where the transcript was, generously padded, and no further. A
disabled-redaction call exposes no audio, exactly as it exposes no transcript.

## The scoring bars map (score-1)/4

A 1 reads empty and a 5 reads full, matching the pipeline's own 1→0 / 3→50 /
5→100 formula; mapping score/5 compressed every real difference into the top of
the track. Gate dimensions are marked `⚑`; a gate dimension at ≤2 turns its bar
amber and says "מגביל" — colour plus words, never colour alone.

## Honest states over faked ones

There is no fallback dataset. Three real states replace the old silent
sample-data leak: **no token** (opened as a file) → "runs only against the local
server"; **server up, 0 calls** → "no calls analysed yet"; **fetch failed** →
"could not load pipeline data". The `running[]` panel reads the live SQLite lock
table (a held lock = processing now), so completed calls no longer appear as
in-flight — a `server.py` fix made in this rebuild.

## Running it

```bash
python dashboard/server.py          # prints a URL with a session token
python dashboard/qa_dashboard.py    # the accessibility harness (0 findings)
```

Read-only by design: it renders what the pipeline already wrote.
