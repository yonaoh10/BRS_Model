# Deploying callqa

**Read this first.** It is written for the engineer who received a link to this
repository and has to get it running on a bank machine. It assumes no contact
with the people who built it.

---

## 0. What this is, in one paragraph

callqa takes one recording of a banker↔customer phone call in Hebrew and
produces a quality-assurance report on how the banker handled it. It
transcribes the call, separates the two speakers, masks every customer
identifier it can find, computes objective conversational features (talk ratio,
interruptions, dead air), asks a locally-served language model to score the call
against an eight-dimension rubric with verbatim evidence quotes, and writes a
Hebrew RTL HTML report. It is a command-line program. There is an optional local
dashboard; there is no web service and nothing listens on a network port unless
you start the dashboard yourself.

**It never contacts the internet at runtime.** The only component that downloads
anything is `scripts/download_models.py`, which you run once, deliberately.

---

## 1. Decisions that are yours

This repository deliberately ships **no infrastructure opinion**. There is no
Dockerfile, no compose file, no orchestration, no cloud anything. It is a Python
package and a CLI. How it gets packaged, scheduled, containerised, monitored and
backed up on the bank's estate is your call, and nothing here will fight you.

Two integration points are worth knowing before you plan:

| You decide | What the software needs |
|---|---|
| How the pipeline is invoked | Any scheduler. `callqa process <file>` handles one call and exits `0` / `1` / `2` (see §7). |
| How the judge model is served | Any HTTP endpoint speaking OpenAI `/v1/chat/completions`. See §4. |

---

## 2. Prerequisites

| | Requirement | Notes |
|---|---|---|
| OS | Linux (x86-64) or macOS | Developed on both. Windows is untested. |
| Python | 3.11 or 3.12 | Pinned in `pyproject.toml`; CI covers both. |
| ffmpeg | Required for real recordings | Not needed for §3. Needed to read `.m4a`, `.mp3`, and anything that is not plain WAV. |
| Disk | ~40 GB | ~25 GB models, the rest for working data. `callqa preflight` refuses to start a batch under 1 GiB free. |
| RAM | 16 GB | For transcription and diarization. |
| GPU | Needed only for the judge | Transcription runs acceptably on CPU; the language model does not. See §4. |

---

## 3. Verify the software before you download anything

This proves the install is correct. It needs no models, no GPU and no network,
and it takes seconds.

```bash
python3 -m venv .venv && source .venv/bin/activate     # recommended; see below
pip install -r requirements.txt
pip install -e .
./scripts/first_run.sh
```

Use a virtual environment. Debian 12, Ubuntu 24.04 and RHEL 9 mark the system
Python "externally managed" and refuse `pip install` into it outright, and RHEL
ships no bare `pip` at all. On an air-gapped machine, `scripts/install_offline.sh`
creates the environment for you from the wheels bundle (§4.3).

It generates six synthetic calls, runs the complete pipeline over them in mock
mode, writes reports, and starts the local dashboard. If
`data/output/reports/index.html` opens and shows six scored calls, everything
except the models is working.

Run the test suite too — it is fast and needs nothing extra:

```bash
python -m pytest -q          # the full suite, about a minute
ruff check src tests scripts dashboard
```

> **Mock mode is not a demo mode.** It is the same pipeline with deterministic
> fake engines behind the same interfaces. Anything that breaks in mock mode is
> broken.

---

## 4. Models

**You download these yourself.** They total roughly 20–25 GB and are not in this
repository — GitHub cannot hold them and a ZIP download of them is not workable.

### 4.1 Accept the gated model's licence first

`pyannote/speaker-diarization-community-1` is **gated**. Before it can be
downloaded, a Hugging Face account must visit its page and accept its
conditions:

> https://huggingface.co/pyannote/speaker-diarization-community-1

Then create a read token at https://huggingface.co/settings/tokens and put it in
`.env`:

```bash
cp .env.example .env
# edit .env:  HF_TOKEN=hf_...
```

This is the single most common thing to get stuck on. The download fails with a
403 that does not mention the licence.

### 4.2 Download

```bash
pip install huggingface_hub                 # only needed for this step
python scripts/download_models.py --all --llm-model <your-choice>
```

| Role | Model | Approx. size |
|---|---|---|
| Transcription | `ivrit-ai/whisper-large-v3-turbo-ct2` | ~1.6 GB |
| Speaker separation | `pyannote/speaker-diarization-community-1` | ~100 MB (gated) |
| Judge | your choice, see below | 15–40 GB |
| Named entities (optional) | `dicta-il/dictabert-ner` | ~700 MB |

Pick the largest judge model that fits the GPU you have:

| VRAM | Candidate |
|---|---|
| 16–24 GB | `dicta-il/dictalm2.0-instruct` (7B, Hebrew-tuned) |
| ~24 GB | a 12–27B instruct model, AWQ/GPTQ quantized |
| ≥48 GB | a 70B instruct model, AWQ quantized |

The downloader writes `models/MODELS_MANIFEST.json` recording each model's
local path, content hash, size **and the licence its own model card declares**.
That file is what your compliance team should be shown; see §9.

### 4.3 Air-gapped target

If the machine that runs the pipeline has no internet at all, run the download
on a connected machine and move the artifacts across:

```bash
./scripts/build_offline_bundle.sh      # pinned wheels into wheels/
python scripts/download_models.py --all --llm-model <your-choice>
# transfer: the repository, wheels/, models/, and the Hugging Face cache
```

Then on the target: `./scripts/install_offline.sh`, and set `HF_HUB_OFFLINE=1`
and `TRANSFORMERS_OFFLINE=1` in the service environment. The pipeline never
downloads, so these only guard against a library trying to phone home.

**The diarization model is the one that does not live in `models/`.** pyannote
loads through the *Hugging Face cache* of whichever user ran the download
(`~/.cache/huggingface/hub` by default). Copy that directory across and point
`HF_HOME` at its parent in the service environment. `callqa preflight` checks
exactly this and says where it looked. Stereo recordings never need it; mono
and dual-mono ones cannot be processed without it.

---

## 5. Serving the judge model

The judge is **an OpenAI-compatible HTTP client and nothing more**. It sends
`POST /v1/chat/completions` with a bearer token. It does not launch, supervise
or assume any particular server, and it will work against vLLM, TGI,
llama.cpp's server, or an internal inference gateway the bank already runs.

`scripts/start_vllm.sh` is a worked example, not a requirement:

```bash
export CALLQA_JUDGE_API_KEY=$(openssl rand -hex 32)     # the SERVER's key
./scripts/start_vllm.sh models/<your-judge-model> 8000
```

Then point the pipeline at whatever you started, and give it the same key:

```yaml
# config/config.yaml
judge:
  base_url: "http://127.0.0.1:8000/v1"
  model: "<the model id the server reports>"
```

```bash
# .env (or the service environment) - never in the YAML
CALLQA_JUDGE__API_KEY=<the same value>
```

The two names differ by one underscore and that is not a typo:
`CALLQA_JUDGE_API_KEY` is read by `start_vllm.sh`; `CALLQA_JUDGE__API_KEY`
(double underscore) is the pipeline's override for `judge.api_key`. Miss the
second and the server answers 401, which the pipeline reports as *refused*,
not *unreachable* - the server is fine, the key is not.

Two things that bite:

- **Set an API key.** An unauthenticated inference server on a bank network
  answers anyone who can reach the port, and what it answers with is built from
  call transcripts. `start_vllm.sh` refuses to start without one.
- **Raise the proxy timeout.** A 27B model takes several minutes on a long call.
  Many reverse proxies cut a request at ~100 seconds, which surfaces as a
  confusing gateway error rather than a timeout.

---

## 6. Configure, then prove it end to end

Everything lives in `config/config.yaml`, and every field can be overridden by
an environment variable prefixed `CALLQA_`, nested keys separated by double
underscores (`CALLQA_JUDGE__BASE_URL=...`). Secrets belong in `.env` or the
environment, never in the YAML.

Check the environment before committing a batch to it:

```bash
python -m callqa preflight          # config, models, hashes, endpoint, disk, inputs
```

Then one real call:

```bash
python -m callqa process --audio data/input/calls/<recording>.wav
echo "exit: $?"
```

---

## 7. Running it

| Command | What it does |
|---|---|
| `callqa process --audio F` | One call. **Exit `0` success, `1` needs human review, `2` failed.** Wire a scheduler straight into this. |
| `callqa watch` | Polls the input directory and processes new recordings as they land. |
| `callqa run` | Processes everything listed in `metadata.csv`. |
| `callqa report` | Per-banker aggregate reports plus an index. |
| `callqa review-queue` / `callqa review` | Lists calls held for a human, and records the human's verdict back into the calibration set. |
| `callqa calibrate` | Agreement between the judge and human raters (QWK). Needs ≥20 human-rated calls to mean anything. |
| `callqa verify <call>` | Is a stored result still reproducible, and if not, which input changed. |
| `callqa drift` | Flags movement in scores, review rate and quality signals against a known-good period. |
| `callqa eval` | Scores the whole system against a golden set. **Read §8 before quoting its numbers.** |
| `callqa retention` | Deletes raw PII-bearing artifacts past the retention window, with an audit log. |

The optional dashboard is `python dashboard/server.py`. It binds to 127.0.0.1
only, requires a session token, and is read-only. **The entire `dashboard/`
directory can be deleted** with no effect on the pipeline; a test enforces that
the core never imports it.

---

## 8. What the evaluation numbers do and do not mean

`callqa eval` against the **synthetic** golden set reports WER 0.0, CER 0.0,
role accuracy 1.0 and judge QWK 1.0. **These are not accuracy measurements.**
The synthetic references are the mock pipeline's own output, so those four
numbers are 0.0/1.0 by construction and will stay there no matter how well or
badly the real system performs. They are *regression sentinels*: they detect a
change from known-good, which is genuinely useful in CI and useless as a
quality claim.

Two exceptions, and they matter:

- **Redaction recall and precision are real**, because the gold identifiers are
  hand-labelled independently of the detector.
- Point `--golden-dir` at a set of real calls with human transcripts and human
  scores, and every number becomes a genuine measurement.

Building that real golden set is the highest-value thing the bank can do with
this system, and it is the one thing that cannot be done without the bank's own
data. Budget roughly 20–30 calls, transcribed and scored by a human QA reviewer.

---

## 9. For the security and compliance review

**What it stores, and where** — all under `data/output/`:

| Artifact | Contains PII? | Lifetime |
|---|---|---|
| `transcripts/*.json` (raw) | **yes** | `retention.raw_days`, default 90 |
| `audio/wav/` (raw) | **yes** | `retention.raw_days`, default 90 |
| `transcripts/*.redacted.json` | no | kept |
| `audio/redacted/` | no — silenced wherever the text was masked | kept |
| `scores/`, `features/`, `reports/`, `runs/`, `drift/` | no | kept |

`callqa retention` enforces the window and logs every deletion. Note that
unlinking a file on an SSD does not overwrite the bytes; full-disk encryption is
the backstop, and the tool records what it removed rather than claiming more
than it does.

**What leaves the machine:** one thing only — the judge request, to the
`judge.base_url` you configured, containing the **redacted** transcript. Nothing
else makes a network call at runtime. `grep -rn "urllib\|http" src/callqa/` will
show you every client in the code.

**PII handling:** masking happens in one place (`src/callqa/redaction.py`) and
everything downstream — the judge prompt, reports, logs, results JSON, the
dashboard — sees only its output. Two classes of identifier are handled
differently:

- *Shaped* identifiers (national ID, phone, payment card, IBAN, email, any run
  of six or more digits) are recognised by form, with checksums where one
  exists.
- *Shapeless* identifiers (mother's name, date of birth, address) have no form
  at all and are recognised by the verification question that precedes them,
  or by the caller naming themselves ("קוראים לי ...", "שמי ...").
- Account numbers are masked even when a currency word follows, with one
  deliberate exception: "בחשבון" (*in* the account) introduces a balance, and
  the amounts a banker quotes are what the compliance dimension is scored on.

The redacted **audio** is silenced wherever the redacted **text** is masked, by
whatever rule or model masked it. Where the audio stage cannot place a silence
precisely, it silences the whole turn rather than risk an audible identifier.

**Known limit, stated plainly:** a name mentioned in passing that nobody asked
for — a third party named mid-conversation — is not masked by the rules.
Nothing about the string marks it as an identifier and no question anchors it.
Setting `redaction.ner: true` adds a Hebrew NER model for exactly this, at a
performance cost; download it with `download_models.py --ner`. If it is enabled
and the model is missing, the pipeline **refuses to start** rather than
quietly running without it. This is a real residual risk with NER off, and the
bank should decide about it consciously.

**Name recordings by call, not by customer.** A recording's file name becomes
the call id, and the call id appears in every artifact path, report title and
log line. A file named after the customer's phone or ID number puts that
number everywhere. The pipeline warns when it sees a long digit run in a call
id, and masks it in validation messages, but it cannot rename your files.

**Licences:**

- This software: see `LICENSE` (proprietary, single client).
- Dependencies: run `python scripts/license_inventory.py -o licences.md` **on
  the deployed machine**. It reads what is actually installed and flags anything
  copyleft, restricted or undeclared. A list written by hand would be out of
  date; this is not.
- Models: `models/MODELS_MANIFEST.json` records the licence each model's own
  card declares, captured at download time. Several candidate judge models
  (notably Gemma- and Llama-family ones) ship under **bespoke community
  licences with usage restrictions, not OSI open-source licences**. Someone at
  the bank must read the licence for the judge model you actually choose.

---

## 10. Troubleshooting

| Symptom | Cause |
|---|---|
| 403 downloading the diarization model | Its licence has not been accepted on the HF account behind `HF_TOKEN`. §4.1. |
| `could not find rubric.yaml` | Running from outside the project directory. Pass `--config`, or set `CALLQA_CONFIG_DIR`. |
| Every call exits `2` on ingestion | ffmpeg is missing and the recordings are not plain WAV. |
| Judge times out | Raise `judge.request_timeout_sec` and the reverse proxy's timeout. §5. |
| Judge *refused* (401/403) | The server is running; `CALLQA_JUDGE__API_KEY` is missing or differs from the key the server was started with. |
| "redaction.ner is on, but DictaBERT-NER could not be loaded" | Working as designed: NER was requested and is missing. `download_models.py --ner`, or set `redaction.ner: false`. |
| Mono calls fail on diarization | The Hugging Face cache is not on this machine. §4.3. |
| Both speakers attributed to one side | The recording is dual-mono (one microphone copied to both channels). The pipeline detects this and falls back to diarization; check `attribution_mode` in the transcript. |
| Calls held with `needs_human_review` | Working as designed: role confidence below `speakers.min_role_confidence`, or the judge could not produce verified evidence. Use `callqa review-queue`. |
| A call cannot be reproduced | `callqa verify <call>` names the input that changed. |

---

## 11. What is not here

Stated so nobody discovers it late:

- **No authentication, authorisation or multi-tenancy.** It is a CLI. Whoever
  can run it sees everything it produces. Access control is filesystem-level and
  is yours to impose.
- **No web service.** The dashboard is a local single-user console bound to
  loopback, and it is deletable.
- **No emotion recognition.** Explicitly out of scope.
- **No judge accuracy figure.** There is no measured agreement between this
  system and human QA reviewers yet, because that requires human-rated calls the
  project has never had. `callqa calibrate` computes it as soon as there are
  ≥20. Until then, treat the scores as a triage aid, not a verdict on a banker.
