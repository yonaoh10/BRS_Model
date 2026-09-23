# Call-QA PoC — Hebrew Conversation Intelligence for Banker Calls

**🌐 [English](#english) · [עברית](#hebrew)**

<a id="english"></a>

A production-grade, single-call pipeline that takes one banker↔customer
recording (Hebrew) and produces the complete per-call output:

1. **Transcription** — ivrit.ai Whisper (faster-whisper CT2), Hebrew forced.
2. **Speaker attribution** — stereo channel split (primary); pyannote
   diarization fallback for mono recordings.
3. **PII redaction** — Hebrew-aware (Israeli ID with checksum, phones,
   payment cards, aggressive account-like numbers (any run of 6+ digits), names).
4. **Objective features** — talk ratio, interruptions, patience, questions,
   monologue length, dead air, speech rate.
5. **LLM judge** — a weighted 8-dimension rubric scored by a locally served
   vLLM model, with verbatim evidence quotes (verified, never fabricated).
6. **Per-call HTML report** — Hebrew, RTL, fully offline (inline CSS).

Around that core sit three thin layers: drivers (`process` / `watch` / `run`),
per-banker aggregation (`report`), and judge calibration vs. human QA ratings
(`calibrate`, QWK).

**Everything runs end-to-end in mock mode on a machine with no GPU and no
models** — deterministic fake engines behind the same interfaces, switched to
real engines by config only. Emotion recognition is explicitly out of scope.

## Quick start (any machine, mock mode)

No GPU, no models, no network. This proves the software is installed and the
whole chain works before anything is downloaded:

```bash
./scripts/first_run.sh            # mock pipeline on synthetic calls, then the dashboard
```

Deploying it for real? Read **[docs/DEPLOYMENT.md](docs/DEPLOYMENT.md)** — it is
the front door: prerequisites, models, serving, verification and licensing.

Step by step:

```bash
pip install -r requirements.txt
pip install -e .

python scripts/generate_sample_data.py          # 6 synthetic stereo WAVs + CSVs

# The atomic unit of work - ONE call (primary acceptance test):
python -m callqa process --mock --audio data/input/calls/CALL001.wav

# PoC batch + aggregate reports + calibration:
python -m callqa run --mock
python -m callqa report --mock
python -m callqa calibrate --mock
# open data/output/reports/index.html

make test                                        # pytest suite
```

Exit codes for `process`: `0` success · `1` needs_human_review · `2` failed —
wire any external scheduler or recording-system hook straight into it.

## CLI

| Command | Purpose |
|---|---|
| `callqa process --audio F [--call-id --banker-id --banker-channel]` | one call, full pipeline |
| `callqa watch` | production ingestion: poll input dir, process each stable file once, move to `processed/`/`failed/` |
| `callqa run [--max-workers N]` | PoC driver over `data/input/metadata.csv` |
| `callqa report` | per-banker reports + `reports/index.html` |
| `callqa calibrate` | QWK vs `human_ratings.csv` → `reports/calibration.html` |
| `callqa validate-inputs` | strict metadata / ratings validation |

All commands accept `--config`, `--mock`, `--force`. Any config field is
env-overridable: `CALLQA_SECTION__FIELD` (e.g. `CALLQA_JUDGE__BASE_URL`).

## Input contract

- `data/input/calls/*.{wav,mp3}` — file stem (or metadata) = `call_id`.
- `data/input/metadata.csv` — required: `call_id, banker_id, file_name`;
  optional: `call_date, call_type, banker_channel (L/R), banker_name`.
- `data/input/human_ratings.csv` — `call_id, rater_id,` one column per rubric
  dimension id, scores 1–5 (1–2 raters per call).

## Design guarantees

- **Idempotent + resumable**: per call×stage state in SQLite; re-runs skip
  completed stages; `--force` reruns. Artifacts written atomically.
- **Per-call atomicity**: an SQLite lock per `call_id` prevents double
  processing; every call ends in an explicit status envelope; one bad call
  never stops a driver.
- **Privacy**: only redacted text reaches the judge, reports, and logs. Raw
  transcripts live only under `data/output/transcripts/` (with a warning
  README). Logs carry call_ids and stage names, never transcript content.
- **Determinism/audit**: judge temperature 0.0; every scorecard stores model
  id, prompt SHA-256, prompt version, retries, latency, timestamp.
- **Evidence verification**: every judge quote must appear verbatim in the
  redacted transcript or the response is rejected and retried; after final
  failure the call is marked `needs_human_review` — never fabricated.
- **Gate rule**: a gate dimension (identification, compliance) scored ≤2 caps
  the total at 59 and flags the call.

---


---

# Bank-server deployment runbook

The developer machine never downloads models. All model files are fetched on
the bank server by `scripts/download_models.py`.

1. **Transfer** the repository plus the wheels bundle. Build the bundle on an
   internet-connected machine first:
   `./scripts/build_offline_bundle.sh` (creates `wheels/`).
2. **Install offline** on the server:
   `./scripts/install_offline.sh --server`
   (installs `requirements.txt` + `requirements-server.txt` from `wheels/`,
   then the `callqa` package — no network).
3. **Install ffmpeg**: `apt/yum install ffmpeg`, or drop a static build of
   `ffmpeg`/`ffprobe` into `$PATH` on a fully offline host.
4. **Download models** per available VRAM (this step needs internet or
   pre-staged files):

   | VRAM | ASR | Judge LLM (`--llm-model`) | vLLM flags |
   |---|---|---|---|
   | 24 GB | `--asr` (ivrit CT2, ~1.6GB) | `dicta-il/dictalm2.0-instruct` (7B fp16), or a 12–27B AWQ/GPTQ build | `--quantization awq` for AWQ |
   | ≥48 GB | `--asr` | `meta-llama/Llama-3.3-70B-Instruct` AWQ (gated) | `--quantization awq --tensor-parallel-size 2` |

   ```bash
   python scripts/download_models.py --asr
   python scripts/download_models.py --llm --llm-model dicta-il/dictalm2.0-instruct
   # only if recordings turn out to be mono (gated; accept terms + set HF_TOKEN):
   python scripts/download_models.py --diarization
   # optional NER-based person redaction:
   python scripts/download_models.py --ner    # then set redaction.ner: true
   ```
   The script writes `models/MODELS_MANIFEST.json` and points `judge.model`
   in `config/config.yaml` at the downloaded LLM.
5. **Go offline**: `export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1`.
6. **Serve the judge**: `./scripts/start_vllm.sh <model-id-or-path> 8000`
   (see the script header for quantization flags). The pipeline only checks
   connectivity to `judge.base_url`; it never launches vLLM.
7. **Place inputs**: recordings under `data/input/calls/`, plus
   `metadata.csv` and (for calibration) `human_ratings.csv`.
8. **Validate**: `python -m callqa validate-inputs` — fails with a problem
   table on any metadata issue.
9. **Process the PoC batch**: `python -m callqa run` (add
   `--max-workers 2` if the GPU has headroom). In ongoing production use
   `python -m callqa watch` instead — it polls the input directory and
   processes each new recording once its file size is stable.
10. **Calibrate**: `python -m callqa calibrate` →
    `data/output/reports/calibration.html` (PASS at overall QWK ≥ 0.70;
    dimensions with QWK < 0.60 are flagged "do not deploy without human
    review").
11. **Reports**: `python -m callqa report` →
    `data/output/reports/index.html` linking every per-call and per-banker
    report (all static HTML, no external assets).

## Troubleshooting

| Symptom | Fix |
|---|---|
| CUDA OOM (ASR) | set `asr.compute_type: int8` in config |
| CUDA OOM (vLLM) | use an AWQ/GPTQ model, lower `--max-model-len`, `--gpu-memory-utilization 0.9` |
| "mono recording but no diarizer" / mono files detected | install server deps, `download_models.py --diarization` with `HF_TOKEN` (accept pyannote terms on HF) |
| vLLM endpoint unreachable | start `scripts/start_vllm.sh`; check `judge.base_url` port; `curl localhost:8000/v1/models` |
| Low ASR confidence (`quality` block flags many segments) | check recording sample rate/noise; confirm `asr.language: he`; consider re-recording setup |
| ASR model directory missing/empty | run `download_models.py --asr` and check `paths.models_dir` |
| Call stuck as "already being processed" | previous process died mid-call: the lock is auto-stolen when the pid is dead; otherwise delete the row from the `locks` table in the state DB |

## Repository layout

See `src/callqa/` — `pipeline.py` (`process_call()`, the production core),
`engines.py` (DI container built once per process), stage modules
(`ingestion`, `audio`, `asr/`, `speakers/`, `redaction`, `features`,
`judge/`), `aggregation.py`, `calibration.py`, `reporting/`, and `cli.py`.
Config lives in `config/` (`config.yaml`, `rubric.yaml`,
`recommendations_he.yaml`). Tests in `tests/` (`make test`).

<a id="hebrew"></a>

---
---

# עברית — מערכת בקרת איכות שיחות בנקאיות

**🌐 [English](#english) · [עברית](#hebrew)**

מערכת ברמת ייצור שמקבלת הקלטה **אחת** של שיחה בין בנקאי ללקוח (בעברית)
ומפיקה את מלוא התוצר עבור אותה שיחה:

1. **תמלול** — מודל ivrit.ai Whisper (faster-whisper CT2), עברית כפויה.
2. **שיוך דוברים** — פיצול ערוצי סטריאו (המסלול העיקרי); pyannote כגיבוי
   להקלטות מונו.
3. **הסרת פרטים מזהים (PII)** — מותאם לעברית (תעודת זהות עם ספרת ביקורת,
   טלפונים, כרטיסי אשראי, מספרי חשבון (כל רצף של 6 ספרות ומעלה), שמות).
4. **מדדים אובייקטיביים** — יחס דיבור, קטיעות, סבלנות, שאלות, אורך מונולוג,
   זמן שקט, קצב דיבור.
5. **שופט LLM** — מחוון משוקלל בן 8 ממדים, מדורג על ידי מודל המוגש מקומית
   ב-vLLM, עם ציטוטי ראיה מילה במילה (מאומתים, לעולם לא מומצאים).
6. **דוח HTML לכל שיחה** — בעברית, RTL, לחלוטין אופליין (CSS מוטמע).

סביב הליבה יושבות שלוש שכבות דקות: מנועי הרצה
(`process` / `watch` / `run`), אגרגציה ברמת הבנקאי (`report`), וכיול השופט
מול הערכות אנושיות (`calibrate`, QWK).

**הכול רץ מקצה לקצה במצב mock על מכונה ללא GPU וללא מודלים** — מנועים
דמה דטרמיניסטיים מאחורי אותם ממשקים בדיוק, שמתחלפים למנועים אמיתיים
באמצעות קונפיגורציה בלבד. זיהוי רגשות מוחרג במפורש מהמערכת.

## התחלה מהירה (כל מחשב, מצב mock)

בלי GPU, בלי מודלים, בלי רשת. זה מוכיח שהתוכנה מותקנת ושכל השרשרת עובדת,
עוד לפני שמורידים משהו:

```bash
./scripts/first_run.sh            # פייפליין mock על שיחות סינתטיות, ואז הדשבורד
```

פורסים לסביבה אמיתית? קראו את **[docs/DEPLOYMENT.md](docs/DEPLOYMENT.md)** —
זו דלת הכניסה: דרישות מוקדמות, מודלים, הרצת שירות, אימות ורישוי.

שלב אחר שלב:

```bash
pip install -r requirements.txt
pip install -e .

python scripts/generate_sample_data.py          # 6 קבצי WAV סטריאו סינתטיים + CSV

# יחידת העבודה האטומית - שיחה אחת (מבחן הקבלה העיקרי):
python -m callqa process --mock --audio data/input/calls/CALL001.wav

# אצוות ה-PoC + דוחות מסכמים + כיול:
python -m callqa run --mock
python -m callqa report --mock
python -m callqa calibrate --mock
# פתח את data/output/reports/index.html

make test                                        # חבילת הבדיקות
```

קודי יציאה עבור `process`: `0` הצלחה · `1` נדרשת בדיקה אנושית ·
`2` כישלון — ניתן לחבר ישירות כל מתזמן חיצוני או hook ממערכת ההקלטות.

## פקודות CLI

| פקודה | תפקיד |
|---|---|
| `callqa process --audio F [--call-id --banker-id --banker-channel]` | שיחה אחת, כל הצינור |
| `callqa watch` | קליטה בייצור: סריקת תיקיית הקלט, עיבוד כל קובץ יציב פעם אחת, העברה ל-`processed/`/`failed/` |
| `callqa run [--max-workers N]` | מנוע ההרצה של ה-PoC על `data/input/metadata.csv` |
| `callqa report` | דוחות לכל בנקאי + `reports/index.html` |
| `callqa calibrate` | QWK מול `human_ratings.csv` ← `reports/calibration.html` |
| `callqa validate-inputs` | ולידציה קפדנית של המטא-דאטה והדירוגים |

כל הפקודות מקבלות `--config`, `--mock`, `--force`. כל שדה בקונפיגורציה
ניתן לדריסה דרך משתני סביבה: `CALLQA_SECTION__FIELD`
(למשל `CALLQA_JUDGE__BASE_URL`).

## חוזה הקלט

- `data/input/calls/*.{wav,mp3}` — שם הקובץ (או המטא-דאטה) = `call_id`.
- `data/input/metadata.csv` — חובה: `call_id, banker_id, file_name`;
  רשות: `call_date, call_type, banker_channel (L/R), banker_name`.
- `data/input/human_ratings.csv` — `call_id, rater_id,` ועמודה אחת לכל ממד
  במחוון, ציונים 1–5 (1–2 מעריכים לכל שיחה).

## הבטחות התכנון

- **אידמפוטנטי וניתן להמשך**: מצב לכל שיחה×שלב ב-SQLite; הרצה חוזרת מדלגת
  על שלבים שהושלמו; `--force` מריץ מחדש. כל תוצר נכתב אטומית.
- **אטומיות ברמת השיחה**: נעילת SQLite לכל `call_id` מונעת עיבוד כפול;
  כל שיחה מסתיימת במעטפת סטטוס מפורשת; שיחה בעייתית אחת לעולם לא עוצרת
  את מנוע ההרצה.
- **פרטיות**: רק טקסט מצונזר מגיע לשופט, לדוחות וללוגים. תמלילים גולמיים
  שמורים רק תחת `data/output/transcripts/` (עם קובץ אזהרה). הלוגים מכילים
  מזהי שיחה ושמות שלבים בלבד, לעולם לא תוכן תמליל.
- **דטרמיניזם וניתן לביקורת**: טמפרטורת השופט 0.0; כל כרטיס ציונים שומר
  מזהה מודל, SHA-256 של הפרומפט, גרסת פרומפט, מספר ניסיונות חוזרים, זמן
  תגובה וחותמת זמן.
- **אימות ראיות**: כל ציטוט של השופט חייב להופיע מילה במילה בתמליל המצונזר,
  אחרת התשובה נדחית ומתבצע ניסיון חוזר; לאחר כישלון סופי השיחה מסומנת
  `needs_human_review` — לעולם לא מומצא מידע.
- **כלל שער**: ממד שער (זיהוי, ציות) שקיבל ציון 2 ומטה מגביל את הציון
  הכולל ל-59 ומסמן את השיחה.

---


---

# מדריך התקנה בשרת הבנק

מכונת הפיתוח לעולם אינה מורידה מודלים. כל קבצי המודלים נמשכים בשרת הבנק
באמצעות `scripts/download_models.py`.

1. **העברה** של המאגר יחד עם חבילת ה-wheels. יש לבנות את החבילה קודם על
   מכונה עם אינטרנט: `./scripts/build_offline_bundle.sh` (יוצר `wheels/`).
2. **התקנה אופליין** בשרת: `./scripts/install_offline.sh --server`
   (מתקין `requirements.txt` + `requirements-server.txt` מתוך `wheels/`,
   ולאחר מכן את חבילת `callqa` — ללא רשת).
3. **התקנת ffmpeg**: `apt/yum install ffmpeg`, או הצבת בינארי סטטי של
   `ffmpeg`/`ffprobe` בתוך `$PATH` בשרת מנותק לחלוטין.
4. **הורדת מודלים** לפי נפח ה-VRAM הזמין (שלב זה דורש אינטרנט או קבצים
   שהוכנו מראש):

   | VRAM | ASR | מודל השופט (`--llm-model`) | דגלי vLLM |
   |---|---|---|---|
   | 24 GB | `--asr` (ivrit CT2, ~1.6GB) | `dicta-il/dictalm2.0-instruct` (7B fp16), או מודל 12–27B בכימות AWQ/GPTQ | `--quantization awq` עבור AWQ |
   | ≥48 GB | `--asr` | `meta-llama/Llama-3.3-70B-Instruct` AWQ (מוגבל גישה) | `--quantization awq --tensor-parallel-size 2` |

   ```bash
   python scripts/download_models.py --asr
   python scripts/download_models.py --llm --llm-model dicta-il/dictalm2.0-instruct
   # רק אם מתברר שההקלטות במונו (מוגבל גישה; יש לאשר תנאים ולהגדיר HF_TOKEN):
   python scripts/download_models.py --diarization
   # רשות: הסרת שמות אנשים מבוססת NER:
   python scripts/download_models.py --ner    # ואז יש להגדיר redaction.ner: true
   ```
   הסקריפט כותב `models/MODELS_MANIFEST.json` ומעדכן את `judge.model`
   בקובץ `config/config.yaml` כך שיצביע על המודל שהורד.
5. **מעבר למצב אופליין**: `export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1`.
6. **הפעלת השופט**: `./scripts/start_vllm.sh <model-id-or-path> 8000`
   (דגלי הכימות מפורטים בכותרת הסקריפט). הצינור רק בודק קישוריות אל
   `judge.base_url`; הוא לעולם אינו מפעיל את vLLM בעצמו.
7. **הצבת הקלט**: הקלטות תחת `data/input/calls/`, יחד עם `metadata.csv`
   ו-(לצורך כיול) `human_ratings.csv`.
8. **ולידציה**: `python -m callqa validate-inputs` — נכשל עם טבלת בעיות
   בכל תקלה במטא-דאטה.
9. **עיבוד אצוות ה-PoC**: `python -m callqa run` (ניתן להוסיף
   `--max-workers 2` אם יש עודף משאבים ב-GPU). בייצור שוטף יש להשתמש
   ב-`python -m callqa watch` — הוא סורק את תיקיית הקלט ומעבד כל הקלטה
   חדשה ברגע שגודל הקובץ שלה מתייצב.
10. **כיול**: `python -m callqa calibrate` ←
    `data/output/reports/calibration.html` (עובר כאשר QWK כולל ≥ 0.70;
    ממדים עם QWK < 0.60 מסומנים "אין לפרוס ללא בקרה אנושית").
11. **דוחות**: `python -m callqa report` ←
    `data/output/reports/index.html` שמקשר לכל דוחות השיחות והבנקאים
    (הכול HTML סטטי, ללא נכסים חיצוניים).

## פתרון תקלות

| תסמין | פתרון |
|---|---|
| CUDA OOM (תמלול) | הגדר `asr.compute_type: int8` בקונפיגורציה |
| CUDA OOM (vLLM) | השתמש במודל AWQ/GPTQ, הקטן את `--max-model-len`, הוסף `--gpu-memory-utilization 0.9` |
| "mono recording but no diarizer" / זוהו קבצי מונו | התקן את תלויות השרת, הרץ `download_models.py --diarization` עם `HF_TOKEN` (יש לאשר את תנאי pyannote ב-HF) |
| נקודת הקצה של vLLM אינה זמינה | הפעל את `scripts/start_vllm.sh`; בדוק את הפורט ב-`judge.base_url`; `curl localhost:8000/v1/models` |
| ביטחון תמלול נמוך (בלוק `quality` מסמן הרבה מקטעים) | בדוק את קצב הדגימה והרעש בהקלטה; ודא `asr.language: he`; שקול לשנות את מערך ההקלטה |
| תיקיית מודל ה-ASR חסרה או ריקה | הרץ `download_models.py --asr` ובדוק את `paths.models_dir` |
| שיחה תקועה במצב "already being processed" | תהליך קודם קרס באמצע: הנעילה נגנבת אוטומטית כאשר ה-pid מת; אחרת מחק את השורה מטבלת `locks` במסד המצב |

## מבנה המאגר

ראה `src/callqa/` — `pipeline.py` (`process_call()`, ליבת הייצור),
`engines.py` (מכל הזרקת תלויות שנבנה פעם אחת לכל תהליך), מודולי השלבים
(`ingestion`, `audio`, `asr/`, `speakers/`, `redaction`, `features`,
`judge/`), `aggregation.py`, `calibration.py`, `reporting/`, ו-`cli.py`.
הקונפיגורציה נמצאת ב-`config/` (`config.yaml`, `rubric.yaml`,
`recommendations_he.yaml`). הבדיקות ב-`tests/` (`make test`).
