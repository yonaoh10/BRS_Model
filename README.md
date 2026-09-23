# callqa

**🌐 [עברית](#hebrew) · [English](#english)**

<a id="hebrew"></a>

<div dir="rtl">

## מה זה

תוכנה שמקבלת הקלטה של שיחה טלפונית בעברית בין בנקאי ללקוח, ומפיקה דוח בקרת איכות על השיחה.
היא מתמללת את השיחה, מפרידה בין הדוברים, מסתירה פרטים מזהים של הלקוח (ת"ז, טלפון, שם וכו'), ונותנת ציון לשיחה בעזרת מודל שפה שרץ על התשתית של הבנק.

- זו תוכנת שורת פקודה (terminal). אין בה אתר ואין בה שרת.
- **היא לא פונה לאינטרנט בזמן ריצה.** יש לה רק שני חיבורי רשת: שרת מודל השיפוט שאתם מגדירים (שלב 5), והורדה חד־פעמית של המודלים (שלב 4).

כל הפקודות במדריך מריצים ב־terminal, **מתוך תיקיית הפרויקט**.

---

## מה צריך לפני שמתחילים

| | דרישה | איך בודקים |
|---|---|---|
| מערכת הפעלה | Linux (מומלץ) או macOS | — |
| Python | גרסה 3.11 או 3.12 | `python3 --version` |
| ffmpeg | נדרש כדי לקרוא הקלטות אמיתיות | `ffmpeg -version` |
| דיסק פנוי | 40GB לפחות | `df -h .` |
| זיכרון | 16GB לפחות | — |
| כרטיס מסך (GPU) של NVIDIA | נדרש **רק** למודל השיפוט (שלב 5) | `nvidia-smi` |

אם ffmpeg חסר:
- Ubuntu/Debian: `sudo apt install ffmpeg`
- RHEL/Rocky: `sudo dnf install ffmpeg`
- macOS: `brew install ffmpeg`

---

## שלב 1 — הורדה ופתיחה

1. ב־GitHub לוחצים **Code ← Download ZIP**.
2. פותחים את הקובץ. נוצרת תיקייה בשם `BRS_Model-main`.
3. נכנסים אליה ב־terminal:

```bash
cd BRS_Model-main
```

---

## שלב 2 — התקנה

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

- הפקודה הראשונה יוצרת "סביבה וירטואלית": תיקייה נפרדת שמחזיקה את כל ספריות ה־Python של הפרויקט, בלי לגעת במערכת.
- **בכל פעם שפותחים terminal חדש** צריך להפעיל אותה מחדש עם `source .venv/bin/activate`.

---

## שלב 3 — בדיקה שהכול עובד (בלי מודלים)

```bash
./scripts/first_run.sh
```

הסקריפט יוצר 6 שיחות לדוגמה ומריץ עליהן את כל התהליך במצב בדיקה (mock). המצב הזה לא דורש מודלים, כרטיס מסך או אינטרנט. בסוף הוא פותח את לוח הבקרה (עוצרים עם `Ctrl+C`).

**הבדיקה הצליחה אם** הקובץ `data/output/reports/index.html` נפתח בדפדפן ומציג 6 שיחות עם ציונים.

כדאי להריץ גם את חבילת הבדיקות האוטומטיות (לוקח כדקה):

```bash
python -m pytest -q
```

---

## שלב 4 — הורדת המודלים

המודלים (בערך 20–25GB) **לא** נמצאים בפרויקט, וצריך להוריד אותם פעם אחת. זה השלב היחיד שדורש אינטרנט.

**4.1 — אישור הרישיון של מודל הפרדת הדוברים** (בלי זה ההורדה תיכשל עם שגיאה 403):

1. נכנסים עם חשבון Hugging Face לעמוד https://huggingface.co/pyannote/speaker-diarization-community-1 ומאשרים את התנאים.
2. יוצרים טוקן קריאה בעמוד https://huggingface.co/settings/tokens.
3. שומרים אותו בקובץ `.env`:

```bash
cp .env.example .env
# פתחו את .env בעורך טקסט וכתבו בשורה המתאימה:  HF_TOKEN=hf_...
```

**4.2 — ההורדה:**

```bash
pip install -r requirements-server.txt
pip install huggingface_hub
python scripts/download_models.py --all --llm-model dicta-il/dictalm2.0-instruct
```

- `--llm-model` קובע איזה מודל ישמש לשיפוט. הדוגמה מתאימה לכרטיס מסך של 16–24GB. אפשרויות נוספות מופיעות ב־`python scripts/download_models.py --help`.
- בסוף ההורדה מודפסת טבלה שמסכמת אילו מודלים ירדו בהצלחה (OK) ואילו נכשלו (FAILED), ומה עושים אם משהו נכשל.
- הסקריפט רושם לכל מודל את הרישיון שלו בקובץ `models/MODELS_MANIFEST.json`, ומעדכן לבד את מודל השיפוט בקובץ `config/config.yaml`.
- אם אין במחשב כרטיס מסך של NVIDIA, צריך לשנות בקובץ `config/config.yaml` את השורה `compute_type: float16` ל־`compute_type: int8`.

---

## שלב 5 — הפעלת מודל השיפוט

מודל השיפוט צריך לרוץ כשרת נפרד. אפשר להשתמש בכל שרת שתומך בממשק OpenAI (`/v1/chat/completions`), למשל vLLM, TGI או שרת פנימי שכבר קיים בבנק.

vLLM **לא** מותקן כחלק מהפרויקט. אם בוחרים בו, מתקינים אותו בנפרד (`pip install vllm`) ומריצים:

```bash
export CALLQA_JUDGE_API_KEY=$(openssl rand -hex 32)
./scripts/start_vllm.sh models/dicta-il--dictalm2.0-instruct 8000
```

אחר כך מעתיקים את אותו מפתח לקובץ `.env`, בשורה:

```
CALLQA_JUDGE__API_KEY=<אותו ערך>
```

⚠️ שימו לב: בשם הראשון יש קו תחתון **אחד** (הוא מיועד לשרת), ובשני יש **שניים** (הוא מיועד לתוכנה). אם המפתחות לא זהים, התוכנה תדווח שהשרת "סירב (401)".

---

## שלב 6 — הגדרות

כל ההגדרות נמצאות בקובץ `config/config.yaml`, ולכל אחת יש הסבר בגוף הקובץ. אלה ההגדרות שכדאי לבדוק:

| הגדרה | מה היא עושה |
|---|---|
| `judge.base_url` | הכתובת של שרת מודל השיפוט. ברירת המחדל היא `http://localhost:8000/v1`. |
| `retention.raw_days` | אחרי כמה ימים נמחקים הנתונים הגולמיים עם הפרטים המזהים. ברירת המחדל היא 90. |
| `redaction.ner` | הסתרה של שמות שאף אחד לא שאל עליהם. מחייב את `python scripts/download_models.py --ner`. |

סיסמאות ומפתחות שמים **רק** בקובץ `.env`, אף פעם לא בקובץ `config.yaml`.

---

## שלב 7 — הכנת השיחות

```bash
mkdir -p data/input/calls
```

1. מעתיקים את ההקלטות לתיקייה `data/input/calls/`.
2. יוצרים את הקובץ `data/input/metadata.csv`, עם שורה אחת לכל הקלטה:

```
call_id,banker_id,file_name,banker_channel,banker_name
C0001,B17,C0001.wav,L,דנה כהן
```

| עמודה | חובה | משמעות |
|---|---|---|
| `call_id` | כן | מזהה ייחודי לשיחה (אותיות באנגלית, ספרות, `_` `-` `.`) |
| `banker_id` | כן | מזהה הבנקאי |
| `file_name` | כן | שם קובץ ההקלטה בתוך `calls/` |
| `banker_channel` | לא | `L` או `R`: באיזה ערוץ נמצא הבנקאי, בהקלטת סטריאו |
| `banker_name` | לא | שם הבנקאי. יוסתר בתמלול. |

⚠️ **אל תקראו לקובצי ההקלטה על שם הלקוח** (למשל לפי מספר ת"ז או טלפון). שם הקובץ מופיע בכל הדוחות ובכל הלוגים.

---

## שלב 8 — בדיקה והרצה

```bash
python -m callqa preflight          # בודק שהמודלים, השרת, הדיסק והקלט מוכנים
python -m callqa run                # מעבד את כל השיחות שב-metadata.csv
python -m callqa report             # מפיק דוח לכל בנקאי
```

את `preflight` כדאי להריץ לפני כל הרצה. אם אחת השורות שלו מסומנת `FAIL`, מתקנים אותה לפני שממשיכים.

**הרצה אוטומטית:**
- `python -m callqa watch` — מאזין לתיקייה `data/input/calls/` ומעבד כל הקלטה חדשה שמגיעה.
- `python -m callqa process --audio <קובץ>` — מעבד שיחה אחת. קוד היציאה: `0` הצלחה · `1` נדרשת בדיקה אנושית · `2` כישלון. כך אפשר לחבר את זה לכל מתזמן משימות.

---

## מה נוצר ואיפה

| מיקום | מה יש שם |
|---|---|
| `data/output/reports/index.html` | הדוחות. **מתחילים מכאן.** |
| `data/output/transcripts/` | ⚠️ תמלול **גולמי** עם פרטים מזהים. `python -m callqa retention --apply` מוחק אותו אחרי מספר הימים שמוגדר ב־`retention.raw_days`. |
| `data/output/redacted/` | תמלול אחרי הסתרת הפרטים המזהים |
| `data/output/redacted_audio/` | הקלטה שבה הפרטים המזהים מושתקים |

לוח הבקרה, אופציונלי: `python dashboard/server.py`. הפקודה מדפיסה כתובת שפותחים בדפדפן. אפשר למחוק את כל התיקייה `dashboard/` בלי שום השפעה על שאר המערכת.

---

## שרת בלי חיבור לאינטרנט

מכינים הכול על מחשב שיש בו אינטרנט, ומעבירים לשרת.

**על המחשב עם האינטרנט:**

```bash
./scripts/build_offline_bundle.sh
python scripts/download_models.py --all --llm-model dicta-il/dictalm2.0-instruct
```

**מעבירים לשרת:** את תיקיית הפרויקט (כולל `wheels/` ו־`models/`), ואת התיקייה `~/.cache/huggingface`.

**על השרת:**

```bash
./scripts/install_offline.sh --server
source .venv/bin/activate
export HF_HOME=<המיקום שאליו הועתקה התיקייה huggingface>
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
python -m callqa preflight
```

---

## תקלות נפוצות

| הודעה או מצב | מה עושים |
|---|---|
| `403` בהורדת מודל | לא אושר הרישיון בשלב 4.1, או ש־`HF_TOKEN` חסר בקובץ `.env` |
| שרת השיפוט "unreachable" | השרת לא רץ, או שהכתובת ב־`judge.base_url` שגויה |
| שרת השיפוט "refused (401/403)" | המפתח ב־`CALLQA_JUDGE__API_KEY` שונה מהמפתח של השרת (שלב 5) |
| `size MISMATCH` ב־preflight | מודל הועתק רק באופן חלקי. מעתיקים אותו שוב. |
| `model:diarization` מסומן WARN | ההקלטות שלכם בסטריאו? אז אפשר להתעלם. אחרת, ראו "שרת בלי חיבור לאינטרנט". |
| כל השיחות נכשלות בשלב הראשון | חסר ffmpeg |
| `externally-managed-environment` | שכחתם להפעיל את `source .venv/bin/activate` |

---

## מסמכים נוספים

- `docs/DEPLOYMENT.md` — פירוט מלא: אבטחה, רישוי, מה נשמר ולכמה זמן, ומגבלות ידועות.
- `docs/MLOPS.md` — תפעול לאורך זמן: הערכה, ניטור שינויים, שחזור תוצאות.
- `CHANGELOG.md` — מה השתנה בכל גרסה.

</div>

---

<a id="english"></a>

## What this is

Software that takes a recording of a Hebrew phone call between a banker and a customer and produces a quality-assurance report on it.
It transcribes the call, separates the two speakers, hides the customer's identifiers (ID, phone, name, etc.), and scores the call with a language model that runs on the bank's own infrastructure.

- It is a command-line (terminal) program. There is no website and no server.
- **It never contacts the internet at runtime.** It has only two network connections: the judge model server you configure (step 5), and a one-time model download (step 4).

Run every command in this guide in a terminal, **from inside the project folder**.

---

## Before you start

| | Requirement | How to check |
|---|---|---|
| Operating system | Linux (recommended) or macOS | — |
| Python | 3.11 or 3.12 | `python3 --version` |
| ffmpeg | Needed to read real recordings | `ffmpeg -version` |
| Free disk | At least 40 GB | `df -h .` |
| Memory | At least 16 GB | — |
| NVIDIA GPU | Needed **only** for the judge model (step 5) | `nvidia-smi` |

If ffmpeg is missing:
- Ubuntu/Debian: `sudo apt install ffmpeg`
- RHEL/Rocky: `sudo dnf install ffmpeg`
- macOS: `brew install ffmpeg`

---

## Step 1 — Download and unpack

1. On GitHub, click **Code → Download ZIP**.
2. Unpack it. This creates a folder named `BRS_Model-main`.
3. Go into it in a terminal:

```bash
cd BRS_Model-main
```

---

## Step 2 — Install

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

- The first command creates a "virtual environment": a separate folder that holds all of the project's Python libraries, without touching the system.
- **Every time you open a new terminal**, activate it again with `source .venv/bin/activate`.

---

## Step 3 — Check that everything works (no models needed)

```bash
./scripts/first_run.sh
```

The script creates 6 sample calls and runs the whole process on them in test mode (mock). Test mode needs no models, no GPU and no internet. At the end it opens the dashboard (stop it with `Ctrl+C`).

**The check passed if** `data/output/reports/index.html` opens in a browser and shows 6 scored calls.

It is also worth running the automated test suite (about a minute):

```bash
python -m pytest -q
```

---

## Step 4 — Download the models

The models (about 20–25 GB) are **not** included in the project and have to be downloaded once. This is the only step that needs the internet.

**4.1 — Accept the speaker-separation model's licence** (without this, the download fails with a 403 error):

1. Using a Hugging Face account, go to https://huggingface.co/pyannote/speaker-diarization-community-1 and accept the conditions.
2. Create a read token at https://huggingface.co/settings/tokens.
3. Save it in the `.env` file:

```bash
cp .env.example .env
# open .env in a text editor and fill in the matching line:  HF_TOKEN=hf_...
```

**4.2 — Download:**

```bash
pip install -r requirements-server.txt
pip install huggingface_hub
python scripts/download_models.py --all --llm-model dicta-il/dictalm2.0-instruct
```

- `--llm-model` sets which model is used for judging. The example fits a 16–24 GB GPU. Other options are listed in `python scripts/download_models.py --help`.
- At the end it prints a table of which models downloaded successfully (OK) and which failed (FAILED), with what to do about each failure.
- The script records each model's licence in `models/MODELS_MANIFEST.json` and updates the judge model in `config/config.yaml` for you.
- If the machine has no NVIDIA GPU, change the line `compute_type: float16` to `compute_type: int8` in `config/config.yaml`.

---

## Step 5 — Start the judge model

The judge model has to run as a separate server. Any server that supports the OpenAI interface (`/v1/chat/completions`) works: for example vLLM, TGI, or an internal server the bank already runs.

vLLM is **not** installed as part of this project. If you choose it, install it separately (`pip install vllm`) and run:

```bash
export CALLQA_JUDGE_API_KEY=$(openssl rand -hex 32)
./scripts/start_vllm.sh models/dicta-il--dictalm2.0-instruct 8000
```

Then copy the same key into `.env`, on this line:

```
CALLQA_JUDGE__API_KEY=<the same value>
```

⚠️ Note: the first name has **one** underscore (it is for the server); the second has **two** (it is for the program). If the keys are not identical, the program reports that the server "refused (401)".

---

## Step 6 — Settings

All settings are in `config/config.yaml`, and each one is explained inside the file. These are the settings worth checking:

| Setting | What it does |
|---|---|
| `judge.base_url` | The address of the judge model server. The default is `http://localhost:8000/v1`. |
| `retention.raw_days` | After how many days the raw data containing identifiers is deleted. The default is 90. |
| `redaction.ner` | Hides names that nobody asked for. Requires `python scripts/download_models.py --ner`. |

Put passwords and keys **only** in `.env`, never in `config.yaml`.

---

## Step 7 — Prepare the calls

```bash
mkdir -p data/input/calls
```

1. Copy the recordings into `data/input/calls/`.
2. Create `data/input/metadata.csv`, with one line per recording:

```
call_id,banker_id,file_name,banker_channel,banker_name
C0001,B17,C0001.wav,L,דנה כהן
```

| Column | Required | Meaning |
|---|---|---|
| `call_id` | yes | A unique ID for the call (English letters, digits, `_` `-` `.`) |
| `banker_id` | yes | The banker's ID |
| `file_name` | yes | The recording's file name inside `calls/` |
| `banker_channel` | no | `L` or `R`: which channel the banker is on, in a stereo recording |
| `banker_name` | no | The banker's name. It is hidden in the transcript. |

⚠️ **Do not name recording files after the customer** (for example by ID or phone number). The file name appears in every report and every log.

---

## Step 8 — Check and run

```bash
python -m callqa preflight          # checks that models, server, disk and input are ready
python -m callqa run                # processes every call in metadata.csv
python -m callqa report             # produces a report for each banker
```

Run `preflight` before every run. If any of its lines is marked `FAIL`, fix it before you continue.

**Automated running:**
- `python -m callqa watch` — watches `data/input/calls/` and processes every new recording that arrives.
- `python -m callqa process --audio <file>` — processes one call. Exit code: `0` success · `1` needs human review · `2` failed. This lets you connect it to any task scheduler.

---

## What is created, and where

| Location | What is there |
|---|---|
| `data/output/reports/index.html` | The reports. **Start here.** |
| `data/output/transcripts/` | ⚠️ **Raw** transcripts containing identifiers. `python -m callqa retention --apply` deletes them once they are older than `retention.raw_days`. |
| `data/output/redacted/` | Transcripts after the identifiers are hidden |
| `data/output/redacted_audio/` | Recordings with the identifiers silenced |

Optional dashboard: `python dashboard/server.py`. It prints an address to open in a browser. You can delete the whole `dashboard/` folder with no effect on the rest of the system.

---

## A server with no internet connection

Prepare everything on a machine that has internet, then move it to the server.

**On the machine with internet:**

```bash
./scripts/build_offline_bundle.sh
python scripts/download_models.py --all --llm-model dicta-il/dictalm2.0-instruct
```

**Move to the server:** the project folder (including `wheels/` and `models/`), and the folder `~/.cache/huggingface`.

**On the server:**

```bash
./scripts/install_offline.sh --server
source .venv/bin/activate
export HF_HOME=<where you copied the huggingface folder>
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
python -m callqa preflight
```

---

## Common problems

| Message or symptom | What to do |
|---|---|
| `403` while downloading a model | The licence in step 4.1 was not accepted, or `HF_TOKEN` is missing from `.env` |
| Judge server "unreachable" | The server is not running, or the address in `judge.base_url` is wrong |
| Judge server "refused (401/403)" | The key in `CALLQA_JUDGE__API_KEY` differs from the server's key (step 5) |
| `size MISMATCH` in preflight | A model was only partly copied. Copy it again. |
| `model:diarization` shows WARN | Are your recordings stereo? Then you can ignore it. Otherwise, see "A server with no internet connection". |
| Every call fails at the first stage | ffmpeg is missing |
| `externally-managed-environment` | You forgot to run `source .venv/bin/activate` |

---

## More documents

- `docs/DEPLOYMENT.md` — full detail: security, licensing, what is stored and for how long, and known limits.
- `docs/MLOPS.md` — running it over time: evaluation, drift monitoring, reproducing results.
- `CHANGELOG.md` — what changed in each version.
