# callqa

**🌐 [עברית](#hebrew) · [English](#english)**

<a id="hebrew"></a>

<div dir="rtl">

## מה זה

תוכנה שמקבלת הקלטה של שיחה טלפונית בעברית בין בנקאי ללקוח, ומפיקה דוח בקרת איכות על השיחה.
היא מתמללת את השיחה, מפרידה בין הדוברים, מסתירה פרטים מזהים של הלקוח (ת"ז, טלפון, שם וכו'), ונותנת ציון לשיחה בעזרת מודל שפה שרץ על התשתית של הבנק.

- זו תוכנת שורת פקודה (terminal). אין בה אתר ואין בה שרת.
- **היא לא פונה לאינטרנט בזמן ריצה.** יש לה רק שני חיבורי רשת: שרת מודל השיפוט שאתם מגדירים (שלב 6), והורדה חד־פעמית של המודלים (שלב 5).
- **עובדת על Windows (כולל VDI של Microsoft, בלי הרשאות מנהל) ועל Linux.** כל פקודה במדריך מופיעה בגרסת Windows; ב־Linux כותבים `/` במקום `\` ו־`.venv/bin/python` במקום `.venv\Scripts\python`.

כל הפקודות במדריך מריצים ב־terminal, **מתוך תיקיית הפרויקט**.

---

## מה צריך לפני שמתחילים

| | דרישה | הערות |
|---|---|---|
| מערכת הפעלה | Windows 10/11 או Linux | ב־Windows לא צריך הרשאות מנהל ולא צריך WSL |
| Python | גרסה 3.11 או 3.12 | שלב 1 |
| דיסק פנוי | 40GB לפחות | רובו למודלים |
| זיכרון | 16GB לפחות | |
| כרטיס מסך (GPU) | **לא חובה** | בלי GPU הכול רץ על המעבד. מודל השיפוט יהיה איטי (כמה דקות לשיחה). |
| ffmpeg | רק להקלטות שאינן WAV | שלב 8. אין צורך להתקין: פורסים לתיקייה. |

---

## שלב 1 — Python

**Windows:**
1. נכנסים ל־https://www.python.org/downloads/windows ומורידים את **Windows installer (64-bit)** של גרסה **3.12**.
2. במסך הראשון של ההתקנה:
   - **מסמנים** ✅ `Add python.exe to PATH`
   - **מבטלים** את הסימון ⬜ `Use admin privileges when installing py.exe`
   - לוחצים **Install Now**. כך ההתקנה היא למשתמש שלכם בלבד ולא דורשת הרשאות מנהל.
3. פותחים חלון `cmd` חדש ובודקים: `py -3.12 --version`

אם ההתקנה חסומה אצלכם במחשב — מבקשים ממחלקת ה־IT להתקין Python 3.12.

**Linux:** `python3 --version` (צריך 3.11 או 3.12; ב־Debian/Ubuntu מתקינים גם `python3-venv`).

---

## שלב 2 — הורדה ופתיחה

1. ב־GitHub לוחצים **Code ← Download ZIP**.
2. **Windows:** לוחצים על הקובץ בכפתור ימני ← **Extract All**, ובתור יעד כותבים את תיקיית המשתמש שלכם: `C:\Users\<שם המשתמש>`. נוצרת התיקייה `C:\Users\<שם המשתמש>\BRS_Model-main`.
3. פותחים terminal בתיקייה: פותחים אותה בסייר הקבצים, לוחצים על שורת הכתובת, כותבים `cmd` ולוחצים Enter.

⚠️ **איפה לא לשים את הפרויקט:** לא על שולחן העבודה, לא ב־Documents ולא ב־OneDrive. בהרבה מחשבים התיקיות האלה מסונכרנות לענן, והפרויקט ישמור בו תמלולים עם פרטים מזהים. גם לא על כונן רשת: מסד הנתונים של התוכנה לא עובד שם באופן אמין. תיקיית המשתמש עצמה (`C:\Users\<שם>`) נשמרת גם ב־VDI, היא פרטית, והיא לא מסונכרנת.

---

## שלב 3 — התקנה

```
py -3.12 scripts\install.py
```

ב־Linux: `python3 scripts/install.py`

הסקריפט יוצר בתוך הפרויקט תיקייה בשם `.venv` ("סביבה וירטואלית") ומתקין לתוכה את כל הספריות. הוא לא נוגע בשום דבר מחוץ לתיקיית הפרויקט.

**מכאן והלאה מריצים הכול עם `.venv\Scripts\python`** (ב־Linux: `.venv/bin/python`). אין צורך "להפעיל" את הסביבה, ולכן חסימת סקריפטים של PowerShell לא משנה.

---

## שלב 4 — בדיקה שהכול עובד (בלי מודלים)

```
.venv\Scripts\python scripts\first_run.py
```

הסקריפט יוצר 6 שיחות לדוגמה ומריץ עליהן את כל התהליך במצב בדיקה (mock). המצב הזה לא דורש מודלים, כרטיס מסך או אינטרנט. בסוף הוא פותח את לוח הבקרה בדפדפן (עוצרים עם `Ctrl+C`).

**הבדיקה הצליחה אם** הקובץ `data/output/reports/index.html` נפתח בדפדפן ומציג 6 שיחות עם ציונים.

כדאי להריץ גם את חבילת הבדיקות האוטומטיות (לוקח כדקה):

```
.venv\Scripts\python -m pytest -q
```

---

## שלב 5 — הורדת המודלים

המודלים **לא** נמצאים בפרויקט, וצריך להוריד אותם פעם אחת. זה השלב היחיד שדורש אינטרנט.

**5.1 — מנועי המודלים** (כ־2GB של ספריות):

```
py -3.12 scripts\install.py --server
.venv\Scripts\python -m pip install huggingface_hub
```

**5.2 — אישור הרישיון של מודל הפרדת הדוברים** (בלי זה ההורדה שלו תיכשל עם שגיאה 403; נדרש רק להקלטות מונו):

1. נכנסים עם חשבון Hugging Face לעמוד https://huggingface.co/pyannote/speaker-diarization-community-1 ומאשרים את התנאים.
2. יוצרים טוקן קריאה בעמוד https://huggingface.co/settings/tokens.
3. יוצרים את קובץ ההגדרות הפרטי ופותחים אותו:

```
copy .env.example .env
notepad .env
```

וכותבים בשורה המתאימה: `HF_TOKEN=hf_...` (ב־Linux: `cp .env.example .env` ועורך טקסט).

**5.3 — ההורדה.** בוחרים לפי המחשב שעליו ירוץ מודל השיפוט:

**בלי כרטיס מסך (למשל VDI):** קובץ מודל אחד מכווץ (כ־4.4GB) שרץ על המעבד:

```
.venv\Scripts\python scripts\download_models.py --asr --diarization --llm --llm-model dicta-il/dictalm2.0-instruct-GGUF --llm-gguf dictalm2.0-instruct.Q4_K_M.gguf
```

**עם כרטיס מסך NVIDIA של 16–24GB (שרת Linux):**

```
.venv/bin/python scripts/download_models.py --all --llm-model dicta-il/dictalm2.0-instruct
```

- בסוף ההורדה מודפסת טבלה שמסכמת אילו מודלים ירדו בהצלחה (OK) ואילו נכשלו (FAILED), ומה עושים אם משהו נכשל.
- הסקריפט רושם לכל מודל את הרישיון שלו בקובץ `models/MODELS_MANIFEST.json`, ומעדכן לבד את מודל השיפוט בקובץ `config/config.yaml`.
- אפשרויות נוספות: `.venv\Scripts\python scripts\download_models.py --help`.

---

## שלב 6 — הפעלת מודל השיפוט

מודל השיפוט רץ כשרת נפרד, בחלון terminal משלו. אפשר להשתמש בכל שרת שתומך בממשק OpenAI (`/v1/chat/completions`), כולל שרת פנימי שכבר קיים בבנק.

**6.1 — מפתח.** השרת מוגן במפתח, כי הוא עונה עם טקסט שנבנה מתמלולי שיחות. יוצרים מפתח:

```
.venv\Scripts\python -c "import secrets; print(secrets.token_hex(32))"
```

ומוסיפים אותו לקובץ `.env` בשורה: `CALLQA_JUDGE__API_KEY=<המפתח>`

**6.2 — Windows או מחשב בלי GPU: llama.cpp.**
1. מורידים מ־https://github.com/ggml-org/llama.cpp/releases את הקובץ שהשם שלו נגמר ב־`bin-win-cpu-x64.zip`.
2. פורסים אותו לתוך תיקייה בשם `tools` בתוך הפרויקט (יוצרים אותה אם אינה קיימת). אין צורך להתקין כלום.
3. מפעילים, **בחלון terminal נפרד שנשאר פתוח**:

```
.venv\Scripts\python scripts\start_llama_server.py models\dicta-il--dictalm2.0-instruct-GGUF\dictalm2.0-instruct.Q4_K_M.gguf
```

הסקריפט קורא את המפתח מ־`.env` בעצמו, ומאזין רק למחשב המקומי (127.0.0.1).

**6.3 — שרת Linux עם GPU: vLLM.** מתקינים אותו בנפרד (`pip install vllm`) ומריצים, עם אותו מפתח:

```bash
export CALLQA_JUDGE_API_KEY=<המפתח>
./scripts/start_vllm.sh models/dicta-il--dictalm2.0-instruct 8000
```

⚠️ שימו לב: בשם הזה יש קו תחתון **אחד** (הוא מיועד לשרת), ובשם שב־`.env` יש **שניים** (הוא מיועד לתוכנה). אם המפתחות לא זהים, התוכנה תדווח שהשרת "סירב (401)".

---

## שלב 7 — הגדרות

כל ההגדרות נמצאות בקובץ `config/config.yaml`, ולכל אחת יש הסבר בגוף הקובץ. אלה ההגדרות שכדאי לבדוק:

| הגדרה | מה היא עושה |
|---|---|
| `judge.base_url` | הכתובת של שרת מודל השיפוט. ברירת המחדל, `http://localhost:8000/v1`, מתאימה לשלב 6. |
| `retention.raw_days` | אחרי כמה ימים נמחקים הנתונים הגולמיים עם הפרטים המזהים. ברירת המחדל היא 90. |
| `redaction.ner` | הסתרה של שמות שאף אחד לא שאל עליהם. מחייב את `scripts/download_models.py --ner`. |
| `asr.compute_type` | `float16` לכרטיס מסך. במחשב בלי כרטיס מסך התוכנה עוברת לבד ל־`int8` ורושמת את זה בלוג. |

סיסמאות ומפתחות שמים **רק** בקובץ `.env`, אף פעם לא בקובץ `config.yaml`.

---

## שלב 8 — הכנת השיחות

```
mkdir data\input\calls
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

- ב־Excel שומרים עם **Save As ← CSV UTF-8**.
- ⚠️ **אל תקראו לקובצי ההקלטה על שם הלקוח** (למשל לפי מספר ת"ז או טלפון). שם הקובץ מופיע בכל הדוחות ובכל הלוגים.
- **הקלטות שאינן WAV** (mp3, m4a ועוד) דורשות ffmpeg. אין צורך להתקין אותו: מורידים מ־https://github.com/BtbN/FFmpeg-Builds/releases את `ffmpeg-master-latest-win64-lgpl.zip` ופורסים לתוך התיקייה `tools` שבפרויקט. התוכנה מוצאת אותו שם לבד. ב־Linux: `sudo apt install ffmpeg`.

---

## שלב 9 — בדיקה והרצה

```
.venv\Scripts\python -m callqa preflight
.venv\Scripts\python -m callqa run
.venv\Scripts\python -m callqa report
```

- `preflight` בודק שהמודלים, השרת, הדיסק והקלט מוכנים. כדאי להריץ אותו לפני כל הרצה. אם אחת השורות שלו מסומנת `FAIL`, מתקנים אותה לפני שממשיכים.
- `run` מעבד את כל השיחות שב־`metadata.csv`.
- `report` מפיק דוח לכל בנקאי.

**הרצה אוטומטית:**
- `python -m callqa watch` — מאזין לתיקייה `data/input/calls/` ומעבד כל הקלטה חדשה שמגיעה.
- `python -m callqa process --audio <קובץ>` — מעבד שיחה אחת. קוד היציאה: `0` הצלחה · `1` נדרשת בדיקה אנושית · `2` כישלון.
- ב־Windows מתזמנים עם **Task Scheduler** (לא דורש הרשאות מנהל): Create Basic Task ← Start a program. בתוכנית שמים את הנתיב המלא של `.venv\Scripts\python.exe`, בארגומנטים `-m callqa run`, וב־**Start in** את תיקיית הפרויקט.

---

## מה נוצר ואיפה

| מיקום | מה יש שם |
|---|---|
| `data/output/reports/index.html` | הדוחות. **מתחילים מכאן.** |
| `data/output/transcripts/` | ⚠️ תמלול **גולמי** עם פרטים מזהים. `python -m callqa retention --apply` מוחק אותו אחרי מספר הימים שמוגדר ב־`retention.raw_days`. |
| `data/output/redacted/` | תמלול אחרי הסתרת הפרטים המזהים |
| `data/output/redacted_audio/` | הקלטה שבה הפרטים המזהים מושתקים |

- התיקיות עם הנתונים נגישות רק למשתמש שהריץ את התוכנה (ולמנהלי המערכת). ב־Windows התוכנה קובעת זאת בעצמה בהרשאות התיקייה, כך שמשתמשים אחרים באותו שרת VDI לא יכולים לקרוא אותן.
- לוח הבקרה, אופציונלי: `python dashboard/server.py`. הפקודה מדפיסה כתובת שפותחים בדפדפן. אפשר למחוק את כל התיקייה `dashboard/` בלי שום השפעה על שאר המערכת.

---

## מחשב בלי חיבור לאינטרנט

מכינים הכול על מחשב שיש בו אינטרנט, **עם אותה מערכת הפעלה ואותה גרסת Python** (למשל Windows עם Python 3.12), ומעבירים.

**על המחשב עם האינטרנט**, בתיקיית הפרויקט:

```
py -3.12 scripts\build_offline_bundle.py
py -3.12 scripts\install.py --server
.venv\Scripts\python -m pip install huggingface_hub
.venv\Scripts\python scripts\download_models.py --asr --diarization --llm --llm-model dicta-il/dictalm2.0-instruct-GGUF --llm-gguf dictalm2.0-instruct.Q4_K_M.gguf
```

**מעבירים:** את כל תיקיית הפרויקט (כולל `wheels` ו־`models`, ובלי `.venv`), ואת התיקייה `C:\Users\<שם>\.cache\huggingface` (ב־Linux: `~/.cache/huggingface`).

**על המחשב בלי האינטרנט:**

```
py -3.12 scripts\install.py --server
```

הסקריפט מזהה את התיקייה `wheels` ומתקין ממנה בלי רשת. אחר כך מוסיפים ל־`.env`:

```
HF_HOME=<המיקום שאליו הועתקה התיקייה huggingface>
HF_HUB_OFFLINE=1
TRANSFORMERS_OFFLINE=1
```

ומריצים `.venv\Scripts\python -m callqa preflight`.

---

## תקלות נפוצות

| הודעה או מצב | מה עושים |
|---|---|
| `'py' is not recognized` | Python לא הותקן, או שלא סימנתם `Add python.exe to PATH`. חוזרים לשלב 1. |
| הפקודה `python` פותחת את Microsoft Store | משתמשים ב־`py -3.12` במקום `python`. |
| `running scripts is disabled on this system` | ניסיתם להפעיל את `Activate.ps1`. אין בזה צורך: מריצים ישירות את `.venv\Scripts\python`. |
| תוכנה נחסמת ע"י מדיניות הארגון (AppLocker / SmartScreen) | מבקשים מה־IT לאשר את Python ואת `llama-server.exe`, או משתמשים בשרת שיפוט שכבר קיים בבנק. |
| `403` בהורדת מודל | לא אושר הרישיון בשלב 5.2, או ש־`HF_TOKEN` חסר בקובץ `.env` |
| שרת השיפוט "unreachable" | השרת לא רץ (שלב 6), או שהכתובת ב־`judge.base_url` שגויה |
| שרת השיפוט "refused (401/403)" | המפתח ב־`CALLQA_JUDGE__API_KEY` שונה מהמפתח של השרת (שלב 6) |
| `size MISMATCH` ב־preflight | מודל הועתק רק באופן חלקי. מעתיקים אותו שוב. |
| `model:diarization` מסומן WARN | ההקלטות שלכם בסטריאו? אז אפשר להתעלם. אחרת, ראו שלב 5.2. |
| שיחות mp3/m4a נכשלות בשלב הראשון | חסר ffmpeg (שלב 8) |
| עברית משובשת ב־metadata.csv | שומרים מחדש ב־Excel בתור **CSV UTF-8** |

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
- **It never contacts the internet at runtime.** It has only two network connections: the judge model server you configure (step 6), and a one-time model download (step 5).
- **It runs on Windows (including Microsoft VDI, with no admin rights) and on Linux.** Every command in this guide is shown for Windows; on Linux write `/` instead of `\` and `.venv/bin/python` instead of `.venv\Scripts\python`.

Run every command in this guide in a terminal, **from inside the project folder**.

---

## Before you start

| | Requirement | Notes |
|---|---|---|
| Operating system | Windows 10/11 or Linux | On Windows no admin rights and no WSL are needed |
| Python | 3.11 or 3.12 | Step 1 |
| Free disk | At least 40 GB | Mostly for the models |
| Memory | At least 16 GB | |
| GPU | **Not required** | Without one everything runs on the CPU. The judge model is then slow (a few minutes per call). |
| ffmpeg | Only for recordings that are not WAV | Step 8. Nothing to install: you unzip it. |

---

## Step 1 — Python

**Windows:**
1. Go to https://www.python.org/downloads/windows and download the **Windows installer (64-bit)** for version **3.12**.
2. On the installer's first screen:
   - **Tick** ✅ `Add python.exe to PATH`
   - **Untick** ⬜ `Use admin privileges when installing py.exe`
   - Click **Install Now**. This installs for your user only and needs no admin rights.
3. Open a new `cmd` window and check: `py -3.12 --version`

If the installer is blocked on your machine, ask IT to install Python 3.12.

**Linux:** `python3 --version` (3.11 or 3.12 is needed; on Debian/Ubuntu also install `python3-venv`).

---

## Step 2 — Download and unpack

1. On GitHub, click **Code → Download ZIP**.
2. **Windows:** right-click the file → **Extract All**, and as the destination type your user folder: `C:\Users\<your user name>`. This creates `C:\Users\<your user name>\BRS_Model-main`.
3. Open a terminal in that folder: open it in File Explorer, click the address bar, type `cmd`, press Enter.

⚠️ **Where not to put the project:** not on the Desktop, not in Documents and not in OneDrive. On many machines those folders are synced to the cloud, and the project stores transcripts containing identifiers. Not on a network drive either: the program's database is not reliable there. Your user folder itself (`C:\Users\<name>`) persists on VDI, is private, and is not synced.

---

## Step 3 — Install

```
py -3.12 scripts\install.py
```

On Linux: `python3 scripts/install.py`

The script creates a folder called `.venv` (a "virtual environment") inside the project and installs every library into it. It touches nothing outside the project folder.

**From here on, run everything with `.venv\Scripts\python`** (on Linux: `.venv/bin/python`). There is no need to "activate" the environment, so PowerShell's script-blocking policy does not matter.

---

## Step 4 — Check that everything works (no models needed)

```
.venv\Scripts\python scripts\first_run.py
```

The script creates 6 sample calls and runs the whole process on them in test mode (mock). Test mode needs no models, no GPU and no internet. At the end it opens the dashboard in the browser (stop it with `Ctrl+C`).

**The check passed if** `data/output/reports/index.html` opens in a browser and shows 6 scored calls.

It is also worth running the automated test suite (about a minute):

```
.venv\Scripts\python -m pytest -q
```

---

## Step 5 — Download the models

The models are **not** included in the project and have to be downloaded once. This is the only step that needs the internet.

**5.1 — The model engines** (about 2 GB of libraries):

```
py -3.12 scripts\install.py --server
.venv\Scripts\python -m pip install huggingface_hub
```

**5.2 — Accept the speaker-separation model's licence** (without this its download fails with a 403 error; it is needed only for mono recordings):

1. Using a Hugging Face account, go to https://huggingface.co/pyannote/speaker-diarization-community-1 and accept the conditions.
2. Create a read token at https://huggingface.co/settings/tokens.
3. Create the private settings file and open it:

```
copy .env.example .env
notepad .env
```

and fill in the matching line: `HF_TOKEN=hf_...` (on Linux: `cp .env.example .env` and a text editor).

**5.3 — Download.** Choose by the machine the judge model will run on:

**No GPU (for example VDI):** one compressed model file (about 4.4 GB) that runs on the CPU:

```
.venv\Scripts\python scripts\download_models.py --asr --diarization --llm --llm-model dicta-il/dictalm2.0-instruct-GGUF --llm-gguf dictalm2.0-instruct.Q4_K_M.gguf
```

**An NVIDIA GPU with 16–24 GB (a Linux server):**

```
.venv/bin/python scripts/download_models.py --all --llm-model dicta-il/dictalm2.0-instruct
```

- At the end it prints a table of which models downloaded successfully (OK) and which failed (FAILED), with what to do about each failure.
- The script records each model's licence in `models/MODELS_MANIFEST.json` and updates the judge model in `config/config.yaml` for you.
- More options: `.venv\Scripts\python scripts\download_models.py --help`.

---

## Step 6 — Start the judge model

The judge model runs as a separate server, in its own terminal window. Any server that supports the OpenAI interface (`/v1/chat/completions`) works, including an internal server the bank already runs.

**6.1 — A key.** The server is protected by a key, because it answers with text built from call transcripts. Create one:

```
.venv\Scripts\python -c "import secrets; print(secrets.token_hex(32))"
```

and add it to `.env` on this line: `CALLQA_JUDGE__API_KEY=<the key>`

**6.2 — Windows, or any machine without a GPU: llama.cpp.**
1. From https://github.com/ggml-org/llama.cpp/releases download the file whose name ends in `bin-win-cpu-x64.zip`.
2. Unzip it into a folder called `tools` inside the project (create it if it does not exist). Nothing is installed.
3. Start it, **in a separate terminal window that stays open**:

```
.venv\Scripts\python scripts\start_llama_server.py models\dicta-il--dictalm2.0-instruct-GGUF\dictalm2.0-instruct.Q4_K_M.gguf
```

The script reads the key from `.env` itself, and listens on this machine only (127.0.0.1).

**6.3 — A Linux server with a GPU: vLLM.** Install it separately (`pip install vllm`) and run it with the same key:

```bash
export CALLQA_JUDGE_API_KEY=<the key>
./scripts/start_vllm.sh models/dicta-il--dictalm2.0-instruct 8000
```

⚠️ Note: this name has **one** underscore (it is for the server); the one in `.env` has **two** (it is for the program). If the keys are not identical, the program reports that the server "refused (401)".

---

## Step 7 — Settings

All settings are in `config/config.yaml`, and each one is explained inside the file. These are the settings worth checking:

| Setting | What it does |
|---|---|
| `judge.base_url` | The address of the judge model server. The default, `http://localhost:8000/v1`, matches step 6. |
| `retention.raw_days` | After how many days the raw data containing identifiers is deleted. The default is 90. |
| `redaction.ner` | Hides names that nobody asked for. Requires `scripts/download_models.py --ner`. |
| `asr.compute_type` | `float16` for a GPU. On a machine without one the program switches to `int8` by itself and logs that it did. |

Put passwords and keys **only** in `.env`, never in `config.yaml`.

---

## Step 8 — Prepare the calls

```
mkdir data\input\calls
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

- In Excel, save with **Save As → CSV UTF-8**.
- ⚠️ **Do not name recording files after the customer** (for example by ID or phone number). The file name appears in every report and every log.
- **Recordings that are not WAV** (mp3, m4a and so on) need ffmpeg. Nothing to install: download `ffmpeg-master-latest-win64-lgpl.zip` from https://github.com/BtbN/FFmpeg-Builds/releases and unzip it into the project's `tools` folder. The program finds it there by itself. On Linux: `sudo apt install ffmpeg`.

---

## Step 9 — Check and run

```
.venv\Scripts\python -m callqa preflight
.venv\Scripts\python -m callqa run
.venv\Scripts\python -m callqa report
```

- `preflight` checks that models, server, disk and input are ready. Run it before every run. If any of its lines is marked `FAIL`, fix it before you continue.
- `run` processes every call in `metadata.csv`.
- `report` produces a report for each banker.

**Automated running:**
- `python -m callqa watch` — watches `data/input/calls/` and processes every new recording that arrives.
- `python -m callqa process --audio <file>` — processes one call. Exit code: `0` success · `1` needs human review · `2` failed.
- On Windows, schedule it with **Task Scheduler** (no admin rights needed): Create Basic Task → Start a program. The program is the full path of `.venv\Scripts\python.exe`, the arguments are `-m callqa run`, and **Start in** is the project folder.

---

## What is created, and where

| Location | What is there |
|---|---|
| `data/output/reports/index.html` | The reports. **Start here.** |
| `data/output/transcripts/` | ⚠️ **Raw** transcripts containing identifiers. `python -m callqa retention --apply` deletes them once they are older than `retention.raw_days`. |
| `data/output/redacted/` | Transcripts after the identifiers are hidden |
| `data/output/redacted_audio/` | Recordings with the identifiers silenced |

- The data folders are accessible only to the user who ran the program (and the system administrators). On Windows the program sets this itself in the folder permissions, so other users of the same VDI host cannot read them.
- Optional dashboard: `python dashboard/server.py`. It prints an address to open in a browser. You can delete the whole `dashboard/` folder with no effect on the rest of the system.

---

## A machine with no internet connection

Prepare everything on a machine that has internet **and the same operating system and Python version** (for example Windows with Python 3.12), then move it across.

**On the machine with internet**, in the project folder:

```
py -3.12 scripts\build_offline_bundle.py
py -3.12 scripts\install.py --server
.venv\Scripts\python -m pip install huggingface_hub
.venv\Scripts\python scripts\download_models.py --asr --diarization --llm --llm-model dicta-il/dictalm2.0-instruct-GGUF --llm-gguf dictalm2.0-instruct.Q4_K_M.gguf
```

**Move across:** the whole project folder (including `wheels` and `models`, without `.venv`), and the folder `C:\Users\<name>\.cache\huggingface` (on Linux: `~/.cache/huggingface`).

**On the machine without internet:**

```
py -3.12 scripts\install.py --server
```

The script sees the `wheels` folder and installs from it with no network. Then add to `.env`:

```
HF_HOME=<where you copied the huggingface folder>
HF_HUB_OFFLINE=1
TRANSFORMERS_OFFLINE=1
```

and run `.venv\Scripts\python -m callqa preflight`.

---

## Common problems

| Message or symptom | What to do |
|---|---|
| `'py' is not recognized` | Python is not installed, or `Add python.exe to PATH` was not ticked. Back to step 1. |
| `python` opens the Microsoft Store | Use `py -3.12` instead of `python`. |
| `running scripts is disabled on this system` | You tried to run `Activate.ps1`. It is not needed: run `.venv\Scripts\python` directly. |
| A program is blocked by company policy (AppLocker / SmartScreen) | Ask IT to allow Python and `llama-server.exe`, or use a judge server the bank already runs. |
| `403` while downloading a model | The licence in step 5.2 was not accepted, or `HF_TOKEN` is missing from `.env` |
| Judge server "unreachable" | The server is not running (step 6), or the address in `judge.base_url` is wrong |
| Judge server "refused (401/403)" | The key in `CALLQA_JUDGE__API_KEY` differs from the server's key (step 6) |
| `size MISMATCH` in preflight | A model was only partly copied. Copy it again. |
| `model:diarization` shows WARN | Are your recordings stereo? Then you can ignore it. Otherwise, see step 5.2. |
| mp3/m4a calls fail at the first stage | ffmpeg is missing (step 8) |
| Garbled Hebrew in metadata.csv | Save it again from Excel as **CSV UTF-8** |

---

## More documents

- `docs/DEPLOYMENT.md` — full detail: security, licensing, what is stored and for how long, and known limits.
- `docs/MLOPS.md` — running it over time: evaluation, drift monitoring, reproducing results.
- `CHANGELOG.md` — what changed in each version.
