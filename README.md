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
| כרטיס מסך (GPU) | **לא במחשב הזה** | תמלול, הפרדת דוברים והסתרת פרטים רצים על המעבד. מודל השיפוט צריך GPU במחשב אחר ברשת (שלב 6). |
| Visual C++ Redistributable (x64) | למודלים בלבד | בדרך כלל כבר מותקן. `preflight` בודק ואומר אם חסר (התקנה שלו דורשת את ה־IT). |

---

## שלב 1 — Python

**Windows:**
1. נכנסים ל־https://www.python.org/downloads/release/python-31210/ ומורידים את **Windows installer (64-bit)** — הקובץ `python-3.12.10-amd64.exe`. (זו גרסת 3.12 האחרונה שיש לה מתקין ל־Windows.)
2. במסך הראשון של ההתקנה:
   - **מסמנים** ✅ `Add python.exe to PATH`
   - **מבטלים** את הסימון ⬜ `Use admin privileges when installing py.exe`
   - לוחצים **Install Now**. כך ההתקנה היא למשתמש שלכם בלבד ולא דורשת הרשאות מנהל.
3. פותחים חלון `cmd` חדש ובודקים: `py -3.12 --version`

אם ההתקנה חסומה במחשב — מבקשים ממחלקת ה־IT להתקין Python 3.12.

**Linux:** `python3 --version` (צריך 3.11 או 3.12; ב־Debian/Ubuntu מתקינים גם `python3-venv`).

---

## שלב 2 — הורדה ופתיחה

1. ב־GitHub לוחצים **Code ← Download ZIP**.
2. **Windows:** לוחצים על הקובץ בכפתור ימני ← **Extract All**, ובתור יעד כותבים את תיקיית המשתמש שלכם: `C:\Users\<שם המשתמש>`. נוצרת התיקייה `C:\Users\<שם המשתמש>\BRS_Model-main`.
3. פותחים terminal בתיקייה: פותחים אותה בסייר הקבצים, לוחצים על שורת הכתובת, כותבים `cmd` ולוחצים Enter.

⚠️ **איפה לא לשים את הפרויקט:** לא על שולחן העבודה, לא ב־Documents ולא ב־OneDrive. בהרבה מחשבים התיקיות האלה מסונכרנות לענן, והפרויקט ישמור בו תמלולים עם פרטים מזהים. גם לא על כונן רשת: מסד הנתונים של התוכנה לא עובד שם באופן אמין. `preflight` בודק את שני המקרים ועוצר.

- **נפח:** יחד עם המודלים הפרויקט תופס כ־30GB. בחלק מסביבות ה־VDI תיקיית המשתמש מוגבלת בנפח. אם כך, שואלים את ה־IT איזה כונן מקומי נשמר בין התחברויות, ושם את המודלים שם: `--models-dir D:\callqa-models` בשלב 5, ובקובץ `.env` את השורה `CALLQA_PATHS__MODELS_DIR=D:\callqa-models`.
- ההתקנה (שלב 3) הופכת את תיקיית הפרויקט לפרטית למשתמש שלכם (ולמנהלי המערכת), גם אם היא נמצאת מחוץ לתיקיית המשתמש.

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

**הבדיקה הצליחה אם** הקובץ `data/demo/output/reports/index.html` נפתח בדפדפן ומציג 6 שיחות עם ציונים. הבדיקה עובדת בתיקייה נפרדת (`data/demo`), כך שאפשר להריץ אותה שוב בכל שלב בלי לגעת בשיחות האמיתיות.

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

**5.3 — ההורדה:**

```
.venv\Scripts\python scripts\download_models.py --asr --diarization
```

- אם בשלב 6 בוחרים באפשרות ב' (llama.cpp על המחשב הזה), מורידים גם את מודל השיפוט, קובץ אחד מכווץ של כ־4.4GB:

```
.venv\Scripts\python scripts\download_models.py --llm --llm-model dicta-il/dictalm2.0-instruct-GGUF --llm-gguf dictalm2.0-instruct.Q4_K_M.gguf
```

- בסוף ההורדה מודפסת טבלה שמסכמת אילו מודלים ירדו בהצלחה (OK) ואילו נכשלו (FAILED), ומה עושים אם משהו נכשל.
- הסקריפט רושם לכל מודל את הרישיון שלו בקובץ `models/MODELS_MANIFEST.json`.
- אפשרויות נוספות: `.venv\Scripts\python scripts\download_models.py --help`.

---

## שלב 6 — מודל השיפוט

מודל השיפוט הוא החלק היחיד שבאמת צריך כרטיס מסך. יש שתי אפשרויות:

| | איכות | זמן לשיחה | למה זה מתאים |
|---|---|---|---|
| **א. שרת עם GPU ברשת הבנק (מומלץ)** | מודל בגודל 27B נמדד ועבר | כ־3.5 דקות | עבודה אמיתית |
| **ב. llama.cpp על המחשב הזה, בלי GPU** | מודל 7B. נמדד מתחת לרף האיכות: רוב השיחות יסומנו "נדרשת בדיקה אנושית" | 10 דקות ויותר | לבדוק שכל השרשרת מחוברת |

**6.1 — מפתח.** השרת מוגן במפתח, כי הוא עונה עם טקסט שנבנה מתמלולי שיחות. יוצרים מפתח:

```
.venv\Scripts\python -c "import secrets; print(secrets.token_hex(32))"
```

ומוסיפים אותו לקובץ `.env` בשורה: `CALLQA_JUDGE__API_KEY=<המפתח>`

**6.2 — אפשרות א': שרת GPU.** צוות התשתיות מקים על מחשב Linux עם כרטיס NVIDIA של 24GB שרת vLLM, לפי `docs/DEPLOYMENT.md` (סעיף 5), מאחורי HTTPS ועם אותו מפתח. על המחשב הזה מוסיפים לקובץ `.env`:

```
CALLQA_JUDGE__BASE_URL=https://<כתובת השרת>/v1
CALLQA_JUDGE__MODEL=<שם המודל שהשרת מדווח>
```

השיחות עצמן לא עוברות לשרת. עובר רק התמלול אחרי הסתרת הפרטים המזהים.

**6.3 — אפשרות ב': llama.cpp על המחשב הזה.**
1. מורידים מ־https://github.com/ggml-org/llama.cpp/releases (תחת Assets של הגרסה האחרונה) את קובץ ה־zip ששמו מכיל `bin-win-cpu-x64`.
2. פורסים אותו לתוך תיקייה בשם `tools` בתוך הפרויקט (יוצרים אותה אם אינה קיימת). אין צורך להתקין כלום.
3. מפעילים, **בחלון terminal נפרד שנשאר פתוח**:

```
.venv\Scripts\python scripts\start_llama_server.py models\dicta-il--dictalm2.0-instruct-GGUF\dictalm2.0-instruct.Q4_K_M.gguf
```

- מוסיפים ל־`.env` את השורה `CALLQA_JUDGE__REQUEST_TIMEOUT_SEC=3600`: על מעבד, שיחה אחת יכולה לקחת יותר מ־10 דקות.
- הסקריפט קורא את המפתח מ־`.env` בעצמו, ומאזין רק למחשב המקומי (127.0.0.1).
- אם מופיעה הודעה שהפורט תפוס (למשל כי משתמש אחר באותו שרת VDI כבר מריץ שרת כזה), מריצים עם `--port 8001` ומוסיפים ל־`.env` את השורה `CALLQA_JUDGE__BASE_URL=http://127.0.0.1:8001/v1`.
- לפני שנשלח תמלול, התוכנה מוודאת שהשרת שעונה לה מריץ את המודל שהוגדר לה, כדי ששיחות לא יישלחו לשרת של מישהו אחר.

---

## שלב 7 — הגדרות

כל ההגדרות נמצאות בקובץ `config/config.yaml`, ולכל אחת יש הסבר בגוף הקובץ. אלה ההגדרות שכדאי לבדוק:

| הגדרה | מה היא עושה |
|---|---|
| `judge.base_url` | הכתובת של שרת מודל השיפוט. ברירת המחדל, `http://127.0.0.1:8000/v1`, מתאימה לשלב 6. |
| `retention.raw_days` | אחרי כמה ימים נמחקים הנתונים הגולמיים עם הפרטים המזהים. ברירת המחדל היא 90. |
| `redaction.ner` | הסתרה של שמות שאף אחד לא שאל עליהם. מחייב את `scripts/download_models.py --ner`. |
| `asr.compute_type` | `float16` לכרטיס מסך. במחשב בלי כרטיס מסך התוכנה עוברת לבד ל־`int8` ורושמת את זה בלוג. |

סיסמאות ומפתחות שמים **רק** בקובץ `.env`, אף פעם לא בקובץ `config.yaml`.

---

## שלב 8 — הכנת השיחות

```
mkdir data\input\calls
```

1. מעתיקים את ההקלטות לתיקייה `data/input/calls/`. יש לשמור כאן עותק: אחרי העיבוד ההקלטות מועברות ל־`data/input/processed/`.
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
- **פורמטים:** אחרי שלב 5 התוכנה קוראת לבד WAV (כולל WAV של מרכזיות טלפוניה, G.711), mp3, m4a ועוד — בלי להתקין שום דבר נוסף. בלי שלב 5 (למשל בבדיקה בלבד) היא קוראת רק WAV רגיל. אפשרות נוספת: להוריד מ־https://github.com/BtbN/FFmpeg-Builds/releases את `ffmpeg-master-latest-win64-lgpl.zip` ולפרוס לתוך התיקייה `tools` שבפרויקט.

---

## שלב 9 — בדיקה והרצה

```
.venv\Scripts\python -m callqa preflight
.venv\Scripts\python -m callqa run
.venv\Scripts\python -m callqa report
```

- `preflight` בודק שהמודלים, השרת, הדיסק והקלט מוכנים. כדאי להריץ אותו לפני כל הרצה. אם אחת השורות שלו מסומנת `FAIL`, מתקנים אותה לפני שממשיכים.
- `run` מעבד את כל השיחות שב־`metadata.csv`.
- `report` מפיק דוח לכל בנקאי ואת **דוח המנהלים** (ראו בהמשך).

**הרצה אוטומטית:**
- `.venv\Scripts\python -m callqa watch` — מאזין לתיקייה `data/input/calls/` ומעבד כל הקלטה חדשה שמגיעה.
- `.venv\Scripts\python -m callqa process --audio <קובץ>` — מעבד שיחה אחת. קוד היציאה: `0` הצלחה · `1` נדרשת בדיקה אנושית · `2` כישלון. ב־cmd רואים אותו עם `echo %ERRORLEVEL%`.
- **תזמון ב־Windows:** Task Scheduler ← Create Basic Task ← Start a program. לא דורש הרשאות מנהל. יוצרים שלוש משימות:

  | משימה | Program | Arguments | Start in |
  |---|---|---|---|
  | עיבוד שיחות | הנתיב המלא של `.venv\Scripts\python.exe` | `-m callqa run --log-file logs\run.log` | תיקיית הפרויקט |
  | עדכון הדוחות ודוח המנהלים (פעם ביום, אחרי העיבוד) | אותו נתיב | `-m callqa report --log-file logs\report.log` | תיקיית הפרויקט |
  | מחיקת נתונים גולמיים (פעם ביום) | אותו נתיב | `-m callqa retention --apply --log-file logs\retention.log` | תיקיית הפרויקט |

  ⚠️ **Start in חובה.** בלעדיו התוכנה מסרבת לרוץ. משימה רצה רק כשהמחשב פועל. ב־VDI שמתאפס בכל התנתקות, צריך לבקש מה־IT מכונה קבועה.

---

## דוח מנהלים

דוח אחד, מקצועי ומוכן להנהלה, על כל השיחות שעובדו (מתאים ל־1,000 שיחות ויותר). הוא נוצר אוטומטית בכל הרצה של `report`, ונשמר בקובץ `data/output/reports/executive.html`. פותחים אותו בלחיצה כפולה (Edge). הוא עובד בלי אינטרנט ואפשר להעביר אותו הלאה כקובץ אחד.

**שלוש רמות, מהכללי לפרטני:**
1. **סיכום מנהלים** — חוות דעת מקצועית בכמה משפטים, 6 מדדי מפתח עם מגמה, 5 הממצאים המרכזיים ו־3 פעולות מומלצות עם ההשפעה הצפויה של כל אחת במספרים.
2. **ניתוח מעמיק** — ממדי ההערכה ו„היכן אובדות הנקודות”, התפלגות, מגמות לאורך זמן, פילוח לפי סוג ומשך שיחה, השוואת בנקאים ומפת חום, גורמים התנהגותיים (יחס דיבור, קטיעות, שאלות ועוד), סיכוני ציות, ואיכות הנתונים.
3. **פירוט ושיחות** — כל שיחה: סינון, מיון וחיפוש; בלחיצה — הציון, הנימוק והציטוט בכל ממד, וקישור לדוח השיחה המלא. בנוסף ספריית מקרים (דוגמאות מצוינות וחלשות לכל ממד) ורשימת השיחות הדורשות תשומת לב.

כל ממצא ברמה 1 מקשר לניתוח שלו ברמה 2 ולשיחות עצמן ברמה 3. כל מספר מגיע עם רווח סמך, וקבוצה מסומנת כחריגה רק כשהפער מובהק סטטיסטית. ההסבר המלא נמצא ב־`docs/executive_report_he.md`.

**דוח לתקופה, לסוג שיחה או לבנקאי:**

```
.venv\Scripts\python -m callqa executive-report --from 01/08/2026 --to 31/08/2026
.venv\Scripts\python -m callqa executive-report --call-type loans --no-quotes
```

- `--from` / `--to` — תאריכי השיחות (בפורמט `31/08/2026` או `2026-08-31`). אפשר גם `--call-type`, `--banker` ו־`--title`.
- `--no-quotes` — גרסה בלי שום ציטוט או נימוק מתוך השיחות, להפצה רחבה.
- הדוח נשמר בשם משלו בתיקייה `data/output/reports/`, לדוגמה `data/output/reports/executive-2026-08-01_2026-08-31.html`, לצד קובץ CSV לאקסל.
- **הדפסה / PDF:** הכפתור בראש הדוח (או Ctrl+P). כל רמה מתחילה בעמוד חדש.

**לראות איך הוא נראה על 1,000 שיחות, לפני שיש נתונים אמיתיים:**

```
.venv\Scripts\python scripts\generate_batch_demo.py --calls 1000 --open
```

הסקריפט יוצר 1,000 שיחות סינתטיות בתיקייה נפרדת (`data/demo-batch`), מפיק מהן דוח מנהלים ופותח אותו. הדוח מסומן בבירור כהדגמה, ואינו נוגע בשיחות האמיתיות.

**דוח לדוגמה, מוכן לצפייה:** [`docs/examples/executive-demo.pdf`](docs/examples/executive-demo.pdf) (נפתח ישירות ב־GitHub) ו־[`docs/examples/executive-demo.html`](docs/examples/executive-demo.html) — הדוח האינטראקטיבי המלא על 1,000 שיחות סינתטיות. את קובץ ה־HTML מורידים (Download raw file) ופותחים ב־Edge.

---

## מה נוצר ואיפה

| מיקום | מה יש שם |
|---|---|
| `data/output/reports/index.html` | הדוחות. **מתחילים מכאן.** |
| `data/output/reports/executive.html` | דוח המנהלים. לצידו `data/output/reports/executive_calls.csv` — כל השיחות והציונים, לאקסל. |
| `data/output/transcripts/` | ⚠️ תמלול **גולמי** עם פרטים מזהים. `.venv\Scripts\python -m callqa retention --apply` מוחק אותו אחרי מספר הימים שמוגדר ב־`retention.raw_days`. |
| `data/output/redacted/` | תמלול אחרי הסתרת הפרטים המזהים |
| `data/output/redacted_audio/` | הקלטה שבה הפרטים המזהים מושתקים |

- התיקיות עם הנתונים נגישות רק למשתמש שהריץ את התוכנה (ולמנהלי המערכת). ב־Windows התוכנה קובעת זאת בעצמה בהרשאות התיקייה, כך שמשתמשים אחרים באותו שרת VDI לא יכולים לקרוא אותן.
- לוח הבקרה, אופציונלי: `.venv\Scripts\python dashboard\server.py`. הפקודה מדפיסה כתובת שפותחים בדפדפן. אפשר למחוק את כל התיקייה `dashboard/` בלי שום השפעה על שאר המערכת.

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

**מעבירים:**
- את מתקין ה־Python (`python-3.12.10-amd64.exe`).
- את כל תיקיית הפרויקט, כולל `wheels`, `models` ו־`tools` (עם llama.cpp). בלי `.venv`.
- את התיקייה `C:\Users\<שם>\.cache\huggingface` (ב־Linux: `~/.cache/huggingface`). מעבירים אותה כקובץ zip: העתקה רגילה עלולה לאבד חלק מהקבצים שבה.

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
| `Visual C++ runtime` מסומן FAIL ב־preflight | חסר Microsoft Visual C++ Redistributable (x64). ההתקנה שלו דורשת את ה־IT. |
| `certificate verify failed` או שגיאת proxy בהורדת המודלים | הרשת של הבנק דורשת הגדרה. ראו את הסעיף "Bank network" בקובץ `.env.example`. |
| `port ... is already in use` | משתמש אחר באותו שרת מריץ שרת שיפוט. ראו שלב 6.2. |
| `the judge server ... serves ..., not judge.model` | השרת שעונה הוא לא השרת שלכם, או שהמודל ב־`config.yaml` לא מעודכן. שום תמלול לא נשלח. |

---

## מסמכים נוספים

- `docs/DEPLOYMENT.md` — פירוט מלא: אבטחה, רישוי, מה נשמר ולכמה זמן, ומגבלות ידועות.
- `docs/executive_report_he.md` — דוח המנהלים: מה כל חלק מראה, איך כל מספר מחושב ואיך לקרוא אותו.
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
| GPU | **Not on this machine** | Transcription, speaker separation and redaction run on the CPU. The judge model needs a GPU on another machine on the network (step 6). |
| Visual C++ Redistributable (x64) | For the models only | Usually installed already. `preflight` checks and says if it is missing (installing it needs IT). |

---

## Step 1 — Python

**Windows:**
1. Go to https://www.python.org/downloads/release/python-31210/ and download the **Windows installer (64-bit)** — the file `python-3.12.10-amd64.exe`. (It is the last 3.12 release that has a Windows installer.)
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

⚠️ **Where not to put the project:** not on the Desktop, not in Documents and not in OneDrive. On many machines those folders are synced to the cloud, and the project stores transcripts containing identifiers. Not on a network drive either: the program's database is not reliable there. `preflight` checks for both and stops.

- **Space:** with the models the project takes about 30 GB. Some VDI setups cap the user folder. If so, ask IT which local disk persists between sign-ins and put the models there: `--models-dir D:\callqa-models` in step 5, and the line `CALLQA_PATHS__MODELS_DIR=D:\callqa-models` in `.env`.
- Installing (step 3) makes the project folder private to your user (and the administrators), even outside your user folder.

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

**The check passed if** `data/demo/output/reports/index.html` opens in a browser and shows 6 scored calls. The check works in a separate folder (`data/demo`), so it can be run again at any time without touching the real calls.

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

**5.3 — Download:**

```
.venv\Scripts\python scripts\download_models.py --asr --diarization
```

- If you choose option B in step 6 (llama.cpp on this machine), also download the judge model, one compressed file of about 4.4 GB:

```
.venv\Scripts\python scripts\download_models.py --llm --llm-model dicta-il/dictalm2.0-instruct-GGUF --llm-gguf dictalm2.0-instruct.Q4_K_M.gguf
```

- At the end it prints a table of which models downloaded successfully (OK) and which failed (FAILED), with what to do about each failure.
- The script records each model's licence in `models/MODELS_MANIFEST.json`.
- More options: `.venv\Scripts\python scripts\download_models.py --help`.

---

## Step 6 — The judge model

The judge model is the one part that really needs a GPU. There are two options:

| | Quality | Time per call | Good for |
|---|---|---|---|
| **A. A GPU server on the bank's network (recommended)** | a 27B model was measured and passed | about 3.5 minutes | real work |
| **B. llama.cpp on this machine, no GPU** | a 7B model. Measured below the quality bar: most calls end up "needs human review" | 10 minutes or more | checking that the whole chain is connected |

**6.1 — A key.** The server is protected by a key, because it answers with text built from call transcripts. Create one:

```
.venv\Scripts\python -c "import secrets; print(secrets.token_hex(32))"
```

and add it to `.env` on this line: `CALLQA_JUDGE__API_KEY=<the key>`

**6.2 — Option A: a GPU server.** The infrastructure team sets up vLLM on a Linux machine with a 24 GB NVIDIA GPU, as described in `docs/DEPLOYMENT.md` (section 5), behind HTTPS and with the same key. On this machine, add to `.env`:

```
CALLQA_JUDGE__BASE_URL=https://<server address>/v1
CALLQA_JUDGE__MODEL=<the model name the server reports>
```

The recordings themselves never go to the server. Only the transcript, after the identifiers are hidden, does.

**6.3 — Option B: llama.cpp on this machine.**
1. From https://github.com/ggml-org/llama.cpp/releases (under Assets of the latest release) download the zip file whose name contains `bin-win-cpu-x64`.
2. Unzip it into a folder called `tools` inside the project (create it if it does not exist). Nothing is installed.
3. Start it, **in a separate terminal window that stays open**:

```
.venv\Scripts\python scripts\start_llama_server.py models\dicta-il--dictalm2.0-instruct-GGUF\dictalm2.0-instruct.Q4_K_M.gguf
```

- Add the line `CALLQA_JUDGE__REQUEST_TIMEOUT_SEC=3600` to `.env`: on a CPU, one call can take more than 10 minutes.
- The script reads the key from `.env` itself, and listens on this machine only (127.0.0.1).
- If it says the port is in use (for example because another user of the same VDI host already runs such a server), run it with `--port 8001` and add `CALLQA_JUDGE__BASE_URL=http://127.0.0.1:8001/v1` to `.env`.
- Before any transcript is sent, the program checks that the server answering it serves the model it was configured with, so calls are never sent to someone else's server.

---

## Step 7 — Settings

All settings are in `config/config.yaml`, and each one is explained inside the file. These are the settings worth checking:

| Setting | What it does |
|---|---|
| `judge.base_url` | The address of the judge model server. The default, `http://127.0.0.1:8000/v1`, matches step 6. |
| `retention.raw_days` | After how many days the raw data containing identifiers is deleted. The default is 90. |
| `redaction.ner` | Hides names that nobody asked for. Requires `scripts/download_models.py --ner`. |
| `asr.compute_type` | `float16` for a GPU. On a machine without one the program switches to `int8` by itself and logs that it did. |

Put passwords and keys **only** in `.env`, never in `config.yaml`.

---

## Step 8 — Prepare the calls

```
mkdir data\input\calls
```

1. Copy the recordings into `data/input/calls/`. Keep a copy: once processed, recordings are moved to `data/input/processed/`.
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
- **Formats:** after step 5 the program reads WAV (including telephony-system WAV, G.711), mp3, m4a and more by itself — nothing else to install. Without step 5 (a test only) it reads plain WAV only. Alternatively: download `ffmpeg-master-latest-win64-lgpl.zip` from https://github.com/BtbN/FFmpeg-Builds/releases and unzip it into the project's `tools` folder.

---

## Step 9 — Check and run

```
.venv\Scripts\python -m callqa preflight
.venv\Scripts\python -m callqa run
.venv\Scripts\python -m callqa report
```

- `preflight` checks that models, server, disk and input are ready. Run it before every run. If any of its lines is marked `FAIL`, fix it before you continue.
- `run` processes every call in `metadata.csv`.
- `report` produces a report for each banker and the **management report** (see below).

**Automated running:**
- `.venv\Scripts\python -m callqa watch` — watches `data/input/calls/` and processes every new recording that arrives.
- `.venv\Scripts\python -m callqa process --audio <file>` — processes one call. Exit code: `0` success · `1` needs human review · `2` failed. In cmd, see it with `echo %ERRORLEVEL%`.
- **Scheduling on Windows:** Task Scheduler → Create Basic Task → Start a program. No admin rights needed. Create three tasks:

  | Task | Program | Arguments | Start in |
  |---|---|---|---|
  | Process calls | full path of `.venv\Scripts\python.exe` | `-m callqa run --log-file logs\run.log` | the project folder |
  | Refresh the reports and the management report (daily, after processing) | the same path | `-m callqa report --log-file logs\report.log` | the project folder |
  | Delete raw data (daily) | the same path | `-m callqa retention --apply --log-file logs\retention.log` | the project folder |

  ⚠️ **Start in is required.** Without it the program refuses to run. A task runs only while the machine is on. On a VDI that is reset at every sign-out, ask IT for a persistent machine.

---

## Management report

One professional, management-ready report over every processed call (made for 1,000 calls and more). It is produced automatically by every `report` run and saved as `data/output/reports/executive.html`. Open it with a double-click (Edge). It works with no internet connection and can be passed on as a single file.

**Three levels, from the overall picture to the detail:**
1. **Executive summary** — a professional opinion in a few sentences, 6 key indicators with their trend, the 5 main findings, and 3 recommended actions, each with its expected impact in numbers.
2. **In-depth analysis** — the rubric dimensions and "where the points are lost", the score distribution, trends over time, breakdowns by call type and length, banker comparison and heat map, behavioural drivers (talk share, interruptions, questions and more), compliance risk, and data quality.
3. **Detail and calls** — every call: filter, sort and search; click a call for its score, reasoning and quote on every dimension, and a link to the full call report. Plus a case library (excellent and weak examples per dimension) and the list of calls that need attention.

Every level-1 finding links to its analysis in level 2 and to the calls themselves in level 3. Every figure carries a confidence interval, and a group is flagged only when its gap is statistically significant. The full explanation is in `docs/executive_report_he.md`.

**A report for a period, a call type or a banker:**

```
.venv\Scripts\python -m callqa executive-report --from 01/08/2026 --to 31/08/2026
.venv\Scripts\python -m callqa executive-report --call-type loans --no-quotes
```

- `--from` / `--to` — call dates (`31/08/2026` or `2026-08-31`). `--call-type`, `--banker` and `--title` also work.
- `--no-quotes` — a version with no quote or reasoning from the calls at all, for wide distribution.
- The report gets its own name in `data/output/reports/`, for example `data/output/reports/executive-2026-08-01_2026-08-31.html`, next to a CSV file for Excel.
- **Print / PDF:** the button at the top of the report (or Ctrl+P). Every level starts on a new page.

**To see it on 1,000 calls before there is real data:**

```
.venv\Scripts\python scripts\generate_batch_demo.py --calls 1000 --open
```

The script creates 1,000 synthetic calls in a separate folder (`data/demo-batch`), produces a management report from them and opens it. The report is clearly marked as a demonstration and does not touch the real calls.

**A sample report, ready to view:** [`docs/examples/executive-demo.pdf`](docs/examples/executive-demo.pdf) (opens directly on GitHub) and [`docs/examples/executive-demo.html`](docs/examples/executive-demo.html) — the full interactive report over 1,000 synthetic calls. Download the HTML file (Download raw file) and open it in Edge.

---

## What is created, and where

| Location | What is there |
|---|---|
| `data/output/reports/index.html` | The reports. **Start here.** |
| `data/output/reports/executive.html` | The management report. Next to it, `data/output/reports/executive_calls.csv` — every call and its scores, for Excel. |
| `data/output/transcripts/` | ⚠️ **Raw** transcripts containing identifiers. `.venv\Scripts\python -m callqa retention --apply` deletes them once they are older than `retention.raw_days`. |
| `data/output/redacted/` | Transcripts after the identifiers are hidden |
| `data/output/redacted_audio/` | Recordings with the identifiers silenced |

- The data folders are accessible only to the user who ran the program (and the system administrators). On Windows the program sets this itself in the folder permissions, so other users of the same VDI host cannot read them.
- Optional dashboard: `.venv\Scripts\python dashboard\server.py`. It prints an address to open in a browser. You can delete the whole `dashboard/` folder with no effect on the rest of the system.

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

**Move across:**
- The Python installer (`python-3.12.10-amd64.exe`).
- The whole project folder, including `wheels`, `models` and `tools` (with llama.cpp). Without `.venv`.
- The folder `C:\Users\<name>\.cache\huggingface` (on Linux: `~/.cache/huggingface`). Move it as a zip file: a plain copy can lose some of the files in it.

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
| `Visual C++ runtime` FAIL in preflight | The Microsoft Visual C++ Redistributable (x64) is missing. Installing it needs IT. |
| `certificate verify failed` or a proxy error while downloading models | The bank's network needs a setting. See the "Bank network" section of `.env.example`. |
| `port ... is already in use` | Another user of the same host runs a judge server. See step 6.2. |
| `the judge server ... serves ..., not judge.model` | The server answering is not yours, or the model in `config.yaml` is out of date. No transcript was sent. |

---

## More documents

- `docs/DEPLOYMENT.md` — full detail: security, licensing, what is stored and for how long, and known limits.
- `docs/executive_report_he.md` — the management report (Hebrew): what each part shows, how each number is computed and how to read it.
- `docs/MLOPS.md` — running it over time: evaluation, drift monitoring, reproducing results.
- `CHANGELOG.md` — what changed in each version.
