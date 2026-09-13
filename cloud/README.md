# Cloud dev option (temporary) · אופציית ענן לפיתוח (זמנית)

**🌐 [English](#english) · [עברית](#hebrew)**

<a id="english"></a>

> **This entire directory is scaffolding for the development phase.**
> When the project moves to the bank's own GPU servers, delete `cloud/`,
> delete `config/config.cloud.yaml`, delete `src/callqa/asr/remote_engine.py`
> and `tests/test_remote_asr.py`, and nothing else changes. See
> [Removal](#removal).

## Why this exists

The pipeline needs a GPU for exactly two stages: transcription (ivrit.ai
Whisper) and the LLM judge (vLLM). Everything else — ingestion, VAD, speaker
attribution, PII redaction, features, the Hebrew reports — is cheap CPU work
that runs fine on a laptop.

So during development you rent one GPU machine by the hour, run those two
stages on it, and keep the rest local:

```
your computer                          rented GPU machine (RunPod)
─────────────                          ──────────────────────────
ingestion, audio/VAD      ──audio──►   ASR server  :8001  (same FasterWhisperEngine
speaker attribution       ◄─JSON───              the bank server will run)
PII redaction
features                  ──text───►   vLLM        :8000  (same `vllm serve` the
judge orchestration       ◄─JSON───              bank server will run)
reports (Hebrew, RTL)
```

**Fidelity is the point.** The rented machine runs the same
`scripts/download_models.py` and the same `vllm serve` command as the bank
runbook, and the ASR server wraps the project's own `FasterWhisperEngine`.
Transcripts and scores are identical to an on-prem run; only a network hop
differs.

## Cost

Prices verified September 2026 ([RunPod pricing](https://www.runpod.io/pricing)).
A pod bills per second **only while running**.

| GPU | $/hr (Community) | Fits | Notes |
|---|---|---|---|
| RTX 4090 24GB | ~$0.34 | 7B judge fp16 + Whisper | default, best value |
| RTX A6000 48GB | ~$0.49 | 12–27B AWQ judge | if you want a stronger judge |
| A100 80GB | ~$1.39 | 70B AWQ judge | only if needed |

Work in sessions: start the box, run a batch, stop it.

| Activity | Time | Cost on a 4090 |
|---|---|---|
| First boot + model downloads | ~20 min | ~$0.12 |
| 100 calls (ASR + judge) | ~1–1.5 h | ~$0.35–0.50 |
| Idle while stopped | — | $0 GPU, ~$0.07/GB/mo volume |

A realistic month of development is **a few dollars**. The danger is not the
rate, it is leaving the pod running: `runpod_cli.py status` warns after four
hours and `down` stops the meter.

## Setup — your steps

You do steps 1–3 (they need a browser and a credit card). Everything after is
commands, and Claude can run them for you if you allowlist `rest.runpod.io` in
the Claude Code web environment's network settings.

**1. Create the account and add credit**
   - Sign up at <https://www.runpod.io>.
   - Billing → add credit. **$10 is plenty** to start.
   - Set a spending alert while you are there.

**2. Create an API key**
   - <https://console.runpod.io/user/settings> → **API Keys** → **+ Create API Key**.
   - Give it Read/Write access, copy the key (shown once).

**3. Hand the key over**
   - On your own machine: `cp .env.example .env`, then paste the key on the
     `RUNPOD_API_KEY=` line. Every command in this project loads `.env`.
   - For Claude to drive it: paste the key into the Claude Code web
     environment as the environment variable `RUNPOD_API_KEY`, and add
     `rest.runpod.io` to the allowed domains
     ([docs](https://code.claude.com/docs/en/claude-code-on-the-web)).
   - Never commit the key. `cloud/.runpod_state.json` is gitignored.

**4. Start the machine**
   ```bash
   python cloud/runpod_cli.py up            # add --gpu "NVIDIA RTX A6000" for 48GB
   python cloud/runpod_cli.py urls          # writes the endpoints + secrets into .env
   ```

**5. Install on the machine** (once per new volume)
   - Open the pod's **web terminal** from the RunPod console.
   - Clone and bootstrap:
     ```bash
     cd /workspace && git clone https://github.com/yonaoh10/BRS_Model.git
     cd BRS_Model && bash cloud/bootstrap_pod.sh
     ```
   - It installs deps, downloads models with the project's own
     `download_models.py`, then starts vLLM and the ASR server. Takes
     10–20 minutes the first time; later starts reuse the volume.

**6. Point your laptop at it**
   Nothing to export: step 4 wrote the four `CALLQA_*` endpoint variables into
   `.env`, and the pipeline reads them from there.
   ```bash
   python -m callqa run --config config/config.cloud.yaml
   python -m callqa report --config config/config.cloud.yaml
   ```

**7. Stop paying**
   ```bash
   python cloud/runpod_cli.py down          # ends GPU billing, keeps the models
   python cloud/runpod_cli.py destroy --yes # removes the volume too
   ```

## Security rules — read before using real recordings

- **Synthetic or consented recordings only.** Audio leaves your machine in
  this setup. Real customer calls are governed by the Privacy Protection Law
  and Bank of Israel cloud directives, and belong on the bank's own
  infrastructure. This option exists to build and tune the system, not to
  process real calls.
- Both endpoints require a bearer token, generated per pod and stored with
  `0600` permissions in `cloud/.runpod_state.json`.
- **The RunPod proxy URL is public.** vLLM's `--api-key` protects `/v1/*` but
  [not the `/invocations` path](https://docs.vllm.ai/en/latest/usage/security/),
  so treat the judge endpoint as exposed: keep the pod up only while you use
  it, and destroy it when a phase ends.
- The ASR server deletes each uploaded file immediately after transcription
  and never logs audio or transcript text.
- Redaction still happens locally, before anything reaches the judge, exactly
  as it will on-prem.

## Removal

When the bank servers are ready:

```bash
make cloud-remove      # deletes cloud/, the cloud config, the remote engine + its tests
python -m pytest -q    # still green
```

Then follow the normal runbook in the main README. The bank config
(`config/config.yaml`) already says `asr.engine: faster_whisper` and points
the judge at `localhost:8000`, so there is nothing to un-wire — the cloud
option was never in that path.

---

<a id="hebrew"></a>

# עברית — אופציית ענן לפיתוח (זמנית)

**🌐 [English](#english) · [עברית](#hebrew)**

> **כל התיקייה הזו היא פיגום לשלב הפיתוח בלבד.**
> כשהפרויקט יעבור לשרתי ה-GPU של הבנק, מוחקים את `cloud/`, את
> `config/config.cloud.yaml`, את `src/callqa/asr/remote_engine.py` ואת
> `tests/test_remote_asr.py` — ושום דבר אחר לא משתנה. ראה [הסרה](#הסרה).

## למה זה קיים

הצינור זקוק ל-GPU עבור שני שלבים בלבד: תמלול (Whisper של ivrit.ai) ושופט
ה-LLM (vLLM). כל השאר — קליטה, VAD, שיוך דוברים, הסרת פרטים מזהים, מדדים
והדוחות בעברית — הוא עבודת מעבד זולה שרצה מצוין על מחשב נייד.

לכן בשלב הפיתוח שוכרים מכונת GPU אחת לפי שעה, מריצים עליה את שני השלבים
האלה, והשאר נשאר מקומי:

```
המחשב שלך                                מכונת GPU שכורה (RunPod)
─────────                                ────────────────────────
קליטה, אודיו/VAD           ──אודיו──►    שרת תמלול  :8001  (אותו מנוע
שיוך דוברים                ◄──JSON───              שירוץ בשרת הבנק)
הסרת פרטים מזהים
מדדים אובייקטיביים          ──טקסט──►    vLLM       :8000  (אותה פקודה
תזמור השופט                ◄──JSON───              שתרוץ בשרת הבנק)
דוחות (עברית, RTL)
```

**הנאמנות היא העיקר.** המכונה השכורה מריצה את אותו
`scripts/download_models.py` ואת אותה פקודת `vllm serve` שבמדריך הבנק, ושרת
התמלול עוטף את `FasterWhisperEngine` של הפרויקט עצמו. התמלילים והציונים
זהים להרצה מקומית בבנק; ההבדל היחיד הוא קפיצה ברשת.

## עלות

המחירים אומתו בספטמבר 2026 ([תמחור RunPod](https://www.runpod.io/pricing)).
החיוב הוא לפי שנייה **רק בזמן שהמכונה פועלת**.

| GPU | $/שעה (Community) | מתאים ל | הערות |
|---|---|---|---|
| RTX 4090 24GB | ~$0.34 | שופט 7B fp16 + Whisper | ברירת המחדל, התמורה הטובה ביותר |
| RTX A6000 48GB | ~$0.49 | שופט 12–27B בכימות AWQ | אם רוצים שופט חזק יותר |
| A100 80GB | ~$1.39 | שופט 70B AWQ | רק אם באמת נדרש |

עובדים במקטעים: מפעילים, מריצים אצווה, מכבים.

| פעילות | זמן | עלות על 4090 |
|---|---|---|
| אתחול ראשון + הורדת מודלים | ~20 דקות | ~$0.12 |
| 100 שיחות (תמלול + שופט) | ~1–1.5 שעות | ~$0.35–0.50 |
| מושבת (לא פועל) | — | $0 על ה-GPU, ~$0.07 לג'יגה בחודש אחסון |

חודש פיתוח ריאלי עולה **כמה דולרים בודדים**. הסכנה אינה התעריף אלא שכחה של
מכונה פועלת: `runpod_cli.py status` מזהיר אחרי ארבע שעות, ו-`down` עוצר
את המונה.

## התקנה — הצעדים שלך

אתה מבצע צעדים 1–3 (דורשים דפדפן וכרטיס אשראי). כל השאר הוא פקודות, וקלוד
יכול להריץ אותן עבורך אם תוסיף את `rest.runpod.io` לרשימת הדומיינים
המורשים בהגדרות הרשת של סביבת Claude Code.

**1. פתיחת חשבון וטעינת יתרה**
   - הרשמה ב-<https://www.runpod.io>.
   - Billing ← טעינת יתרה. **10 דולר יותר מספיקים** להתחלה.
   - הגדר התראת הוצאות בזמן שאתה שם.

**2. יצירת מפתח API**
   - <https://console.runpod.io/user/settings> ← **API Keys** ←
     **+ Create API Key**.
   - הרשאת קריאה/כתיבה, והעתק את המפתח (מוצג פעם אחת בלבד).

**3. מסירת המפתח**
   - במחשב שלך: `cp .env.example .env`, ואז הדבק את המפתח בשורה
     `RUNPOD_API_KEY=`. כל פקודה בפרויקט טוענת את `.env`.
   - כדי שקלוד יפעיל את זה: הדבק את המפתח בסביבת Claude Code כמשתנה הסביבה
     `RUNPOD_API_KEY`, והוסף את `rest.runpod.io` לדומיינים המורשים
     ([תיעוד](https://code.claude.com/docs/en/claude-code-on-the-web)).
   - לעולם אל תכניס את המפתח ל-git. הקובץ `cloud/.runpod_state.json` נמצא
     ב-gitignore.

**4. הפעלת המכונה**
   ```bash
   python cloud/runpod_cli.py up            # הוסף --gpu "NVIDIA RTX A6000" עבור 48GB
   python cloud/runpod_cli.py urls          # כותב את הכתובות והסודות לתוך .env
   ```

**5. התקנה על המכונה** (פעם אחת לכל אחסון חדש)
   - פתח את ה-**web terminal** של הפוד מהקונסולה של RunPod.
   - שכפל והרץ:
     ```bash
     cd /workspace && git clone https://github.com/yonaoh10/BRS_Model.git
     cd BRS_Model && bash cloud/bootstrap_pod.sh
     ```
   - הסקריפט מתקין תלויות, מוריד מודלים באמצעות `download_models.py` של
     הפרויקט עצמו, ואז מפעיל את vLLM ואת שרת התמלול. 10–20 דקות בפעם
     הראשונה; הפעלות הבאות משתמשות באחסון הקיים.

**6. הפניית המחשב שלך אליה**
   אין מה לייצא: שלב 4 כתב את ארבעת משתני `CALLQA_*` לתוך `.env`,
   והפייפליין קורא אותם משם.
   ```bash
   python -m callqa run --config config/config.cloud.yaml
   python -m callqa report --config config/config.cloud.yaml
   ```

**7. להפסיק לשלם**
   ```bash
   python cloud/runpod_cli.py down          # עוצר חיוב GPU, שומר את המודלים
   python cloud/runpod_cli.py destroy --yes # מוחק גם את האחסון
   ```

## כללי אבטחה — לקרוא לפני שימוש בהקלטות אמיתיות

- **הקלטות סינתטיות או בהסכמה בלבד.** בהגדרה הזו האודיו יוצא מהמחשב שלך.
  שיחות לקוחות אמיתיות כפופות לחוק הגנת הפרטיות ולהוראות בנק ישראל בנושא
  מחשוב ענן, ומקומן בתשתית של הבנק. האופציה הזו נועדה לבנות ולכוונן את
  המערכת, לא לעבד שיחות אמיתיות.
- שתי נקודות הקצה דורשות טוקן, שנוצר לכל פוד ונשמר בהרשאות `0600` בקובץ
  `cloud/.runpod_state.json`.
- **כתובת ה-proxy של RunPod היא ציבורית.** הדגל `--api-key` של vLLM מגן על
  `/v1/*` אך [לא על הנתיב `/invocations`](https://docs.vllm.ai/en/latest/usage/security/),
  ולכן יש להתייחס לנקודת הקצה של השופט כחשופה: השאר את הפוד פועל רק בזמן
  שימוש, ומחק אותו בסיום שלב.
- שרת התמלול מוחק כל קובץ שהועלה מיד לאחר התמלול, ולעולם אינו רושם ללוג
  אודיו או תוכן תמליל.
- הסרת הפרטים המזהים עדיין מתבצעת מקומית, לפני שמשהו מגיע לשופט — בדיוק
  כפי שיקרה בבנק.

## הסרה

כששרתי הבנק יהיו מוכנים:

```bash
make cloud-remove      # מוחק את cloud/, קונפיג הענן, מנוע ה-remote והבדיקות שלו
python -m pytest -q    # עדיין ירוק
```

לאחר מכן פשוט עוקבים אחרי המדריך הרגיל ב-README הראשי. קונפיג הבנק
(`config/config.yaml`) כבר מוגדר `asr.engine: faster_whisper` ומפנה את השופט
ל-`localhost:8000`, כך שאין מה לנתק — אופציית הענן מעולם לא הייתה בנתיב הזה.
