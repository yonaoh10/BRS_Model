"""A synthetic batch of customer journeys, shaped like the bank's real exports.

    python scripts/generate_journey_demo.py --stories 100 [--workspace data/journey-demo]
            [--audio] [--seed 7] [--no-report]

Writes, into a folder of its own (never the real data folders):

    input/handoff.xlsx          the three-sheet handoff workbook (contacts, call
                                files, messages) - sheets in the order an EG
                                import leaves them, the files sheet without a header
    input/recordings.zip        synthetic NICE .nmf parts (tones, two streams) - --audio
    input/atlas/ATLR_*.csv      Atlas exports: contacts, banker sessions, operations,
                                operation categories (SAS DATETIME format)
    output/...                  the per-call artifacts of the recorded calls, as the
                                pipeline would leave them after transcription:
                                redacted transcripts, ingestion records, results and
                                (for most calls) scorecards

Every story follows one of a handful of scenarios with a known cause - a
callback that never came, a runaround between the call centre and the
branch, a legitimate process, abandoned calls, a written thread, a new topic
- so the report built from it reads like a real one. Everything is invented:
names are masked tokens, account numbers random, audio is tones.

Then, unless --no-report: `callqa journey import`, `journey process --mock`
(when available) and `journey report`, with the workspace's paths.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
import zipfile
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from callqa.journey import g711, nmf  # noqa: E402
from callqa.journey.importers.common import call_id_for  # noqa: E402
from callqa.journey.xlsx import write_xlsx  # noqa: E402

MARKER = ".callqa-journey-demo"
DEFAULT_WORKSPACE = ROOT / "data" / "journey-demo"
PERIOD_START = datetime(2026, 6, 1)
PERIOD_DAYS = 84
ATLAS_FROM = datetime(2026, 6, 25)      # the Atlas log keeps ~90 days
CENTER = "109"
BACK_OFFICE = "136"
NAME = "<שם:████>"
ID = "<ת\"ז:████>"
ACCT = "<חשבון:████>"

TOPICS = [  # key, weight, what the customer wants, what resolves it
    ("loans_mortgages", 18, "מה קורה עם בקשת ההלוואה שהגשתי, אמרו לי שתאושר תוך יומיים",
     "ההלוואה אושרה והכסף יועבר לחשבון מחר"),
    ("transfers_payments", 17, "העברה של שמונת אלפים שקל נחסמה ואני לא מבינה למה",
     "שחררתי את ההעברה, היא תצא עוד היום"),
    ("account_service", 14, "אני צריך אישור על יתרות לשנת 2025 בשביל רואה החשבון",
     "שלחתי לך את האישור לתיבת ההודעות"),
    ("deposits_investments", 12, "אני רוצה לשבור את הפיקדון ולהבין כמה ריבית אפסיד",
     "הפיקדון שוחרר והכסף בעובר ושב"),
    ("cards_digital", 12, "לא קיבלתי קוד סודי לכרטיס החדש ואני לא יכולה למשוך מזומן",
     "הקוד נשלח שוב ב-SMS ותוכלי למשוך כבר עכשיו"),
    ("credit_lines", 8, "אני צריך הגדלה זמנית של המסגרת עד סוף החודש",
     "אישרתי הגדלה זמנית של המסגרת עד סוף החודש"),
    ("fx_international", 6, "הגיע אליי תקבול מחו\"ל והוא לא נכנס לחשבון",
     "התקבול זוכה בחשבון בשקלים"),
    ("cheques", 6, "צ'ק שהפקדתי חזר ואני לא יודע למה",
     "בדקתי, הצ'ק הופקד מחדש ויכובד"),
    ("other", 3, "אני רוצה לברר משהו לגבי החשבון", "בסדר, טיפלתי בזה"),
]
SCENARIOS = [("broken_callback", 27), ("runaround", 20), ("legit_process", 18),
             ("abandoned", 15), ("thread", 10), ("new_topic", 10)]


@dataclass
class Contact:
    at: datetime
    channel: str                 # call / message
    direction: str               # inbound / outbound
    answered: bool = True
    recorded: bool = False
    talk: float = 0.0
    handler: str = CENTER        # the unit that took it
    lines: list[tuple[str, str]] = field(default_factory=list)   # (speaker, text)
    messages: list[tuple[str, str, str]] = field(default_factory=list)  # (dir, subject, body)
    executes: bool = False
    call_key: str = ""
    um: str = ""
    parts: int = 1


@dataclass
class Story:
    branch: str
    account: str
    topic: str
    scenario: str
    contacts: list[Contact] = field(default_factory=list)
    background_exec: list[tuple[datetime, str]] = field(default_factory=list)  # (when, unit)


def _hex(rng: random.Random) -> str:
    return "007203" + "".join(rng.choice("0123456789abcdef") for _ in range(10))


def _business_time(rng: random.Random, day: datetime) -> datetime:
    while day.weekday() in (4, 5):          # Friday, Saturday
        day += timedelta(days=1)
    return day.replace(hour=rng.randint(8, 17), minute=rng.randint(0, 59), second=rng.randint(0, 59))


def _later(rng: random.Random, t: datetime, hours: tuple[float, float]) -> datetime:
    nt = t + timedelta(hours=rng.uniform(*hours))
    if nt.hour < 8:
        nt = nt.replace(hour=8 + rng.randint(0, 2))
    if nt.hour > 18:
        nt = (nt + timedelta(days=1)).replace(hour=8 + rng.randint(0, 3))
    while nt.weekday() in (4, 5):
        nt += timedelta(days=1)
    return nt


def _opening(banker_first: bool = True) -> tuple[str, str]:
    return ("banker", f"שלום, הגעת למרכז הבנקאות, מדברת {NAME}, במה אפשר לעזור?")


def _ident() -> list[tuple[str, str]]:
    return [("banker", "לפני שנמשיך, אפשר בבקשה מספר תעודת זהות?"),
            ("customer", f"כן, {ID}"),
            ("banker", "תודה, זיהיתי אותך.")]


def _call(t, talk, lines, handler=CENTER, direction="inbound", executes=False):
    return Contact(at=t, channel="call", direction=direction, answered=True, recorded=True,
                   talk=talk, handler=handler, lines=lines, executes=executes)


def build_story(rng: random.Random, n: int) -> Story:
    topic, _w, want, resolve = rng.choices(TOPICS, weights=[t[1] for t in TOPICS])[0]
    scenario = rng.choices([s for s, _ in SCENARIOS], weights=[w for _, w in SCENARIOS])[0]
    branch = str(rng.choice([2, 5, 10, 14, 22, 30, 33, 35, 42, 43, 46, 49, 55, 56, 67, 73, 74,
                             83, 102, 113, 120, 122, 126, 130, 139, 146, 148, 150, 165, 176]))
    story = Story(branch=branch, account=str(rng.randint(100000, 499999)), topic=topic,
                  scenario=scenario)
    t = _business_time(rng, PERIOD_START + timedelta(days=rng.randint(0, PERIOD_DAYS - 14)))
    c = story.contacts

    def abandoned(at):
        c.append(Contact(at=at, channel="call", direction="inbound", answered=False, talk=0))

    def branch_call(at, executes=False):
        c.append(Contact(at=at, channel="call", direction="inbound", answered=True, recorded=False,
                         talk=rng.uniform(60, 400), handler=branch, executes=executes))

    if scenario == "broken_callback":
        c.append(_call(t, rng.uniform(180, 420), [
            _opening(), ("customer", f"שלום, {want}"), *_ident(),
            ("banker", "אני רואה את הבקשה במערכת, אבל אני צריכה לבדוק מול המחלקה."),
            ("banker", "אבדוק ואחזור אלייך עד מחר בצהריים, בסדר?"),
            ("customer", "בסדר, אני מחכה לטלפון.")]))
        t2 = _later(rng, t, (24, 50))
        for _ in range(rng.randint(0, 2)):
            abandoned(t2)
            t2 = _later(rng, t2, (0.1, 1.5))
        c.append(_call(t2, rng.uniform(240, 560), [
            _opening(), ("customer", "שלום, כבר דיברתי איתכם לפני יומיים, אמרו לי שיחזרו אליי "
                                     "ואף אחד לא חזר."),
            ("banker", "אני מתנצלת, אני לא רואה תיעוד של השיחה הקודמת. אפשר להסביר לי מה הבעיה?"),
            ("customer", f"שוב מההתחלה? {want}. זו כבר הפעם השלישית שאני מסביר."), *_ident(),
            ("banker", "אני מעבירה את זה לסניף, הם יטפלו בזה ויחזרו אלייך."),
            ("customer", "אני מקווה שהפעם באמת יחזרו.")]))
        if rng.random() < 0.55:
            t3 = _later(rng, t2, (20, 70))
            c.append(_call(t3, rng.uniform(120, 300), [
                ("banker", f"שלום, מדבר {NAME} מהסניף, חוזר אלייך לגבי הפנייה שלך."),
                ("customer", "סוף סוף, תודה."), ("banker", f"{resolve}."),
                ("customer", "מעולה, תודה רבה.")], handler=branch, direction="outbound",
                executes=True))
        else:
            t3 = _later(rng, t2, (26, 80))
            c.append(_call(t3, rng.uniform(200, 500), [
                _opening(), ("customer", "שלום, אני מתקשר כבר פעם רביעית, הבטיחו לי שהסניף "
                                         "יחזור אליי ואף אחד לא חזר."),
                ("banker", "אני רואה שהפנייה עדיין פתוחה. אעביר שוב לסניף עם דחיפות."),
                ("customer", "זה לא רציני, אני שוקל לעבור בנק.")]))
            if rng.random() < 0.5:
                story.background_exec.append((_later(rng, t3, (30, 90)), branch))
    elif scenario == "runaround":
        c.append(_call(t, rng.uniform(150, 360), [
            _opening(), ("customer", f"שלום, {want}"), *_ident(),
            ("banker", "את זה אני לא יכולה לבצע מכאן, צריך לפנות לסניף."),
            ("customer", "אבל ניסיתי להשיג את הסניף כל הבוקר.")]))
        t2 = _later(rng, t, (0.2, 1.0))
        abandoned(t2)
        t3 = _later(rng, t2, (0.3, 1.5))
        branch_call(t3)
        t4 = _later(rng, t3, (0.5, 3.0))
        c.append(_call(t4, rng.uniform(200, 480), [
            _opening(), ("customer", "שלום, דיברתי עם הסניף והם אמרו שדווקא אתם מטפלים בזה."),
            ("banker", "אוקיי, תסביר לי בבקשה מה הבקשה?"),
            ("customer", f"{want}, סיפרתי את זה כבר פעמיים היום."),
            ("banker", "אני מעבירה אותך לבנקאי אחר שמטפל בזה."),
            ("customer", "עוד העברה...")]))
        if rng.random() < 0.6:
            t5 = _later(rng, t4, (18, 48))
            c.append(_call(t5, rng.uniform(150, 360), [
                _opening(), ("customer", f"שלום, {want}, זו הפעם החמישית שאני פונה."),
                ("banker", "אני רואה את כל ההיסטוריה, אני מטפלת בזה עכשיו."),
                ("banker", f"{resolve}."), ("customer", "תודה, סוף סוף.")], executes=True))
    elif scenario == "legit_process":
        c.append(_call(t, rng.uniform(180, 400), [
            _opening(), ("customer", f"שלום, {want}"), *_ident(),
            ("banker", "כדי להתקדם אני צריכה ממך תלוש שכר אחרון. אפשר לשלוח בהודעה באתר."),
            ("customer", "בסדר, אשלח היום.")]))
        t2 = _later(rng, t, (3, 26))
        um = Contact(at=t2, channel="message", direction="inbound", messages=[
            ("Inbound", "מסמכים", "שלום, מצרף את תלוש השכר כפי שביקשתם. תודה."),
            ("Outbound", "מסמכים", f"שלום רב, קיבלנו את המסמך ונעדכן. בברכה, {NAME}")])
        c.append(um)
        t3 = _later(rng, t2, (20, 60))
        c.append(_call(t3, rng.uniform(120, 260), [
            ("banker", f"שלום, מדברת {NAME} ממרכז הבנקאות, חוזרת אלייך לגבי הבקשה."),
            ("banker", f"{resolve}."), ("customer", "מצוין, תודה.")], direction="outbound",
            executes=True))
    elif scenario == "abandoned" and rng.random() < 0.5:
        # the bank calls back before the customer tries again
        abandoned(t)
        t2 = _later(rng, t, (0.3, 3.0))
        c.append(_call(t2, rng.uniform(150, 360), [
            ("banker", f"שלום, מדברת {NAME} ממרכז הבנקאות, ראינו שניסית להשיג אותנו."),
            ("customer", f"כן, תודה שחזרתם. {want}"), *_ident(), ("banker", f"{resolve}."),
            ("customer", "מעולה, תודה.")], direction="outbound", executes=True))
    elif scenario == "abandoned":
        for _ in range(rng.randint(2, 4)):
            abandoned(t)
            t = _later(rng, t, (0.1, 2.0))
        c.append(_call(t, rng.uniform(200, 420), [
            _opening(), ("customer", "שלום, אני מנסה להשיג אתכם כבר שעתיים, כל פעם זה מתנתק."),
            ("customer", f"{want}"), *_ident(), ("banker", f"{resolve}."),
            ("customer", "טוב, תודה.")], executes=True))
        if rng.random() < 0.4:
            t2 = _later(rng, t, (24, 72))
            abandoned(t2)
            branch_call(_later(rng, t2, (0.5, 2)), executes=True)
    elif scenario == "thread":
        c.append(Contact(at=t, channel="message", direction="inbound", messages=[
            ("Inbound", "פנייה", f"שלום, {want}. אשמח לתשובה."),
            ("Outbound", "פנייה", f"שלום רב, העברנו את פנייתך לגורם המטפל. בברכה, {NAME}")]))
        t2 = _later(rng, t, (26, 70))
        c.append(Contact(at=t2, channel="message", direction="inbound", messages=[
            ("Inbound", "פנייה", "שלום, עברו יומיים ולא קיבלתי שום תשובה. מה קורה?"),
            ("Outbound", "פנייה", f"שלום רב, מתנצלים על העיכוב, נחזור אלייך עד סוף היום. {NAME}")]))
        t3 = _later(rng, t2, (20, 50))
        if rng.random() < 0.5:
            c.append(_call(t3, rng.uniform(150, 300), [
                _opening(), ("customer", "שלום, כתבתי לכם פעמיים בהודעות ואף אחד לא עונה."),
                ("customer", f"{want}"), ("banker", f"{resolve}."), ("customer", "תודה.")],
                executes=True))
        else:
            abandoned(t3)
            branch_call(_later(rng, t3, (0.5, 3)), executes=rng.random() < 0.6)
    else:  # new_topic
        c.append(_call(t, rng.uniform(150, 300), [
            _opening(), ("customer", f"שלום, {want}"), *_ident(), ("banker", f"{resolve}."),
            ("customer", "תודה רבה.")], executes=True))
        other = rng.choice([x for x in TOPICS if x[0] != topic])
        t2 = _later(rng, t, (48, 240))
        c.append(_call(t2, rng.uniform(150, 300), [
            _opening(), ("customer", f"שלום, יש לי שאלה אחרת הפעם: {other[2]}"),
            *_ident(), ("banker", f"{other[3]}."), ("customer", "מצוין, תודה.")], executes=True))
        if rng.random() < 0.5:
            branch_call(_later(rng, t2, (24, 96)))
    # a first contact that is not a return but the start: everything is ordered
    c.sort(key=lambda x: x.at)
    for x in c:
        if x.channel == "call":
            x.call_key = _hex(rng)
            if x.recorded:
                x.parts = rng.choices([1, 2, 3], weights=[53, 34, 13])[0]
        else:
            x.um = f"UM-{rng.randint(10_000_000, 10_999_999)}"
    return story


def _sas(dt: datetime) -> str:
    return dt.strftime("%d%b%Y:%H:%M:%S").upper()


def _fmt(dt: datetime) -> str:
    return dt.strftime("%d%b%Y %H:%M:%S")


def write_workbook(stories: list[Story], path: Path, rng: random.Random) -> dict[str, list[str]]:
    rows = [["מועד אינטראקציה ראשונה", "סניף", "חשבון", "מועד השיחה/התכתבות",
             "מזהה שיחה/התכתבות", "סגמנטציית שירות", "כמות אינטראקציות חוזרות", "שם קובץ שיחה"]]
    files: list[list[str]] = []
    msgs = [["מזהה תהליך התכתבות", "מזהה הודעה", "מועד שליחת ההודעה", "כיוון ההודעה",
             "נושא ההתכתבות", "תוכן ההודעה"]]
    parts_of: dict[str, list[str]] = {}
    for s in stories:
        first = s.contacts[0].at
        seg = rng.randint(1, 3)
        for x in s.contacts:
            name = ""
            if x.channel == "call" and x.recorded:
                base = rng.randint(7_600_000_000_000_000_000, 7_699_999_999_999_999_999)
                names = [f"1_{base}_{base + 900_000 + 50_000 * k}" for k in range(x.parts)]
                parts_of[x.call_key] = names
                name = names[0]
                for n in names:
                    files.append([x.call_key.upper(), n])
            rid = x.call_key if x.channel == "call" else x.um
            rows.append([_fmt(first), s.branch, s.account, _fmt(x.at), rid, seg,
                         len(s.contacts) - 1, name])
            for k, (direction, subject, body) in enumerate(x.messages):
                sent = x.at + timedelta(hours=3 * k, milliseconds=rng.randint(0, 999))
                msgs.append([x.um, f"COR-{rng.randint(5_000_000, 5_999_999)}",
                             sent.strftime("%d%b%Y %H:%M:%S.%f")[:-3], direction, subject, body])
    rng.shuffle(files)
    write_xlsx(path, [("ATL_REPEAT_0000", files), ("ATL_REPEAT_0001", msgs),
                      ("ATL_REPEAT_0002", rows)])
    return parts_of


def write_audio(stories: list[Story], parts_of: dict[str, list[str]], path: Path,
                seconds: float, rng: random.Random) -> None:
    t = np.arange(int(seconds * 8000)) / 8000
    with zipfile.ZipFile(path, "w", zipfile.ZIP_STORED) as zf:
        for s in stories:
            for x in s.contacts:
                for k, name in enumerate(parts_of.get(x.call_key, [])):
                    a = (np.sin(2 * np.pi * (220 + 40 * k) * t) * 6000).astype(np.int16)
                    b = (np.sin(2 * np.pi * 480 * t) * 5000 * (t % 2 < 1)).astype(np.int16)

                    def chunks(pcm, t0):
                        return [(t0 + i / 8000, t0 + (i + 1600) / 8000,
                                 g711.encode_alaw(pcm[i:i + 1600])) for i in range(0, len(pcm), 1600)]
                    data = nmf.write_nmf({0: chunks(a, 1000.0), 1: chunks(b, 1000.0)})
                    zf.writestr(f"הקלטות/{name}.NMF", data)
        # the unexplained extra file every real export seems to have
        zf.writestr("הקלטות/1_7000000000000000001_7000000000000000002.NMF",
                    nmf.write_nmf({0: [(0.0, 0.2, g711.encode_alaw(np.zeros(1600, np.int16)))]}))


def write_atlas(stories: list[Story], folder: Path, rng: random.Random) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    center_bankers = [f"B{n:04d}" for n in range(1, 41)]
    branch_bankers: dict[str, list[str]] = {}
    ints = ["ACC_KEY,INT_SEQ,INT_ID,INT_DT,INT_TYPE,KIND,DIR1,TALK_N,CHURN_N"]
    sess = ["ACC_KEY,SESS_NO,S_START,S_END,S_BANKER,S_UNIT,N_OPS,INT_SEQ,LINKED"]
    rows = ["ROW_ID,ACC_KEY,TS_DT,UNIT_NO,BANKER_CODE,OP_KEY,OP_DESC"]
    row_id = 0
    seq = 0
    next_b = 41
    for acc, s in enumerate(stories, start=1):
        bb = branch_bankers.setdefault(s.branch, [])
        while len(bb) < 3:
            bb.append(f"B{next_b:04d}")
            next_b += 1
        n_sess = 0

        def session(start: datetime, unit: str, banker: str, ops: list[tuple[str, str]],
                    int_seq: int | None, acc: int = acc) -> None:
            nonlocal n_sess, row_id
            if start < ATLAS_FROM:
                return
            n_sess += 1
            at = start
            for code, desc in ops:
                row_id += 1
                rows.append(f"{row_id},{acc},{_sas(at)},{unit},{banker},{code},{desc}")
                at += timedelta(seconds=rng.randint(20, 150))
            sess.append(f"{acc},{n_sess},{_sas(start)},{_sas(at - timedelta(seconds=10))},{banker},"
                        f"{unit},{len(ops)},{int_seq or ''},{1 if int_seq else 0}")
        for x in s.contacts:
            seq += 1
            if x.channel == "call":
                itype = "REC" if x.recorded else ("ABN" if not x.answered else "TALK")
                direction = "O" if x.direction == "outbound" else "I"
                ints.append(f"{acc},{seq},{x.call_key},{_sas(x.at)},{itype},CALL,{direction},"
                            f"{int(x.talk)},{0 if x.answered else 1}")
                if not x.answered:
                    continue
                unit = x.handler
                banker = rng.choice(center_bankers if unit == CENTER else bb)
                ops = [("201", "מצב לקוח - זיהוי לקוח חדש"), ("332", "פרטי חשבון / לקוח")]
                if rng.random() < 0.5:
                    ops.append(("011", "תנועות אחרונות"))
                if x.executes:
                    ops.append(rng.choice([("970", "הזמנה/עדכון לכרטיסים מגנטיים"),
                                           ("675", "העברה לבנק אחר"), ("649", "תשלומים"),
                                           ("505", "מסגרות אשראי")]))
                start = x.at + timedelta(minutes=rng.uniform(-8, 2) if x.direction == "outbound"
                                         else rng.uniform(0, 3))
                session(start, unit, banker, ops, seq)
            else:
                ints.append(f"{acc},{seq},{x.um},{_sas(x.at)},UM,UM,,,")
                if rng.random() < 0.6:
                    session(x.at + timedelta(minutes=rng.uniform(5, 50)), BACK_OFFICE,
                            f"B{rng.randint(300, 310):04d}",
                            [("201", "מצב לקוח - זיהוי לקוח חדש"), ("332", "פרטי חשבון / לקוח")],
                            seq)
            # view-only glances by other bankers, with no contact behind them
            for _ in range(rng.choice([0, 0, 0, 0, 1])):
                start = x.at + timedelta(hours=rng.uniform(1, 30))
                session(start, rng.choice([CENTER, BACK_OFFICE, s.branch]), rng.choice(center_bankers),
                        [("201", "מצב לקוח - זיהוי לקוח חדש")], None)
        for when, unit in s.background_exec:
            session(when, unit, rng.choice(bb), [("201", "מצב לקוח - זיהוי לקוח חדש"),
                                                 ("332", "פרטי חשבון / לקוח"),
                                                 ("649", "תשלומים")], None)
    (folder / "ATLR_INT.csv").write_text("\n".join(ints) + "\n", encoding="utf-8")
    (folder / "ATLR_SESS.csv").write_text("\n".join(sess) + "\n", encoding="utf-8")
    (folder / "ATLR_ROWS.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")
    (folder / "ATLR_CODECAT.csv").write_text(
        "CAT,OP_KEY\nOPEN,201\nLOOK,332\nLOOK,011\nDO,970\nDO,675\nDO,649\nOTHER,505\n",
        encoding="utf-8")


def write_call_artifacts(stories: list[Story], output: Path, rng: random.Random) -> int:
    """What the pipeline leaves after transcribing a recorded call: the
    redacted transcript, the ingestion record, and a result - plus a score
    for most calls, lower in the stories that went badly."""
    from callqa.models import (
        CallMeta,
        CallResult,
        DimensionScore,
        RedactedTranscript,
        RedactedTurn,
        ScoreCard,
    )
    from callqa.rubric import load_rubric, weighted_total
    from callqa.state import atomic_write_model

    rubric = load_rubric()
    n = 0
    for s in stories:
        bad = s.scenario in ("broken_callback", "runaround", "thread")
        for x in s.contacts:
            if not (x.channel == "call" and x.recorded):
                continue
            cid = call_id_for(x.call_key)
            turns, t = [], 0.0
            per = max(4.0, x.talk / max(1, len(x.lines)))
            for speaker, text in x.lines:
                turns.append(RedactedTurn(speaker=speaker, start=round(t, 2),
                                          end=round(t + per * 0.8, 2), text=text))
                t += per
            atomic_write_model(output / "redacted" / f"{cid}.json", RedactedTranscript(
                call_id=cid, engine="synthetic-demo", turns=turns,
                redaction_counts={"PERSON": sum(NAME in tx for _s, tx in x.lines),
                                  "ISRAELI_ID": sum(ID in tx for _s, tx in x.lines)}))
            atomic_write_model(output / "ingestion" / f"{cid}.json", CallMeta(
                call_id=cid, banker_id="unknown", file_name=f"{cid}.wav", duration_sec=x.talk,
                channels=2, sample_rate=8000, codec="pcm_s16le",
                call_date=x.at.date().isoformat(), banker_channel="L"))
            n += 1
            if rng.random() < 0.85:
                scores = {}
                for d in rubric.dimensions:
                    base = 3 if bad else 4
                    scores[d.id] = DimensionScore(score=int(np.clip(base + rng.choice([-1, 0, 0, 1]), 1, 5)),
                                                  reasoning_he="(נתון סינתטי)", evidence=[])
                total, gate_failed, failed = weighted_total(rubric, scores)
                atomic_write_model(output / "scores" / f"{cid}.json", ScoreCard(
                    call_id=cid, banker_id="unknown", scores=scores, weighted_total=total,
                    gate_failed=gate_failed, failed_gates=failed, judge_engine="synthetic-demo",
                    model="synthetic-demo", prompt_sha256="0" * 64, prompt_version="demo",
                    rubric_sha256=rubric.sha256, timestamp=datetime.now(UTC).isoformat()))
                atomic_write_model(output / "results" / f"{cid}.json", CallResult(
                    call_id=cid, status="success",
                    stages_completed=["ingestion", "audio", "asr", "speakers", "redaction",
                                      "features", "judge"]))
            else:
                atomic_write_model(output / "results_partial" / f"{cid}.json", CallResult(
                    call_id=cid, status="success",
                    stages_completed=["ingestion", "audio", "asr", "speakers", "redaction",
                                      "features"]))
    return n


def prepare(workspace: Path) -> Path:
    ws = workspace.resolve()
    real = {(ROOT / "data" / "input").resolve(), (ROOT / "data" / "output").resolve(),
            (ROOT / "data").resolve()}
    if ws in real:
        raise SystemExit(f"refusing to write the demo into {ws}: that is where real data lives")
    if ws.exists() and any(ws.iterdir()) and not (ws / MARKER).exists():
        raise SystemExit(f"{ws} is not empty and not a journey demo folder; choose another")
    if ws.exists():
        import shutil
        shutil.rmtree(ws)
    ws.mkdir(parents=True)
    (ws / MARKER).write_text("synthetic journey demo - safe to delete\n", encoding="utf-8")
    return ws


def generate(workspace: Path, *, stories: int = 100, seed: int = 7, audio: bool = False,
             seconds: float = 2.0) -> dict:
    rng = random.Random(seed)
    ws = prepare(workspace)
    built = [build_story(rng, i) for i in range(stories)]
    inp = ws / "input"
    inp.mkdir()
    parts_of = write_workbook(built, inp / "handoff.xlsx", rng)
    if audio:
        write_audio(built, parts_of, inp / "recordings.zip", seconds, rng)
    write_atlas(built, inp / "atlas", rng)
    calls = write_call_artifacts(built, ws / "output", rng)
    summary = {"stories": len(built), "contacts": sum(len(s.contacts) for s in built),
               "recorded_calls": calls,
               "scenarios": {k: sum(s.scenario == k for s in built) for k, _ in SCENARIOS}}
    (ws / "demo_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1),
                                          encoding="utf-8")
    return summary


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--stories", type=int, default=100)
    ap.add_argument("--workspace", type=Path, default=DEFAULT_WORKSPACE)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--audio", action="store_true", help="also write synthetic .nmf recordings")
    ap.add_argument("--seconds", type=float, default=2.0, help="length of each synthetic part")
    ap.add_argument("--no-report", action="store_true")
    args = ap.parse_args(argv)
    summary = generate(args.workspace, stories=args.stories, seed=args.seed, audio=args.audio,
                       seconds=args.seconds)
    print(json.dumps(summary, ensure_ascii=False))
    if args.no_report:
        return 0
    ws = args.workspace.resolve()
    env = dict(os.environ, CALLQA_PATHS__INPUT_DIR=str(ws / "input"),
               CALLQA_PATHS__OUTPUT_DIR=str(ws / "output"),
               CALLQA_PATHS__STATE_DB=str(ws / "state.db"))
    cmd = [sys.executable, "-m", "callqa", "journey", "import", "--xlsx",
           str(ws / "input" / "handoff.xlsx"), "--atlas", str(ws / "input" / "atlas")]
    if args.audio:
        cmd += ["--audio", str(ws / "input" / "recordings.zip")]
    steps = [cmd, [sys.executable, "-m", "callqa", "journey", "report"]]
    for step in steps:
        proc = subprocess.run(step, env=env, cwd=str(ROOT))
        if proc.returncode not in (0, 1):
            return proc.returncode
    print(f"journey report: {ws / 'output' / 'reports' / 'journey.html'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
