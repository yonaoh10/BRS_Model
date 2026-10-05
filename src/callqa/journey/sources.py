"""Table 0.1 of every report: the sources this batch carries, with counts,
and what each missing one takes away.

The bank's unified report (ATL_R07) opens with the sources and their
counts; a reader must see at once which of the four angles - the contact
list, the call-centre table, Atlas, the CRM - the figures stand on. A
layer that is absent is said so, with the chapters it affects, instead of
leaving the reader to infer it from empty cells.
"""

from __future__ import annotations

from pydantic import BaseModel

from callqa.journey.models import JourneyDataset

PRESENT_HE = "קיים"
ABSENT_HE = "חסר"
PARTIAL_HE = "חלקי"


class SourceRow(BaseModel):
    key: str
    label_he: str                  # the layer, as the bank names it
    status: str                    # present | absent | partial
    status_he: str
    count_he: str = ""             # what was read, as "n ..." text
    effect_he: str = ""            # what the absence (or partiality) takes away
    source_he: str = ""            # where it came from, in the bank's words


def _n(n: int, word: str) -> str:
    return f"{n:,} {word}"


def sources_table(dataset: JourneyDataset, layers: dict[str, bool]) -> list[SourceRow]:
    c = dataset.report.counts
    man = dataset.manifest
    t0 = man is not None and man.tier == "T0"
    rows: list[SourceRow] = []

    # the contact list (the vendor's workbook or the bank's own export)
    if layers.get("contact_list", True):
        rows.append(SourceRow(
            key="contact_list", label_he="רשימת פניות", status="present", status_he=PRESENT_HE,
            count_he=f"{_n(c.interactions, 'פניות')} ב-{_n(c.stories, 'חשבונות')}",
            source_he=("ייצוא במבנה החוזה (גרסה 2)" if dataset.contract_version >= 2
                       else "חוברת ההעברה" if "workbook" in dataset.source
                       else "ייצוא במבנה החוזה")))
    else:
        rows.append(SourceRow(
            key="contact_list", label_he="רשימת פניות", status="absent", status_he=ABSENT_HE,
            count_he=f"{_n(c.stories, 'שיחות')} נגזרו משמות ההקלטות",
            effect_he=("אין מסעות ואין חזרות: כל שיחה היא סיפור של פנייה אחת; זמני השיחות "
                       "משוערים משעת הקובץ; אין כיוון ואין נטישה"),
            source_he="הקלטות בלבד (מדרגה T0)"))

    # recordings
    if c.recorded_calls:
        status, he = ("present", PRESENT_HE) if not c.unrecorded_calls else ("partial", PARTIAL_HE)
        rows.append(SourceRow(
            key="audio", label_he="הקלטות", status=status, status_he=he,
            count_he=f"{_n(c.recorded_calls, 'שיחות מוקלטות')} מ-{_n(c.audio_files_mapped, 'קבצים')}"
                     + (f"; {_n(c.unrecorded_calls, 'שיחות בלי הקלטה')}" if c.unrecorded_calls else ""),
            effect_he=("לשיחות בלי הקלטה אין תוכן: הסיבה לחזרה נקבעת רק מהעובדות"
                       if c.unrecorded_calls else ""),
            source_he="מערכת ההקלטה (NICE)"))
    else:
        rows.append(SourceRow(
            key="audio", label_he="הקלטות", status="absent", status_he=ABSENT_HE,
            count_he=_n(c.calls, "שיחות בלי הקלטה"),
            effect_he="אין תוכן לשיחות: אין סיווג סיבה, אין הבטחות, אין 'סיפר מחדש'"))

    # the call-centre table
    if layers.get("centre"):
        known = sum(1 for i in dataset.interactions if i.channel == "call" and i.answer != "unknown")
        rows.append(SourceRow(
            key="centre", label_he="טבלת המוקד", status="present", status_he=PRESENT_HE,
            count_he=f"{_n(known, 'שיחות')} עם כיוון ומענה ידועים",
            source_he="T4418_CRM_CALLS (דרך הייצוא)"))
    elif not t0:
        rows.append(SourceRow(
            key="centre", label_he="טבלת המוקד", status="absent", status_he=ABSENT_HE,
            effect_he="כיוון השיחה ונטישה לא ידועים לשיחות בלי הקלטה; שיעור הנטישות לא מחושב",
            source_he="T4418_CRM_CALLS"))

    # written correspondence
    if c.correspondences or c.chats:
        rows.append(SourceRow(
            key="messages", label_he="התכתבויות", status="present", status_he=PRESENT_HE,
            count_he=f"{_n(c.correspondences, 'התכתבויות')} עם {_n(c.messages, 'הודעות')}"
                     + (f"; {_n(c.chats, 'שיחות צ׳אט/וואטסאפ')}" if c.chats else ""),
            source_he="תיבת הדואר המאובטחת (T4453)"))

    # Atlas
    if layers.get("atlas"):
        cov = dataset.atlas_rules.coverage_from
        full = sum(1 for s in dataset.stories if s.atlas_coverage == "full")
        rows.append(SourceRow(
            key="atlas", label_he="אטלס", status="present" if full == len(dataset.stories)
            else "partial", status_he=PRESENT_HE if full == len(dataset.stories) else PARTIAL_HE,
            count_he=f"{_n(c.atlas_sessions, 'סשנים')}, {_n(c.atlas_ops, 'פעולות')}; "
                     f"{full:,} מתוך {len(dataset.stories):,} סיפורים בכיסוי מלא"
                     + (f" (הלוג מתחיל ב-{cov:%d.%m.%Y})" if cov else ""),
            effect_he=("סיפורים שהתחילו לפני היום הראשון בלוג אינם במדדי האטלס"
                       if full < len(dataset.stories) else ""),
            source_he="STG_ATLAS_LOG_357 (ATLR_INT / SESS / ROWS)"))
    else:
        rows.append(SourceRow(
            key="atlas", label_he="אטלס", status="absent", status_he=ABSENT_HE,
            effect_he="אין סשנים ובנקאים: פרק המאמץ, ההעברות והביצוע לא מחושב; הבטחות נבחנות "
                      "רק מול פניות",
            source_he="STG_ATLAS_LOG_357"))

    # CRM (the layer this version does not read yet)
    rows.append(SourceRow(
        key="crm", label_he="CRM (תהליכים, סיכומים, חזרות ללקוח)", status="absent",
        status_he=ABSENT_HE,
        effect_he="מצב התהליך בזמן הפנייה לא ידוע; חזרות שנרשמו ב-CRM אינן נספרות",
        source_he="T4459 / T4401 / T4444 — שכבה שגרסה זו עדיין לא קוראת"))

    # units and bankers
    if layers.get("units"):
        rows.append(SourceRow(
            key="units", label_he="טבלת יחידות", status="present", status_he=PRESENT_HE,
            count_he=_n(c.units, "יחידות"), source_he="units.csv (T1604 / T1017)"))
    else:
        rows.append(SourceRow(
            key="units", label_he="טבלת יחידות", status="partial", status_he=PARTIAL_HE,
            count_he="רשימת היחידות הניתנת לעריכה בלבד",
            effect_he="יחידה שאינה ברשימה נחשבת סניף"))

    # content (the reading of calls and messages)
    if layers.get("content"):
        rows.append(SourceRow(key="content", label_he="קריאת התוכן", status="present",
                              status_he=PRESENT_HE, source_he="שלב התוכן של הכלי"))
    else:
        rows.append(SourceRow(
            key="content", label_he="קריאת התוכן", status="absent", status_he=ABSENT_HE,
            effect_he="סיווג הסיבות, ההבטחות ו'סיפר מחדש' ממתינים לשלב התוכן"))
    return rows
