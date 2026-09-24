#!/usr/bin/env python3
"""Generate a realistic synthetic batch of processed calls - no pipeline run.

    python scripts/generate_batch_demo.py                 # 1000 calls + the management report
    python scripts/generate_batch_demo.py --calls 300 --no-report
    python scripts/generate_batch_demo.py --open          # and open the report in the browser

WHY THIS EXISTS. The management report (`callqa executive-report`) is built for
batches of hundreds to thousands of calls: confidence intervals, trends over
weeks, banker comparisons, behavioural drivers. None of that can be shown,
tested or tuned on the six sample calls, and real customer calls may not leave
the bank to do it. This script writes what the pipeline WOULD have written for
such a batch - the stage artifacts under output/ - in seconds, without a model.

The batch has STRUCTURE, so every chart and finding has something true to say:
bankers differ in latent skill (three clear stars, three clear strugglers, a few
new joiners with only a handful of calls), investments and mortgage calls are
harder on disclosure and clarity, long calls lose clarity and closure, empathy
and listening visibly improve after a training in week 7, and the objective
features move with the scores they are supposed to explain. Gate failures are
concentrated in some bankers and in investments, as they would be in life.

The text is built the way the real pipeline's text is: every transcript turn
is already redacted (identifiers appear only as mask tokens, never as values),
and every evidence quote is copied verbatim from a turn of that call, so it
passes the same verification a judge's quote must pass. Nothing here is a real
person, a real identifier or a real bank.

It works in its own folder (default data/demo-batch/), never in data/input or
data/output: a synthetic batch mixed into real results would be averaged into
a real banker's report. A folder it did not create is never cleared.

Deterministic by --seed: the same arguments produce the same batch.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import numbers
import os
import platform
import re
import shlex
import shutil
import subprocess
import sys
import time
import webbrowser
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import TypeVar

import numpy as np
from pydantic import BaseModel

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from callqa import __version__ as CALLQA_VERSION  # noqa: E402
from callqa.judge.prompts import PROMPT_VERSION, mmss  # noqa: E402
from callqa.models import (  # noqa: E402
    CallMeta,
    CallResult,
    DimensionScore,
    Evidence,
    Features,
    RedactedTranscript,
    RedactedTurn,
    ScoreCard,
    SpeechRateWPM,
)
from callqa.ops.models import CallRunSummary, RunManifest  # noqa: E402
from callqa.pipeline import STAGES  # noqa: E402
from callqa.portable import configure_stdio  # noqa: E402
from callqa.redaction import replacement_token  # noqa: E402
from callqa.rubric import Rubric, load_rubric, weighted_total  # noqa: E402

T = TypeVar("T")

DEFAULT_WORKSPACE = REPO_ROOT / "data" / "demo-batch"
# Written into every workspace this script creates. A folder without it is
# somebody else's, and is never cleared.
MARKER = ".callqa-demo-batch"
JUDGE_ENGINE = "synthetic-demo"

END_DATE = date(2026, 9, 17)                 # a Thursday: the last business day in the batch
# The CLI's bounds, enforced by generate_batch() as well: a bad value must be
# refused before the previous batch is cleared, not after.
MAX_CALLS = 20000
MAX_BANKERS = 500
MAX_WEEKS = 52
# Bank holidays that fall on a Sunday-Thursday inside the longest window
# (--weeks 52: 21.09.2025-17.09.2026). A call centre does not take calls on
# Yom Kippur, and a calendar that does looks machine-made. 5786: Rosh Hashana
# (both days), Yom Kippur, Sukkot, Shemini Atzeret, the first and seventh days
# of Pesach and Independence Day; 5787: the second day of Rosh Hashana.
HOLIDAYS = frozenset({
    date(2025, 9, 23), date(2025, 9, 24), date(2025, 10, 2), date(2025, 10, 7),
    date(2025, 10, 14), date(2026, 4, 2), date(2026, 4, 8), date(2026, 4, 22),
    date(2026, 9, 13),
})
TYPE_SHARES = (("service", 0.45), ("loans", 0.20), ("cards", 0.15),
               ("mortgage", 0.12), ("investments", 0.08))
TYPE_HE = {"service": "שירות", "loans": "הלוואות", "cards": "כרטיסי אשראי",
           "mortgage": "משכנתאות", "investments": "השקעות"}
# Duration multipliers: advice calls run long, a blocked card is quick.
TYPE_DURATION = {"service": 0.85, "loans": 1.1, "cards": 0.8, "mortgage": 1.45,
                 "investments": 1.4}
MEDIAN_DURATION_SEC = 330.0
WEEKDAY_WEIGHT = (1.2, 1.05, 1.0, 1.0, 0.9)  # Sunday..Thursday: Sunday is the busiest

KNOWN_DIMENSIONS = ("identification", "compliance", "empathy", "listening", "clarity",
                    "resolution", "suitability", "closure")
NON_GATE = ("empathy", "listening", "clarity", "resolution", "suitability", "closure")

# The latent score model, in 1..5 score units. A call's expected score on a
# dimension is BASE + LOAD * banker skill + banker's own offset + call-type
# effect + a shared per-call term (a hard customer drags everything down).
BASE = {"empathy": 3.15, "listening": 3.1, "clarity": 3.65, "resolution": 3.7,
        "suitability": 3.4, "closure": 3.55}
LOAD = {"empathy": 0.75, "listening": 0.7, "clarity": 0.6, "resolution": 0.65,
        "suitability": 0.6, "closure": 0.5}
TYPE_EFFECT: dict[str, dict[str, float]] = {
    "service": {"resolution": 0.2, "suitability": 0.1},
    "loans": {"suitability": -0.1},
    "cards": {"resolution": 0.3, "clarity": 0.1},
    "mortgage": {"clarity": -0.35, "suitability": -0.1, "resolution": -0.15},
    "investments": {"clarity": -0.45, "suitability": -0.25},
}
TRAINING_EFFECT = {"empathy": 0.6, "listening": 0.5}
DRIFT = 0.12  # the mild improvement of everything else across the whole period
# Gate failures are drawn as events, not as the tail of a normal: in life they
# are a procedure skipped, concentrated in some people and some products.
GATE_BASE_LOGIT = {"identification": -3.7, "compliance": -4.0}
GATE_TYPE_LOGIT = {
    "identification": {"service": 0.2, "loans": 0.0, "cards": 0.1, "mortgage": -0.2,
                       "investments": 0.0},
    "compliance": {"service": -0.6, "loans": 0.2, "cards": -0.2, "mortgage": 0.9,
                   "investments": 2.0},
}
GATE_PASS_TYPE = {"service": 0.0, "loans": -0.1, "cards": 0.0, "mortgage": -0.35,
                  "investments": -0.45}

HELD_SHARE = 0.03
FAILED_SHARE = 0.01
MONO_SHARE = 0.10
MIN_ROLE_CONFIDENCE = 0.34

MASKED_SLOTS = {"ID": "ISRAELI_ID", "DOB": "DOB", "PHONE": "PHONE", "CARD4": "CREDIT_CARD",
                "NAME": "PERSON", "BNAME": "PERSON"}


class WorkspaceError(RuntimeError):
    """The workspace cannot be used safely; nothing was written."""


# -- rendering ---------------------------------------------------------------
#
# Templates carry three kinds of markup, resolved per utterance:
#   $slot          a call-specific value; $UPPER slots become mask tokens
#   [male|female]  the SPEAKER's own grammatical gender
#   {male|female}  the ADDRESSEE's (in judge prose: the customer's)
# and «...» marks the span the judge quotes as evidence for that beat.
# Hebrew inflects verbs and adjectives for gender in the first AND second
# person, so a call centre transcript without it reads as machine-made.

_SLOT_RE = re.compile(r"\$([A-Za-z_][A-Za-z0-9_]*)")
_OWN_RE = re.compile(r"\[([^\[\]|]*)\|([^\[\]|]*)\]")
_OTHER_RE = re.compile(r"\{([^{}|]*)\|([^{}|]*)\}")
_QUOTE_RE = re.compile("«(.*?)»")


def _render(template: str, slots: dict[str, str], own_female: bool, other_female: bool,
            counts: dict[str, int] | None = None) -> tuple[str, str | None]:
    """(text, evidence quote or None). Mask tokens are counted into `counts`."""

    def slot(match: re.Match[str]) -> str:
        key = match.group(1)
        entity = MASKED_SLOTS.get(key)
        if entity is not None:
            if counts is not None:
                counts[entity] = counts.get(entity, 0) + 1
            return replacement_token(entity)
        return slots[key]

    text = _SLOT_RE.sub(slot, template)
    text = _OWN_RE.sub(lambda m: m.group(2 if own_female else 1), text)
    text = _OTHER_RE.sub(lambda m: m.group(2 if other_female else 1), text)
    # Sources use the Hebrew gershayim for readability; transcripts carry the
    # ASCII '"' an ASR engine and a keyboard produce (ש"ח, ת"ז).
    text = text.replace("״", '"')
    marked = _QUOTE_RE.search(text)
    quote = marked.group(1).strip() if marked else None
    return text.replace("«", "").replace("»", ""), quote


def _pick(rng: np.random.Generator, options: str | Sequence[str]) -> str:
    """One phrase: a bare string is itself, a sequence is a choice of variants."""
    if isinstance(options, str):
        return options
    return _choose(rng, options)


def _choose(rng: np.random.Generator, options: Sequence[T]) -> T:
    return options[int(rng.integers(len(options)))]


def _money(value: float) -> str:
    return f"{int(round(value)):,}"


def _annuity(principal: float, annual_rate_pct: float, months: int) -> float:
    r = annual_rate_pct / 100.0 / 12.0
    return principal * r / (1.0 - (1.0 + r) ** -months)


# -- the phrase library ------------------------------------------------------

@dataclass(frozen=True)
class Scenario:
    """One reason a customer calls, with everything each quality level says about it."""

    key: str
    call_type: str
    topic: str                                   # "שיחת <סוג> בנושא <topic>"
    disclosure: str                              # what a full disclosure covers
    values: Callable[[np.random.Generator], dict[str, str]]
    request: tuple[str, ...]
    empathy: str                                 # levels 4-5: reflects THIS situation
    summary: str                                 # levels 4-5 listening: the paraphrase
    details1: tuple[str, ...]
    details2: tuple[str, ...]
    clarity: dict[str, str]                      # low / mid / high
    compliance: dict[str, str]                   # low (omission) / mid (partial) / high
    resolution: dict[str, str]                   # low (referral) / mid (open) / high
    next_step: str
    mislead: tuple[str, ...] = ()                # compliance 1; the type's default otherwise
    suitability: dict[int, str] = field(default_factory=dict)   # overrides levels 3-5
    probe: str | None = None                     # what a good needs-probe covers


def _v_service_fee(rng: np.random.Generator) -> dict[str, str]:
    fee = int(_pick(rng, ("15", "18", "22", "25", "32")))
    return {"fee": str(fee), "fee2": str(2 * fee), "track": _pick(rng, ("9.90", "11.90", "14.90"))}


def _v_service_transfer(rng: np.random.Generator) -> dict[str, str]:
    return {"amount": _pick(rng, ("1,850", "3,400", "4,750", "6,200")),
            "days": _pick(rng, ("יומיים", "שלושה ימים"))}


def _v_service_overdraft(rng: np.random.Generator) -> dict[str, str]:
    limit = int(_pick(rng, ("5000", "8000", "10000", "12000")))
    return {"limit": _money(limit), "limit2": _money(limit + 3000),
            "over": _pick(rng, ("900", "1,200", "1,750", "2,300"))}


def _v_service_standing(rng: np.random.Generator) -> dict[str, str]:
    amount = int(_pick(rng, ("149", "189", "229", "259")))
    return {"amount": str(amount), "amount2": str(2 * amount)}


def _v_loans_car(rng: np.random.Generator) -> dict[str, str]:
    amount = int(_pick(rng, ("60000", "75000", "90000", "110000")))
    months = int(_pick(rng, ("48", "60", "72")))
    rate = float(_pick(rng, ("5.4", "5.9", "6.4", "6.9")))
    payment = _annuity(amount, rate, months)
    return {"amount": _money(amount), "months": str(months), "rate": f"{rate:.1f}%",
            "payment": _money(payment), "total": _money(round(payment) * months),
            "car_price": str(amount // 1000 + int(_pick(rng, ("35", "45", "60")))),
            "salary": _pick(rng, ("11", "13", "14", "16", "18")),
            "max_pay": _money(math.ceil((payment + 250) / 100) * 100)}


def _v_loans_hardship(rng: np.random.Generator) -> dict[str, str]:
    payment = int(_pick(rng, ("2300", "2800", "3400")))
    extra = int(_pick(rng, ("12", "18", "24")))
    rate, left = 6.5, 36
    r = rate / 100 / 12
    balance = payment * (1 - (1 + r) ** -left) / r
    new_payment = _annuity(balance, rate, left + extra)
    extra_interest = round(new_payment) * (left + extra) - payment * left
    return {"payment": _money(payment), "months_extra": str(extra),
            "new_payment": _money(new_payment),
            "extra_interest": _money(round(extra_interest, -2))}


def _v_loans_prepay(rng: np.random.Generator) -> dict[str, str]:
    balance = int(_pick(rng, ("35000", "42000", "48000", "62000")))
    fee = int(_pick(rng, ("180", "240", "310", "365")))
    return {"balance": _money(balance), "fee_early": str(fee), "total_early": _money(balance + fee)}


def _v_cards_theft(rng: np.random.Generator) -> dict[str, str]:
    return {"amount_tx": _pick(rng, ("320", "890", "1,450", "2,180"))}


def _v_cards_limit(rng: np.random.Generator) -> dict[str, str]:
    old = int(_pick(rng, ("8000", "12000", "15000")))
    add = int(_pick(rng, ("4000", "5000", "7000")))
    return {"limit_old": _money(old), "limit_new": _money(old + add), "add": _money(add)}


def _v_cards_fee(rng: np.random.Generator) -> dict[str, str]:
    return {"card_fee": _pick(rng, ("15", "19", "25")),
            "spend": _pick(rng, ("1,800", "2,100", "2,400"))}


def _v_mortgage_refi(rng: np.random.Generator) -> dict[str, str]:
    payment = int(_pick(rng, ("4900", "5600", "6300")))
    saving = int(_pick(rng, ("380", "450", "520", "690")))
    return {"balance": _pick(rng, ("720,000", "850,000", "960,000", "1,050,000")),
            "payment_now": _money(payment), "saving": str(saving),
            "payment_new": _money(payment - saving),
            "exit_fee": _pick(rng, ("3,800", "4,200", "5,600")),
            "income": _pick(rng, ("24", "28", "32"))}


def _v_mortgage_new(rng: np.random.Generator) -> dict[str, str]:
    return {"price": _pick(rng, ("1,850,000", "2,200,000", "2,600,000")),
            "equity_pct": _pick(rng, ("25", "30", "35")),
            "income": _pick(rng, ("22", "26", "31"))}


def _v_mortgage_deferral(rng: np.random.Generator) -> dict[str, str]:
    return {"payment_now": _pick(rng, ("4,200", "5,100", "5,900")),
            "delta": _pick(rng, ("160", "210", "280")),
            "extra": _pick(rng, ("5,200", "7,400", "9,800"))}


def _v_investments_deposit(rng: np.random.Generator) -> dict[str, str]:
    return {"amount": _pick(rng, ("150,000", "240,000", "400,000")),
            "dep_rate": _pick(rng, ("3.6%", "3.9%", "4.1%")),
            "mgmt": _pick(rng, ("0.6%", "0.7%", "0.8%"))}


def _v_investments_drop(rng: np.random.Generator) -> dict[str, str]:
    return {"amount": _pick(rng, ("180,000", "320,000", "540,000")),
            "drop": _pick(rng, ("8", "12", "15")),
            "years": _pick(rng, ("חמש עשרה", "עשרים"))}


SCENARIOS: tuple[Scenario, ...] = (
    # -- service -------------------------------------------------------------
    Scenario(
        key="service_fee", call_type="service", topic="עמלה לא מוכרת בדף החשבון",
        disclosure="עלות המסלול החודשית והחיוב על פעולות נוספות",
        values=_v_service_fee,
        request=(
            "ראיתי בדף החשבון עמלה של $fee ש״ח ואני בכלל לא [מבין|מבינה] על מה היא. "
            "זו כבר הפעם השנייה החודש, וזה ממש מעצבן אותי.",
            "יש לי בחשבון חיוב של $fee ש״ח שנקרא 'עמלת פעולה', ואני לא [זוכר|זוכרת] "
            "שעשיתי משהו כזה. זה כבר פעם שנייה החודש.",
        ),
        empathy="«אני [מבין|מבינה] את התסכול, במיוחד כשזה כבר חיוב שני החודש על משהו שלא "
                "ברור לך.»",
        summary="הופיעו החודש שני חיובים של $fee ש״ח שלא ברור לך על מה הם, ו{אתה רוצה|את "
                "רוצה} להבין אותם ולבדוק אם אפשר לבטל",
        details1=(
            "כן, זה מופיע כ'עמלת פעולה על ידי פקיד'. אני עושה כמעט הכול באפליקציה, אז לא "
            "ברור לי מאיפה זה.",
            "אני כמעט לא [מדבר|מדברת] עם הבנק בטלפון, אז באמת לא ברור לי על מה זה.",
        ),
        details2=(
            "רוב הפעולות שלי הן העברות ותשלומים באפליקציה, אולי עשר בחודש. פעם-פעמיים "
            "ביקשתי העברה בטלפון כי האפליקציה נתקעה.",
            "זה קרה רק החודש. בדרך כלל אני עושה הכול לבד באפליקציה.",
        ),
        clarity={
            "high": "«שני החיובים הם על העברות שבוצעו דרך נציג טלפוני ולא באפליקציה: כל "
                    "העברה כזו עולה $fee ש״ח, ובאפליקציה אותה פעולה היא ללא עלות.»",
            "mid": "«אלה עמלות על פעולות שנעשו דרך נציג, יש על זה תעריף.» באפליקציה זה בדרך "
                   "כלל זול יותר.",
            "low": "«זו עמלת פעולה לפי התעריפון, בהתאם לסוג הערוץ ולמסלול העמלות שמשויך "
                   "לחשבון.»",
        },
        compliance={
            "high": "«אם {תעבור|תעברי} למסלול הבסיסי, העלות היא $track ש״ח לחודש והיא כוללת "
                    "עד עשר פעולות בערוץ ישיר; פעולה דרך נציג תמשיך להיות מחויבת לפי "
                    "התעריפון.» אפשר לבטל את המסלול בכל עת, ללא עלות.",
            "mid": "«יש גם מסלול עמלות חודשי שיכול להתאים לך, יש בו עלות חודשית קטנה.»",
            "low": "«אני [ממליץ|ממליצה] לך לעבור למסלול הבסיסי, אני [מעביר|מעבירה] אותך "
                   "אליו עכשיו.»",
        },
        mislead=("«במסלול החדש לא {תשלם|תשלמי} יותר אף שקל עמלות, על שום פעולה, מובטח.»",),
        resolution={
            "high": "«ביטלתי את שני החיובים, סך הכול $fee2 ש״ח, והזיכוי יופיע בחשבון תוך "
                    "שלושה ימי עסקים.» {תקבל|תקבלי} על זה הודעת SMS.",
            "mid": "«פתחתי בקשה לבדיקת החיובים, יחזרו אליך מהמחלקה.»",
            "low": "«את החיוב אני לא [יכול|יכולה] לבטל, זה לפי התעריף. {תנסה|תנסי} לפנות "
                   "לסניף.»",
        },
        next_step="הזיכוי של $fee2 ש״ח יופיע בחשבון תוך שלושה ימי עסקים",
    ),
    Scenario(
        key="service_transfer", call_type="service",
        topic="העברה בנקאית שעוד לא התקבלה אצל המוטב",
        disclosure="עמלת ההעברה המיידית ומגבלת הסכום שלה",
        values=_v_service_transfer,
        request=(
            "ביצעתי העברה של $amount ש״ח לבעל הדירה לפני $days, והוא טוען שהכסף לא הגיע. "
            "אני כבר ממש [לחוץ|לחוצה], הוא מאיים לחייב אותי בריבית פיגורים.",
        ),
        empathy="«אני [שומע|שומעת] שזה ממש מלחיץ, במיוחד כשבעל הדירה לוחץ ו{אתה חושש|את "
                "חוששת} מריבית פיגורים.»",
        summary="העברת $amount ש״ח לבעל הדירה לפני $days, הכסף עוד לא הגיע אליו, ו{אתה "
                "צריך|את צריכה} לדעת איפה הוא עומד",
        details1=(
            "עשיתי את ההעברה דרך האתר, ביום ראשון בערב. יש לי גם אישור על ההעברה במייל.",
            "ההעברה יצאה ביום ראשון בלילה, ואני רואה שהסכום כבר ירד מהיתרה.",
        ),
        details2=(
            "זה תשלום שכירות קבוע, אני [מעביר|מעבירה] לו כל חודש באותו תאריך בערך.",
            "זה שכר דירה, כל חודש אותו סכום. עד עכשיו זה תמיד הגיע בזמן.",
        ),
        clarity={
            "high": "«העברה שמבוצעת בערב נקלטת רק ביום העסקים הבא, ולכן ההעברה שלך נקלטה "
                    "ביום שני; משם לוקח עד יום עסקים נוסף עד שהיא מופיעה בחשבון המוטב.» "
                    "כלומר, הכסף אמור להופיע בחשבון שלו היום, עד סוף היום.",
            "mid": "«ההעברה יצאה מאיתנו, לפעמים לוקח קצת זמן עד שזה מגיע לבנק השני.»",
            "low": "«ההעברה עוברת במערכת הסליקה הבין-בנקאית לפי מועדי ערך ומועדי קליטה, זה "
                   "תלוי בכמה פרמטרים.»",
        },
        compliance={
            "high": "«אם {תרצה|תרצי} בעתיד שהכסף יגיע באותו יום, אפשר לבצע העברה מיידית: "
                    "העמלה עליה היא 6 ש״ח לפעולה, והיא אפשרית עד 50,000 ש״ח ביום.»",
            "mid": "«יש גם אפשרות להעברה מיידית, יש עליה עמלה קטנה.»",
            "low": "«בפעם הבאה {תעשה|תעשי} העברה מיידית וזהו, ככה זה מגיע מיד.»",
        },
        mislead=("«בפעם הבאה {תעשה|תעשי} העברה מיידית, היא תמיד בחינם ובלי שום הגבלה.»",),
        resolution={
            "high": "«בדקתי מול המערכת: ההעברה התקבלה בבנק של המוטב הבוקר, והכסף יופיע "
                    "בחשבון שלו עד השעה 18:00 היום.» אני [שולח|שולחת] לך עכשיו אישור העברה "
                    "חתום למייל, כדי {שתוכל|שתוכלי} להעביר לו.",
            "mid": "«פתחתי בדיקה מול הבנק של המוטב, יחזרו אליך כשתהיה תשובה.»",
            "low": "«אין לי אפשרות לראות מה קורה בבנק השני, {תבדוק|תבדקי} את זה מולם.»",
        },
        next_step="הכסף יופיע בחשבון של בעל הדירה עד 18:00 היום, ואישור ההעברה בדרך למייל שלך",
    ),
    Scenario(
        key="service_overdraft", call_type="service", topic="חריגה ממסגרת האשראי בחשבון",
        disclosure="הריבית על המסגרת המוגדלת ומשך התוקף שלה",
        values=_v_service_overdraft,
        request=(
            "קיבלתי הודעה שחרגתי ממסגרת האשראי בחשבון. המשכורת נכנסת רק בעוד שבוע, ויש לי "
            "הוראות קבע שיורדות השבוע. אני לא [יודע|יודעת] מה לעשות.",
        ),
        empathy="«אני [מבין|מבינה] את הלחץ, במיוחד כשהמשכורת רק בעוד שבוע והוראות הקבע כבר "
                "עומדות לרדת.»",
        summary="החשבון חרג מהמסגרת, המשכורת נכנסת רק בעוד שבוע, ו{אתה רוצה|את רוצה} לוודא "
                "שהוראות הקבע של השבוע לא יחזרו",
        details1=(
            "יש לי הוראת קבע לגן של הילדה ולחברת החשמל, ושתיהן יורדות ביום רביעי.",
            "הוראות הקבע של הגן ושל הביטוח יורדות ביום רביעי, ואני לא רוצה שיחזרו.",
        ),
        details2=(
            "בדרך כלל אני לא במינוס בכלל. החודש היה לי תיקון גדול ברכב.",
            "זה חד-פעמי, בגלל טיפול שיניים יקר שהיה החודש.",
        ),
        clarity={
            "high": "«כרגע החשבון נמצא $over ש״ח מעבר למסגרת של $limit ש״ח.» כל עוד היתרה "
                    "מעבר למסגרת, הוראות קבע עלולות לחזור, ולכן חשוב לסגור את הפער לפני יום "
                    "רביעי.",
            "mid": "«יש חריגה מהמסגרת, צריך לכסות אותה כדי שלא יהיו בעיות עם החיובים.»",
            "low": "«החריגה נובעת מהפער בין יתרת העו״ש למסגרת החח״ד המאושרת, בתוספת ריבית "
                   "חריגה מצטברת.»",
        },
        compliance={
            "high": "«אני [יכול|יכולה] להגדיל את המסגרת זמנית לחודש, בריבית של פריים ועוד "
                    "6.5% בשנה, רק על הסכום שמנוצל בפועל.» אם בסוף התקופה היתרה עדיין תהיה "
                    "מעבר למסגרת המקורית, תחול עליה ריבית חריגה גבוהה יותר.",
            "mid": "«אפשר להגדיל לך את המסגרת זמנית, יש על זה ריבית.»",
            "low": "«אני [מגדיל|מגדילה] לך את המסגרת עכשיו, ככה החשבון יהיה מכוסה.»",
        },
        mislead=("«ההגדלה הזאת בלי ריבית בכלל, אין לך מה לדאוג.»",),
        resolution={
            "high": "«הגדלתי את המסגרת ל-$limit2 ש״ח עד ה-15 לחודש הבא, כך שהוראות הקבע של "
                    "יום רביעי ירדו כרגיל.» אישור יישלח אליך בהודעה בעוד כמה דקות.",
            "mid": "«הגשתי בקשה להגדלת המסגרת, היא צריכה אישור של הסניף.»",
            "low": "«הגדלת מסגרת מאשרים רק בסניף, {תתקשר|תתקשרי} אליהם מחר.»",
        },
        next_step="המסגרת המוגדלת בתוקף עד ה-15 לחודש הבא, והוראות הקבע ירדו כרגיל",
    ),
    Scenario(
        key="service_standing_order", call_type="service",
        topic="חיוב בהוראת קבע אחרי ביטול מנוי",
        disclosure="עלות ביטול ההרשאה ותנאי ההחזר",
        values=_v_service_standing,
        request=(
            "ביטלתי את המנוי לחדר הכושר לפני חודשיים, והם עדיין מחייבים אותי $amount ש״ח "
            "כל חודש דרך הוראת קבע. זה פשוט לא נגמר.",
        ),
        empathy="«אני [מבין|מבינה] כמה זה מתסכל: ביטלת את המנוי כמו שצריך, ובכל זאת הכסף "
                "ממשיך לרדת כל חודש.»",
        summary="ביטלת את המנוי לחדר הכושר לפני חודשיים, ובכל זאת ממשיכה לרדת הוראת קבע של "
                "$amount ש״ח בחודש, ו{אתה רוצה|את רוצה} לעצור אותה ולקבל את הכסף בחזרה",
        details1=(
            "יש לי אפילו אישור ביטול מהם במייל. הם פשוט ממשיכים לחייב.",
            "ביטלתי מולם בטלפון וגם שלחתי מייל, ועדיין זה יורד.",
        ),
        details2=(
            "זה כבר ירד פעמיים, בחודש שעבר ובחודש הזה.",
            "זו הוראת הקבע היחידה שיש לי מולם, ואין לי שום קשר איתם יותר.",
        ),
        clarity={
            "high": "«ברגע שמבטלים את ההרשאה לחיוב דרך הבנק, החיוב הבא כבר לא יורד, גם אם "
                    "בית העסק ממשיך לשלוח דרישה.» ביטול מול בית העסק בלבד לא מבטל את ההרשאה "
                    "בבנק, ולכן החיובים המשיכו.",
            "mid": "«אפשר לבטל את ההרשאה בבנק, ואז החיובים ייפסקו.»",
            "low": "«זה תלוי בסוג ההרשאה לחיוב ובמועד הקליטה במס״ב, צריך לבדוק את זה מול "
                   "הגורם המחייב.»",
        },
        compliance={
            "high": "«ביטול ההרשאה בבנק הוא ללא עלות; החזר על החיובים שכבר ירדו מותנה באישור "
                    "הביטול מבית העסק, ויכול לקחת עד 14 ימי עסקים.»",
            "mid": "«הביטול עצמו בלי עלות, וההחזר תלוי בבית העסק.»",
            "low": "«אני [מבטל|מבטלת] לך עכשיו את ההרשאה, וזהו.»",
        },
        mislead=("«ההחזר מובטח, הכסף יחזור אליך תוך יומיים בלי שום בעיה.»",),
        resolution={
            "high": "«ביטלתי את ההרשאה לחיוב והגשתי בקשה להחזר של שני החיובים, סך הכול "
                    "$amount2 ש״ח, מול בית העסק.» {תקבל|תקבלי} עדכון בתוך 14 ימי עסקים.",
            "mid": "«ביטלתי את ההרשאה, ואת הבקשה להחזר העברתי לבדיקה; יחזרו אליך.»",
            "low": "«את זה {תצטרך|תצטרכי} לסגור מול חדר הכושר, זה לא מול הבנק.»",
        },
        next_step="ההרשאה בוטלה, ועדכון על ההחזר של $amount2 ש״ח יגיע תוך 14 ימי עסקים",
    ),
    # -- loans ---------------------------------------------------------------
    Scenario(
        key="loans_car", call_type="loans", topic="הלוואה לרכישת רכב",
        disclosure="הריבית, העלות הכוללת ועמלת הפירעון המוקדם",
        values=_v_loans_car,
        request=(
            "אני [מתכנן|מתכננת] לקנות רכב ורציתי לבדוק הלוואה של $amount ש״ח. זו פעם ראשונה "
            "שאני [לוקח|לוקחת] הלוואה כזאת גדולה, וזה קצת מלחיץ אותי.",
        ),
        empathy="«זה לגמרי טבעי להרגיש ככה לפני הלוואה ראשונה בסכום כזה, וטוב ש{אתה בודק|את "
                "בודקת} את הכול מראש.»",
        summary="{אתה מתכנן|את מתכננת} לקנות רכב ו{צריך|צריכה} הלוואה של $amount ש״ח, וחשוב "
                "לך להבין בדיוק לאן {אתה נכנס|את נכנסת}",
        details1=("הרכב עולה בערך $car_price אלף, ואת השאר אני [משלם|משלמת] מהחסכונות.",),
        details2=(
            "המשכורת שלי בערך $salary אלף נטו, ואין לי הלוואות אחרות. החזר של עד $max_pay "
            "ש״ח בחודש נראה לי סביר.",
        ),
        clarity={
            "high": "«על הלוואה של $amount ש״ח ל-$months חודשים בריבית שנתית קבועה של $rate, "
                    "ההחזר החודשי יהיה $payment ש״ח.» בסך הכול {תחזיר|תחזירי} $total ש״ח "
                    "לאורך התקופה, וההחזר הראשון ירד ב-10 לחודש הבא.",
            "mid": "«ההחזר ייצא בערך $payment ש״ח בחודש, תלוי בתקופה שנבחר.»",
            "low": "«ההחזר מחושב בשיטת שפיצר על בסיס ריבית פריים בתוספת מרווח, בכפוף לדירוג "
                   "האשראי.»",
        },
        compliance={
            "high": "«הריבית היא $rate בשנה, קבועה לכל התקופה, ויש עמלת פתיחת תיק של 250 ש״ח; "
                    "בפירעון מוקדם עשויה לחול עמלה, ואיחור בתשלום יגרור ריבית פיגורים.» "
                    "הכול מופיע בהסכם, ו{תקבל|תקבלי} אותו לעיון לפני החתימה.",
            "mid": "«יש גם עמלת פתיחת תיק, והכול כתוב בהסכם.»",
            "low": "«אז אני [שולח|שולחת] לך את ההסכם לחתימה באפליקציה, ו{תקבל|תקבלי} את "
                   "הכסף מחר.»",
        },
        resolution={
            "high": "«אישרתי את ההלוואה: ההסכם כבר באפליקציה לחתימה, ומיד אחרי החתימה הכסף "
                    "יופקד בחשבון, עד מחר ב-12:00.»",
            "mid": "«הבקשה הועברה לחיתום, יחזרו אליך בהמשך.»",
            "low": "«את זה צריך לעשות מול הסניף, אני לא [יכול|יכולה] לאשר מכאן.»",
        },
        next_step="אחרי החתימה באפליקציה, הכסף יופקד בחשבון עד מחר ב-12:00",
    ),
    Scenario(
        key="loans_hardship", call_type="loans",
        topic="קושי בהחזר הלוואה ובקשה לפריסה מחדש",
        disclosure="תוספת הריבית הכוללת ועמלת הפירעון המוקדם",
        values=_v_loans_hardship,
        request=(
            "יש לי הלוואה, ואני כבר לא [עומד|עומדת] בהחזר החודשי. פוטרתי לפני חודש ואני פשוט "
            "לא [יודע|יודעת] מה לעשות.",
        ),
        empathy="«אני מאוד [מצטער|מצטערת] לשמוע על הפיטורים. ברור לי שזו תקופה לא פשוטה, "
                "והחזר חודשי כבד רק מוסיף לחץ.»",
        summary="{אתה לא עומד|את לא עומדת} כרגע בהחזר של $payment ש״ח בחודש בגלל הפיטורים, "
                "ו{מחפש|מחפשת} דרך להקל עליו עד {שתמצא|שתמצאי} עבודה",
        details1=(
            "ההחזר עכשיו $payment ש״ח בחודש, ונשארו בערך שלוש שנים. אני [מקבל|מקבלת] דמי "
            "אבטלה, אבל זה בקושי מכסה את השכירות.",
        ),
        details2=(
            "אני מקווה לחזור לעבוד תוך כמה חודשים, יש לי כבר ראיונות. חוץ מדמי האבטלה אין "
            "כרגע הכנסה נוספת.",
        ),
        clarity={
            "high": "«אפשר לפרוס מחדש את יתרת ההלוואה ולהאריך אותה ב-$months_extra חודשים, כך "
                    "שההחזר החודשי ירד ל-$new_payment ש״ח.» השינוי ייכנס לתוקף כבר מהחיוב "
                    "הבא.",
            "mid": "«אפשר לפרוס מחדש, ההחזר ירד, אבל התקופה תתארך.»",
            "low": "«אפשר לבצע מחזור עם שינוי לוח הסילוקין ופריסה מחדש של הקרן והריבית.»",
        },
        compliance={
            "high": "«בפריסה מחדש ההחזר יורד, אבל סך הריבית לאורך התקופה יגדל בכ-"
                    "$extra_interest ש״ח, ואין עמלה על הפריסה עצמה.» אם {תחזור|תחזרי} לעבוד, "
                    "אפשר יהיה לפרוע מוקדם, בכפוף לעמלת פירעון מוקדם לפי ההסכם.",
            "mid": "«כן, בסך הכול זה יעלה קצת יותר ריבית.»",
            "low": "«אז אני [מעדכן|מעדכנת] את זה עכשיו, ומהחודש הבא ההחזר יהיה נמוך יותר.»",
        },
        mislead=("«הפריסה לא תעלה לך אף שקל נוסף, היא רק מקטינה את ההחזר.»",),
        resolution={
            "high": "«עדכנתי את הפריסה: מהחיוב הבא, ב-10 לחודש, ההחזר יהיה $new_payment ש״ח.» "
                    "לוח הסילוקין המעודכן יישלח אליך במייל עוד היום.",
            "mid": "«העברתי את הבקשה למחלקת האשראי, הם יבדקו ויחזרו אליך.»",
            "low": "«מכאן אין לי מה לעשות, ההלוואה משולמת לפי ההסכם. {תפנה|תפני} לסניף.»",
        },
        next_step="מהחיוב הבא ההחזר יעמוד על $new_payment ש״ח, ולוח הסילוקין המעודכן יגיע "
                  "אליך במייל",
        suitability={
            5: "«כדי למצוא פתרון שבאמת יתאים: מה ההכנסה של משק הבית עכשיו, ולכמה זמן בערך "
               "{אתה צופה|את צופה} שהמצב יימשך?»",
            4: "«יש לך הערכה לכמה זמן {תצטרך|תצטרכי} את ההקלה?»",
            3: "«יש לך עוד הכנסה כרגע?»",
        },
        probe="ההכנסה הנוכחית ומשך הקושי הצפוי",
    ),
    Scenario(
        key="loans_prepay", call_type="loans", topic="פירעון מוקדם של הלוואה",
        disclosure="עמלת הפירעון המוקדם והרכב שלה",
        values=_v_loans_prepay,
        request=(
            "קיבלתי מענק מהעבודה ואני רוצה לסגור את ההלוואה שנשארה לי. רציתי לדעת כמה זה "
            "יעלה לי.",
        ),
        empathy="«קודם כול, מזל טוב על המענק! ואני [מבין|מבינה] למה חשוב לך לסגור את ההלוואה "
                "ולהוריד את ההחזר החודשי מהראש.»",
        summary="{אתה רוצה|את רוצה} לפרוע את כל יתרת ההלוואה מהמענק, ולפני כן לדעת בדיוק כמה "
                "זה יעלה",
        details1=("נשאר לי בערך $balance ש״ח, ככה רשום באפליקציה.",),
        details2=(
            "הכסף כבר בעו״ש, ויש לי עוד חיסכון בצד, אז אני לא [צריך|צריכה] אותו לשום דבר "
            "אחר.",
        ),
        clarity={
            "high": "«יתרת ההלוואה לסילוק היום היא $balance ש״ח, ובתוספת עמלת פירעון מוקדם של "
                    "$fee_early ש״ח, הסכום הכולל הוא $total_early ש״ח.» אם {תסגור|תסגרי} "
                    "היום, לא יהיו חיובים נוספים בחודש הבא.",
            "mid": "«יש יתרה של בערך $balance ש״ח, ועוד עמלה קטנה על פירעון מוקדם.»",
            "low": "«הסכום לסילוק תלוי במועד הפירעון, בהפרשי הריבית ובעמלת ההיוון, הוא משתנה "
                   "מיום ליום.»",
        },
        compliance={
            "high": "«על פירעון מוקדם חלה עמלה של $fee_early ש״ח, שמורכבת מעמלה תפעולית "
                    "ומהפרשי ריבית; אם {תיתן|תיתני} הודעה מוקדמת של עשרה ימים, חלק מהעמלה "
                    "יורד.»",
            "mid": "«יש עמלת פירעון מוקדם, אבל היא לא גבוהה.»",
            "low": "«אין בעיה, אני [סוגר|סוגרת] לך את זה עכשיו מהעו״ש.»",
        },
        mislead=("«אין שום עמלה על פירעון מוקדם, זה בחינם לגמרי.»",),
        resolution={
            "high": "«ביצעתי את הפירעון: ההלוואה סגורה, $total_early ש״ח ירדו מהחשבון היום, "
                    "ואישור סגירה יישלח אליך בדואר ובמייל תוך שבוע.»",
            "mid": "«רשמתי בקשה לפירעון, היא תטופל בימים הקרובים.»",
            "low": "«פירעון מוקדם עושים רק בסניף, {תקבע|תקבעי} שם תור.»",
        },
        next_step="ההלוואה סגורה, ואישור הסגירה יגיע אליך תוך שבוע",
        suitability={
            5: "«לפני שסוגרים: הכסף מהמענק לא נדרש לך לשום דבר אחר בקרוב? ותישאר לך כרית "
               "ביטחון אחרי הפירעון?»",
            4: "«יש לך צורך אחר בכסף הזה בחודשים הקרובים?»",
            3: "«הכסף כבר נמצא בעו״ש?»",
        },
        probe="צורכי הנזילות וכרית הביטחון לאחר הפירעון",
    ),
    # -- cards ---------------------------------------------------------------
    Scenario(
        key="cards_theft", call_type="cards", topic="גניבת כרטיס אשראי ועסקה לא מוכרת",
        disclosure="היקף האחריות על עסקאות לא מורשות ומשך הבירור",
        values=_v_cards_theft,
        request=(
            "נראה לי שגנבו לי את הארנק עם כרטיס האשראי. אני רואה באפליקציה עסקה של "
            "$amount_tx ש״ח שלא אני עשיתי, ואני ממש בלחץ.",
        ),
        empathy="«אני [מבין|מבינה] כמה זה מלחיץ: גם הארנק נגנב וגם מופיעה עסקה ש{אתה לא "
                "מכיר|את לא מכירה}.»",
        summary="הארנק נגנב, מופיעה עסקה של $amount_tx ש״ח שלא ביצעת, ו{אתה צריך|את צריכה} "
                "לחסום את הכרטיס ולבטל את העסקה",
        details1=("זה קרה כנראה אתמול בערב, באוטובוס. ראיתי את העסקה רק הבוקר.",),
        details2=(
            "אני [משתמש|משתמשת] בכרטיס בעיקר בסופר ובדלק, ולא קניתי כלום באינטרנט השבוע.",
        ),
        clarity={
            "high": "«אני [חוסם|חוסמת] את הכרטיס עכשיו, ומרגע החסימה אי אפשר לבצע בו אף "
                    "עסקה.» את העסקה של $amount_tx ש״ח אני [פותח|פותחת] לבירור, וכרטיס חדש "
                    "יגיע אליך בדואר רשום תוך חמישה ימי עסקים.",
            "mid": "«אני [חוסם|חוסמת] את הכרטיס, ואת העסקה צריך לבדוק.»",
            "low": "«צריך להגיש הכחשת עסקה במסגרת הליך צ'ארג'בק מול חברת הסליקה, ואז זה נבדק "
                   "לפי הנהלים.»",
        },
        compliance={
            "high": "«מרגע ההודעה על הגניבה לא {תחויב|תחויבי} על עסקאות שיבוצעו בכרטיס, ועל "
                    "העסקה שבוצעה לפני ההודעה ייפתח בירור שיכול להימשך עד 30 יום.» הנפקת "
                    "כרטיס חלופי במקרה של גניבה היא ללא עלות.",
            "mid": "«הכרטיס החדש בלי עלות, ואת העסקה בודקים.»",
            "low": "«טוב, אני [מזמין|מזמינה] לך כרטיס חדש.»",
        },
        mislead=("«{אל תדאג|אל תדאגי}, הכסף בטוח יחזור אליך תוך יומיים, בלי שום בירור.»",),
        resolution={
            "high": "«הכרטיס חסום, הבירור על העסקה נפתח, וכרטיס חדש יוצא אליך היום בדואר "
                    "רשום.» {תקבל|תקבלי} הודעה ברגע שהבירור יסתיים.",
            "mid": "«הכרטיס חסום, ולגבי העסקה פתחתי פנייה; יחזרו אליך.»",
            "low": "«אני לא [יכול|יכולה] לטפל בזה מכאן, {תתקשר|תתקשרי} ישירות לחברת "
                   "האשראי.»",
        },
        next_step="הכרטיס החדש יגיע בדואר רשום תוך חמישה ימי עסקים, ותשובה על הבירור תוך 30 יום",
    ),
    Scenario(
        key="cards_limit", call_type="cards", topic="הגדלת מסגרת בכרטיס לקראת נסיעה לחו״ל",
        disclosure="עמלת ההמרה במטבע חוץ והריבית על עסקאות בתשלומים",
        values=_v_cards_limit,
        request=(
            "אני [טס|טסה] לחו״ל בעוד שבועיים ורציתי להגדיל את המסגרת בכרטיס לתקופת הנסיעה. "
            "בפעם הקודמת נתקעתי שם עם כרטיס חסום, וזה היה סיוט.",
        ),
        empathy="«אני [מבין|מבינה] לגמרי: להיתקע בחו״ל עם כרטיס חסום זו חוויה לא נעימה, וחשוב "
                "לך שזה לא יחזור.»",
        summary="{אתה טס|את טסה} לחו״ל בעוד שבועיים ורוצה מסגרת גבוהה יותר בכרטיס, כדי לא "
                "להיתקע שוב כמו בפעם הקודמת",
        details1=(
            "המסגרת עכשיו $limit_old ש״ח, ואני [חושש|חוששת] שזה לא יספיק למלון ולהשכרת רכב.",
        ),
        details2=(
            "אני [משתמש|משתמשת] בכרטיס בעיקר לקניות שוטפות. לנסיעה אני [חושב|חושבת] שעוד "
            "$add ש״ח יספיקו.",
        ),
        clarity={
            "high": "«אני [יכול|יכולה] להגדיל את המסגרת מ-$limit_old ל-$limit_new ש״ח באופן "
                    "זמני, לחודש, ובסוף התקופה היא תחזור אוטומטית לגובה הנוכחי.» ההגדלה "
                    "תיכנס לתוקף כבר מחר בבוקר.",
            "mid": "«אפשר להגדיל זמנית, ואחר כך זה חוזר למסגרת הרגילה.»",
            "low": "«ההגדלה תלויה בדירוג האשראי ובניצול המסגרת הכוללת, צריך לראות מה המערכת "
                   "מאשרת.»",
        },
        compliance={
            "high": "«עסקאות במטבע חוץ מחויבות בעמלת המרה של 2.5% מסכום העסקה, והגדלת המסגרת "
                    "עצמה היא ללא עלות; עסקה {שתפרוס|שתפרסי} לתשלומים תישא ריבית לפי תנאי "
                    "הכרטיס.»",
            "mid": "«יש עמלה על עסקאות בחו״ל, כמו תמיד.»",
            "low": "«סגרנו, ההגדלה תיכנס לתוקף מחר.»",
        },
        resolution={
            "high": "«ביצעתי: המסגרת תעמוד על $limit_new ש״ח מחר בבוקר ועד סוף החודש הבא, "
                    "ואישור יישלח אליך ב-SMS.»",
            "mid": "«שלחתי בקשה להגדלה, היא אמורה להיות מאושרת בימים הקרובים.»",
            "low": "«אין לי הרשאה להגדיל מסגרת, {תפנה|תפני} לחברת האשראי.»",
        },
        next_step="המסגרת המוגדלת בתוקף מחר בבוקר ועד סוף החודש הבא",
    ),
    Scenario(
        key="cards_fee", call_type="cards", topic="חיוב דמי כרטיס שלא היה צפוי",
        disclosure="דמי הכרטיס ותנאי הפטור מהם",
        values=_v_cards_fee,
        request=(
            "חייבו אותי בדמי כרטיס של $card_fee ש״ח, ובהצטרפות אמרו לי שהכרטיס בחינם. אני "
            "ממש [מאוכזב|מאוכזבת].",
        ),
        empathy="«אני [מבין|מבינה] את האכזבה: אמרו לך שהכרטיס ללא עלות, ופתאום מופיע חיוב.»",
        summary="חויבת בדמי כרטיס של $card_fee ש״ח, למרות שבהצטרפות נאמר לך שהכרטיס ללא "
                "עלות",
        details1=("הצטרפתי לפני שנה בערך, בסניף, ואמרו לי במפורש שאין דמי כרטיס.",),
        details2=("אני [משתמש|משתמשת] בכרטיס כל חודש, בסביבות $spend ש״ח בחודש.",),
        clarity={
            "high": "«הפטור מדמי כרטיס ניתן לשנה הראשונה בלבד, והוא הסתיים בחודש שעבר, ולכן "
                    "החודש נגבו $card_fee ש״ח.» מהמחזור הבא, בכל חודש שבו ההוצאות בכרטיס "
                    "יעברו 2,500 ש״ח, דמי הכרטיס לא ייגבו.",
            "mid": "«הפטור היה לתקופה מוגבלת, ועכשיו התחיל החיוב.»",
            "low": "«דמי הכרטיס נקבעים לפי מסלול ההטבות ותנאי המועדון, זה מופיע בתעריפון.»",
        },
        compliance={
            "high": "«דמי הכרטיס הם $card_fee ש״ח בחודש, והם לא נגבים בחודש שבו ההוצאות "
                    "עוברות 2,500 ש״ח; אפשר גם לעבור לכרטיס בלי דמי כרטיס בכלל, אבל בלי "
                    "צבירת נקודות.»",
            "mid": "«יש דמי כרטיס, אבל יש דרך לקבל פטור.»",
            "low": "«אני [מעביר|מעבירה] אותך למסלול אחר ונגמר הסיפור.»",
        },
        resolution={
            "high": "«זיכיתי אותך על החיוב של החודש, $card_fee ש״ח, ועדכנתי את המסלול כך "
                    "שמהחודש הבא הפטור ייבדק אוטומטית לפי היקף ההוצאות.»",
            "mid": "«אני [מעביר|מעבירה] בקשה לזיכוי, יחזרו אליך.»",
            "low": "«זה לפי התנאים, אין לי אפשרות לזכות. {תפנה|תפני} לסניף שבו הצטרפת.»",
        },
        next_step="הזיכוי של $card_fee ש״ח יופיע בפירוט החיובים הבא",
    ),
    # -- mortgage ------------------------------------------------------------
    Scenario(
        key="mortgage_refi", call_type="mortgage", topic="מיחזור משכנתא",
        disclosure="עמלת הפירעון המוקדם וסיכון עליית הריבית",
        values=_v_mortgage_refi,
        request=(
            "יש לנו משכנתא, ושמעתי שכדאי עכשיו למחזר. ההחזר החודשי עלה מאוד בשנה האחרונה, "
            "וזה ממש מכביד עלינו.",
        ),
        empathy="«אני [שומע|שומעת] שההחזר שעלה מכביד עליכם, וזה באמת לא פשוט כשזה חוזר כל "
                "חודש מחדש.»",
        summary="ההחזר החודשי במשכנתא עלה ל-$payment_now ש״ח, ו{אתה רוצה|את רוצה} לבדוק אם "
                "מיחזור יוריד אותו",
        details1=("נשארו בערך $balance ש״ח, וההחזר היום הוא $payment_now ש״ח בחודש.",),
        details2=(
            "אנחנו לא מתכננים למכור את הדירה בשנים הקרובות. ההכנסה המשותפת בערך $income "
            "אלף נטו.",
        ),
        clarity={
            "high": "«אם נעביר את המסלול הצמוד למדד למסלול קבוע לא צמוד, ההחזר החודשי ירד "
                    "בכ-$saving ש״ח, ל-$payment_new ש״ח.» היתרה ותקופת ההלוואה יישארו כפי "
                    "שהן.",
            "mid": "«מיחזור יכול להוריד את ההחזר בכמה מאות שקלים, תלוי במסלולים.»",
            "low": "«צריך לבחון את תמהיל המסלולים, את ההצמדה למדד ואת נקודות היציאה, זה חישוב "
                   "מורכב.»",
        },
        compliance={
            "high": "«על פירעון המסלול הקיים תחול עמלת פירעון מוקדם, שמוערכת כרגע בכ-$exit_fee "
                    "ש״ח, ובמסלול בריבית משתנה ההחזר עלול לעלות אם הריבית תעלה.» "
                    "{תקבל|תקבלי} ממני בכתב את החישוב המלא, כולל העמלות, לפני כל חתימה.",
            "mid": "«יש עמלת פירעון מוקדם על המסלול הקיים, צריך לבדוק כמה.»",
            "low": "«אז אני [מכין|מכינה] לכם הצעת מיחזור, ונתקדם.»",
        },
        resolution={
            "high": "«הכנתי הצעת מיחזור מפורטת ושלחתי אותה למייל, ויועץ המשכנתאות יחזור "
                    "אליכם ביום שלישי ב-10:00 כדי לעבור עליה ולהתקדם לחתימה.»",
            "mid": "«אני [מעביר|מעבירה] את הפרטים למחלקת המשכנתאות, ומישהו יחזור אליכם.»",
            "low": "«את זה צריך לבדוק מול יועץ משכנתאות, {תקבע|תקבעי} פגישה בסניף.»",
        },
        next_step="יועץ המשכנתאות יחזור אליכם ביום שלישי ב-10:00 עם ההצעה המפורטת",
    ),
    Scenario(
        key="mortgage_new", call_type="mortgage", topic="בקשה למשכנתא לדירה ראשונה",
        disclosure="דמי פתיחת התיק, עלות השמאות וסיכוני ההצמדה והריבית המשתנה",
        values=_v_mortgage_new,
        request=(
            "אנחנו בתהליך של קניית דירה ראשונה, ורצינו להתחיל לבדוק משכנתא. אין לנו מושג "
            "מאיפה מתחילים, וזה קצת מבהיל.",
        ),
        empathy="«דירה ראשונה זה צעד גדול, ולגמרי טבעי שזה מרגיש מבהיל כשכל התהליך חדש "
                "לכם.»",
        summary="אתם קונים דירה ראשונה במחיר של $price ש״ח, ורוצים להבין מאיפה מתחילים את "
                "תהליך המשכנתא",
        details1=("הדירה עולה $price ש״ח, ויש לנו הון עצמי של בערך $equity_pct אחוז.",),
        details2=("שנינו שכירים, ההכנסה המשותפת כ-$income אלף נטו, ואין לנו הלוואות אחרות.",),
        clarity={
            "high": "«השלב הראשון הוא אישור עקרוני: מעלים תלושי שכר ודפי חשבון, ותוך שלושה "
                    "ימי עסקים מקבלים אישור עם סכום המשכנתא המקסימלי.» אחרי זה בוחרים תמהיל, "
                    "עושים שמאות לדירה, ורק אז חותמים.",
            "mid": "«צריך קודם אישור עקרוני, ואחר כך ממשיכים משם.»",
            "low": "«זה תלוי ביחס ההחזר, בשיעור המימון ובתמהיל שיאושר בחיתום, קשה להגיד משהו "
                   "לפני זה.»",
        },
        compliance={
            "high": "«דמי פתיחת התיק הם עד 360 ש״ח, השמאות בתשלום נפרד, ובמסלולים בריבית "
                    "משתנה או צמודה ההחזר עלול לעלות לאורך השנים.» כל התנאים יופיעו בהצעה "
                    "המחייבת, לפני החתימה.",
            "mid": "«יש גם עלויות של פתיחת תיק ושל שמאי.»",
            "low": "«אני [פותח|פותחת] לכם תיק ונתחיל, זה פשוט.»",
        },
        resolution={
            "high": "«פתחתי בקשה לאישור עקרוני ושלחתי לך למייל את רשימת המסמכים; ברגע "
                    "ש{תעלה|תעלי} אותם לאתר, התשובה תגיע תוך שלושה ימי עסקים.»",
            "mid": "«אני [שולח|שולחת] לך רשימת מסמכים, {תעביר|תעבירי} כשיהיה לכם.»",
            "low": "«בשלב הזה אין לי מה לעשות, {תחזור|תחזרי} אלינו כשיהיה לכם חוזה.»",
        },
        next_step="האישור העקרוני יגיע תוך שלושה ימי עסקים מרגע העלאת המסמכים",
    ),
    Scenario(
        key="mortgage_deferral", call_type="mortgage",
        topic="דחיית תשלומי משכנתא בשל מצב משפחתי",
        disclosure="הצטברות הריבית בתקופת הדחייה והעלות הכוללת",
        values=_v_mortgage_deferral,
        request=(
            "[אשתי אושפזה|בעלי אושפז] לפני שבועיים, ואנחנו לא עומדים בתשלום המשכנתא החודש. "
            "אני לא [יודע|יודעת] מה עושים במצב כזה.",
        ),
        empathy="«אני מאוד [מצטער|מצטערת] לשמוע על האשפוז. ברור שהראש שלך עכשיו במקום אחר "
                "לגמרי, ואני כאן כדי להוריד ממך לפחות את הדאגה הזאת.»",
        summary="בגלל האשפוז אתם לא עומדים בתשלום המשכנתא החודש, ו{אתה צריך|את צריכה} פתרון "
                "זמני עד שהמצב יתייצב",
        details1=("התשלום הוא $payment_now ש״ח, והוא יורד ב-15 לחודש.",),
        details2=(
            "כרגע נכנסת רק המשכורת שלי, ואנחנו מקווים שתוך שלושה חודשים בערך המצב יחזור "
            "לשגרה.",
        ),
        clarity={
            "high": "«אפשר לדחות את ההחזרים לשלושה חודשים: בתקופה הזו לא ירד תשלום, והסכום "
                    "שנדחה ייפרס על יתרת התקופה.» בפועל, ההחזר החודשי אחרי הדחייה יעלה בכ-"
                    "$delta ש״ח.",
            "mid": "«יש אפשרות לדחות תשלומים לכמה חודשים, ואז זה מתחלק על ההמשך.»",
            "low": "«דחיית תשלומים יוצרת גרייס חלקי עם היוון הריבית לקרן, בכפוף לאישור "
                   "החיתום.»",
        },
        compliance={
            "high": "«בתקופת הדחייה הריבית ממשיכה להצטבר ומתווספת ליתרה, כך שסך התשלומים "
                    "לאורך המשכנתא יגדל בכ-$extra ש״ח; על הדחייה עצמה אין עמלה.»",
            "mid": "«הדחייה עצמה בלי עמלה, אבל הריבית ממשיכה לרוץ.»",
            "low": "«אין בעיה, אני דוחה לכם את התשלומים וזהו.»",
        },
        mislead=("«הדחייה לא עולה לכם כלום, לא עמלה ולא ריבית.»",),
        resolution={
            "high": "«אישרתי דחייה של שלושה חודשים: התשלום של ה-15 לחודש לא ירד, והתשלום "
                    "הבא יהיה בעוד שלושה חודשים.» אישור בכתב יישלח אליך עוד היום.",
            "mid": "«העברתי את הבקשה לדחייה, יחזרו אליך תוך כמה ימים.»",
            "low": "«דחייה צריך לבקש בכתב דרך הסניף, אין לי אפשרות מכאן.»",
        },
        next_step="התשלום של ה-15 לחודש לא ירד, ואישור בכתב יגיע אליך עוד היום",
        suitability={
            5: "«כדי להתאים את הפתרון: מה ההכנסה של משק הבית כרגע, ולכמה זמן בערך {אתה "
               "צופה|את צופה} שהמצב יימשך?»",
            4: "«לכמה זמן בערך {אתה צופה|את צופה} שתצטרכו את הדחייה?»",
            3: "«יש עוד הכנסה בבית כרגע?»",
        },
        probe="הכנסת משק הבית והמשך הצפוי של הקושי",
    ),
    # -- investments ---------------------------------------------------------
    Scenario(
        key="investments_deposit", call_type="investments",
        topic="השקעת כספים מפיקדון שמסתיים", disclosure="דמי הניהול והסיכון להפסד",
        values=_v_investments_deposit,
        request=(
            "יש לי פיקדון של $amount ש״ח שמסתיים בשבוע הבא, ואני [מתלבט|מתלבטת] מה לעשות "
            "איתו. אני לא [מבין|מבינה] הרבה בהשקעות, ואני [חושש|חוששת] להפסיד.",
        ),
        empathy="«אני [מבין|מבינה] את החשש: זה כסף שחסכת לאורך זמן, ומאוד טבעי לא לרצות "
                "לסכן אותו.»",
        summary="הפיקדון של $amount ש״ח מסתיים בשבוע הבא, ו{אתה מחפש|את מחפשת} אפיק השקעה בלי "
                "לקחת סיכון שלא נוח לך איתו",
        details1=("זה כסף שחסכתי לאורך שנים, ואין לי תוכניות להשתמש בו בקרוב.",),
        details2=(
            "אני [חושב|חושבת] על שלוש-ארבע שנים לפחות, ואני לא רוצה שבשום מצב זה יירד ביותר "
            "מכמה אחוזים.",
        ),
        clarity={
            "high": "«יש שלוש אפשרויות עיקריות: לחדש את הפיקדון בריבית קבועה של $dep_rate "
                    "לשנה, קרן כספית עם נזילות יומית, או תיק מנוהל סולידי שרובו באג״ח.» ככל "
                    "שהסיכון נמוך יותר, גם התשואה הצפויה נמוכה יותר.",
            "mid": "«אפשר לחדש פיקדון, ויש גם קרנות ותיקים מנוהלים, כל אחד עם תשואה אחרת.»",
            "low": "«אפשר לשקול אג״ח ממשלתיות במח״מ קצר, קרן כספית או פוליסת חיסכון, תלוי "
                   "בפרופיל הסיכון.»",
        },
        compliance={
            "high": "«בתיק מנוהל יש דמי ניהול של $mgmt בשנה, והשקעה בשוק ההון כרוכה בסיכון: "
                    "הערך יכול לרדת, ואין הבטחה לתשואה.» לפני כל השקעה נמלא יחד שאלון "
                    "התאמה, ו{תקבל|תקבלי} את מסמך הגילוי המלא.",
            "mid": "«יש דמי ניהול, ויש סיכון מסוים כמו בכל השקעה.»",
            "low": "«אני [ממליץ|ממליצה] על התיק המנוהל, הוא עשה שנה טובה מאוד.»",
        },
        resolution={
            "high": "«קבעתי לך שיחה עם יועץ השקעות ליום רביעי ב-11:00, ושלחתי למייל את שאלון "
                    "ההתאמה ואת מסמך הגילוי, כדי {שתוכל|שתוכלי} לעבור עליהם לפני כן.»",
            "mid": "«אני [מעביר|מעבירה] את הפרטים ליועץ השקעות, והוא יחזור אליך.»",
            "low": "«אני לא [יועץ|יועצת] השקעות, {תבדוק|תבדקי} באתר מה מתאים לך.»",
        },
        next_step="יועץ ההשקעות יתקשר אליך ביום רביעי ב-11:00",
    ),
    Scenario(
        key="investments_drop", call_type="investments",
        topic="ירידת ערך בתיק ההשקעות ורצון למכור",
        disclosure="היעדר הוודאות, עמלת המכירה והמס על רווחי הון",
        values=_v_investments_drop,
        request=(
            "ראיתי שהתיק שלי ירד ב-$drop% בחודש האחרון, ואני ממש בלחץ. אני [חושב|חושבת] "
            "למכור הכול לפני שזה יורד עוד.",
        ),
        empathy="«אני [שומע|שומעת] כמה הירידה הזאת מלחיצה אותך, ובאמת לא נעים לראות את "
                "החיסכון יורד ככה בתוך חודש.»",
        summary="התיק ירד ב-$drop% בחודש האחרון, ו{אתה שוקל|את שוקלת} למכור את הכול כדי "
                "לעצור את ההפסד",
        details1=("יש שם בערך $amount ש״ח, רובו בקרנות מחקות מדד.",),
        details2=(
            "הכסף הזה מיועד לפנסיה, בעוד $years שנה בערך. אבל קשה לי לראות אותו יורד.",
        ),
        clarity={
            "high": "«מכירה עכשיו תהפוך את הירידה של $drop% להפסד בפועל, ואם השוק יתאושש, "
                    "הכסף כבר לא יהיה מושקע כדי ליהנות מזה.» בטווח של $years שנה, לתנודות של "
                    "חודש אחד יש השפעה קטנה יחסית על התוצאה.",
            "mid": "«מכירה עכשיו מקבעת את ההפסד, ובדרך כלל עדיף לא לפעול בלחץ.»",
            "low": "«זה תלוי בסטיית התקן של התיק, בבטא ובמתאם לשוק, קשה לדעת.»",
        },
        compliance={
            "high": "«אני לא [יכול|יכולה] להבטיח שהשוק יעלה, והוא יכול גם להמשיך לרדת; על "
                    "מכירה תחול עמלה של 0.1% ומס רווחי הון על רווח, אם יש.» כדאי לקבל את "
                    "ההחלטה אחרי שיחה עם יועץ שמכיר את התיק שלך.",
            "mid": "«יש עמלה על מכירה, וכמובן שאין ודאות לגבי השוק.»",
            "low": "«אני [מוכר|מוכרת] לך את הכול עכשיו, ככה לפחות זה לא יירד יותר.»",
        },
        mislead=("«{אל תדאג|אל תדאגי}, השוק תמיד חוזר תוך חודש-חודשיים, זה בטוח.»",),
        resolution={
            "high": "«סיכמנו לא לפעול בלחץ: קבעתי לך שיחה עם יועץ ההשקעות מחר ב-9:30, ושלחתי "
                    "לך למייל את ביצועי התיק בחמש השנים האחרונות.»",
            "mid": "«אני [רושם|רושמת] בקשה לשיחה עם יועץ, יחזרו אליך.»",
            "low": "«זה בשיקול דעתך, אני לא [יכול|יכולה] להגיד לך מה לעשות.»",
        },
        next_step="יועץ ההשקעות יתקשר אליך מחר ב-9:30",
    ),
)

SCENARIOS_BY_TYPE: dict[str, tuple[Scenario, ...]] = {
    t: tuple(s for s in SCENARIOS if s.call_type == t) for t, _ in TYPE_SHARES
}

DEPARTMENT = {"service": "שירות הלקוחות", "loans": "האשראי", "cards": "כרטיסי האשראי",
              "mortgage": "המשכנתאות", "investments": "ההשקעות"}
GREETINGS = (
    "בוקר טוב, הגעת למוקד $dept של הבנק, [מדבר|מדברת] $BNAME. במה אוכל לעזור?",
    "שלום, מוקד $dept, שמי $BNAME. איך אפשר לעזור?",
    "צהריים טובים, כאן $BNAME ממוקד $dept. במה אפשר לעזור היום?",
)
CUSTOMER_OPENERS = ("שלום, [מדבר|מדברת] $NAME. ", "היי, כאן $NAME. ", "שלום. ", "היי, ")


@dataclass(frozen=True)
class IdVariant:
    """One way an identification beat goes, with the reasoning it earns."""

    turns: tuple[tuple[str, str | tuple[str, ...]], ...]
    reasoning: str
    carry: str | None = None      # said at the start of the banker's next turn


IDENTIFICATION: dict[int, tuple[IdVariant, ...]] = {
    5: (
        IdVariant(
            turns=(("banker", "אשמח לעזור. לפני שנתחיל, אני [צריך|צריכה] לזהות אותך. מה מספר "
                              "תעודת הזהות שלך?"),
                   ("customer", ("$ID.", "כן, $ID.")),
                   ("banker", "תודה. ולאימות נוסף, מה תאריך הלידה שלך?"),
                   ("customer", "$DOB.")),
            carry="«תודה, הזיהוי הושלם.»",
            reasoning="[הבנקאי|הבנקאית] [ביקש|ביקשה] שני פרטים מזהים, מספר תעודת זהות ותאריך "
                      "לידה, [המתין|המתינה] לתשובות ו[אישר|אישרה] במפורש שהזיהוי הושלם לפני "
                      "[שמסר|שמסרה] מידע כלשהו.",
        ),
        IdVariant(
            turns=(("banker", "«כדי לזהות אותך אני [צריך|צריכה] שני פרטים: מספר תעודת זהות "
                              "ותאריך לידה.»"),
                   ("customer", "כן. תעודת זהות $ID, ותאריך לידה $DOB.")),
            carry="תודה, זיהיתי אותך.",
            reasoning="זיהוי מלא בפתיחת השיחה: [הבנקאי|הבנקאית] [ביקש|ביקשה] שני פרטים "
                      "מזהים ו[אישר|אישרה] את השלמת הזיהוי לפני [שעבר|שעברה] לטיפול בפנייה.",
        ),
    ),
    4: (
        IdVariant(
            turns=(("banker", "לצורך זיהוי, אפשר בבקשה מספר תעודת זהות?"),
                   ("customer", "$ID."),
                   ("banker", "«ומה מספר הטלפון שמעודכן אצלנו?»"),
                   ("customer", "$PHONE.")),
            reasoning="[הבנקאי|הבנקאית] [ביקש|ביקשה] שני פרטים מזהים ו[המתין|המתינה] לתשובה, "
                      "אך לא [אישר|אישרה] במפורש שהזיהוי הושלם לפני [שעבר|שעברה] לטיפול "
                      "בפנייה.",
        ),
        IdVariant(
            turns=(("banker", "«אפשר בבקשה תעודת זהות ותאריך לידה?»"),
                   ("customer", "$ID, ותאריך לידה $DOB.")),
            reasoning="נאספו שני פרטים מזהים כנדרש, אך [הבנקאי|הבנקאית] [המשיך|המשיכה] ישר "
                      "לפנייה בלי לאשר במפורש שהזיהוי הושלם.",
        ),
    ),
    3: (
        IdVariant(
            turns=(("banker", "«אפשר בבקשה מספר תעודת זהות?»"),
                   ("customer", ("$ID.", "כן, $ID."))),
            reasoning="בוצע זיהוי חלקי: [הבנקאי|הבנקאית] [הסתפק|הסתפקה] במספר תעודת זהות "
                      "בלבד ו[המשיך|המשיכה] בשיחה בלי פרט אימות נוסף, כנדרש בנוהל.",
        ),
        IdVariant(
            turns=(("banker", "«לצורך זיהוי, מה מספר תעודת הזהות?»"),
                   ("customer", "$ID.")),
            reasoning="[הבנקאי|הבנקאית] [ביקש|ביקשה] פרט מזהה אחד בלבד (תעודת זהות) ולא "
                      "[השלים|השלימה] אימות בפרט נוסף לפני [שהמשיך|שהמשיכה] בטיפול.",
        ),
    ),
    2: (
        IdVariant(
            turns=(("banker", "«אני כבר רואה אותך במערכת לפי מספר הטלפון.» רק {תגיד|תגידי} "
                              "לי את השם המלא."),
                   ("customer", "$NAME.")),
            reasoning="הזיהוי לא עמד בנוהל: [הבנקאי|הבנקאית] [הסתמך|הסתמכה] על מספר הטלפון "
                      "המזוהה ו[הסתפק|הסתפקה] בשם {הלקוח|הלקוחה}, שאינו פרט מזהה מספק. זהו "
                      "כשל בשער חובה.",
        ),
        IdVariant(
            turns=(("banker", "«{תגיד|תגידי} לי רק ארבע ספרות אחרונות של הכרטיס, ונתקדם.»"),
                   ("customer", "$CARD4.")),
            reasoning="[הבנקאי|הבנקאית] [הסתפק|הסתפקה] בארבע הספרות האחרונות של כרטיס, "
                      "שאינן מהוות זיהוי לפי הנוהל, ו[המשיך|המשיכה] לטפל בפנייה. זהו כשל "
                      "בשער חובה.",
        ),
    ),
    1: (
        IdVariant(
            turns=(("banker", "«אין צורך בזיהוי, אני כבר רואה את החשבון שלך מול העיניים.»"),
                   ("customer", "אה, מצוין, זה חוסך זמן.")),
            reasoning="לא בוצע זיהוי כלל: [הבנקאי|הבנקאית] [הודיע|הודיעה] שאין צורך בזיהוי "
                      "ו[ניגש|ניגשה] לפרטי החשבון על סמך מספר הטלפון בלבד. זהו כשל בשער "
                      "חובה.",
        ),
        IdVariant(
            turns=(("banker", "«לא צריך פרטים, אני רואה לפי המספר שממנו התקשרת שזה {אתה|את}.»"),
                   ("customer", "אוקיי, יופי.")),
            reasoning="[הבנקאי|הבנקאית] [ויתר|ויתרה] במפורש על זיהוי ו[טיפל|טיפלה] בפנייה על "
                      "סמך המספר המזוהה בלבד, בלי לבקש אף פרט מזהה. זהו כשל בשער חובה.",
        ),
    ),
}

EMPATHY_GENERIC = {
    3: ("«אוקיי, אני [מבין|מבינה]. {בוא|בואי} נראה מה אפשר לעשות.»",
        "«אני [מבין|מבינה] שזה לא נעים. נבדוק את זה.»",
        "«בסדר, אני [מבין|מבינה]. נטפל בזה.»"),
    2: ("«טוב, זה קורה. {בוא|בואי} נתקדם.»",
        "«אוקיי, זה לא משהו חריג, זה קורה כל יום.»"),
    1: ("«זה לא באחריות שלי, אני רק עונה לפי מה שכתוב במערכת.»",
        "«אין סיבה להילחץ, {תקשיב|תקשיבי} רק למה שאני [אומר|אומרת].»"),
}
ACCOMPANY = (
    "אני [נשאר|נשארת] איתך על הקו עד שנסגור את זה.",
    "{אל תדאג|אל תדאגי}, נעבור על זה יחד, צעד אחרי צעד.",
    "אני אלווה אותך בזה עד שהעניין יסתדר.",
)
_LISTEN_OK = ("כן, בדיוק ככה.", "נכון, זה בדיוק העניין.", "כן, הבנת נכון.")
LISTENING: dict[int, tuple[tuple[tuple[str, str | tuple[str, ...]], ...], ...]] = {
    5: ((("banker", "«אם הבנתי נכון, $summary. {תקן|תקני} אותי אם פספסתי משהו.»"),
         ("customer", _LISTEN_OK)),
        (("banker", "«רק כדי לוודא שהבנתי: $summary. נכון?»"),
         ("customer", _LISTEN_OK))),
    4: ((("banker", "«כלומר, $summary.»"), ("customer", ("כן.", "נכון.", "כן, בגדול."))),
        (("banker", "«אוקיי, אז $summary.»"), ("customer", ("כן.", "נכון.")))),
    3: ((("banker", "«אוקיי, הבנתי בגדול. יש עוד משהו?»"),
         ("customer", ("לא, זה העיקר.", "לא, זה בעיקר זה."))),
        (("banker", "«הבנתי. עוד משהו שחשוב שאדע?»"), ("customer", "לא, זה הכול."))),
    2: ((("banker", "«כן, כן, הבנתי כבר, זה קורה הרבה.»"),
         ("customer", "רגע, עוד לא סיימתי להסביר...")),
        (("banker", "«אוקיי, הבנתי, אין צורך בכל הפרטים.»"),
         ("customer", "אבל הפרטים חשובים פה..."))),
    1: ((("banker", "רגע, רגע, {תן|תני} לי לדבר."),
         ("customer", "«{אתה לא נותן|את לא נותנת} לי לסיים אפילו משפט אחד.»")),
        (("banker", "«לא צריך להסביר, אני כבר [יודע|יודעת] מה הבעיה.»"),
         ("customer", "אבל לא סיימתי להגיד מה קרה..."))),
}

SUITABILITY: dict[str, dict[int, str]] = {
    "service": {
        5: "«כדי שאתאים לך את הפתרון הנכון: זה משהו שחוזר על עצמו, או מקרה חד-פעמי? ואיך "
           "{אתה מנהל|את מנהלת} בדרך כלל את הפעולות בחשבון?»",
        4: "«זה קורה באופן קבוע, או שזו הפעם הראשונה?»",
        3: "«זה קרה גם בעבר?»",
        2: "«{אתה רוצה|את רוצה} אולי לשמוע גם על ההטבות החדשות שלנו בכרטיס האשראי?»",
        1: "«{תשמע|תשמעי}, יש לנו עכשיו מבצע על פיקדון לשנה, כדאי לך להצטרף.»",
    },
    "loans": {
        5: "«כדי להתאים לך את ההלוואה: מה ההכנסה החודשית נטו, יש התחייבויות נוספות, ואיזה "
           "החזר חודשי יהיה לך נוח לאורך זמן?»",
        4: "«מה ההכנסה החודשית שלך, ואיזה החזר חודשי נוח לך?»",
        3: "«ויש לך עוד הלוואות פתוחות?»",
        2: "«יש לנו מסלול הלוואה קבוע לחמש שנים, זה מה שרוב הלקוחות לוקחים.»",
        1: "«אני [ממליץ|ממליצה] לך לקחת כבר את הסכום המקסימלי שמאושר לך, ליתר ביטחון.»",
    },
    "cards": {
        5: "«כדי להתאים את הפתרון: באילו עסקאות {אתה משתמש|את משתמשת} בכרטיס בדרך כלל, "
           "ובערך באיזה היקף בחודש?»",
        4: "«בשביל מה {אתה משתמש|את משתמשת} בכרטיס בעיקר?»",
        3: "«{אתה משתמש|את משתמשת} בכרטיס הרבה?»",
        2: "«יש לנו כרטיס פרימיום עם הרבה הטבות, רוצה לשמוע?»",
        1: "«אני [יכול|יכולה] להציע לך הלוואה בתנאים מעולים, מעניין אותך?»",
    },
    "mortgage": {
        5: "«כדי להתאים את המסלולים אני [צריך|צריכה] להבין כמה דברים: מה ההכנסה המשותפת, "
           "כמה זמן אתם מתכננים להישאר בדירה, ועד כמה נוח לכם עם החזר שיכול להשתנות?»",
        4: "«מה ההכנסה המשותפת שלכם, ואתם מתכננים להישאר בדירה לאורך זמן?»",
        3: "«מה ההכנסה החודשית שלכם, בערך?»",
        2: "«רוב הלקוחות לוקחים שליש פריים, שליש קבועה ושליש משתנה, ככה נעשה גם לכם.»",
        1: "«אני [ממליץ|ממליצה] לכם לקחת את המסלול הכי ארוך שיש, ככה ההחזר הכי נמוך וזהו.»",
    },
    "investments": {
        5: "«לפני שנדבר על אפיקים, אני [צריך|צריכה] להכיר אותך: לכמה זמן הכסף יכול להיות "
           "מושקע, מה המטרה שלו, ואיך {תרגיש|תרגישי} אם הוא יירד זמנית בעשרה אחוזים?»",
        4: "«לכמה זמן {אתה רוצה|את רוצה} להשקיע את הכסף, ומה רמת הסיכון שנוחה לך?»",
        3: "«לכמה זמן {אתה רוצה|את רוצה} להשקיע?»",
        2: "«התיק המנייתי שלנו הכי מבוקש עכשיו, רוב הלקוחות הולכים עליו.»",
        1: "«{תשים|תשימי} הכול במניות טכנולוגיה, שם נמצא הכסף עכשיו.»",
    },
}
PROBE = {"service": "האם מדובר בתופעה חוזרת ואיך החשבון מנוהל",
         "loans": "הכנסה, התחייבויות והחזר חודשי נוח",
         "cards": "דפוסי השימוש בכרטיס והיקף ההוצאות",
         "mortgage": "הכנסת משק הבית, אופק המגורים והנכונות לתנודות בהחזר",
         "investments": "טווח ההשקעה, מטרת הכסף והסבילות לירידות"}
SUITABILITY_REJECT = ("לא, תודה, זה בכלל לא מה שביקשתי.",
                      "אני לא [בטוח|בטוחה] שזה קשור, אבל בסדר.",
                      "זה לא ממש רלוונטי אליי כרגע.")
MISLEAD_BY_TYPE = {
    "service": ("«במסלול החדש לא {תשלם|תשלמי} יותר אף שקל עמלות, על שום פעולה, מובטח.»",),
    "loans": ("«{אל תדאג|אל תדאגי}, זו הריבית הכי נמוכה בשוק ואין פה שום עמלות, מובטח.»",
              "«ההלוואה הזאת בעצם לא עולה לך כלום, הריבית זניחה.»"),
    "cards": ("«אין על זה שום עמלות ושום ריבית, מה ש{תעשה|תעשי} בכרטיס זה בחינם לגמרי.»",),
    "mortgage": ("«הריבית לא תעלה בשנים הקרובות, זה ידוע, אז אין שום סיכון במסלול המשתנה.»",
                 "«אין שום עמלות במהלך הזה, זה חיסכון נטו בשבילכם.»"),
    "investments": ("«התיק הזה מניב בערך 8% בשנה, זה כמעט בטוח, אין פה באמת סיכון.»",
                    "«אני [מבטיח|מבטיחה] לך שתוך שנה {תהיה|תהיי} ברווח, זה תמיד עולה "
                    "בסוף.»"),
}
CHECK_UNDERSTANDING = ("האם זה ברור עד כאן?", "רוצה שאחזור על זה שוב?",
                       "זה ברור, או שאפרט יותר?")
CLARITY_MUDDLE = ("אבל זה בכפוף לתנאים, ויכול להיות שבסוף זה בכלל אחרת.",
                  "בכל מקרה, זה מה שמופיע במערכת, אני לא [נכנס|נכנסת] לפרטים.")
CLARITY_REACT = {
    "high": ("ברור, תודה על ההסבר.", "אוקיי, עכשיו זה ברור לי.", "מצוין, הבנתי."),
    "mid": ("אוקיי... ומה זה אומר בפועל מבחינתי?", "הבנתי בערך. אבל כמה זה יוצא בסוף?"),
    "low": ("רגע, לא הבנתי. מה זה אומר בפועל?", "סליחה, אבל לא הבנתי כלום ממה שאמרת עכשיו.",
            "אני לא [בטוח|בטוחה] שהבנתי. אפשר בפשטות?"),
}
# Disclosure given only because the customer asked is what separates 4 from 5
# (and 3 from 5): so at those levels the customer asks first.
ASK_FEES = ("ויש פה עמלות או עלויות שאני [צריך|צריכה] לדעת עליהן?", "ורגע, זה עולה לי משהו?")
ASK_LEAD_IN = {"high": ("ברור, תודה.", "אוקיי, הבנתי."), "mid": ("אוקיי...", "טוב."),
               "low": ("לא ממש הבנתי, אבל בסדר.", "אני לא [בטוח|בטוחה] שהבנתי.")}
PROACTIVE = ("לפני שנתקדם, חשוב לי {שתכיר|שתכירי} את כל התנאים.",
             "ורק כדי שהתמונה תהיה מלאה ושקופה:",
             "לפני שמחליטים, יש כמה פרטים חשובים {שתדע|שתדעי}.")
COMPLIANCE_REACT = {
    "full": ("טוב שאמרת, תודה על השקיפות.", "אוקיי, זה חשוב לדעת.",
             "תודה, זה עוזר לי להחליט."),
    "partial": ("אוקיי. כמה בערך?", "טוב... אני אבדוק את זה בהסכם."),
    "none": ("אוקיי, סבבה.", "טוב, בסדר."),
    "misled": ("מעולה, אז אין מה לחשוש.", "נשמע מצוין, אם {אתה אומר|את אומרת}."),
}
OWNERSHIP = ("אני [ממשיך|ממשיכה] לעקוב אחרי זה, ואם משהו לא יסתדר עד אז, {תחזור|תחזרי} אליי "
             "ישירות.",
             "אני [רושם|רושמת] לעצמי לבדוק מחר שהכול עבר כמו שצריך.")
RESOLUTION_NONE = ("«אני לא [יכול|יכולה] לעזור בזה, אין לי מה לעשות עם הבקשה הזאת.»",
                   "«זה לא משהו שמטופל במוקד, אין לי מה להציע לך.»")
RESOLUTION_REACT = {
    "full": ("מעולה, תודה רבה!", "איזה יופי, הורדת לי אבן מהלב.",
             "מצוין, זה בדיוק מה שהייתי [צריך|צריכה]."),
    "partial": ("אוקיי... ומתי בערך יחזרו אליי?", "טוב, אני מקווה שזה יטופל מהר."),
    "none": ("אז בעצם לא קיבלתי שום פתרון?", "חבל, קיוויתי {שתוכל|שתוכלי} לעזור לי כבר עכשיו."),
}
NEXT_PARTIAL = "הבקשה פתוחה, ו{תקבל|תקבלי} עדכון ברגע שתטופל"
NEXT_REFERRED = "את ההמשך {תצטרך|תצטרכי} לברר מול הגורם שהפניתי אליו"
NEXT_NONE = "כרגע אין פעולה נוספת מצדנו"
CLOSURE: dict[int, tuple[tuple[tuple[str, str | tuple[str, ...]], ...], ...]] = {
    5: ((("banker", "«אז לסיכום: $next. יש עוד משהו שאוכל לעזור לך בו?»"),
         ("customer", ("לא, זה הכול. תודה רבה!", "לא, תודה, עזרת לי מאוד.")),
         ("banker", ("תודה לך, שיהיה לך יום נעים.", "בשמחה, שיהיה המשך יום טוב."))),
        (("banker", "«רק כדי שנסכם: $next. נשארו לך שאלות נוספות?»"),
         ("customer", "לא, הכול ברור. תודה!"),
         ("banker", "תודה שפנית אלינו, יום טוב."))),
    4: ((("banker", "«אז סיכמנו: $next. שיהיה לך יום טוב.»"),
         ("customer", ("תודה, גם לך.", "תודה רבה, ביי."))),),
    3: ((("banker", "«טוב, אז אם אין עוד משהו, יום טוב.»"), ("customer", "תודה, ביי.")),
        (("banker", "«בסדר גמור, שיהיה לך יום נעים.»"), ("customer", "תודה, להתראות."))),
    2: ((("banker", "«טוב, זהו להיום.»"), ("customer", "אה... אוקיי, ביי.")),
        (("banker", "«אוקיי, זהו. ביי.»"), ("customer", "טוב... ביי."))),
    1: ((("banker", "«אני [חייב|חייבת] לעבור לשיחה הבאה, ביי.»"),
         ("customer", "רגע, אבל עוד לא...")),
        (("banker", "«טוב, אין לי עוד מה להוסיף.»"), ("customer", "רגע, אבל יש לי עוד שאלה..."))),
}
# (banker, customer, whether the pause after it is dead air: a system lookup)
FILLERS = (
    ("רגע, אני [בודק|בודקת] את זה במערכת, זה ייקח כמה שניות.", "בסדר, אני [ממתין|ממתינה].", True),
    ("{תן|תני} לי רגע לפתוח את התיק שלך במערכת.", "כן, בטח.", True),
    ("אני רק [מוודא|מוודאת] שאין פניות קודמות בנושא הזה.", "בסדר.", True),
    ("יש עוד משהו שחשוב לך שאבדוק במקביל?", "לא, רק את זה בינתיים.", False),
)
# What the banker says on coming back from a lookup.
AFTER_LOOKUP = ("תודה על ההמתנה.", "תודה על הסבלנות.", "תודה שהמתנת.")
MAX_TURNS = 24

# Judge prose. [..] is the banker, {..} the customer.
REASONING: dict[str, dict[int, tuple[str, ...]]] = {
    "compliance": {
        5: ("גילוי נאות מלא ויזום: [הבנקאי|הבנקאית] [פירט|פירטה] [ביוזמתו|ביוזמתה] את "
            "$disclosure, בשפה ברורה ובלי הבטחות, לפני ש{הלקוח נדרש|הלקוחה נדרשה} לשאול.",
            "[הבנקאי|הבנקאית] [הקדים|הקדימה] לפרט את $disclosure לפני כל החלטה, בלי הבטחות "
            "לתוצאה. גילוי נאות מלא."),
        4: ("המידע המהותי נמסר במלואו ($disclosure), אך רק לאחר ש{הלקוח שאל|הלקוחה שאלה} "
            "על עלויות, ולא [ביוזמת הבנקאי|ביוזמת הבנקאית].",),
        3: ("הגילוי היה חלקי: [הבנקאי|הבנקאית] [הזכיר|הזכירה] שקיימות עלויות, אך בלי לנקוב "
            "בסכומים ובהשלכות, ורק לאחר ש{הלקוח שאל|הלקוחה שאלה}.",),
        2: ("חובת הגילוי לא קוימה: [הבנקאי|הבנקאית] [התקדם|התקדמה] לביצוע בלי להזכיר את "
            "$disclosure. זהו כשל בשער חובה.",),
        1: ("נמסר מידע מטעה: [הבנקאי|הבנקאית] [הבטיח|הבטיחה] תוצאה ו[הציג|הציגה] את ההצעה "
            "כנטולת סיכון או עלות, בניגוד לחובת הגילוי הנאות. זהו כשל חמור בשער חובה.",),
    },
    "empathy": {
        5: ("[הבנקאי|הבנקאית] [שיקף|שיקפה] במילים את מצב {הלקוח|הלקוחה}, [הכיר|הכירה] "
            "בקושי הספציפי ו[הציע|הציעה] ליווי אקטיבי עד לפתרון.",
            "שיקוף רגשי מדויק: [הבנקאי|הבנקאית] [התייחס|התייחסה] לקושי ש{הלקוח תיאר|הלקוחה "
            "תיארה} ו[הבהיר|הבהירה] [שילווה|שתלווה] את הטיפול עד הסוף."),
        4: ("[הבנקאי|הבנקאית] [שיקף|שיקפה] את הקושי הספציפי ש{הלקוח תיאר|הלקוחה תיארה}, "
            "אך בלי הצעה מפורשת ללוות את הטיפול.",),
        3: ("תגובה אדיבה ברמה בסיסית ('[אני מבין|אני מבינה]'), בלי שיקוף של הקושי הספציפי "
            "ש{הלקוח העלה|הלקוחה העלתה} ובלי התאמת הטון למצב.",),
        2: ("[הבנקאי|הבנקאית] [הגיב|הגיבה] לתחושות {הלקוח|הלקוחה} באופן מבטל ('זה קורה') "
            "ו[עבר|עברה] מיד לטיפול הטכני.",),
        1: ("[הבנקאי|הבנקאית] [התעלם|התעלמה] מהמצוקה ש{הלקוח הביע|הלקוחה הביעה} ו[הגיב|הגיבה] "
            "בנוסח מתגונן, בלי כל הכרה בקושי.",),
    },
    "listening": {
        5: ("[הבנקאי|הבנקאית] [נתן|נתנה] ל{לקוח|לקוחה} לסיים, [סיכם|סיכמה] במילים [שלו|שלה] "
            "את הצורך ו[וידא|וידאה] [שהבין|שהבינה] נכון.",),
        4: ("[הבנקאי|הבנקאית] [סיכם|סיכמה] את הצורך ש{הלקוח תיאר|הלקוחה תיארה}, אך בלי שאלות "
            "המשך שמעמיקות את ההבנה.",),
        3: ("הקשבה בסיסית: [הבנקאי|הבנקאית] [אישר|אישרה] [שהבין|שהבינה] בכלליות, בלי לסכם "
            "את הצורך ובלי שאלות המשך ממוקדות.",),
        2: ("[הבנקאי|הבנקאית] [קטע|קטעה] את {הלקוח|הלקוחה} לפני {שסיים|שסיימה} להסביר, "
            "ו[הניח|הניחה] [שהוא כבר יודע|שהיא כבר יודעת] מה הבעיה.",),
        1: ("[הבנקאי|הבנקאית] [קטע|קטעה] את {הלקוח|הלקוחה} שוב ושוב ו[השיב|השיבה] לפני "
            "{שהלקוח סיים|שהלקוחה סיימה} לתאר את הצורך.",),
    },
    "clarity": {
        5: ("ההסבר היה מדויק ומובנה: [הבנקאי|הבנקאית] [נקב|נקבה] בסכומים ובמועדים, "
            "[פירק|פירקה] את התהליך לשלבים ו[וידא|וידאה] הבנה.",),
        4: ("המידע נמסר באופן מדויק וברור, כולל סכומים ומועדים, אך בלי לוודא ש{הלקוח "
            "הבין|הלקוחה הבינה}.",),
        3: ("המידע היה נכון אך כללי: חסרו סכומים ומועדים מדויקים, ו{הלקוח נזקק|הלקוחה "
            "נזקקה} לשאלת הבהרה.",),
        2: ("ההסבר היה עמוס במונחים מקצועיים ללא פישוט, ו{הלקוח נותר|הלקוחה נותרה} בלי תשובה "
            "ברורה.",),
        1: ("ההסבר היה מבלבל: מונחים מקצועיים ללא פישוט והסתייגות שרוקנה את התשובה מתוכן, "
            "כך ש{הלקוח לא קיבל|הלקוחה לא קיבלה} מענה ברור.",),
    },
    "resolution": {
        5: ("הפנייה נפתרה במלואה: [הבנקאי|הבנקאית] [ביצע|ביצעה] את הפעולה בשיחה, "
            "[הגדיר|הגדירה] מועד ברור ו[לקח|לקחה] אחריות על המשך המעקב.",),
        4: ("הפנייה טופלה בשיחה עצמה ונקבע מועד ברור להשלמה, אם כי בלי לקיחת אחריות מפורשת "
            "על המשך המעקב.",),
        3: ("הפנייה טופלה חלקית: נפתחה בקשה להמשך טיפול, אך בלי מועד ובלי גורם מטפל "
            "מוגדר.",),
        2: ("הפנייה לא טופלה בשיחה: {הלקוח הופנה|הלקוחה הופנתה} לגורם אחר בלי "
            "ש[הבנקאי|הבנקאית] [ניסה|ניסתה] לטפל בה [בעצמו|בעצמה].",),
        1: ("הפנייה נותרה ללא מענה: [הבנקאי|הבנקאית] [הודיע|הודיעה] [שאין באפשרותו|שאין "
            "באפשרותה] לעזור, בלי הפניה ובלי מסלול המשך.",),
    },
    "suitability": {
        5: ("[הבנקאי|הבנקאית] [בירר|ביררה] את הצורך בשאלות ממוקדות ($probe) לפני "
            "[שהציע|שהציעה] מענה, והמענה נשען על הנתונים ש{הלקוח מסר|הלקוחה מסרה}.",),
        4: ("[הבנקאי|הבנקאית] [שאל|שאלה] שאלות רלוונטיות, אך הבירור היה חלקי ולא כיסה את כל "
            "ההיבטים ($probe).",),
        3: ("המענה רלוונטי באופן כללי, אך [הבנקאי|הבנקאית] [הסתפק|הסתפקה] בשאלה כללית אחת "
            "ולא [בירר|ביררה] $probe.",),
        2: ("[הבנקאי|הבנקאית] [הציע|הציעה] מענה כללי שאינו נשען על בירור, בלי לברר את "
            "{מצבו וצרכיו|מצבה וצרכיה} של {הלקוח|הלקוחה}.",),
        1: ("[הבנקאי|הבנקאית] [הציע|הציעה] מוצר שאינו קשור לצורך ש{הלקוח תיאר|הלקוחה "
            "תיארה}, בלי כל בירור של {מצבו|מצבה}.",),
    },
    "closure": {
        5: ("השיחה נסגרה בסיכום מלא: [הבנקאי|הבנקאית] [חזר|חזרה] על הצעד הבא, [שאל|שאלה] אם "
            "נותרו שאלות ו[נפרד|נפרדה] באדיבות.",),
        4: ("[הבנקאי|הבנקאית] [סיכם|סיכמה] את הצעד הבא, אך לא [שאל|שאלה] אם נותרו שאלות "
            "נוספות.",),
        3: ("סגירה אדיבה אך כללית: לא נאמר מהו הצעד הבא ומתי.",),
        2: ("השיחה הסתיימה בחטף, בלי סיכום ובלי הגדרת צעד הבא.",),
        1: ("השיחה נקטעה בפתאומיות כש{הלקוח עוד ניסה|הלקוחה עוד ניסתה} לשאול, בלי סיכום "
            "ובלי צעד הבא.",),
    },
}
# A strength names what was actually done: a level-4 disclosure given only
# when asked is not "proactive", and the prose must not say it was.
STRENGTH = {
    "identification": {5: "זיהוי מלא עם אישור מפורש, לפני מסירת מידע כלשהו",
                       4: "איסוף שני פרטים מזהים בפתיחת השיחה"},
    "compliance": {5: "גילוי נאות יזום של עמלות, תנאים וסיכונים",
                   4: "מסירת מידע מלא על עלויות ותנאים כשנשאל[|ה]"},
    "empathy": {5: "שיקוף אמיתי של מצב {הלקוח|הלקוחה} והצעת ליווי עד לפתרון",
                4: "הכרה בקושי הספציפי ש{הלקוח תיאר|הלקוחה תיארה}"},
    "listening": {5: "הקשבה פעילה וסיכום הצורך במילים [שלו|שלה], עם וידוא הבנה",
                  4: "סיכום הצורך במילים [שלו|שלה]"},
    "clarity": {5: "הסבר מובנה עם סכומים ומועדים מדויקים, כולל וידוא הבנה",
                4: "הסבר מדויק עם סכומים ומועדים"},
    "resolution": {5: "פתרון מלא בשיחה ולקיחת אחריות על המשך המעקב",
                   4: "טיפול בפנייה בשיחה עצמה, עם מועד ברור להשלמה"},
    "suitability": {5: "בירור צרכים ממוקד לפני הצעת מענה",
                    4: "שאלות רלוונטיות לפני הצעת מענה"},
    "closure": {5: "סגירה מסודרת: סיכום הצעד הבא, בדיקה אם נותרו שאלות ופרידה אדיבה",
                4: "סיכום הצעד הבא בסיום השיחה"},
}
STRENGTH_FALLBACK = "שמירה על שיחה מנומסת ועניינית"
DEVELOPMENT = {
    "identification": "להקפיד על זיהוי מלא, שני פרטים מזהים ואישור מפורש, לפני מסירת מידע "
                      "כלשהו.",
    "compliance": "לפרט באופן יזום עמלות, ריבית וסיכונים לפני שהלקוח מתבקש להחליט, ולהימנע "
                  "מכל הבטחה לתוצאה.",
    "empathy": "לשקף במילים את מצב הלקוח ואת הקושי הספציפי שתיאר, לפני המעבר לפתרון.",
    "listening": "לאפשר ללקוח לסיים את דבריו, לצמצם קטיעות ולסכם את הצורך לפני מתן מענה.",
    "clarity": "למסור מידע מובנה עם סכומים ומועדים מדויקים, להימנע ממונחים מקצועיים ולוודא "
               "הבנה.",
    "resolution": "לטפל בפנייה עד סופה בשיחה, או להגדיר מסלול המשך ברור: מי מטפל ועד מתי.",
    "suitability": "לברר את צורכי הלקוח בשאלות ממוקדות לפני הצעת מוצר או פתרון.",
    "closure": "לסיים כל שיחה בסיכום הצעד הבא והמועד, ולשאול אם נותרו שאלות.",
}


# -- one call ----------------------------------------------------------------

@dataclass
class _Turn:
    speaker: str
    parts: list[str]
    quotes: dict[str, str] = field(default_factory=dict)
    start: float = 0.0
    end: float = 0.0

    @property
    def text(self) -> str:
        return " ".join(self.parts)


class _Dialog:
    """Accumulates utterances into strictly alternating turns."""

    def __init__(self, slots: dict[str, str], banker_female: bool, customer_female: bool,
                 rng: np.random.Generator) -> None:
        self.slots = slots
        self.banker_female = banker_female
        self.customer_female = customer_female
        self.rng = rng
        self.turns: list[_Turn] = []
        self.counts: dict[str, int] = {}
        self.dead_air_after: list[int] = []      # turn indices followed by a system lookup
        self.after_lookup: str | None = None     # opens the banker's next turn

    def say(self, speaker: str, template: str | Sequence[str], dim: str | None = None) -> None:
        if speaker == "banker" and self.after_lookup is not None:
            prefix, self.after_lookup = self.after_lookup, None
            self.say("banker", prefix)
        own, other = ((self.banker_female, self.customer_female) if speaker == "banker"
                      else (self.customer_female, self.banker_female))
        text, quote = _render(_pick(self.rng, template), self.slots, own, other, self.counts)
        # Consecutive utterances of one speaker are one turn, as a diarizer
        # would deliver them; the quote still lies inside that one turn.
        if self.turns and self.turns[-1].speaker == speaker:
            turn = self.turns[-1]
            turn.parts.append(text)
        else:
            turn = _Turn(speaker, [text])
            self.turns.append(turn)
        if dim is not None and quote is not None:
            turn.quotes[dim] = quote

    def script(self, lines: Sequence[tuple[str, str | Sequence[str]]], dim: str) -> None:
        for speaker, template in lines:
            self.say(speaker, template, dim)


def build_slots(rng: np.random.Generator, scenario: Scenario, *, banker_female: bool,
                customer_female: bool) -> dict[str, str]:
    """The call-specific values a scenario's templates refer to."""
    slots = dict(scenario.values(rng), dept=DEPARTMENT[scenario.call_type])
    # The paraphrase is the banker's, so its gender markup resolves for the banker.
    slots["summary"] = _render(scenario.summary, slots, banker_female, customer_female)[0]
    return slots


def _bucket(level: int) -> str:
    """The phrase tier a 1..5 level draws from (1 and 5 add their own extras)."""
    return "high" if level >= 4 else "mid" if level == 3 else "low"


# Phrase tier -> the customer's reaction to it.
_REACTION = {"high": "full", "mid": "partial", "low": "none"}


def compose_dialog(rng: np.random.Generator, scenario: Scenario, levels: dict[str, int],
                   slots: dict[str, str], *, banker_female: bool, customer_female: bool,
                   n_fillers: int = 0) -> tuple[_Dialog, str]:
    """The call's turns for these per-dimension levels, and the identification
    reasoning that belongs to the variant used. Every dimension gets exactly
    one marked evidence span."""
    d = _Dialog(slots, banker_female, customer_female, rng)
    ctype = scenario.call_type

    variant = _choose(rng, IDENTIFICATION[levels["identification"]])
    listening = _choose(rng, LISTENING[levels["listening"]])
    closing = _choose(rng, CLOSURE[levels["closure"]])
    # Greeting, empathy, suitability, clarity, compliance and resolution are one
    # banker and one customer turn each; the identification carry joins the
    # empathy turn. A filler is two turns.
    fixed = 12 + len(variant.turns) + len(listening) + len(closing)
    n_fillers = max(0, min(n_fillers, (MAX_TURNS - fixed) // 2))

    d.say("banker", GREETINGS)
    d.say("customer", _pick(rng, CUSTOMER_OPENERS) + _pick(rng, scenario.request))

    d.script(variant.turns, "identification")
    if variant.carry:
        d.say("banker", variant.carry, "identification")

    lvl = levels["empathy"]
    if lvl >= 4:
        d.say("banker", scenario.empathy, "empathy")
        if lvl == 5:
            d.say("banker", ACCOMPANY)
    else:
        d.say("banker", EMPATHY_GENERIC[lvl], "empathy")
    d.say("customer", scenario.details1)

    d.script(listening, "listening")

    fillers = [FILLERS[i] for i in rng.permutation(len(FILLERS))[:n_fillers]]

    def filler() -> None:
        if fillers:
            banker, customer, dead_air = fillers.pop(0)
            d.say("banker", banker)
            d.say("customer", customer)
            if dead_air:
                d.dead_air_after.append(len(d.turns) - 1)
                d.after_lookup = _pick(rng, AFTER_LOOKUP)

    filler()
    lvl = levels["suitability"]
    d.say("banker", scenario.suitability.get(lvl) or SUITABILITY[ctype][lvl], "suitability")
    d.say("customer", scenario.details2 if lvl >= 3 else SUITABILITY_REJECT)
    filler()

    lvl, comp = levels["clarity"], levels["compliance"]
    d.say("banker", scenario.clarity[_bucket(lvl)], "clarity")
    if lvl == 5:
        d.say("banker", CHECK_UNDERSTANDING)
    elif lvl == 1:
        d.say("banker", CLARITY_MUDDLE)
    if comp in (3, 4):
        d.say("customer", _pick(rng, ASK_LEAD_IN[_bucket(lvl)]) + " " + _pick(rng, ASK_FEES))
    else:
        d.say("customer", CLARITY_REACT[_bucket(lvl)])

    if comp == 5:
        d.say("banker", PROACTIVE)
    if comp == 1:
        d.say("banker", scenario.mislead or MISLEAD_BY_TYPE[ctype], "compliance")
    else:
        d.say("banker", scenario.compliance[_bucket(comp)], "compliance")
    d.say("customer", COMPLIANCE_REACT[_REACTION[_bucket(comp)] if comp > 1 else "misled"])

    lvl = levels["resolution"]
    if lvl == 1:
        d.say("banker", RESOLUTION_NONE, "resolution")
    else:
        d.say("banker", scenario.resolution[_bucket(lvl)], "resolution")
        if lvl == 5:
            d.say("banker", OWNERSHIP)
    d.say("customer", RESOLUTION_REACT[_REACTION[_bucket(lvl)]])

    # The closing recap can only promise what the resolution delivered.
    recap = (scenario.next_step if lvl >= 4 else NEXT_PARTIAL if lvl == 3
             else NEXT_REFERRED if lvl == 2 else NEXT_NONE)
    d.slots = dict(slots, next=_render(recap, slots, banker_female, customer_female)[0])
    d.script(closing, "closure")

    missing = [dim for dim in KNOWN_DIMENSIONS
               if not any(dim in t.quotes for t in d.turns)]
    if missing:  # pragma: no cover - a phrase-library bug, caught by the tests
        raise AssertionError(f"{scenario.key}: no evidence span for {missing}")
    return d, variant.reasoning


def _timeline(rng: np.random.Generator, turns: list[_Turn], duration: float, talk_ratio: float,
              patience: float, dead_air: float, dead_air_after: list[int]) -> float:
    """Place the turns on the clock; returns the dead air actually laid out.

    The features and the transcript must describe the same call: banker and
    customer speech add up to the talk ratio, the turn gaps to the patience,
    and every gap over three seconds counts as dead air (features.py)."""
    n = len(turns)
    gaps = []
    for turn in turns[1:]:
        if turn.speaker == "banker":
            gaps.append(float(np.clip(rng.normal(patience, 0.3), 0.15, 2.9)))
        else:
            gaps.append(float(rng.uniform(0.2, 0.8)))
    if dead_air >= 3.2:
        slots = [i for i in dead_air_after if i < n - 1]
        if not slots:
            k = 1 if dead_air < 8 else 2
            slots = sorted(int(i) for i in rng.choice(n - 1, size=min(k, n - 1), replace=False))
        share = max(3.2, dead_air / len(slots))
        for i in slots:
            gaps[i] = share
    lead, trail = float(rng.uniform(0.3, 1.2)), float(rng.uniform(0.3, 1.5))
    speech = duration - lead - trail - sum(gaps)
    if speech < 0.45 * duration:
        # A short call cannot hold this much silence: shrink every gap alike.
        factor = max(0.1, (0.55 * duration - lead - trail) / sum(gaps))
        gaps = [g * factor for g in gaps]
        speech = duration - lead - trail - sum(gaps)
    banker_total, customer_total = talk_ratio * speech, (1 - talk_ratio) * speech
    weight = [len(t.text) + 15 for t in turns]
    banker_w = sum(w for w, t in zip(weight, turns, strict=True) if t.speaker == "banker")
    customer_w = sum(w for w, t in zip(weight, turns, strict=True) if t.speaker == "customer")
    clock = lead
    for i, turn in enumerate(turns):
        share = (banker_total * weight[i] / banker_w if turn.speaker == "banker"
                 else customer_total * weight[i] / customer_w)
        turn.start, turn.end = round(clock, 2), round(clock + share, 2)
        clock += share + (gaps[i] if i < n - 1 else 0.0)
    return round(sum(g for g in gaps if g > 3.0), 1)


# -- the batch ---------------------------------------------------------------

@dataclass(frozen=True)
class Banker:
    banker_id: str
    kind: str                    # star | struggler | new | regular
    skill: float
    female: bool
    offsets: dict[str, float]
    id_risk: float
    comp_risk: float


@dataclass
class _Call:
    index: int
    call_id: str
    banker: Banker
    call_date: date
    call_type: str
    duration: float
    mono: bool
    week: int
    status: str = "success"
    error: str | None = None
    stages: list[str] = field(default_factory=list)
    has_card: bool = True
    role_confidence: float = 1.0


@dataclass(frozen=True)
class GenerationSummary:
    workspace: Path
    input_dir: Path
    output_dir: Path
    n_calls: int
    counts: dict[str, int]              # status -> calls
    n_bankers: int
    n_scored: int                       # success scorecards (the report's cohort)
    gate_failed: int                    # among them
    mean_total: float | None
    first_date: date
    last_date: date
    run_ids: list[str]
    seconds: float


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def _make_bankers(rng: np.random.Generator, n: int) -> list[Banker]:
    n_special = min(3, max(0, (n - 2) // 4))
    n_new = max(1, round(n * 0.1)) if n >= 8 else 0
    order = [int(i) for i in rng.permutation(n)]
    kinds = {i: "regular" for i in range(n)}
    for i in order[:n_special]:
        kinds[i] = "star"
    for i in order[n_special:2 * n_special]:
        kinds[i] = "struggler"
    for i in order[2 * n_special:2 * n_special + n_new]:
        kinds[i] = "new"
    bankers: list[Banker] = []
    for i in range(n):
        kind = kinds[i]
        skill = float(rng.normal(0.0, 0.45))
        if kind == "star":
            skill = float(rng.uniform(1.05, 1.35))
        elif kind == "struggler":
            skill = -float(rng.uniform(1.05, 1.35))
        elif kind == "new":
            skill -= 0.3
        sloppy_id = kind == "struggler" or (kind != "star" and rng.random() < 0.15)
        sloppy_comp = kind == "struggler" or (kind != "star" and rng.random() < 0.12)
        bankers.append(Banker(
            banker_id=f"B{101 + i}", kind=kind, skill=skill, female=bool(rng.random() < 0.6),
            offsets={d: float(rng.normal(0.0, 0.28)) for d in NON_GATE},
            id_risk=float(rng.normal(0.0, 0.3)) + (1.4 if sloppy_id else 0.0)
            - (0.6 if kind == "star" else 0.0),
            comp_risk=float(rng.normal(0.0, 0.3)) + (1.3 if sloppy_comp else 0.0)
            - (0.6 if kind == "star" else 0.0),
        ))
    return bankers


def business_days(weeks: int) -> tuple[date, list[date]]:
    """(first Sunday, every Sunday-Thursday business day) of the window ending END_DATE."""
    last_sunday = END_DATE - timedelta(days=(END_DATE.weekday() + 1) % 7)
    first_sunday = last_sunday - timedelta(weeks=weeks - 1)
    days = [first_sunday + timedelta(days=7 * w + k) for w in range(weeks) for k in range(5)]
    return first_sunday, [d for d in days if d <= END_DATE and d not in HOLIDAYS]


def _plan_calls(rng: np.random.Generator, n_calls: int, bankers: list[Banker],
                weeks: int) -> list[_Call]:
    first_sunday, days = business_days(weeks)
    weights = np.array([WEEKDAY_WEIGHT[(d.weekday() + 1) % 7] for d in days])
    weights /= weights.sum()
    recent = days[-10:]
    recent_w = weights[-10:] / weights[-10:].sum()

    counts = np.zeros(len(bankers), dtype=int)
    new = [i for i, b in enumerate(bankers) if b.kind == "new"]
    for i in new:
        counts[i] = int(rng.integers(3, 7))
    if counts.sum() > n_calls // 4:
        # A tiny batch: new joiners with their own handful would be most of it.
        counts[:] = 0
        new = []
    regular = [i for i in range(len(bankers)) if counts[i] == 0]
    share = rng.gamma(16.0, 1.0, len(regular))
    counts[regular] += rng.multinomial(n_calls - int(counts.sum()), share / share.sum())

    drawn: list[tuple[date, float, int]] = []
    for i, count in enumerate(counts):
        if count == 0:
            continue
        pool, pool_w = (recent, recent_w) if i in new else (days, weights)
        for k in rng.choice(len(pool), size=int(count), p=pool_w):
            drawn.append((pool[int(k)], float(rng.random()), i))
    drawn.sort()

    type_ids = [t for t, _ in TYPE_SHARES]
    types = rng.choice(len(type_ids), size=n_calls, p=[s for _, s in TYPE_SHARES])
    width = max(4, len(str(n_calls)))
    calls: list[_Call] = []
    for idx, (day, _tiebreak, b_idx) in enumerate(drawn):
        ctype = type_ids[int(types[idx])]
        duration = MEDIAN_DURATION_SEC * TYPE_DURATION[ctype] * math.exp(rng.normal(0.0, 0.5))
        calls.append(_Call(
            index=idx, call_id=f"DEMO{idx + 1:0{width}d}", banker=bankers[b_idx],
            call_date=day, call_type=ctype,
            duration=round(float(np.clip(duration, 60.0, 1800.0)), 1),
            mono=bool(rng.random() < MONO_SHARE), week=(day - first_sunday).days // 7,
        ))
    return calls


def _assign_statuses(rng: np.random.Generator, calls: list[_Call]) -> None:
    """~3% held for review (half with a scorecard, half without) and ~1% failed."""
    n = len(calls)
    n_failed = round(n * FAILED_SHARE)
    n_held = round(n * HELD_SHARE)
    n_held_card = n_held // 2 + n_held % 2
    order = [int(i) for i in rng.permutation(n)]
    mono_first = sorted(order, key=lambda i: not calls[i].mono)
    held_card = [i for i in mono_first[:n_held_card] if calls[i].mono]
    taken = set(held_card)
    rest = [i for i in order if i not in taken]
    held_nocard = rest[:n_held - len(held_card)]
    failed = rest[len(held_nocard):len(held_nocard) + n_failed]
    for call in calls:
        call.stages = list(STAGES)
    for i in held_card:
        # The pipeline's own reason for holding a scored mono call.
        c = calls[i]
        c.status, c.has_card = "needs_human_review", True
        c.role_confidence = round(float(rng.uniform(0.12, 0.31)), 2)
        c.error = (f"speaker roles inferred with low confidence "
                   f"({c.role_confidence:.2f} < {MIN_ROLE_CONFIDENCE:.2f})")
    for i in held_nocard:
        c = calls[i]
        c.status, c.has_card = "needs_human_review", False
        c.stages = list(STAGES[:STAGES.index("judge")])
        c.error = ("judge output failed after 3 attempts: evidence verification failed: "
                   "dimension 'compliance': no evidence quote provided")
    for i in failed:
        c = calls[i]
        c.status, c.has_card = "failed", False
        c.stages = list(STAGES[:int(rng.integers(1, 6))])
        c.error = "RuntimeError: synthetic failure"


def _draw_levels(rng: np.random.Generator, call: _Call, span_days: int,
                 training_week: int | None) -> tuple[dict[str, int], dict[str, float]]:
    """Per-dimension scores and the objective features that go with them."""
    b = call.banker
    t = call.call_type
    frac = (call.week * 7) / max(1, span_days)
    trained = training_week is not None and call.week >= training_week
    long_call = max(0.0, math.log(call.duration / MEDIAN_DURATION_SEC))
    shared = float(rng.normal(0.0, 0.3))
    mu = {d: BASE[d] + LOAD[d] * b.skill + b.offsets[d] + TYPE_EFFECT[t].get(d, 0.0)
          + shared + DRIFT * frac for d in NON_GATE}
    if trained:
        for d, effect in TRAINING_EFFECT.items():
            mu[d] += effect
    mu["clarity"] -= 0.4 * long_call
    mu["closure"] -= 0.5 * long_call
    mu["resolution"] -= 0.15 * long_call

    # Features follow the same latent state, plus their own noise - and that
    # noise feeds back into the score, so the feature carries information of
    # its own (a talk ratio that happens to be high DID crowd the customer out).
    stereo = not call.mono
    z_listen = mu["listening"] - BASE["listening"]
    tr_noise = float(rng.normal(0.0, 0.055))
    talk_ratio = float(np.clip(0.58 - 0.05 * z_listen + tr_noise, 0.35, 0.80))
    scale = math.sqrt(call.duration / MEDIAN_DURATION_SEC)
    lam = 1.1 * math.exp(-0.6 * z_listen) * scale
    interruptions = int(rng.poisson(lam)) if stereo else 0
    interrupted_by_customer = int(rng.poisson(0.5 * scale)) if stereo else 0
    patience = float(np.clip(0.9 + 0.22 * z_listen + rng.normal(0.0, 0.35), 0.15, 6.0))
    z_suit = mu["suitability"] - BASE["suitability"]
    qpm = float(np.clip(1.0 + 0.28 * z_suit + rng.normal(0.0, 0.3), 0.15, 3.2))
    z_clar = mu["clarity"] - BASE["clarity"]
    da_noise = float(rng.normal(0.0, 0.013))
    dead_air_frac = float(np.clip(0.03 - 0.014 * z_clar + da_noise, 0.0, 0.2))

    observed = dict(mu)
    observed["listening"] += -2.5 * tr_noise - (0.12 * (interruptions - lam) if stereo else 0.0)
    observed["clarity"] += -6.0 * da_noise
    levels = {d: int(np.clip(round(observed[d] + rng.normal(0.0, 0.5)), 1, 5)) for d in NON_GATE}

    for gate in ("identification", "compliance"):
        risk = b.id_risk if gate == "identification" else b.comp_risk
        p_fail = _sigmoid(GATE_BASE_LOGIT[gate] + risk - 0.5 * b.skill
                          + GATE_TYPE_LOGIT[gate][t])
        if rng.random() < p_fail:
            levels[gate] = 2 if rng.random() < 0.6 else 1
        else:
            base = 4.35 if gate == "identification" else 4.05
            load = 0.45 if gate == "identification" else 0.4
            pass_type = GATE_PASS_TYPE[t] if gate == "compliance" else 0.0
            levels[gate] = int(np.clip(round(base + load * b.skill + pass_type
                                             + rng.normal(0.0, 0.55)), 3, 5))

    # The listening reasoning quotes the measured interruptions next to what the
    # judge saw, so the two must not contradict each other: a banker who cut
    # the customer off (2) - "again and again" (1) - interrupted at least once
    # (twice), and one who let the customer finish (5) at most once. Without
    # this, ~2% of calls read "interrupted the customer ... with no
    # interruptions". No draw is added, so every other value stays as it was.
    if stereo:
        if levels["listening"] <= 2:
            interruptions = max(interruptions, 3 - levels["listening"])
        elif levels["listening"] == 5:
            interruptions = min(interruptions, 1)

    features = {
        "talk_ratio": talk_ratio, "interruptions_by_banker": interruptions,
        "interruptions_by_customer": interrupted_by_customer, "patience": patience,
        "qpm": qpm, "dead_air": dead_air_frac * call.duration,
        "monologue": float(np.clip(18 + 100 * (talk_ratio - 0.45) + rng.normal(0.0, 12.0)
                                   + 6 * long_call, 6.0, 0.45 * call.duration)),
        "wpm_banker": float(np.clip(rng.normal(148, 16), 95, 210)),
        "wpm_customer": float(np.clip(rng.normal(132, 20), 80, 200)),
    }
    return {d: levels[d] for d in KNOWN_DIMENSIONS}, features


def _listening_evidence_clause(features: Features) -> str:
    pct = round(features.talk_ratio * 100)
    clause = f" [הבנקאי|הבנקאית] [דיבר|דיברה] כ-{pct}% מזמן הדיבור"
    if not features.overlap_metrics_available:
        return clause + " (בהקלטה חד-ערוצית לא ניתן למדוד קטיעות)."
    n = features.interruptions_by_banker
    if n == 0:
        return clause + ", ללא קטיעות."
    if n == 1:
        return clause + ", עם קטיעה אחת של {הלקוח|הלקוחה}."
    return clause + f", עם {n} קטיעות של {{הלקוח|הלקוחה}}."


def _prose(template: str, call: _Call, customer_female: bool,
           slots: dict[str, str] | None = None) -> str:
    return _render(template, slots or {}, call.banker.female, customer_female)[0]


def _scorecard(rng: np.random.Generator, rubric: Rubric, call: _Call, scenario: Scenario,
               levels: dict[str, int], dialog: _Dialog, id_reasoning: str,
               features: Features, customer_female: bool, timestamp: str,
               latency: float, prompt_sha: str) -> ScoreCard:
    by_turn: dict[str, _Turn] = {}
    for turn in dialog.turns:
        for dim in turn.quotes:
            by_turn.setdefault(dim, turn)
    probe = scenario.probe or PROBE[scenario.call_type]
    scores: dict[str, DimensionScore] = {}
    for dim in (d.id for d in rubric.dimensions):
        level = levels[dim]
        if dim == "identification":
            reasoning = id_reasoning
        else:
            reasoning = _pick(rng, REASONING[dim][level])
        if dim == "listening":
            reasoning += _listening_evidence_clause(features)
        if dim == "clarity" and features.dead_air_total_sec >= 45:
            reasoning += f" בשיחה נרשמו כ-{int(features.dead_air_total_sec)} שניות של שקט ממושך."
        turn = by_turn[dim]
        scores[dim] = DimensionScore(
            score=level,
            reasoning_he=_prose(reasoning, call, customer_female,
                                {"disclosure": scenario.disclosure, "probe": probe}),
            evidence=[Evidence(quote=turn.quotes[dim], timestamp=mmss(turn.start),
                               speaker=turn.speaker)],
        )
    total, gate_failed, failed_gates = weighted_total(rubric, scores)

    weight = {d.id: d.weight for d in rubric.dimensions}
    gate_ids = set(rubric.gate_ids)
    # On a tie a judge names the distinctive behaviour, not the procedure
    # every call is expected to follow.
    ranked = sorted(levels, key=lambda d: (-levels[d], d in gate_ids, -weight[d]))
    strengths = [_prose(STRENGTH[d][levels[d]], call, customer_female)
                 for d in ranked[:2] if levels[d] >= 4]
    weakest = sorted(levels, key=lambda d: (levels[d], d not in gate_ids, -weight[d]))
    name = {d.id: d.name_he for d in rubric.dimensions}
    opening = f"שיחת {TYPE_HE[call.call_type]} בנושא {scenario.topic}."
    if gate_failed:
        verdict = (f"השיחה נכשלה בשער חובה ({', '.join(name[g] for g in failed_gates)}), "
                   f"ולכן הציון הכולל הוגבל.")
    elif total >= 80:
        verdict = f"טיפול מקצועי ברמה גבוהה, בולט במיוחד ב{name[ranked[0]]}."
    elif total >= 60:
        verdict = f"טיפול טוב ברובו; התחום העיקרי לשיפור הוא {name[weakest[0]]}."
    elif total >= 40:
        verdict = f"טיפול ברמה גבולית, עם פערים בעיקר ב{name[weakest[0]]} וב{name[weakest[1]]}."
    else:
        verdict = f"טיפול חלש שמחייב ליווי צמוד, בעיקר ב{name[weakest[0]]} וב{name[weakest[1]]}."
    return ScoreCard(
        call_id=call.call_id, banker_id=call.banker.banker_id, scores=scores,
        weighted_total=total, gate_failed=gate_failed, failed_gates=failed_gates,
        strengths_he=strengths or [STRENGTH_FALLBACK],
        development_area_he=DEVELOPMENT[weakest[0]],
        summary_he=_prose(f"{opening} {verdict}", call, customer_female),
        judge_engine=JUDGE_ENGINE, model=JUDGE_ENGINE, prompt_sha256=prompt_sha,
        prompt_version=PROMPT_VERSION, rubric_sha256=rubric.sha256,
        retries=int(rng.choice(3, p=(0.9, 0.08, 0.02))), latency_sec=round(latency, 2),
        timestamp=timestamp,
    )


def _stage_seconds(rng: np.random.Generator, call: _Call) -> dict[str, float]:
    minutes = call.duration / 60.0
    full = {
        "ingestion": rng.uniform(0.05, 0.3),
        "audio": rng.uniform(0.4, 1.8),
        "asr": minutes * rng.uniform(4.0, 7.0),          # ~0.08 real-time factor on a GPU
        "speakers": rng.uniform(4.0, 14.0) if call.mono else rng.uniform(0.2, 0.8),
        "redaction": rng.uniform(0.1, 0.6),
        "features": rng.uniform(0.05, 0.25),
        "judge": rng.uniform(12.0, 40.0),
        "report": rng.uniform(0.15, 0.5),
    }
    return {s: round(float(full[s]), 3) for s in call.stages}


# -- the workspace -----------------------------------------------------------

def _norm(path: Path) -> Path:
    return Path(os.path.normcase(str(path)))


def _overlaps(a: Path, b: Path) -> bool:
    a, b = _norm(a), _norm(b)
    return a == b or a.is_relative_to(b) or b.is_relative_to(a)


def prepare_workspace(workspace: Path) -> Path:
    """Resolve, vet and clear the workspace; returns its absolute path.

    Refuses (WorkspaceError, before anything is touched) a folder that would
    overlap the checkout's real data/input or data/output, and a non-empty
    folder this script did not create. Inside its own workspace it removes the
    previous generation - input/, output/ and the state database - so no stale
    call from a larger earlier batch survives into this one."""
    ws = workspace.expanduser().resolve()
    for real in (REPO_ROOT / "data" / "input", REPO_ROOT / "data" / "output"):
        real = real.resolve()
        if any(_overlaps(p, real) for p in (ws, ws / "input", ws / "output")):
            raise WorkspaceError(
                f"{ws} overlaps {real}, where real calls live. A synthetic batch there would "
                f"be averaged into real bankers' reports. Choose another --workspace.")
    marker = ws / MARKER
    if ws.exists():
        if not ws.is_dir():
            raise WorkspaceError(f"{ws} exists and is not a folder.")
        if any(ws.iterdir()) and not marker.is_file():
            raise WorkspaceError(
                f"{ws} is not empty and was not created by this script, so it will not be "
                f"cleared. Choose an empty or new folder with --workspace.")
    targets = [ws / "input", ws / "output"]
    # Every refusal comes before the first deletion, so a refused workspace is
    # left exactly as it was.
    for target in targets:
        if target.is_symlink():
            raise WorkspaceError(f"{target} is a link; refusing to clear what it points to.")
    # The state database first: Windows refuses to delete one another process
    # has open, and a workspace that is in use must not lose its batch.
    for name in ("callqa_state.db", "callqa_state.db-wal", "callqa_state.db-shm",
                 "callqa_state.db-journal"):
        try:
            (ws / name).unlink(missing_ok=True)
        except OSError as exc:
            raise WorkspaceError(
                f"could not remove {ws / name} ({exc}). Close any program using it (the "
                f"dashboard, a running callqa command) and run again.") from exc
    for target in targets:
        if target.exists():
            try:
                shutil.rmtree(target)
            except OSError as exc:
                raise WorkspaceError(
                    f"could not clear {target} ({exc}). Close any program using it (the "
                    f"dashboard, a report open in a viewer) and run again.") from exc
    ws.mkdir(parents=True, exist_ok=True)
    return ws


def _whole(name: str, value: object, low: int, high: int | None = None) -> int:
    """`value` as an int in [low, high], or ValueError naming the argument."""
    if (isinstance(value, bool) or not isinstance(value, numbers.Integral)
            or value < low or (high is not None and value > high)):
        bounds = f"between {low} and {high}" if high is not None else f"of {low} or more"
        raise ValueError(f"{name} must be a whole number {bounds}, not {value!r}")
    return int(value)


def _write(path: Path, model: BaseModel) -> None:
    # LF on every OS, so a batch generated on Windows is byte-identical to Linux.
    path.write_text(model.model_dump_json(indent=2), encoding="utf-8", newline="\n")


def generate_batch(workspace: Path, *, calls: int = 1000, seed: int = 7, bankers: int = 40,
                   weeks: int = 12) -> GenerationSummary:
    """Write a complete synthetic batch under `workspace` and describe it.

    Arguments are checked before the workspace is touched: a value the
    generator cannot use raises ValueError and leaves the previous batch as
    it was, instead of clearing it and then failing inside numpy."""
    calls = _whole("calls", calls, 1, MAX_CALLS)
    bankers = _whole("bankers", bankers, 1, MAX_BANKERS)
    weeks = _whole("weeks", weeks, 1, MAX_WEEKS)
    seed = _whole("seed", seed, 0)
    started = time.perf_counter()
    rubric = load_rubric()
    dims = [d.id for d in rubric.dimensions]
    if sorted(dims) != sorted(KNOWN_DIMENSIONS):
        raise WorkspaceError(
            f"the rubric's dimensions {dims} are not the eight this generator writes text "
            f"for ({', '.join(KNOWN_DIMENSIONS)}); run it against the shipped rubric.")
    ws = prepare_workspace(workspace)
    input_dir, output_dir = ws / "input", ws / "output"

    rng = np.random.default_rng(seed)
    staff = _make_bankers(rng, bankers)
    plan = _plan_calls(rng, calls, staff, weeks)
    _assign_statuses(rng, plan)
    first_sunday, days = business_days(weeks)
    span_days = max(1, (days[-1] - first_sunday).days)
    training_week = round(weeks * 7 / 12) if weeks >= 3 else None
    prompt_sha = hashlib.sha256(
        f"{JUDGE_ENGINE}|{PROMPT_VERSION}|{rubric.sha256}".encode()).hexdigest()

    for sub in ("ingestion", "features", "redacted", "scores", "results", "runs"):
        (output_dir / sub).mkdir(parents=True, exist_ok=True)
    input_dir.mkdir(parents=True, exist_ok=True)
    (ws / MARKER).write_text(json.dumps(
        {"generator": "scripts/generate_batch_demo.py", "calls": calls, "seed": seed,
         "bankers": bankers, "weeks": weeks}, indent=2) + "\n", encoding="utf-8", newline="\n")

    # One weekly batch run, Friday night after the week's last business day:
    # that is how the calls would have been processed, and it gives --run a
    # meaningful unit.
    by_week: dict[int, list[_Call]] = {}
    for call in plan:
        by_week.setdefault(call.week, []).append(call)
    run_ids: list[str] = []
    scored: list[ScoreCard] = []
    for week, members in sorted(by_week.items()):
        friday = first_sunday + timedelta(days=7 * week + 5)
        run_start = datetime(friday.year, friday.month, friday.day, 21, 0, tzinfo=UTC)
        clock = run_start
        summaries: list[CallRunSummary] = []
        for call in members:
            crng = np.random.default_rng([seed, call.index])
            levels, raw = _draw_levels(crng, call, span_days, training_week)
            stage_s = _stage_seconds(crng, call)
            processed = clock + timedelta(seconds=sum(stage_s.values()))
            clock = processed + timedelta(seconds=float(crng.uniform(0.5, 3.0)))
            summaries.append(CallRunSummary(
                call_id=call.call_id, status=call.status, error=call.error,
                seconds=round(sum(stage_s.values()), 3), stage_seconds=stage_s))
            card = _write_call(crng, rubric, call, levels, raw, output_dir,
                               processed.isoformat(timespec="seconds"), stage_s, prompt_sha)
            if card is not None and call.status == "success":
                scored.append(card)
        counts: dict[str, int] = {}
        for s in summaries:
            counts[s.status] = counts.get(s.status, 0) + 1
        gpu_hours = round(sum(s.seconds for s in summaries) / 3600.0, 4)
        run_id = f"{run_start:%Y%m%dT%H%M%SZ}-demo"
        manifest = RunManifest(
            run_id=run_id, command="run", started_at=run_start.isoformat(timespec="seconds"),
            finished_at=clock.isoformat(timespec="seconds"),
            fingerprint={"code_version": CALLQA_VERSION, "git_sha": "none",
                         "python": platform.python_version(), "rubric_sha256": rubric.sha256,
                         "prompt_version": PROMPT_VERSION, "models": [],
                         "judge_engine": JUDGE_ENGINE},
            counts=counts, cost={"gpu_hours": gpu_hours, "rate_per_hour": 0.0, "estimated": 0.0},
            calls=summaries)
        _write(output_dir / "runs" / f"{run_id}.json", manifest)
        run_ids.append(run_id)

    # utf-8-sig, like the sample data: Excel opens it as UTF-8. No banker
    # names: they exist only to be redacted, and these bankers have none.
    with (input_dir / "metadata.csv").open("w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.writer(fh)
        writer.writerow(["call_id", "banker_id", "file_name", "call_date", "call_type",
                         "banker_channel"])
        for call in plan:
            writer.writerow([call.call_id, call.banker.banker_id, f"{call.call_id}.wav",
                             call.call_date.isoformat(), call.call_type,
                             "" if call.mono else "L"])

    status_counts: dict[str, int] = {}
    for call in plan:
        status_counts[call.status] = status_counts.get(call.status, 0) + 1
    return GenerationSummary(
        workspace=ws, input_dir=input_dir, output_dir=output_dir, n_calls=len(plan),
        counts=status_counts, n_bankers=len({c.banker.banker_id for c in plan}),
        n_scored=len(scored), gate_failed=sum(c.gate_failed for c in scored),
        mean_total=round(sum(c.weighted_total for c in scored) / len(scored), 1)
        if scored else None,
        first_date=min(c.call_date for c in plan), last_date=max(c.call_date for c in plan),
        run_ids=run_ids, seconds=round(time.perf_counter() - started, 2),
    )


def _write_call(rng: np.random.Generator, rubric: Rubric, call: _Call, levels: dict[str, int],
                raw: dict[str, float], output_dir: Path, timestamp: str,
                stage_s: dict[str, float], prompt_sha: str) -> ScoreCard | None:
    """Every artifact the stages this call completed would have written."""
    scenarios = SCENARIOS_BY_TYPE[call.call_type]
    scenario = scenarios[int(rng.integers(len(scenarios)))]
    customer_female = bool(rng.random() < 0.5)
    slots = build_slots(rng, scenario, banker_female=call.banker.female,
                        customer_female=customer_female)
    n_fillers = 0 if call.duration < 240 else 1 if call.duration < 480 else 2
    dialog, id_reasoning = compose_dialog(
        rng, scenario, levels, slots, banker_female=call.banker.female,
        customer_female=customer_female, n_fillers=n_fillers)
    dead_air = _timeline(rng, dialog.turns, call.duration, raw["talk_ratio"], raw["patience"],
                         raw["dead_air"], dialog.dead_air_after)

    speech_b = sum(t.end - t.start for t in dialog.turns if t.speaker == "banker")
    speech_c = sum(t.end - t.start for t in dialog.turns if t.speaker == "customer")
    minutes = call.duration / 60.0
    questions = max(0, round(raw["qpm"] * minutes))
    features = Features(
        call_id=call.call_id, talk_ratio=round(speech_b / (speech_b + speech_c), 3),
        overlap_metrics_available=not call.mono,
        longest_banker_monologue_sec=round(raw["monologue"], 1),
        interruptions_by_banker=int(raw["interruptions_by_banker"]),
        interruptions_by_customer=int(raw["interruptions_by_customer"]),
        patience_median_sec=round(raw["patience"], 2),
        banker_question_count=questions,
        banker_questions_per_minute=round(questions / minutes, 3),
        speech_rate_wpm=SpeechRateWPM(banker=round(raw["wpm_banker"], 1),
                                      customer=round(raw["wpm_customer"], 1)),
        dead_air_total_sec=dead_air, call_duration_sec=call.duration,
        banker_speech_sec=round(speech_b, 1), customer_speech_sec=round(speech_c, 1),
    )
    meta = CallMeta(
        call_id=call.call_id, banker_id=call.banker.banker_id, file_name=f"{call.call_id}.wav",
        duration_sec=call.duration, channels=1 if call.mono else 2, sample_rate=16000,
        codec="pcm_s16le", call_date=call.call_date.isoformat(), call_type=call.call_type,
        banker_channel=None if call.mono else "L")
    redacted = RedactedTranscript(
        call_id=call.call_id, engine=JUDGE_ENGINE,
        turns=[RedactedTurn(speaker=t.speaker, start=t.start, end=t.end, text=t.text)
               for t in dialog.turns],
        redaction_counts=dict(sorted(dialog.counts.items())), enabled=True)

    card = None
    if call.has_card:
        card = _scorecard(rng, rubric, call, scenario, levels, dialog, id_reasoning, features,
                          customer_female, timestamp, stage_s.get("judge", 0.0), prompt_sha)
    _write(output_dir / "ingestion" / f"{call.call_id}.json", meta)
    if "redaction" in call.stages:
        _write(output_dir / "redacted" / f"{call.call_id}.json", redacted)
    if "features" in call.stages:
        _write(output_dir / "features" / f"{call.call_id}.json", features)
    score_path = output_dir / "scores" / f"{call.call_id}.json"
    if card is not None:
        _write(score_path, card)
    # report_path stays None: no per-call HTML exists for a synthetic call.
    _write(output_dir / "results" / f"{call.call_id}.json", CallResult(
        call_id=call.call_id, status=call.status, error=call.error, stages_completed=call.stages,
        scorecard_path=str(score_path) if card is not None else None))
    return card


# -- the report --------------------------------------------------------------

def report_command(workspace: Path) -> tuple[list[str], dict[str, str]]:
    """The `callqa executive-report` invocation for a workspace: argv and environment.

    The workspace's folders reach the CLI as the same CALLQA_PATHS__* overrides
    scripts/first_run.py uses, so no config file is edited and the operator's
    own data/ is never the target."""
    ws = workspace.resolve()
    env = dict(os.environ)
    env.update({
        "CALLQA_PATHS__INPUT_DIR": str(ws / "input"),
        "CALLQA_PATHS__OUTPUT_DIR": str(ws / "output"),
        "CALLQA_PATHS__STATE_DB": str(ws / "callqa_state.db"),
        "PYTHONUTF8": "1",
    })
    src = REPO_ROOT / "src"
    if (src / "callqa").is_dir():
        # Works from a checkout before `pip install -e .`, as this script does.
        env["PYTHONPATH"] = os.pathsep.join(p for p in (str(src), env.get("PYTHONPATH")) if p)
    return [sys.executable, "-m", "callqa", "executive-report"], env


def _positive(limit: int) -> Callable[[str], int]:
    def parse(value: str) -> int:
        number = int(value)
        if not 1 <= number <= limit:
            raise argparse.ArgumentTypeError(f"must be between 1 and {limit}")
        return number
    return parse


def _seed(value: str) -> int:
    # numpy seeds are non-negative; a negative one used to raise from inside
    # numpy after the previous batch had already been cleared.
    number = int(value)
    if number < 0:
        raise argparse.ArgumentTypeError("must be 0 or more")
    return number


def _command_line(argv: Sequence[str]) -> str:
    """argv as one line the user can paste into this OS's shell. Windows user
    folders often contain spaces ("C:\\Users\\Dana Levi\\..."), and an
    unquoted path there splits into several arguments."""
    return subprocess.list2cmdline(argv) if os.name == "nt" else shlex.join(argv)


def main(argv: list[str] | None = None) -> int:
    configure_stdio()
    parser = argparse.ArgumentParser(
        description=(__doc__ or "").split("\n\n")[0],
        epilog="Writes into its own workspace only; never into data/input or data/output.")
    parser.add_argument("--calls", type=_positive(MAX_CALLS), default=1000,
                        help="number of calls (default 1000)")
    parser.add_argument("--workspace", type=Path, default=DEFAULT_WORKSPACE,
                        help="folder to write into (default data/demo-batch)")
    parser.add_argument("--seed", type=_seed, default=7,
                        help="random seed, 0 or more (default 7)")
    parser.add_argument("--bankers", type=_positive(MAX_BANKERS), default=40,
                        help="number of bankers (default 40)")
    parser.add_argument("--weeks", type=_positive(MAX_WEEKS), default=12,
                        help="weeks of calls ending 2026-09-17 (default 12)")
    parser.add_argument("--no-report", action="store_true",
                        help="write the batch only; do not build the management report")
    parser.add_argument("--open", action="store_true",
                        help="open the report in the default browser when it is built")
    args = parser.parse_args(argv)

    try:
        summary = generate_batch(args.workspace, calls=args.calls, seed=args.seed,
                                 bankers=args.bankers, weeks=args.weeks)
    except WorkspaceError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    statuses = ", ".join(f"{k} {v}" for k, v in sorted(summary.counts.items()))
    gate_rate = summary.gate_failed / summary.n_scored if summary.n_scored else 0.0
    print(f"Synthetic batch written to {summary.output_dir} in {summary.seconds:.1f}s")
    print(f"  {summary.n_calls} calls ({statuses}), {summary.n_bankers} bankers, "
          f"{summary.first_date:%d.%m.%Y}-{summary.last_date:%d.%m.%Y}, "
          f"{len(summary.run_ids)} weekly runs")
    if summary.mean_total is not None:
        print(f"  scored cohort: {summary.n_scored} calls, mean index {summary.mean_total}, "
              f"gate failures {gate_rate:.1%}")

    cmd, env = report_command(summary.workspace)
    if args.no_report:
        print("\nTo build the management report for this batch, run with these settings:")
        for key in ("CALLQA_PATHS__INPUT_DIR", "CALLQA_PATHS__OUTPUT_DIR",
                    "CALLQA_PATHS__STATE_DB"):
            print(f"  {key}={env[key]}")
        print("  " + _command_line(cmd))
        return 0

    print("\n> python -m callqa executive-report", flush=True)
    code = subprocess.call(cmd, cwd=REPO_ROOT, env=env)
    report = summary.output_dir / "reports" / "executive.html"
    if code != 0 or not report.is_file():
        print("error: the management report could not be built (see the messages above). "
              "The batch itself is complete; --no-report skips this step.", file=sys.stderr)
        return 1
    print(f"\nManagement report: {report}")
    server = REPO_ROOT / "dashboard" / "server.py"
    if server.is_file():  # the dashboard is optional and may have been deleted
        print("In the dashboard: " + _command_line(
            [sys.executable, str(server), "--output-dir", str(summary.output_dir)]))
    if args.open:
        webbrowser.open(report.as_uri())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
