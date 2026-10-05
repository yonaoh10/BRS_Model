"""Timestamps as bank exports write them, read the same on every machine.

strptime("%b") reads month names in the process locale, and on a Windows
machine set to Hebrew "Aug" is not a month - so month names are mapped here,
by hand. Everything stays naive local time, the time the bank's systems
recorded; nothing here converts time zones.

Accepted:
    02Aug2026 16:04:34      the vendor workbook (also 02AUG2026:16:04:34, SAS DATETIME)
    15Jul2026 11:00:28.557  with milliseconds
    2026-08-02 16:04:34[.123] / 2026-08-02T16:04:34
    2026-08-02T16:04:34Z / +03:00   a zone suffix is accepted and dropped: the
                            clock reading is kept as the bank's local time
    02/08/2026 16:04[:34]   day first by default, also 02.08.2026 and 2.8.2026;
                            `day_first=False` reads a month-first (US) file, and
                            `detect_day_first` tells which order a column uses
    20260802                eight digits are a yyyymmdd date
    2026-08-02              a date alone is midnight
    a datetime / date object
    a number (a typed cell): an Excel serial day count (1900 or 1904 system),
              or SAS seconds since 1960-01-01 when too large to be a day count
    a numeric string: SAS seconds when large enough; a smaller one is refused
              as ambiguous (an Excel serial? SAS days since 1960?) - the file
              must say which, instead of the reader guessing a decade wrong
"""

from __future__ import annotations

import math
import re
from collections.abc import Iterable
from datetime import date, datetime, timedelta

MONTHS = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
    "jul": 7, "aug": 8, "sep": 9, "sept": 9, "oct": 10, "nov": 11, "dec": 12,
    "january": 1, "february": 2, "march": 3, "april": 4, "june": 6, "july": 7,
    "august": 8, "september": 9, "october": 10, "november": 11, "december": 12,
}

_EXCEL_1900 = datetime(1899, 12, 30)
_EXCEL_1904 = datetime(1904, 1, 1)
_SAS_EPOCH = datetime(1960, 1, 1)

_TIME = r"(?:[ T:]+(\d{1,2}):(\d{2})(?::(\d{2})(?:[.,](\d{1,6}))?)?)?"
_MON_NAME = re.compile(r"^(\d{1,2})[-\s]?([A-Za-z]{3,9})[-\s]?(\d{4})" + _TIME + r"$")
_ISO = re.compile(r"^(\d{4})-(\d{1,2})-(\d{1,2})" + _TIME + r"$")
_SLASHED = re.compile(r"^(\d{1,2})[/.](\d{1,2})[/.](\d{4})" + _TIME + r"$")
_YMD8 = re.compile(r"^(\d{4})(\d{2})(\d{2})$")
_TZ_SUFFIX = re.compile(r"(?:Z|[+-]\d{2}:?\d{2})$")
# Below this, a numeric string could be an Excel day count or SAS days since
# 1960 (24300 is 2026 as the one and 1966 as the other); above it, only SAS
# seconds make a date in living memory.
_NUMERIC_TEXT_FLOOR = 1e7


class TimeParseError(ValueError):
    pass


def tz_suffix(value: object) -> bool:
    """Whether a text timestamp carries a zone suffix (Z, +03:00, -0500)."""
    text = str(value or "").strip()
    return bool(text) and bool(_TZ_SUFFIX.search(text)) and bool(
        _ISO.match(_TZ_SUFFIX.sub("", text)))


def detect_day_first(values: Iterable[object]) -> bool | None:
    """Which order the slashed dates of one column use. True = day first,
    False = month first, None = nothing decides it: either no slashed date is
    there, or every one has both fields at 12 or below (ambiguous - the file
    must say). The two orders can't both be present; when they are, the
    column is inconsistent and None is returned too."""
    day_first = month_first = False
    for value in values:
        if isinstance(value, datetime | date | int | float | bool):
            continue
        m = _SLASHED.match(str(value or "").strip())
        if not m:
            continue
        a, b = int(m.group(1)), int(m.group(2))
        if a > 12:
            day_first = True
        if b > 12:
            month_first = True
    if day_first == month_first:
        return None
    return day_first


def has_slashed_dates(values: Iterable[object]) -> bool:
    return any(_SLASHED.match(str(v or "").strip())
               for v in values if not isinstance(v, datetime | date | int | float | bool))


def _clock(groups: tuple) -> tuple[int, int, int, int]:
    hh, mm, ss, frac = groups
    micro = int((frac or "0").ljust(6, "0")[:6])
    return int(hh or 0), int(mm or 0), int(ss or 0), micro


def _build(y: int, mo: int, d: int, clock: tuple[int, int, int, int]) -> datetime:
    try:
        return datetime(y, mo, d, *clock)
    except ValueError as exc:
        raise TimeParseError(f"bad_date: {exc}") from None


def from_number(value: float, *, date1904: bool = False) -> datetime:
    """An Excel serial (days) or, when implausibly large for one, SAS seconds."""
    if not math.isfinite(value):
        raise TimeParseError("bad_date: not a finite number")
    if value > _NUMERIC_TEXT_FLOOR:
        # Days since 1900 reach 1e7 only in the year 29,000; seconds since
        # 1960 pass 1e9 in 1991. A number this large is SAS seconds.
        return _SAS_EPOCH + timedelta(seconds=round(value, 3))
    if value <= 0:
        raise TimeParseError("bad_date: a day count must be positive")
    base = _EXCEL_1904 if date1904 else _EXCEL_1900
    # Rounded to the millisecond: Excel stores 16:04:34 as 0.66984953...
    return base + timedelta(milliseconds=round(value * 86_400_000))


def parse_dt(value: object, *, date1904: bool = False, day_first: bool = True) -> datetime:
    """A timestamp in any of the accepted shapes. Raises TimeParseError, whose
    message starts with a stable code (ambiguous_numeric_date, bad_date,
    unknown_month, unrecognised, empty) for the import report."""
    if isinstance(value, datetime):
        return value.replace(tzinfo=None)
    if isinstance(value, date):
        return datetime(value.year, value.month, value.day)
    if isinstance(value, bool):
        raise TimeParseError("bad_date: a boolean is not a time")
    if isinstance(value, int | float):
        return from_number(float(value), date1904=date1904)
    text = str(value or "").strip()
    if not text:
        raise TimeParseError("empty")
    m = _MON_NAME.match(text)
    if m:
        month = MONTHS.get(m.group(2).lower())
        if month is None:
            raise TimeParseError(f"unknown_month: {m.group(2)!r}")
        return _build(int(m.group(3)), month, int(m.group(1)), _clock(m.groups()[3:]))
    m = _ISO.match(_TZ_SUFFIX.sub("", text))
    if m:
        return _build(int(m.group(1)), int(m.group(2)), int(m.group(3)), _clock(m.groups()[3:]))
    m = _SLASHED.match(text)
    if m:
        a, b = int(m.group(1)), int(m.group(2))
        d, mo = (a, b) if day_first else (b, a)
        return _build(int(m.group(3)), mo, d, _clock(m.groups()[3:]))
    m = _YMD8.match(text)
    if m:
        return _build(int(m.group(1)), int(m.group(2)), int(m.group(3)), (0, 0, 0, 0))
    try:
        number = float(text)
    except ValueError:
        raise TimeParseError(f"unrecognised: {text[:40]!r}") from None
    if 0 < number < _NUMERIC_TEXT_FLOOR:
        raise TimeParseError(f"ambiguous_numeric_date: {text[:20]!r} could be an Excel day "
                             "count or SAS days since 1960; the file must say which")
    return from_number(number, date1904=date1904)


def try_parse_dt(value: object, *, date1904: bool = False,
                 day_first: bool = True) -> datetime | None:
    try:
        return parse_dt(value, date1904=date1904, day_first=day_first)
    except TimeParseError:
        return None


def iso(value: datetime | None) -> str:
    return value.isoformat(timespec="seconds") if value else ""


# Sunday..Thursday are business days in Israel (Friday and Saturday are not).
_WEEKEND = {4, 5}  # datetime.weekday(): Monday=0 ... Friday=4, Saturday=5


def add_business_days(start: datetime, days: int, holidays: frozenset[date] = frozenset()
                      ) -> datetime:
    """`days` Sunday-to-Thursday working days after `start`, at the same clock
    time; holidays are skipped too."""
    moment = start
    left = days
    while left > 0:
        moment += timedelta(days=1)
        if moment.weekday() not in _WEEKEND and moment.date() not in holidays:
            left -= 1
    return moment
