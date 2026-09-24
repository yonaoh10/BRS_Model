"""Timestamps as bank exports write them, read the same on every machine.

strptime("%b") reads month names in the process locale, and on a Windows
machine set to Hebrew "Aug" is not a month - so month names are mapped here,
by hand. Everything stays naive local time, the time the bank's systems
recorded; nothing here converts time zones.

Accepted:
    02Aug2026 16:04:34      the vendor workbook (also 02AUG2026:16:04:34, SAS DATETIME)
    15Jul2026 11:00:28.557  with milliseconds
    2026-08-02 16:04:34[.123] / 2026-08-02T16:04:34
    02/08/2026 16:04[:34]   day first, also 02.08.2026 and 2.8.2026
    2026-08-02              a date alone is midnight
    a datetime / date object
    a number: an Excel serial day count (1900 or 1904 system), or SAS
              seconds since 1960-01-01 when it is too large to be a day count
"""

from __future__ import annotations

import math
import re
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
_DAY_FIRST = re.compile(r"^(\d{1,2})[/.](\d{1,2})[/.](\d{4})" + _TIME + r"$")


class TimeParseError(ValueError):
    pass


def _clock(groups: tuple) -> tuple[int, int, int, int]:
    hh, mm, ss, frac = groups
    micro = int((frac or "0").ljust(6, "0")[:6])
    return int(hh or 0), int(mm or 0), int(ss or 0), micro


def _build(y: int, mo: int, d: int, clock: tuple[int, int, int, int]) -> datetime:
    try:
        return datetime(y, mo, d, *clock)
    except ValueError as exc:
        raise TimeParseError(str(exc)) from None


def from_number(value: float, *, date1904: bool = False) -> datetime:
    """An Excel serial (days) or, when implausibly large for one, SAS seconds."""
    if not math.isfinite(value):
        raise TimeParseError("not a finite number")
    if value > 1e7:
        # Days since 1900 reach 1e7 only in the year 29,000; seconds since
        # 1960 pass 1e9 in 1991. A number this large is SAS seconds.
        return _SAS_EPOCH + timedelta(seconds=round(value, 3))
    if value <= 0:
        raise TimeParseError("a day count must be positive")
    base = _EXCEL_1904 if date1904 else _EXCEL_1900
    # Rounded to the millisecond: Excel stores 16:04:34 as 0.66984953...
    return base + timedelta(milliseconds=round(value * 86_400_000))


def parse_dt(value: object, *, date1904: bool = False) -> datetime:
    """A timestamp in any of the accepted shapes. Raises TimeParseError."""
    if isinstance(value, datetime):
        return value.replace(tzinfo=None)
    if isinstance(value, date):
        return datetime(value.year, value.month, value.day)
    if isinstance(value, bool):
        raise TimeParseError("a boolean is not a time")
    if isinstance(value, int | float):
        return from_number(float(value), date1904=date1904)
    text = str(value or "").strip()
    if not text:
        raise TimeParseError("empty")
    m = _MON_NAME.match(text)
    if m:
        month = MONTHS.get(m.group(2).lower())
        if month is None:
            raise TimeParseError(f"unknown month name {m.group(2)!r}")
        return _build(int(m.group(3)), month, int(m.group(1)), _clock(m.groups()[3:]))
    m = _ISO.match(text)
    if m:
        return _build(int(m.group(1)), int(m.group(2)), int(m.group(3)), _clock(m.groups()[3:]))
    m = _DAY_FIRST.match(text)
    if m:
        return _build(int(m.group(3)), int(m.group(2)), int(m.group(1)), _clock(m.groups()[3:]))
    try:
        number = float(text)
    except ValueError:
        raise TimeParseError(f"unrecognised time {text[:40]!r}") from None
    return from_number(number, date1904=date1904)


def try_parse_dt(value: object, *, date1904: bool = False) -> datetime | None:
    try:
        return parse_dt(value, date1904=date1904)
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
