"""One way to print a number, everywhere in the management report.

Half-up on the shortest decimal representation (what a reader expects: 74.25
shows as 74.3, where Python's round gives 74.2), a true minus sign, never
"-0". Charts, tables, KPI tiles and the written opinion all go through here,
so the same value never appears as 51.2 in one place and 51.3 in another.
"""

from __future__ import annotations

import math
from decimal import ROUND_HALF_UP, Decimal, InvalidOperation

MINUS = "−"


def _quantize(value: float, places: int) -> Decimal | None:
    try:
        return Decimal(repr(float(value))).quantize(Decimal(1).scaleb(-places),
                                                    rounding=ROUND_HALF_UP)
    except (InvalidOperation, ValueError, TypeError):
        return None


def fmt(value: float | None, digits: int = 1) -> str:
    """At most `digits` decimals: 74.0 -> '74', 74.25 -> '74.3', -0.04 -> '0'."""
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return "—"
    q = _quantize(value, max(0, digits))
    if q is None:
        return "—"
    if q.is_zero():
        return "0"
    text = f"{abs(q):,.{max(0, digits)}f}"
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return (MINUS if q < 0 else "") + text


def fixed(value: float | None, places: int = 2) -> str:
    """Exactly `places` decimals (for 1-5 means: 3.98, 4.00)."""
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return "—"
    q = _quantize(value, max(0, places))
    if q is None:
        return "—"
    if q.is_zero():
        q = abs(q)
    return (MINUS if q < 0 else "") + f"{abs(q):,.{max(0, places)}f}"


def signed(value: float | None, digits: int = 1) -> str:
    """'+2.1' / '−3.4' / '0'."""
    text = fmt(value, digits)
    if text in ("—", "0") or text.startswith(MINUS):
        return text
    return "+" + text


def percent(p: float | None, digits: int = 1) -> str:
    """A 0..1 share as '14.8%'."""
    return "—" if p is None else fmt(p * 100, digits) + "%"
