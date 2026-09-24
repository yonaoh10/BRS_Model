"""Evidence the model gives is checked against the line it names.

A quote is accepted when the line exists, was shown to the model, is not
marked ⚠, and the quote is a run of the line's own WORDS - matched word by
word, so "5 ימים" is not found inside "15 ימים" - or a near-quote the judge's
verifier would snap to a real run of words without changing a number or a
negation. What is stored is always the line's own words, never the model's
version of them, and a negation just before the quoted words ("לא", "אינו",
"אי", up to two words back) is kept in front of them.
"""

from __future__ import annotations

from callqa.journey.models import Evidence
from callqa.journey.transcript_view import ContentView
from callqa.judge.validation import _POLARITY_WORDS, _snap_quote_to_turn, normalize_for_match

MIN_QUOTE_CHARS = 8
# One-letter prefixes a Hebrew word takes ("ו", "ה", "ש", ...): a quote may
# start without the prefix its first word carries on the line.
_PREFIXES = "והשבלמכ"
NEGATIONS = frozenset(_POLARITY_WORDS) | {
    "אי", "אינו", "אינה", "איננה", "אינם", "אינן", "איני", "אינני", "מעולם", "בלתי"}


def verify_evidence(view: ContentView, line_no: object, quote: object
                    ) -> tuple[Evidence | None, str]:
    """(evidence with the real words, "") or (None, why it was refused)."""
    if not isinstance(line_no, int) or isinstance(line_no, bool):
        return None, "line חייב להיות מספר שורה"
    line = view.line(line_no)
    if line is None:
        return None, f"אין שורה L{line_no}"
    if view.shown and line_no not in view.shown:
        return None, f"L{line_no} לא הוצגה לך; צטט רק שורות מהטקסט"
    if line.uncertain:
        return None, f"L{line_no} מסומנת ⚠ ואסור לצטט אותה"
    if not isinstance(quote, str) or len(normalize_for_match(quote)) < MIN_QUOTE_CHARS:
        return None, f"הציטוט מ-L{line_no} קצר מדי; צטט כמה מילים שלמות"
    words = line.text.split()
    span = _word_run(quote, words)
    if span is None:
        snapped = _snap_quote_to_turn(quote, line.text)
        span = _word_run(snapped, words) if snapped else None
    if span is None:
        return None, f"הציטוט אינו מופיע ב-L{line_no}; העתק מילים שלמות מהשורה עצמה"
    start, end = span
    start = _with_negation(words, start)
    return Evidence(interaction_id=view.interaction_id, line=line_no,
                    quote=" ".join(words[start:end]), speaker=line.who), ""


def _norm(word: str) -> str:
    return normalize_for_match(word)


def _word_run(quote: str, words: list[str]) -> tuple[int, int] | None:
    """(start, end) of the words of the line that the quote is, word for word
    (punctuation-only tokens ignored on both sides). The first quoted word may
    lack a one-letter prefix the line's word has; anything with a digit must
    match exactly."""
    q = [w for w in (_norm(x) for x in quote.split()) if w]
    line = [(i, n) for i, n in ((i, _norm(x)) for i, x in enumerate(words)) if n]
    if not q:
        return None
    for i in range(len(line) - len(q) + 1):
        window = [n for _i, n in line[i:i + len(q)]]
        if window[1:] != q[1:]:
            continue
        first, want = window[0], q[0]
        if first == want or (not any(ch.isdigit() for ch in first)
                             and len(first) == len(want) + 1 and first[0] in _PREFIXES
                             and first[1:] == want):
            return line[i][0], line[i + len(q) - 1][0] + 1
    return None


def _is_negation(word: str) -> bool:
    w = _norm(word)
    return w in NEGATIONS or (len(w) > 2 and w[0] in "וש" and w[1:] in NEGATIONS)


def _with_negation(words: list[str], start: int) -> int:
    """Move the start back over a negation one or two words before it."""
    for back in (1, 2):
        i = start - back
        if i >= 0 and _is_negation(words[i]):
            return i
    return start
