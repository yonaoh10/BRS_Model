"""Evidence the model gives is checked against the line it names.

A quote is accepted when the line exists, is not marked ⚠, and the quote is
on it - verbatim after normalisation, or as a near-quote that the existing
judge verifier would snap to the real words (never across lines, never
changing a number or a negation). What is stored is always the line's own
words, not the model's version of them.
"""

from __future__ import annotations

from callqa.journey.models import Evidence
from callqa.journey.transcript_view import ContentView
from callqa.judge.validation import _POLARITY_WORDS, _snap_quote_to_turn, normalize_for_match

MIN_QUOTE_CHARS = 4


def verify_evidence(view: ContentView, line_no: object, quote: object
                    ) -> tuple[Evidence | None, str]:
    """(evidence with the real words, "") or (None, why it was refused)."""
    if not isinstance(line_no, int) or isinstance(line_no, bool):
        return None, "line חייב להיות מספר שורה"
    line = view.line(line_no)
    if line is None:
        return None, f"אין שורה L{line_no}"
    if line.uncertain:
        return None, f"L{line_no} מסומנת ⚠ ואסור לצטט אותה"
    if not isinstance(quote, str) or len(normalize_for_match(quote)) < MIN_QUOTE_CHARS:
        return None, f"הציטוט מ-L{line_no} קצר מדי"
    target = normalize_for_match(quote)
    text = line.text
    if target in normalize_for_match(text):
        real = _span_of(quote, text) or quote.strip()
    else:
        real = _snap_quote_to_turn(quote, text)
        if real is None:
            return None, f"הציטוט אינו מופיע ב-L{line_no}; העתק מילים מהשורה עצמה"
    return Evidence(interaction_id=view.interaction_id, line=line_no,
                    quote=_keep_negation(real, text), speaker=line.who), ""


def _is_polarity(word: str) -> bool:
    w = normalize_for_match(word)
    return w in _POLARITY_WORDS or (len(w) > 2 and w[0] in "וש" and w[1:] in _POLARITY_WORDS)


def _keep_negation(span: str, text: str) -> str:
    """A quote that starts right after "לא" (or "ולא", "אין", ...) says the
    opposite of the line: "אישרו לי אותה" out of "ולא אישרו לי אותה". The
    negation is put back in front of it."""
    words, part = text.split(), span.split()
    if not part:
        return span
    for i in range(len(words) - len(part) + 1):
        if words[i:i + len(part)] == part:
            if i > 0 and _is_polarity(words[i - 1]):
                return " ".join(words[i - 1:i + len(part)])
            return span
    return span


def _span_of(quote: str, text: str) -> str | None:
    """The line's own words that match the quote (the quote may differ in
    punctuation or spacing)."""
    words = text.split()
    target = normalize_for_match(quote)
    n = len(quote.split())
    for size in range(max(1, n - 1), n + 2):
        for start in range(0, max(0, len(words) - size) + 1):
            span = " ".join(words[start:start + size])
            if normalize_for_match(span) == target:
                return span
    return None
