"""PII redaction (stage 5), Hebrew-aware.

The project's hardest rule: raw customer identifiers must not survive past
this stage. Everything downstream - the judge prompt, the HTML reports, the
aggregate views, the logs - sees only what comes out of here.

HOW DETECTION WORKS, and why it is not a list of regexes over the raw text.

Speech is transcribed, not typed, so an identifier arrives in whatever shape
the speaker and the ASR engine produced it: "123456782", "123-456-782",
"123 456 782", digits split across two transcript segments because the speaker
paused, or digits carrying an invisible bidirectional control character that
Hebrew text routinely picks up. A contiguous-digits pattern misses every one
of those, and a miss here is a leak.

So detection runs in three steps:

1. **Normalise.** Invisible formatting characters are dropped and Arabic-Indic
   and full-width digits are folded to ASCII, keeping an index map back to the
   original text so the output is the original minus the masked spans.
2. **Find digit runs**, allowing separators inside them, over the WHOLE dialog
   rather than one turn at a time - which is what catches a number split
   across a pause.
3. **Classify each run by its digits**, with a checksum where one exists.
   A checksum is what makes separator tolerance safe: only one in ten
   nine-digit sequences is a valid Israeli ID, so accepting spaces inside the
   run costs very little and catches a very common way of saying a number.

Where no checksum exists the bias is privacy: any run of six or more
contiguous digits is masked. The one exception is an amount, recognised by
the currency word that follows it, because masking every price in the call
would blind the compliance and clarity scoring.

Replacement format: <סוג:████> e.g. <ת"ז:████>. Turn timestamps are untouched,
so evidence-quote timestamps stay valid.
"""

from __future__ import annotations

import logging
import re
import unicodedata
from dataclasses import dataclass
from typing import Protocol

from callqa.config import RedactionConfig
from callqa.models import DialogTranscript, RedactedTranscript, RedactedTurn

logger = logging.getLogger(__name__)

# Entity type -> Hebrew label used in the replacement token.
ENTITY_LABELS_HE = {
    "ISRAELI_ID": 'ת"ז',
    "PHONE": "טלפון",
    "CREDIT_CARD": "כרטיס אשראי",
    "ACCOUNT_LIKE": "חשבון",
    "IBAN": "חשבון",
    "EMAIL": 'דוא"ל',
    "DOB": "תאריך לידה",
    "PERSON": "שם",
}

MASK = "████"


def replacement_token(entity_type: str) -> str:
    return f"<{ENTITY_LABELS_HE.get(entity_type, entity_type)}:{MASK}>"


# -- validators --------------------------------------------------------------

def is_valid_israeli_id(value: str) -> bool:
    """Israeli ID (teudat zehut) checksum.

    Left-pad to 9 digits; multiply digits alternately x1, x2; if a product
    exceeds 9, sum its digits (equivalent to product - 9); valid iff the
    total is divisible by 10.
    """
    digits = re.sub(r"\D", "", value)
    if not 8 <= len(digits) <= 9:
        return False
    digits = digits.zfill(9)
    total = 0
    for i, ch in enumerate(digits):
        product = int(ch) * (1 if i % 2 == 0 else 2)
        if product > 9:
            product -= 9
        total += product
    return total % 10 == 0


def is_valid_luhn(value: str) -> bool:
    digits = re.sub(r"\D", "", value)
    if not 13 <= len(digits) <= 19:
        return False
    total = 0
    for i, ch in enumerate(reversed(digits)):
        d = int(ch)
        if i % 2 == 1:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return total % 10 == 0


LANDLINE_AREA = set("234689")


def is_israeli_phone(digits: str) -> bool:
    """Phone shape from digits alone, so separators cannot hide it."""
    if digits.startswith("972"):
        digits = "0" + digits[3:].lstrip("0")
    if len(digits) == 10 and digits[0] == "0" and digits[1] in "57":
        return True                      # mobile 05X, and 07X virtual ranges
    return len(digits) == 9 and digits[0] == "0" and digits[1] in LANDLINE_AREA


# -- normalisation -----------------------------------------------------------

# Bidi controls, zero-width characters and the soft hyphen. Hebrew text picks
# these up constantly, and one of them inside a digit run defeats any pattern
# written against the raw string.
INVISIBLE = frozenset(
    "­​‌‍‎‏⁠﻿"
    "‪‫‬‭‮⁦⁧⁨⁩"
)


def _fold_digit(ch: str) -> str:
    """Arabic-Indic, extended Arabic-Indic and full-width digits -> ASCII."""
    if ch.isdigit() and not ch.isascii():
        try:
            return str(unicodedata.digit(ch))
        except (TypeError, ValueError):      # pragma: no cover - defensive
            return ch
    return ch


# Unicode categories dropped before detection: format controls (Cf, e.g.
# bidi/zero-width), non-spacing marks (Mn, e.g. Hebrew niqqud), enclosing marks
# (Me). A single such codepoint inside a digit run - realistic from an ASR or a
# keyboard emitting niqqud - would otherwise split the run into two fragments
# below the masking threshold and leak the whole identifier. Detection runs on
# the stripped text; masks map back through `index` to the original, so the
# stripped char is still covered by (masked within) the surrounding span.
_STRIP_CATEGORIES = frozenset({"Cf", "Mn", "Me"})


def normalize_for_detection(text: str) -> tuple[str, list[int]]:
    """Return (normalised text, index map back into the original string)."""
    chars: list[str] = []
    index: list[int] = []
    for i, ch in enumerate(text):
        if ch in INVISIBLE or unicodedata.category(ch) in _STRIP_CATEGORIES:
            continue
        chars.append(_fold_digit(ch))
        index.append(i)
    return "".join(chars), index


# -- Hebrew number-word folding (Pass B) -------------------------------------

# Customers dictate identifiers digit by digit, and the real ivrit.ai ASR
# writes those digits as Hebrew WORDS ("ארבע חמש שמונה אפס"): no digit regex
# can ever see them. This pass folds runs of dictation words to digits between
# normalisation and detection, carrying an index map back so masking always
# lands on the original words. Only the words for 0-9 fold - quantity words
# (עשרים, מאות, אלף) stay text, so amounts in words are untouched.
DIGIT_WORDS = {
    "אפס": "0",
    "אחת": "1", "אחד": "1",
    "שתיים": "2", "שניים": "2", "שתי": "2", "שני": "2",
    "שלוש": "3", "שלושה": "3",
    "ארבע": "4", "ארבעה": "4",
    "חמש": "5", "חמישה": "5",
    "שש": "6", "שישה": "6",
    "שבע": "7", "שבעה": "7",
    "שמונה": "8",
    "תשע": "9", "תשעה": "9",
}
# Fewer than four consecutive digit-words is ordinary speech ("שתי דקות",
# "שלוש ארבע פעמים"); four or more is dictation. Card last-four is exactly
# four, so the threshold sits at the shortest identifier fragment that matters.
MIN_DIGIT_WORD_RUN = 4

_WORD_TOKEN_RE = re.compile(r"[֐-׿]+|\S+")
_TRAILING_PUNCT = ".,?!:;"
# A comma is a dictation pause ("שמונה, אפס, ארבע, וחמש") and stays inside
# the run; sentence-terminal punctuation ends it, so an unrelated number in
# the next sentence never glues onto this one.
_RUN_ENDING_PUNCT = ".?!:;"


def _word_digit(token: str) -> str | None:
    core = token.strip(_TRAILING_PUNCT + "\"'")
    if core in DIGIT_WORDS:
        return DIGIT_WORDS[core]
    # The conjunction prefix ו- ("שמונה, אפס, וחמש") is part of dictation
    # cadence; the definite article ה- ("הארבע") is not.
    if len(core) > 1 and core[0] == "ו" and core[1:] in DIGIT_WORDS:
        return DIGIT_WORDS[core[1:]]
    return None


def fold_number_words(text: str) -> tuple[str, list[tuple[int, int]]]:
    """Fold runs of >=MIN_DIGIT_WORD_RUN Hebrew digit-words to digit strings.

    Returns (folded_text, spans) where spans[i] is the (start, end) range in
    `text` that folded character i came from. Every digit produced by a fold
    maps to the WHOLE word run, so masking any part of the folded number
    removes all of the words it came from - privacy bias over precision.
    """
    tokens = [(m.group(), m.start(), m.end()) for m in _WORD_TOKEN_RE.finditer(text)]
    folded: list[str] = []
    spans: list[tuple[int, int]] = []
    cursor = 0

    def passthrough(upto: int) -> None:
        nonlocal cursor
        for k in range(cursor, upto):
            folded.append(text[k])
            spans.append((k, k + 1))
        cursor = upto

    i, n = 0, len(tokens)
    while i < n:
        if _word_digit(tokens[i][0]) is None:
            i += 1
            continue
        j = i
        last_word = i
        digits: list[str] = []
        while j < n:
            d = _word_digit(tokens[j][0])
            if d is not None:
                digits.append(d)
                last_word = j
                ended = tokens[j][0] != tokens[j][0].rstrip(_RUN_ENDING_PUNCT)
                j += 1
                if ended:
                    break
                continue
            # The tokenizer splits "שמונה," into a word and a bare comma;
            # a comma between two digit words is a dictation pause.
            if (digits and tokens[j][0] == ","
                    and j + 1 < n and _word_digit(tokens[j + 1][0]) is not None):
                j += 1
                continue
            break
        if len(digits) >= MIN_DIGIT_WORD_RUN:
            run_start, run_end = tokens[i][1], tokens[last_word][2]
            passthrough(run_start)
            for d in digits:
                folded.append(d)
                spans.append((run_start, run_end))
            cursor = run_end
        i = max(j, i + 1)
    passthrough(len(text))
    return "".join(folded), spans


# -- patterns ----------------------------------------------------------------

# A run of digits that may carry separators: spaces (including a newline,
# which is how a number split across two transcript turns appears once the
# dialog is joined), hyphens, dots and the various dashes an ASR may emit.
SEPARATORS = " \t\n -.‐‑‒–—/"
# Up to TWO separator chars between digit groups: the real ivrit.ai ASR
# writes dictated numbers as "314 -15992 -6265" (space before the hyphen),
# and with only one separator allowed the run split into fragments below the
# masking threshold - a full national ID left redaction unmasked on the
# first real recording.
DIGIT_RUN_RE = re.compile(rf"\d(?:[{re.escape(SEPARATORS)}]{{0,2}}\d)*")
# Separators that a real account or card number can contain. Spaces and
# newlines are excluded here so that "500 300 שקל" is not read as one number,
# while a checksum-backed identifier is still allowed to contain them.
STRUCTURAL_SEPARATORS = frozenset("-.‐‑‒–—/")

SHORT_CODE_RE = re.compile(r"\*\d{3,5}(?!\d)")
EMAIL_RE = re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}")
IBAN_RE = re.compile(r"(?<![A-Za-z0-9])IL\d{2}(?:[ \-]?[A-Za-z0-9]){19}(?![A-Za-z0-9])")
DATE_RE = re.compile(r"(?<!\d)\d{1,2}[./\-]\d{1,2}[./\-]\d{2,4}(?!\d)")
# "שם האם" / "שם האם שלך הוא" followed by the name itself.
MOTHER_NAME_RE = re.compile(
    r"שם\s+ה?(?:אמא|אימא|אם)\s*(?:שלך\s*)?"
    r"(?:הוא\s+|היא\s+|:\s*)([\u0590-\u05FF]{2,})"
)

# Kept for backward compatibility and for the presidio recognizers; detection
# no longer relies on them alone.
PHONE_RE = re.compile(r"(?<!\d)(?:\+972[-\s]?0?|0)(?:5\d|7\d|[23489])(?:[-\s]?\d){7}(?!\d)")
ISRAELI_ID_RE = re.compile(r"(?<!\d)\d{8,9}(?!\d)")
CREDIT_CARD_RE = re.compile(r"(?<!\d)(?:\d[-\s]?){12,18}\d(?!\d)")
ACCOUNT_LIKE_RE = re.compile(r"(?<!\d)\d{6,}(?!\d)")

# Context windows, in characters, searched before a number.
CONTEXT_BEFORE = 40
ID_CONTEXT = ("תעודת זהות", "תעודת הזהות", "מספר זהות", "ת.ז", 'ת"ז', "תז ")
PHONE_CONTEXT = ("טלפון", "נייד", "סלולרי", "פלאפון", "לחזור אלי")
ACCOUNT_CONTEXT = ("חשבון", "סניף", "מספר לקוח", "כרטיס", "הלוואה", "פיקדון", "תיק")
BIRTH_CONTEXT = ("לידה", "נולד", "נולדת", "נולדתי")
# An amount is not an account number; masking every price would blind the
# compliance and clarity dimensions, which are scored on what was quoted.
AMOUNT_AFTER = ("שקל", "שקלים", 'ש"ח', "ש״ח", "₪", "אגורות", "דולר", "יורו", "אלף", "מיליון")
AMOUNT_WINDOW = 14
# Phrases that explicitly announce "the number that follows is an IDENTIFIER,
# not a price". When one of these precedes the run, the amount exception below
# is overridden: "מספר החשבון 761534892 שקל" is an account number spoken next
# to a currency word, not a 761-million-shekel amount. Bare "חשבון" is NOT in
# this list - "יש בחשבון 120000 שקל" is a balance, a real amount - so only the
# unambiguous "number of ..." / "ends in ..." phrasings force masking.
STRONG_ID_PHRASES = (
    "מספר החשבון", "מספר חשבון", "חשבון מספר", "מספר הלקוח", "מספר לקוח",
    "מספר כרטיס", "מספר הכרטיס", "מספר תעודת", "תעודת זהות", "תעודת הזהות",
    "מספר זהות", "מספר הזהות", "מספר סניף", 'ת"ז', "ת.ז",
    "מסתיים ב", "שמסתיים ב", "המסתיים ב", "הספרות האחרונות",
)


@dataclass
class PIIMatch:
    entity_type: str
    start: int
    end: int


def _context_before(text: str, start: int) -> str:
    return text[max(0, start - CONTEXT_BEFORE):start]


def _looks_like_amount(text: str, end: int) -> bool:
    window = text[end:end + AMOUNT_WINDOW]
    return any(word in window for word in AMOUNT_AFTER)


def _has_strong_id_phrase(before: str) -> bool:
    return any(phrase in before for phrase in STRONG_ID_PHRASES)


def _classify_run(text: str, start: int, end: int) -> str | None:
    """Decide what a digit run is, from its digits and its surroundings."""
    raw = text[start:end]
    digits = re.sub(r"\D", "", raw)
    # A run is "structural" when every gap between digit groups contains at
    # least one structural character. Two amounts joined by a bare space
    # ("500 300 שקל") stay two numbers, but the real ASR's dictation style
    # "314 -15992 -6265" (space before each hyphen) is one identifier - the
    # old separators-as-a-set test called any run containing a space
    # non-structural and let a complete national ID through unmasked.
    gaps = [g for g in re.split(r"\d+", raw) if g]
    structural_only = all(
        any(ch in STRUCTURAL_SEPARATORS for ch in gap) for gap in gaps
    )

    if len(digits) < 4:
        return None

    before = _context_before(text, start)
    phone_hint = any(word in before for word in PHONE_CONTEXT)
    id_hint = any(word in before for word in ID_CONTEXT)

    # Checksum-backed identifiers first: they are the ones separator tolerance
    # is safe for, and the ones that matter most.
    if is_valid_israeli_id(digits) and len(digits) == 9 and id_hint:
        return "ISRAELI_ID"
    if is_israeli_phone(digits) and not id_hint:
        return "PHONE"
    if is_valid_israeli_id(digits):
        return "ISRAELI_ID"
    if is_valid_luhn(digits):
        return "CREDIT_CARD"
    if phone_hint and 7 <= len(digits) <= 12:
        return "PHONE"

    # The amount exception (a currency word right after the run -> a price, not
    # an identifier). Two escapes restore masking for identifiers spoken near
    # money: a 9+ digit run is never a retail amount (it is an ID/account/card),
    # and an explicit identifier phrase ("מספר החשבון", "מסתיים ב", ...) before
    # it overrides the currency word. Without either, a <=8-digit run before a
    # currency word is treated as an amount so balances/fees stay scoreable.
    if (_looks_like_amount(text, end)
            and len(digits) < 9
            and not _has_strong_id_phrase(before)):
        return None
    if len(digits) >= 6 and structural_only:
        return "ACCOUNT_LIKE"
    if 4 <= len(digits) <= 5 and any(word in before for word in ACCOUNT_CONTEXT):
        return "ACCOUNT_LIKE"
    # Dictation context lifts the structural-separator requirement: a customer
    # reading out an identifier pauses between groups, which the ASR writes as
    # bare spaces ("926 9265") or which number-word folding produces directly.
    # An ID/account word before the run says it is an identifier, not a list
    # of quantities; the amount check above has already had its turn.
    if 6 <= len(digits) <= 20 and (
        id_hint or any(word in before for word in ACCOUNT_CONTEXT)
    ):
        return "ISRAELI_ID" if id_hint else "ACCOUNT_LIKE"
    return None


def _spans_in_normalized(text: str) -> list[PIIMatch]:
    """All PII spans, in coordinates of the NORMALISED text."""
    found: list[PIIMatch] = []

    def add(entity_type: str, start: int, end: int) -> None:
        if any(start < f.end and f.start < end for f in found):
            return                     # an earlier, higher-priority match wins
        found.append(PIIMatch(entity_type, start, end))

    for m in EMAIL_RE.finditer(text):
        add("EMAIL", m.start(), m.end())
    for m in IBAN_RE.finditer(text):
        add("IBAN", m.start(), m.end())
    for m in SHORT_CODE_RE.finditer(text):
        add("PHONE", m.start(), m.end())
    for m in MOTHER_NAME_RE.finditer(text):
        add("PERSON", m.start(1), m.end(1))
    for m in DATE_RE.finditer(text):
        window = _context_before(text, m.start()) + text[m.end():m.end() + CONTEXT_BEFORE]
        if any(word in window for word in BIRTH_CONTEXT):
            add("DOB", m.start(), m.end())
    for m in DIGIT_RUN_RE.finditer(text):
        # Trim separators that the run may have swallowed at either edge.
        start, end = m.start(), m.end()
        entity = _classify_run(text, start, end)
        if entity:
            add(entity, start, end)

    found.sort(key=lambda f: f.start)
    return found


def _repeated_fragments(text: str, spans: list[PIIMatch]) -> list[PIIMatch]:
    """Short digit runs that read back the TAIL of an already-masked identifier.

    A banker reads back "the last four digits, 9265" after the customer
    dictated the full number: masking the full number while leaving its tail in
    the clear undoes the mask. The match is a SUFFIX, not any substring - "ends
    in 9265" means the last digits - and a run that looks like an amount
    ("...1534 שקל") is never swallowed, so a price that merely shares a few
    digits with a masked identifier stays scoreable. Substring matching (the
    earlier version) over-masked amounts and incidental years/dates.
    """
    masked_digits = [re.sub(r"\D", "", text[s.start:s.end])
                     for s in spans if s.end - s.start > 0]
    long_masked = [d for d in masked_digits if len(d) >= 6]
    if not long_masked:
        return []
    extra: list[PIIMatch] = []
    for m in DIGIT_RUN_RE.finditer(text):
        if any(m.start() < s.end and s.start < m.end() for s in spans):
            continue
        if _looks_like_amount(text, m.end()):
            continue
        digits = re.sub(r"\D", "", m.group())
        if 4 <= len(digits) <= 5 and any(d.endswith(digits) for d in long_masked):
            extra.append(PIIMatch("ACCOUNT_LIKE", m.start(), m.end()))
    return extra


def find_pii(text: str) -> list[PIIMatch]:
    """Find PII spans in `text`, in the coordinates of `text` itself."""
    normalized, index = normalize_for_detection(text)
    folded, fold_spans = fold_number_words(normalized)
    spans = _spans_in_normalized(folded)
    spans.extend(_repeated_fragments(folded, spans))
    spans.sort(key=lambda s: s.start)
    out: list[PIIMatch] = []
    for span in spans:
        if span.start >= len(fold_spans) or span.end - 1 >= len(fold_spans):
            continue                                  # pragma: no cover
        # folded coords -> normalized coords -> original coords
        n_start = fold_spans[span.start][0]
        n_end = fold_spans[span.end - 1][1]
        if n_start >= len(index) or n_end - 1 >= len(index):
            continue                                  # pragma: no cover
        out.append(PIIMatch(span.entity_type, index[n_start], index[n_end - 1] + 1))
    return out


# -- name redaction ----------------------------------------------------------

# A name must match as a whole word. Substring replacement turned "אור" into a
# mask inside "האורח" and "באורך", which corrupts the transcript the judge
# reads and the reviewer trusts.
def _name_pattern(name: str) -> re.Pattern[str] | None:
    parts = [re.escape(p) for p in name.split() if p]
    if not parts:
        return None
    body = r"\s+".join(parts)
    # Hebrew has no case and no ASCII word boundary that behaves here, so the
    # boundary is "not a letter or digit on either side".
    return re.compile(rf"(?<![\w֐-׿]){body}(?![\w֐-׿])")


def redact_names(text: str, names: list[str]) -> tuple[str, int]:
    count = 0
    for name in names:
        cleaned = " ".join((name or "").split())
        if len(cleaned) < 2:
            continue
        pattern = _name_pattern(cleaned)
        if pattern is None:
            continue
        text, n = pattern.subn(replacement_token("PERSON"), text)
        count += n
    return text, count


def _apply(text: str, matches: list[PIIMatch]) -> tuple[str, dict[str, int]]:
    counts: dict[str, int] = {}
    out: list[str] = []
    cursor = 0
    for m in sorted(matches, key=lambda x: x.start):
        if m.start < cursor:
            continue
        out.append(text[cursor:m.start])
        out.append(replacement_token(m.entity_type))
        counts[m.entity_type] = counts.get(m.entity_type, 0) + 1
        cursor = m.end
    out.append(text[cursor:])
    return "".join(out), counts


def redact_text(text: str, extra_names: list[str] | None = None) -> tuple[str, dict[str, int]]:
    """Redact one string; returns (redacted_text, counts_by_entity_type)."""
    redacted, counts = _apply(text, find_pii(text))
    redacted, n_names = redact_names(redacted, extra_names or [])
    if n_names:
        counts["PERSON"] = counts.get("PERSON", 0) + n_names
    return redacted, counts


def sanitize_error(value: object, limit: int = 500) -> str:
    """Make an error message safe to log, store and show.

    Exceptions raised before or during redaction routinely carry transcript
    text - a pydantic ValidationError prints the offending value, a judge
    rejection quotes the model's output. Those messages are written to
    results/<call>.json and to the log, which are exactly the places raw PII
    must never reach.
    """
    text = str(value)
    redacted, _ = redact_text(text)
    if len(redacted) > limit:
        redacted = redacted[:limit] + "...[truncated]"
    return redacted


# -- engines -----------------------------------------------------------------

class Redactor(Protocol):
    name: str

    def redact_dialog(
        self, dialog: DialogTranscript, extra_names: list[str] | None = None
    ) -> RedactedTranscript: ...


TURN_SEPARATOR = "\n"


class RegexRedactor:
    """Dependency-free redactor implementing all mandatory recognizers."""

    name = "regex"

    def __init__(self, config: RedactionConfig | None = None) -> None:
        self.config = config or RedactionConfig()

    def redact_dialog(
        self, dialog: DialogTranscript, extra_names: list[str] | None = None
    ) -> RedactedTranscript:
        if not self.config.enabled:
            # Deliberately loud. A silent pass-through here would send raw
            # customer identifiers to the judge, the reports and the logs while
            # every artifact still claimed to be redacted.
            logger.error(
                "REDACTION IS DISABLED: call_id=%s is being processed with RAW "
                "transcript text. This must never be used on real recordings.",
                dialog.call_id,
            )
            return RedactedTranscript(
                call_id=dialog.call_id, engine=f"{self.name}:DISABLED",
                turns=[RedactedTurn(speaker=t.speaker, start=t.start, end=t.end, text=t.text)
                       for t in dialog.turns],
                redaction_counts={}, enabled=False,
            )
        return self._redact(dialog, extra_names or [])

    def _redact(self, dialog: DialogTranscript, extra_names: list[str]) -> RedactedTranscript:
        """Detect over the whole dialog, then write the masks back per turn.

        Detection has to see the joined text: a number the speaker paused in
        the middle of arrives as two transcript turns, and neither half is
        recognisable on its own.
        """
        texts = [turn.text for turn in dialog.turns]
        offsets: list[int] = []
        cursor = 0
        for text in texts:
            offsets.append(cursor)
            cursor += len(text) + len(TURN_SEPARATOR)
        document = TURN_SEPARATOR.join(texts)

        matches = find_pii(document)
        totals: dict[str, int] = {}
        turns: list[RedactedTurn] = []
        for i, turn in enumerate(dialog.turns):
            start, end = offsets[i], offsets[i] + len(texts[i])
            local: list[PIIMatch] = []
            for m in matches:
                if m.end <= start or m.start >= end:
                    continue
                clipped_start = max(m.start, start) - start
                clipped_end = min(m.end, end) - start
                # A span crossing a turn boundary is masked once, in the turn
                # where it starts; the tail is removed rather than masked twice.
                entity = m.entity_type if m.start >= start else "_CONTINUATION"
                local.append(PIIMatch(entity, clipped_start, clipped_end))
            text, counts = _apply_with_continuations(texts[i], local)
            text, n_names = redact_names(text, extra_names)
            if n_names:
                counts["PERSON"] = counts.get("PERSON", 0) + n_names
            for k, v in counts.items():
                totals[k] = totals.get(k, 0) + v
            turns.append(
                RedactedTurn(speaker=turn.speaker, start=turn.start, end=turn.end, text=text)
            )

        logger.info("redaction done: call_id=%s entities=%d turns=%d",
                    dialog.call_id, sum(totals.values()), len(turns))
        return RedactedTranscript(
            call_id=dialog.call_id, engine=self.name, turns=turns,
            redaction_counts=totals, enabled=True,
        )


def _apply_with_continuations(
    text: str, matches: list[PIIMatch]
) -> tuple[str, dict[str, int]]:
    counts: dict[str, int] = {}
    out: list[str] = []
    cursor = 0
    for m in sorted(matches, key=lambda x: x.start):
        if m.start < cursor:
            continue
        out.append(text[cursor:m.start])
        if m.entity_type != "_CONTINUATION":
            out.append(replacement_token(m.entity_type))
            counts[m.entity_type] = counts.get(m.entity_type, 0) + 1
        cursor = m.end
    out.append(text[cursor:])
    return "".join(out), counts


class PresidioRedactor(RegexRedactor):
    """Presidio layered on top of the built-in recognizers.

    The built-in pass runs first and does the Hebrew-specific work; presidio
    adds its own pattern recognizers for the international entities it knows
    (payment cards, e-mail, IBAN), and optional Hebrew NER for person names.
    Presidio is asked for `language="en"` on purpose: the recognizers used
    here are pattern-based and language-independent, and presidio ships no
    Hebrew NLP engine.
    """

    name = "presidio"

    PRESIDIO_ENTITIES = ("CREDIT_CARD", "EMAIL_ADDRESS", "IBAN_CODE", "IP_ADDRESS")
    ENTITY_MAP = {
        "CREDIT_CARD": "CREDIT_CARD",
        "EMAIL_ADDRESS": "EMAIL",
        "IBAN_CODE": "IBAN",
        "IP_ADDRESS": "ACCOUNT_LIKE",
    }

    def __init__(self, config: RedactionConfig) -> None:
        from presidio_analyzer import AnalyzerEngine

        super().__init__(config)
        self.analyzer = AnalyzerEngine()
        self._ner = self._load_ner() if config.ner else None

    def _load_ner(self):  # noqa: ANN202
        try:
            from transformers import pipeline as hf_pipeline  # lazy import

            return hf_pipeline("ner", model="models/dictabert-ner", aggregation_strategy="simple")
        except Exception as exc:  # pragma: no cover - server-only path
            logger.warning("DictaBERT-NER unavailable (%s); NER redaction disabled", exc)
            return None

    def _redact(self, dialog: DialogTranscript, extra_names: list[str]) -> RedactedTranscript:
        base = super()._redact(dialog, extra_names)
        turns: list[RedactedTurn] = []
        totals = dict(base.redaction_counts)
        for turn in base.turns:
            text = turn.text
            for res in sorted(
                self.analyzer.analyze(text=text, language="en",
                                      entities=list(self.PRESIDIO_ENTITIES)),
                key=lambda r: r.start, reverse=True,
            ):
                entity = self.ENTITY_MAP.get(res.entity_type, res.entity_type)
                text = text[:res.start] + replacement_token(entity) + text[res.end:]
                totals[entity] = totals.get(entity, 0) + 1
            if self._ner is not None:
                for ent in sorted(self._ner(text), key=lambda e: e["start"], reverse=True):
                    if ent.get("entity_group") in ("PER", "PERSON"):
                        text = (text[:ent["start"]] + replacement_token("PERSON")
                                + text[ent["end"]:])
                        totals["PERSON"] = totals.get("PERSON", 0) + 1
            turns.append(RedactedTurn(speaker=turn.speaker, start=turn.start,
                                      end=turn.end, text=text))
        return RedactedTranscript(call_id=dialog.call_id, engine=self.name, turns=turns,
                                  redaction_counts=totals, enabled=True)


def build_redactor(config: RedactionConfig, mock: bool) -> Redactor:
    if mock:
        return RegexRedactor(config)
    try:
        return PresidioRedactor(config)
    except ImportError:
        logger.warning("presidio not installed; using built-in regex redactor")
        return RegexRedactor(config)
    except Exception as exc:  # noqa: BLE001 - degrade, never block redaction
        # Presidio imports fine but cannot construct - typically the spaCy
        # model its AnalyzerEngine loads (en_core_web_lg) is not installed,
        # which raises OSError, not ImportError. Redaction itself must never
        # die with it: the built-in recognizers do all the Hebrew-specific
        # work and presidio only layers international extras on top.
        logger.warning("presidio unavailable (%s); using built-in regex redactor", exc)
        return RegexRedactor(config)
