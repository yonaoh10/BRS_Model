"""Unit tests: Israeli ID checksum, phone regexes, redaction behavior."""

from __future__ import annotations

import pytest

from callqa.models import DialogTranscript, DialogTurn
from callqa.redaction import (
    RegexRedactor,
    is_valid_israeli_id,
    is_valid_luhn,
    redact_text,
)

# Known valid Israeli IDs (checksum-correct synthetic numbers).
VALID_IDS = ["123456782", "305567661", "12345674", "000000000"]
INVALID_IDS = ["123456789", "123456780", "12345", "1234567890", "305567663", "12345675"]


@pytest.mark.parametrize("value", VALID_IDS)
def test_valid_israeli_ids(value: str) -> None:
    assert is_valid_israeli_id(value)


@pytest.mark.parametrize("value", INVALID_IDS)
def test_invalid_israeli_ids(value: str) -> None:
    assert not is_valid_israeli_id(value)


@pytest.mark.parametrize(
    "text",
    [
        "התקשר אליי ל-052-1234567 בבקשה",
        "הטלפון 0521234567 זמין",
        "מספר קווי 03-6123456",
        "חייג +972-52-123-4567 עכשיו",
        "+972521234567",
        "04 812 3456",
    ],
)
def test_phone_redacted(text: str) -> None:
    redacted, counts = redact_text(text)
    assert counts.get("PHONE", 0) >= 1, redacted
    assert "<טלפון:████>" in redacted


def test_id_redacted_only_when_checksum_valid() -> None:
    redacted, counts = redact_text("תעודת זהות 123456782 בבקשה")
    assert counts == {"ISRAELI_ID": 1}
    assert '<ת"ז:████>' in redacted
    # Invalid checksum, 9 digits -> falls through to ACCOUNT_LIKE (privacy bias).
    redacted, counts = redact_text("מספר 123456789 כלשהו")
    assert counts == {"ACCOUNT_LIKE": 1}


def test_credit_card_luhn() -> None:
    assert is_valid_luhn("4580458045804580")  # synthetic Luhn-valid
    assert not is_valid_luhn("4580458045804581")
    redacted, counts = redact_text("כרטיס 4580-4580-4580-4580 בתוקף")
    assert counts == {"CREDIT_CARD": 1}
    assert "<כרטיס אשראי:████>" in redacted


def test_account_like_aggressive() -> None:
    redacted, counts = redact_text("חשבון מספר 765432 בסניף")
    assert counts == {"ACCOUNT_LIKE": 1}
    assert "<חשבון:████>" in redacted


def test_a_short_number_is_masked_only_where_it_identifies_something() -> None:
    """Four and five digit runs are ambiguous. Next to an account word they
    are an identifier and must go; on their own they are a quantity."""
    _, counts = redact_text("מספר החשבון הוא 54321")
    assert counts == {"ACCOUNT_LIKE": 1}
    _, counts = redact_text("היו שם 54321 אנשים")
    assert counts == {}


def test_name_redaction() -> None:
    redacted, counts = redact_text("מדבר רון לוי מהבנק", extra_names=["רון לוי"])
    assert counts == {"PERSON": 1}
    assert "<שם:████>" in redacted
    assert "רון לוי" not in redacted


def test_redact_dialog_counts_and_offsets() -> None:
    dialog = DialogTranscript(
        call_id="T1",
        attribution_mode="mock",
        turns=[
            DialogTurn(speaker="customer", start=1.0, end=4.0,
                       text="תעודת הזהות שלי היא 123456782 והנייד 052-1234567"),
            DialogTurn(speaker="banker", start=5.0, end=8.0, text="תודה רבה"),
        ],
    )
    redacted = RegexRedactor().redact_dialog(dialog)
    assert redacted.redaction_counts == {"ISRAELI_ID": 1, "PHONE": 1}
    # Turn timestamps are preserved so evidence timestamps stay valid.
    assert redacted.turns[0].start == 1.0 and redacted.turns[0].end == 4.0
    assert "123456782" not in redacted.turns[0].text
    assert "1234567" not in redacted.turns[0].text


# -- identifiers with no shape ----------------------------------------------
#
# A national ID passes a checksum, a phone has a prefix, a card passes Luhn.
# A mother's name, a date of birth and a street address have no form at all -
# the only thing that marks them as identifiers is that a banker asked for
# them. These pin the question-and-answer path that masks them, in BOTH
# directions: the leak must close, and ordinary speech must survive.


def _dialog(*turns: tuple[str, str]) -> DialogTranscript:
    return DialogTranscript(
        call_id="QA", attribution_mode="stereo",
        turns=[DialogTurn(speaker=s, start=float(i * 5), end=float(i * 5 + 4), text=t)
               for i, (s, t) in enumerate(turns)],
    )


@pytest.mark.parametrize(("question", "answer", "label"), [
    ("ומה שם האם לאימות נוסף?", "רות.", "שם"),
    ("מה שם האם שלך?", "קוראים לה מרים.", "שם"),
    ("מה השם המלא שלך?", "דוד בן ארי.", "שם"),
    ("ומה תאריך הלידה?", "נולדתי בחמישי למרץ שמונים ושתיים.", "תאריך לידה"),
    ("מתי נולדת?", "ב-5.3.1982.", "תאריך לידה"),
    ("מה כתובת המגורים שלך?", "רחוב הרצל 15 בחיפה.", "כתובת"),
    ("איפה אתה גר?", "שדרות רוטשילד 40 בתל אביב.", "כתובת"),
])
def test_an_answer_to_an_identity_question_is_masked(
    question: str, answer: str, label: str
) -> None:
    """The answer arrives in its OWN turn, so nothing in the string itself is
    recognisable - the preceding question is the entire signal."""
    redacted = RegexRedactor().redact_dialog(_dialog(("banker", question), ("customer", answer)))
    assert f"<{label}:████>" in redacted.turns[1].text, redacted.turns[1].text


def test_the_question_itself_is_never_masked() -> None:
    """A reviewer has to be able to read what the banker asked; masking the
    question would also hide whether identification was performed at all,
    which is one of the scored rubric dimensions."""
    redacted = RegexRedactor().redact_dialog(_dialog(
        ("banker", "ולאימות הכתובת, מה כתובת המגורים שלך?"),
        ("customer", "רחוב הרצל 15 בחיפה."),
    ))
    assert redacted.turns[0].text == "ולאימות הכתובת, מה כתובת המגורים שלך?"
    assert "████" in redacted.turns[1].text


def test_the_askers_own_next_words_are_not_the_answer() -> None:
    """Question and answer are matched across SPEAKERS. Without that, the rest
    of the banker's own sentence reads as the answer."""
    redacted = RegexRedactor().redact_dialog(_dialog(
        ("banker", "מה שם האם לאימות נוסף?"),
        ("banker", "אני שואל רק לצורך הזיהוי."),
        ("customer", "רות."),
    ))
    assert "████" not in redacted.turns[1].text
    assert "<שם:████>" in redacted.turns[2].text


def test_a_refused_answer_does_not_swallow_ordinary_speech() -> None:
    redacted = RegexRedactor().redact_dialog(_dialog(
        ("banker", "מה שם האם שלך?"),
        ("customer", "אני לא זוכר."),
    ))
    assert "████" not in redacted.turns[1].text


def test_a_refusal_followed_by_a_real_answer_is_still_masked() -> None:
    redacted = RegexRedactor().redact_dialog(_dialog(
        ("banker", "מה שם האם שלך?"),
        ("customer", "לא בטוח, אולי רות."),
    ))
    assert "<שם:████>" in redacted.turns[1].text


@pytest.mark.parametrize("line", [
    "הסניף ברחוב דיזנגוף פתוח היום עד חמש.",
    "אני לקוח כבר עשר שנים וזו פעם ראשונה.",
    "ההחזר החודשי המשוער הוא כאלף וחמש מאות שקל.",
    "זה חיוב של שלושים שקלים מלפני שבוע.",
    "המסלול עולה עשרה שקלים בחודש.",
    "הכרטיס יגיע אליך עד שלושה ימי עסקים.",
])
def test_ordinary_speech_is_not_masked(line: str) -> None:
    """Recall is trivially 1.0 for a redactor that masks everything. These are
    the sentences that must survive: amounts the compliance and clarity
    dimensions are scored on, and a branch location that is not a home address."""
    redacted, counts = redact_text(line)
    assert "████" not in redacted, f"over-masked: {redacted}"
    assert counts == {}


def test_a_landmark_without_a_house_number_is_not_an_address() -> None:
    """"הסניף ברחוב דיזנגוף" is where the branch is; "רחוב הרצל 15" is where
    the customer lives. The house number is what separates them."""
    assert "████" not in redact_text("הסניף ברחוב דיזנגוף פתוח היום.")[0]
    assert "<כתובת:████>" in redact_text("רחוב הרצל 15 בחיפה.")[0]


# -- NER is a privacy control: on means on -----------------------------------

def test_ner_on_without_the_model_stops_instead_of_silently_running_without_it(
    tmp_path, monkeypatch,
) -> None:
    """NER used to live only inside the presidio redactor, so once presidio was
    made opt-in, `redaction.ner: true` did nothing at all - and before that, a
    missing model degraded to a warning. Either way names the operator believed
    were covered went through. Enabled-but-unavailable must be a hard stop at
    engine build, before a single call is processed."""
    from callqa.config import RedactionConfig
    from callqa.redaction import RedactionConfigError, build_redactor

    with pytest.raises(RedactionConfigError, match="download_models.py --ner"):
        build_redactor(RedactionConfig(ner=True), mock=False, models_dir=tmp_path)


def test_ner_masks_names_without_presidio_and_around_existing_masks(monkeypatch) -> None:
    """NER must work on the default (presidio-off) path, and must never write a
    mask over an existing one - that corrupts the token into something neither
    the judge nor the reviewer can read."""
    from callqa.config import RedactionConfig

    def fake_ner(text: str) -> list[dict]:
        out = []
        for name in ("מיכל אברמוביץ", "████"):
            i = text.find(name)
            if i >= 0:
                out.append({"entity_group": "PER", "start": i, "end": i + len(name)})
        return out

    monkeypatch.setattr(RegexRedactor, "_load_ner", lambda self: fake_ner)
    redactor = RegexRedactor(RedactionConfig(ner=True))
    dialog = DialogTranscript(call_id="N", attribution_mode="stereo", turns=[
        DialogTurn(speaker="customer", start=0, end=5,
                   text="שלום, מדברת מיכל אברמוביץ, ת.ז 123456782")])
    text = redactor.redact_dialog(dialog).turns[0].text
    assert "מיכל אברמוביץ" not in text
    assert "<שם:████>" in text
    assert '<ת"ז:████>' in text                  # the existing mask survived intact


# -- "the account" names an identifier; "in the account" introduces an amount -

@pytest.mark.parametrize("line", [
    "החשבון שלי 481902 שקלים",
    "החשבון 481902 אלף",
    "העברתי לחשבון 481902 עוד אלף שקל",
    "והחשבון הוא 7654321 שקל",
])
def test_an_account_number_next_to_a_currency_word_is_masked(line: str) -> None:
    """6-8 digits is the ordinary length of an Israeli account number, and the
    amount exception let every one of these through because a currency word
    followed. Found by adversarial review after being wrongly dismissed here -
    the dismissal tested sentences that really were amounts."""
    redacted, counts = redact_text(line)
    assert "████" in redacted, redacted
    assert counts.get("ACCOUNT_LIKE") == 1


@pytest.mark.parametrize("line", [
    "יש לך בחשבון 150000 שקל",
    "בחשבון החיסכון 250000 שקל",
    "הלוואה של 150000 שקל",
    "החשבון נסגר. 150000 שקל הועברו",
])
def test_a_balance_or_loan_amount_is_still_an_amount(line: str) -> None:
    """The other direction. "בחשבון" - IN the account - introduces a balance,
    and the compliance and clarity dimensions are scored on the amounts a
    banker quotes. A sentence boundary also breaks the link to the account."""
    redacted, counts = redact_text(line)
    assert "████" not in redacted, redacted


@pytest.mark.parametrize(("line", "name"), [
    ("שלום, קוראים לי מיכל אברמוביץ, אני מתקשרת בקשר לחשבון", "מיכל אברמוביץ"),
    ("שמי דוד כהן.", "דוד כהן"),
    ("השם שלי הוא רונית לוי", "רונית לוי"),
])
def test_a_customer_who_names_themselves_is_masked(line: str, name: str) -> None:
    """Only the banker's name reached the redactor (from metadata), and the
    question-driven rules need a question. A customer usually says who they are
    before anyone asks."""
    redacted, _ = redact_text(line)
    assert name not in redacted and "<שם:████>" in redacted, redacted


@pytest.mark.parametrize("line", [
    "שלום, הגעת לבנק, מדבר יועץ מהמוקד. במה אפשר לעזור?",
    "בוקר טוב, מדבר בנקאי מצוות ההלוואות.",
    "קוראים לזה מסלול חודשי",
])
def test_a_banker_opening_is_not_mistaken_for_a_name(line: str) -> None:
    """"מדבר X" is how a banker opens, and X is as often a role as a name.
    Masking it erases the opening the rubric scores and protects nobody."""
    assert "████" not in redact_text(line)[0]
