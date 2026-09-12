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
