"""The diarization scorer.

The scorer exists so the mono path can be measured rather than trusted, which
only works if the scorer itself is right. The decisive case is the third test:
a perfect separation with the two roles swapped must show up as a role failure,
not as a separation failure, because the two have completely different fixes.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from eval_diarization import Turn, load_reference, score  # noqa: E402

REFERENCE = [
    Turn(0.0, 10.0, "banker"),
    Turn(10.0, 20.0, "customer"),
    Turn(20.0, 30.0, "banker"),
]


def test_a_perfect_hypothesis_scores_zero() -> None:
    result = score(REFERENCE, list(REFERENCE))
    assert result["der"] == 0.0
    assert result["role_accuracy"] == 1.0


def test_swapped_roles_are_a_role_failure_not_a_separation_failure() -> None:
    swapped = [Turn(t.start, t.end, "customer" if t.speaker == "banker" else "banker")
               for t in REFERENCE]
    result = score(REFERENCE, swapped)
    assert result["der"] > 0.9, "as labelled, almost every frame is wrong"
    assert result["der_best_pairing"] == 0.0, "the separation itself was perfect"
    assert result["role_accuracy"] == 0.0


def test_missed_speech_is_counted() -> None:
    partial = [Turn(0.0, 10.0, "banker"), Turn(10.0, 20.0, "customer")]
    result = score(REFERENCE, partial)
    assert result["missed_speech"] > 0.2
    assert result["false_alarm"] == 0.0


def test_false_alarm_is_counted() -> None:
    noisy = [*REFERENCE, Turn(30.0, 40.0, "banker")]
    result = score(REFERENCE, noisy)
    assert result["false_alarm"] > 0.2
    assert result["missed_speech"] == 0.0


def test_the_collar_forgives_a_small_boundary_shift() -> None:
    """Diarizers are routinely a tenth of a second out at turn edges; that is
    not what anyone means by a speaker error."""
    shifted = [Turn(t.start + 0.1, t.end + 0.1, t.speaker) for t in REFERENCE]
    assert score(REFERENCE, shifted, collar=0.25)["der"] == 0.0
    assert score(REFERENCE, shifted, collar=0.0)["der"] > 0.0


def test_csv_and_rttm_references_agree(tmp_path: Path) -> None:
    csv_path = tmp_path / "ref.csv"
    csv_path.write_text("start,end,speaker\n0,10,banker\n10,20,customer\n20,30,banker\n",
                        encoding="utf-8")
    rttm_path = tmp_path / "ref.rttm"
    rttm_path.write_text(
        "SPEAKER call 1 0.0 10.0 <NA> <NA> banker <NA> <NA>\n"
        "SPEAKER call 1 10.0 10.0 <NA> <NA> customer <NA> <NA>\n"
        "SPEAKER call 1 20.0 10.0 <NA> <NA> banker <NA> <NA>\n",
        encoding="utf-8")
    assert load_reference(csv_path) == load_reference(rttm_path) == REFERENCE
