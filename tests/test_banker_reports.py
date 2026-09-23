"""Banker aggregate reports: links must resolve, and distinct bankers must not
overwrite each other's report."""

from __future__ import annotations

import re
from datetime import UTC, datetime
from pathlib import Path

from callqa.models import CallResult, DimensionScore, ScoreCard
from callqa.reporting.banker_report import _banker_slugs, generate_banker_reports
from callqa.rubric import load_rubric


def _write_card(out: Path, call_id: str, banker: str) -> None:
    rubric = load_rubric("config/rubric.yaml")
    card = ScoreCard(
        call_id=call_id, banker_id=banker,
        scores={d.id: DimensionScore(score=3, reasoning_he="x") for d in rubric.dimensions},
        weighted_total=50.0, judge_engine="mock", model="mock",
        prompt_sha256="0" * 64, prompt_version="1.0.0", rubric_sha256="R",
        timestamp=datetime.now(UTC).isoformat())
    (out / "scores").mkdir(parents=True, exist_ok=True)
    (out / "scores" / f"{call_id}.json").write_text(card.model_dump_json(), encoding="utf-8")
    (out / "results").mkdir(parents=True, exist_ok=True)
    (out / "results" / f"{call_id}.json").write_text(
        CallResult(call_id=call_id, status="success").model_dump_json(), encoding="utf-8")


def test_banker_slugs_disambiguates_collisions() -> None:
    # "B 001" and "B/001" both reduce to "B_001"; ".." reduces to the fallback
    slugs = _banker_slugs(["B 001", "B/001", ".."])
    assert len(set(slugs.values())) == 3, "distinct bankers must get distinct files"
    assert all("/" not in s and ".." not in s for s in slugs.values())


def test_index_links_resolve_to_written_files(tmp_path: Path) -> None:
    out = tmp_path / "output"
    _write_card(out, "C1", "B 001")
    _write_card(out, "C2", "B/001")      # collides with "B 001" under safe_filename
    generate_banker_reports(out, load_rubric("config/rubric.yaml"))

    index = (out / "reports" / "index.html").read_text(encoding="utf-8")
    hrefs = re.findall(r'href="(bankers/[^"]+\.html)"', index)
    assert len(hrefs) == 2
    for href in hrefs:
        assert (out / "reports" / href).is_file(), f"index links to a missing file: {href}"


def test_a_rubric_change_names_itself_instead_of_raising_a_keyerror() -> None:
    """Resume reuses a stored scorecard, so editing config/rubric.yaml and
    re-running `report` renders old scores against new dimensions. That came
    out as a bare KeyError from inside a Jinja render - an error naming a
    dictionary key and nothing an operator could act on."""
    import pytest

    from callqa.models import CallMeta, Features, RedactedTranscript
    from callqa.reporting.call_report import render_call_report
    from callqa.reporting.common import ReportError

    rubric = load_rubric("config/rubric.yaml")
    dims = [d.id for d in rubric.dimensions]
    card = ScoreCard(
        call_id="STALE", banker_id="B1",
        # scored under a rubric that did not yet have the last dimension
        scores={d: DimensionScore(score=3, reasoning_he="x") for d in dims[:-1]},
        weighted_total=60.0, judge_engine="mock", model="mock",
        prompt_sha256="old", prompt_version="0.0.0",
        timestamp=datetime.now(UTC).isoformat(),
    )
    with pytest.raises(ReportError, match=dims[-1]):
        render_call_report(
            rubric,
            CallMeta(call_id="STALE", banker_id="B1", file_name="a.wav",
                     duration_sec=60.0, channels=2, sample_rate=16000),
            card,
            # model_construct: the guard must fire before anything downstream
            # is touched, so these stand-ins never need to be valid. If the
            # check is ever moved below them the test fails loudly rather than
            # passing for the wrong reason.
            Features.model_construct(call_id="STALE"),
            RedactedTranscript(call_id="STALE", engine="regex", turns=[],
                               redaction_counts={}, enabled=True),
        )
