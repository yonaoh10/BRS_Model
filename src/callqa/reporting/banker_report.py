"""Per-banker aggregate reports + reports/index.html (via `callqa report`)."""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from pathlib import Path

from callqa.aggregation import BankerAggregate, aggregate_bankers, load_scorecards
from callqa.models import ScoreCard
from callqa.reporting.common import (
    jinja_env,
    load_recommendations,
    pick_recommendation,
    safe_filename,
    score_color,
)
from callqa.rubric import Rubric
from callqa.state import atomic_write_text

GROUP_LABEL_HE = {"median": "חציון", "mean": "ממוצע"}


def _footer_meta(cards: list[ScoreCard]) -> str:
    if not cards:
        return ""
    sample = cards[0]
    return (
        f"מודל שיפוט: {sample.model} · מנוע: {sample.judge_engine} · "
        f"גרסת פרומפט: {sample.prompt_version} · "
        f"חתימת פרומפט: {sample.prompt_sha256[:16]} · "
        f"נוצר: {datetime.now(UTC).isoformat(timespec='seconds')}"
    )


def _banker_slugs(banker_ids: list[str]) -> dict[str, str]:
    """A unique, path-safe filename stem per banker id.

    safe_filename alone collides: distinct ids reduce to the same name
    ("B 001", "B/001" -> "B_001") or to the "unknown" fallback, and one
    banker's report then silently overwrote another's. Disambiguate a collision
    with a short hash of the raw id. Deterministic (sorted) so filenames are
    stable across runs.
    """
    from callqa.ingestion import is_windows_reserved

    slugs: dict[str, str] = {}
    owner: dict[str, str] = {}   # casefolded stem -> the banker_id that claimed it
    for bid in sorted(set(banker_ids)):
        stem = safe_filename(bid)
        if is_windows_reserved(stem):
            stem = f"b_{stem}"            # "NUL.html" is a device on Windows
        # Casefolded: on Windows "B001.html" and "b001.html" are one file.
        if owner.get(stem.casefold(), bid) != bid:
            stem = f"{stem}-{hashlib.sha256(bid.encode('utf-8')).hexdigest()[:8]}"
        owner[stem.casefold()] = bid
        slugs[bid] = stem
    return slugs


def render_banker_report(
    agg: BankerAggregate,
    rubric: Rubric,
    group_dim: dict[str, float],
    group_total: float,
    group_comparison: str,
    recommendations: dict[str, list[str]],
    footer_meta: str,
) -> str:
    template = jinja_env().get_template("banker_report.html.j2")
    return template.render(
        agg=agg,
        dimensions=rubric.dimensions,
        dim_names={d.id: d.name_he for d in rubric.dimensions},
        dim_colors={d.id: score_color(agg.dim_means[d.id]) for d in rubric.dimensions},
        total_color=score_color(agg.mean_total, maximum=100.0),
        group_dim=group_dim,
        group_total=group_total,
        group_label=GROUP_LABEL_HE.get(group_comparison, group_comparison),
        recommendation=pick_recommendation(recommendations, agg.weakest_dim, agg.banker_id),
        footer_meta=footer_meta,
    )


def generate_banker_reports(
    output_dir: Path,
    rubric: Rubric,
    group_comparison: str = "median",
    recommendations_path: str | None = None,
) -> list[Path]:
    """Render all banker reports + index.html. Returns the written paths."""
    cards = load_scorecards(output_dir)
    if not cards:
        raise FileNotFoundError(
            f"no scorecards found under {output_dir / 'scores'} - run the pipeline first"
        )
    recommendations = load_recommendations(recommendations_path)
    aggregates, group_dim, group_total = aggregate_bankers(cards, rubric, group_comparison)
    footer = _footer_meta(cards)

    written: list[Path] = []
    bankers_dir = output_dir / "reports" / "bankers"
    # One unique, path-safe stem per banker, used for BOTH the written file and
    # the index link so they can never diverge.
    slugs = _banker_slugs([agg.banker_id for agg in aggregates.values()])
    for agg in aggregates.values():
        html = render_banker_report(
            agg, rubric, group_dim, group_total, group_comparison, recommendations, footer
        )
        path = bankers_dir / f"{slugs[agg.banker_id]}.html"
        atomic_write_text(path, html)
        written.append(path)

    written.append(_write_index(output_dir, cards, aggregates, slugs, footer))
    return written


def _write_index(output_dir: Path, cards: list[ScoreCard],
                 aggregates: dict[str, BankerAggregate], slugs: dict[str, str],
                 footer: str) -> Path:
    reports = output_dir / "reports"
    executive = sorted(p.name for p in reports.glob("executive*.html")) if reports.is_dir() else []
    index_template = jinja_env().get_template("index.html.j2")
    index_html = index_template.render(
        generated_at=datetime.now(UTC).isoformat(timespec="seconds"),
        n_calls=len(cards),
        bankers=list(aggregates.values()),
        banker_slugs=slugs,
        cards=sorted(cards, key=lambda c: c.call_id),
        calibration_exists=(reports / "calibration.html").exists(),
        executive_reports=executive,
        footer_meta=footer,
    )
    index_path = reports / "index.html"
    atomic_write_text(index_path, index_html)
    return index_path


def refresh_index(output_dir: Path, rubric: Rubric, group_comparison: str = "median") -> Path:
    """Re-render only reports/index.html - after a management report was
    written, so the index links it."""
    cards = load_scorecards(output_dir)
    if not cards:
        raise FileNotFoundError(
            f"no scorecards found under {output_dir / 'scores'} - run the pipeline first"
        )
    aggregates, _, _ = aggregate_bankers(cards, rubric, group_comparison)
    slugs = _banker_slugs([agg.banker_id for agg in aggregates.values()])
    return _write_index(output_dir, cards, aggregates, slugs, _footer_meta(cards))
