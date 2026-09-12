"""Per-banker aggregate reports + reports/index.html (via `callqa report`)."""

from __future__ import annotations

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
    for agg in aggregates.values():
        html = render_banker_report(
            agg, rubric, group_dim, group_total, group_comparison, recommendations, footer
        )
        # A banker_id comes from a CSV the bank edits; without this it chose
        # the output path, and "../../x" wrote outside output_dir entirely.
        path = bankers_dir / f"{safe_filename(agg.banker_id)}.html"
        atomic_write_text(path, html)
        written.append(path)

    index_template = jinja_env().get_template("index.html.j2")
    index_html = index_template.render(
        generated_at=datetime.now(UTC).isoformat(timespec="seconds"),
        n_calls=len(cards),
        bankers=list(aggregates.values()),
        cards=sorted(cards, key=lambda c: c.call_id),
        calibration_exists=(output_dir / "reports" / "calibration.html").exists(),
        footer_meta=footer,
    )
    index_path = output_dir / "reports" / "index.html"
    atomic_write_text(index_path, index_html)
    written.append(index_path)
    return written
