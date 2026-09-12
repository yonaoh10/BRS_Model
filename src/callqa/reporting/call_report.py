"""Per-call RTL HTML report - part of the process_call output (stage 8)."""

from __future__ import annotations

from callqa.models import (
    CallMeta,
    DialogTranscript,
    Features,
    RedactedTranscript,
    ScoreCard,
)
from callqa.reporting.common import (
    jinja_env,
    load_recommendations,
    pick_recommendation,
    score_color,
)
from callqa.rubric import Rubric

SIGNAL_NAMES_HE = {
    "opening": "פתיחת השיחה",
    "identity_request": "בקשת פרטי זיהוי",
    "identity_supply": "מסירת פרטי זיהוי",
    "question_rate": "שיעור השאלות",
    "service_language": "שפת שירות",
    "closing": "סגירת השיחה",
    "first_speaker": "מי פתח את השיחה",
}


def _banker_index(dialog: DialogTranscript | None) -> int:
    """Which speaker index the pipeline decided was the banker."""
    if dialog is None or dialog.banker_index is None:
        return 0
    return dialog.banker_index


def weakest_non_gate_dimension(rubric: Rubric, scorecard: ScoreCard) -> str | None:
    """The weakest non-gate dimension (lowest score; rubric order breaks ties)."""
    candidates = [d for d in rubric.dimensions if not d.gate and d.id in scorecard.scores]
    if not candidates:
        return None
    return min(candidates, key=lambda d: scorecard.scores[d.id].score).id


def render_call_report(
    rubric: Rubric,
    meta: CallMeta,
    scorecard: ScoreCard,
    features: Features,
    redacted: RedactedTranscript,
    recommendations: dict[str, list[str]] | None = None,
    dialog: DialogTranscript | None = None,
) -> str:
    if recommendations is None:
        recommendations = load_recommendations()
    weakest = weakest_non_gate_dimension(rubric, scorecard)
    recommendation = (
        pick_recommendation(recommendations, weakest, scorecard.call_id) if weakest else None
    )
    by_id = rubric.by_id
    # Signals are recorded as votes for a speaker index; the report needs them
    # as votes for a role, so they can be read without the index.
    banker_index = _banker_index(dialog)
    template = jinja_env().get_template("call_report.html.j2")
    return template.render(
        meta=meta,
        scorecard=scorecard,
        features=features,
        dimensions=rubric.dimensions,
        dim_colors={d.id: score_color(scorecard.scores[d.id].score) for d in rubric.dimensions},
        total_color=score_color(scorecard.weighted_total, maximum=100.0),
        failed_gate_names=[by_id[g].name_he for g in scorecard.failed_gates if g in by_id],
        weakest_dim_name=by_id[weakest].name_he if weakest else None,
        recommendation=recommendation,
        # Who-said-what is certain on a stereo recording and inferred on a mono
        # one. A score built on an inferred split must say so on its face.
        attribution_mode=dialog.attribution_mode if dialog else None,
        role_confidence=dialog.role_confidence if dialog else None,
        role_signals=[
            {
                "he": SIGNAL_NAMES_HE.get(s.name, s.name),
                "weight": s.weight,
                "role": "בנקאי" if s.votes_for == banker_index else "לקוח",
                "agrees": s.votes_for == banker_index,
            }
            for s in (dialog.role_signals if dialog else [])
        ],
        diarization=dialog.diarization if dialog else None,
    )
