"""Shared reporting utilities: Jinja2 environment + recommendation loading."""

from __future__ import annotations

import re
from pathlib import Path

import yaml
from jinja2 import Environment, FileSystemLoader, select_autoescape

from callqa.judge.prompts import mmss
from callqa.resources import find_config

TEMPLATES_DIR = Path(__file__).parent / "templates"

SPEAKER_HE = {"banker": "בנקאי", "customer": "לקוח"}


def jinja_env() -> Environment:
    env = Environment(
        loader=FileSystemLoader(TEMPLATES_DIR),
        autoescape=select_autoescape(["html", "j2"]),
        trim_blocks=True,
        lstrip_blocks=True,
    )
    env.filters["mmss"] = mmss
    env.globals["speaker_he"] = SPEAKER_HE
    return env


def load_recommendations(path: str | Path | None = None) -> dict[str, list[str]]:
    """Coaching lines per dimension. Missing is survivable; the report simply
    carries no recommendation."""
    try:
        p = find_config("recommendations_he.yaml", path)
    except FileNotFoundError:
        return {}
    data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    return {k: list(v) for k, v in data.items() if isinstance(v, list)}


def pick_recommendation(
    recommendations: dict[str, list[str]], dimension_id: str, seed_text: str
) -> str | None:
    options = recommendations.get(dimension_id) or []
    if not options:
        return None
    index = sum(ord(c) for c in seed_text) % len(options)
    return options[index]


def score_color(score: float, maximum: float = 5.0) -> str:
    """Traffic-light color for a score (used inline in reports)."""
    ratio = score / maximum if maximum else 0
    if ratio >= 0.8:
        return "#1a7f37"
    if ratio >= 0.6:
        return "#9a6700"
    return "#c0392b"


_UNSAFE_FILENAME = re.compile(r"[^A-Za-z0-9._-]+")


def safe_filename(value: str, fallback: str = "unknown") -> str:
    """A filename component built from data, never a path.

    Report filenames are built from identifiers that arrive in a CSV the bank
    maintains, so they have to be treated as untrusted text.
    """
    cleaned = _UNSAFE_FILENAME.sub("_", (value or "").strip()).strip("._-")
    return cleaned[:64] or fallback
