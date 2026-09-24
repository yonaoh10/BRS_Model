"""Shared reporting utilities: Jinja2 environment + recommendation loading."""

from __future__ import annotations

import re
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, select_autoescape
from markupsafe import Markup

from callqa.judge.prompts import mmss
from callqa.resources import find_config, load_yaml

TEMPLATES_DIR = Path(__file__).parent / "templates"

SPEAKER_HE = {"banker": "בנקאי", "customer": "לקוח"}


class ReportError(RuntimeError):
    """A report cannot be rendered from the artifacts it was given."""


_ROOT_BLOCK_RE = re.compile(r":root\s*\{([^}]*)\}")
_VAR_DEF_RE = re.compile(r"(--[\w-]+)\s*:\s*([^;]+)(?:;|$)")
_VAR_USE_RE = re.compile(r"var\(\s*(--[\w-]+)\s*(?:,\s*([^()]*))?\)")
# A declaration (after "{" or ";") whose value uses a custom property.
_DECL_RE = re.compile(r"(?<=[;{])(\s*)([a-zA-Z][a-zA-Z-]*)(\s*:\s*)([^;{}]*var\(--[^;{}]*)(?=[;}])")


def legacy_css(css: str) -> str:
    """Give every declaration that uses a CSS variable a literal fallback in
    front of it, resolved from the light-theme :root blocks.

    Internet Explorer - and Edge's "IE mode", which bank desktops often apply
    to local files - does not know CSS variables and drops each declaration
    that uses one: the page came out white, unstyled, in the default font. With
    `color:#1a1c22; color:var(--ink)` an old engine keeps the first and a
    modern one the second (dark mode and the theme button keep working)."""
    # every plain `:root{...}` block is the light theme (the dark ones are
    # `:root:not(...)` / `:root[data-theme]` and are not matched)
    values: dict[str, str] = {}
    for block in _ROOT_BLOCK_RE.findall(css):
        values.update({name: value.strip() for name, value in _VAR_DEF_RE.findall(block)})
    if not values:
        return css

    def resolve(value: str) -> str:
        for _ in range(6):
            new = _VAR_USE_RE.sub(lambda m: values.get(m.group(1), (m.group(2) or "").strip()),
                                  value)
            if new == value:
                break
            value = new
        return value

    def add_fallback(m: re.Match) -> str:
        ws, prop, colon, value = m.groups()
        literal = resolve(value)
        if "var(" in literal or not literal.strip():
            return m.group(0)
        return f"{ws}{prop}{colon}{literal};{prop}{colon}{value}"

    return _DECL_RE.sub(add_fallback, css)


def jinja_env() -> Environment:
    env = Environment(
        loader=FileSystemLoader(TEMPLATES_DIR),
        autoescape=select_autoescape(["html", "j2"]),
        trim_blocks=True,
        lstrip_blocks=True,
    )
    env.filters["legacy_css"] = lambda text: Markup(legacy_css(str(text)))
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
    data = load_yaml(p) or {}
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
