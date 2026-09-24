"""The reports on the machines they are really opened on: Internet Explorer /
Edge's IE mode (no CSS variables), a mail gateway that strips styles, and a
PDF printed by the browser already installed."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from callqa.reporting.common import jinja_env, legacy_css
from callqa.reporting.pdf import find_browser, print_pdf

TEMPLATES = Path(__file__).resolve().parent.parent / "src" / "callqa" / "reporting" / "templates"
_DECL = re.compile(r"(?<=[;{])\s*([a-zA-Z][a-zA-Z-]*)\s*:\s*([^;{}]*)(?=[;}])")


def _css() -> str:
    return (TEMPLATES / "executive_report.css.j2").read_text(encoding="utf-8") + \
        (TEMPLATES / "journey_report.css.j2").read_text(encoding="utf-8")


def test_every_variable_declaration_gets_a_literal_twin_before_it():
    out = legacy_css(_css())
    decls = _DECL.findall(out)
    for i, (prop, value) in enumerate(decls):
        if "var(--" not in value:
            continue
        prev_prop, prev_value = decls[i - 1]
        assert prev_prop == prop and "var(" not in prev_value, (prop, value)


def test_literals_come_from_the_light_theme_of_every_root_block():
    out = legacy_css(":root{--ink:#111; --x:var(--ink)}\n@media (prefers-color-scheme: dark)"
                     "{ :root:not([data-theme]){--ink:#eee} }\n:root{--cat:#c00}\n"
                     ".a{color:var(--x)} .b{fill:var(--cat)} .c{color:var(--none, red)}")
    assert "color:#111;color:var(--x)" in out
    assert "fill:#c00;fill:var(--cat)" in out
    assert "color:red;color:var(--none, red)" in out


def test_css_without_variables_is_unchanged():
    css = "a{color:#111} b{margin:0}"
    assert legacy_css(css) == css


def test_both_reports_carry_the_fallbacks_and_the_notice():
    env = jinja_env()
    for name in ("executive_report.html.j2", "journey_report.html.j2"):
        src = (TEMPLATES / name).read_text(encoding="utf-8")
        assert "{% filter legacy_css %}" in src
        assert 'http-equiv="X-UA-Compatible" content="IE=edge"' in src
        assert 'class="compat-note"' in src
        # the notice is hidden only where CSS variables work
        assert "@supports (--callqa: 1) { .compat-note{display:none !important} }" in src
    assert env.filters["legacy_css"]("a{color:#1}") == "a{color:#1}"


def test_the_notice_is_readable_without_any_stylesheet():
    src = (TEMPLATES / "journey_report.html.j2").read_text(encoding="utf-8")
    note = re.search(r'<div class="compat-note"[^>]*style="([^"]+)"', src)
    assert note and "background" in note.group(1) and "direction:rtl" in note.group(1)


def test_no_browser_is_reported_clearly(tmp_path, monkeypatch):
    monkeypatch.setenv("CALLQA_BROWSER", str(tmp_path / "missing.exe"))
    assert find_browser() is None
    from callqa.reporting.pdf import PDFError
    with pytest.raises(PDFError, match="no Edge or Chrome"):
        print_pdf(tmp_path / "x.html", tmp_path / "x.pdf")


@pytest.mark.skipif(find_browser() is None and not Path("/opt/pw-browsers/chromium").exists(),
                    reason="no Chromium-family browser on this machine")
def test_a_report_prints_to_pdf(tmp_path, monkeypatch):
    if find_browser() is None:
        monkeypatch.setenv("CALLQA_BROWSER", "/opt/pw-browsers/chromium")
    html = tmp_path / "r.html"
    html.write_text('<!DOCTYPE html><html dir="rtl" lang="he"><meta charset="utf-8">'
                    "<body><h1>דוח</h1><p>שלום</p></body></html>", encoding="utf-8")
    pdf = print_pdf(html, tmp_path / "r.pdf")
    assert pdf.read_bytes()[:5] == b"%PDF-"
