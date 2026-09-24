"""CI check of the management report in a real browser (not shipped logic).

    python .github/check_executive_browser.py REPORT.html --channel msedge

Drives the browser the bank uses - Edge on Windows (`--channel msedge`),
Chrome on Linux - through Playwright, with nothing downloaded: the channel is
the browser already installed on the runner. Opens the report from disk,
exactly as a manager would, and fails on any script error, or if the explorer
does not draw, a finding's link does not filter to its calls, a row does not
open its drill-down, or printing throws.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

try:
    from callqa.portable import configure_stdio

    configure_stdio()
except ImportError:
    pass


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("report", type=Path)
    ap.add_argument("--channel", default=None, help="msedge | chrome (installed browser)")
    ap.add_argument("--executable", default=None, help="or a browser executable path")
    ap.add_argument("--screenshot", type=Path, default=None)
    args = ap.parse_args()

    from playwright.sync_api import sync_playwright

    html = args.report.read_text(encoding="utf-8")
    data = json.loads(re.search(r'id="xr-data">(.*?)</script>', html, re.S).group(1))
    scored = [c for c in data["calls"] if c["s"] == "success" and c["tot"] is not None]
    gated = [c for c in scored if c["g"]]
    problems: list[str] = []
    errors: list[str] = []
    with sync_playwright() as p:
        launch = {"channel": args.channel} if args.channel else {}
        if args.executable:
            launch = {"executable_path": args.executable}
        browser = p.chromium.launch(**launch)
        print("browser:", args.channel or args.executable or "chromium", browser.version)
        page = browser.new_context(viewport={"width": 1366, "height": 900},
                                   locale="he-IL").new_page()
        page.on("pageerror", lambda e: errors.append(f"pageerror: {e}"))
        page.on("console", lambda m: errors.append(f"console: {m.text}")
                if m.type == "error" else None)
        page.goto(args.report.resolve().as_uri())
        page.wait_for_timeout(500)
        if page.evaluate("document.documentElement.getAttribute('data-explorer')") != "ready":
            problems.append("the explorer script did not finish")
        rows = page.evaluate("document.querySelectorAll('#xp-body tr.row').length")
        if rows != min(50, len(scored)):
            problems.append(f"explorer drew {rows} rows, expected {min(50, len(scored))}")
        if gated and page.query_selector("#sec-risk a[data-filter]"):
            page.click("#sec-risk a[data-filter] >> nth=0")
            page.wait_for_timeout(200)
            shown = page.inner_text("#xp-count")
            if not shown.startswith(str(len(gated))) and not (len(gated) == 1
                                                               and "אחת" in shown):
                problems.append(f"the gate-failure link shows '{shown}', expected {len(gated)}")
        if scored:
            page.click("#xp-body tr.row >> nth=0")
            page.wait_for_timeout(200)
            cards = page.evaluate("document.querySelectorAll('#xp-body tr.drawer .dcard').length")
            if cards != len(data["dims"]):
                problems.append(f"the drill-down shows {cards} dimensions, expected "
                                f"{len(data['dims'])}")
        page.click("#btn-theme")
        page.emulate_media(media="print")
        page.wait_for_timeout(200)
        page.emulate_media(media="screen")
        if args.screenshot:
            page.screenshot(path=str(args.screenshot), full_page=False)
        browser.close()
    problems.extend(errors)
    if problems:
        print("\nMANAGEMENT REPORT BROWSER CHECK FAILED:")
        for item in problems:
            print("  -", item)
        return 1
    print(f"browser check: OK ({len(scored)} scored calls, {rows} rows drawn)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
