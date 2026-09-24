"""CI assertions for the journey report (not shipped logic).

    python .github/check_journey_report.py REPORT.html --stories N [--channel msedge|chrome]

Static checks of the file the bank would open: the levels, sections and story
cards are there, the embedded data parses and covers every story, every
in-page anchor has a target and every relative link a file, Hebrew is literal,
no forbidden word, and the page is of a size Edge opens comfortably.

With --channel (or --executable), the installed browser opens the report from
disk through Playwright and fails on any script error, or if the explorer does
not draw, a finding's link does not filter the stories it names, a row does
not open its story, a timeline mark does not open its contact, or printing
throws.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from html.parser import HTMLParser
from pathlib import Path

try:
    from callqa.portable import configure_stdio

    configure_stdio()
except ImportError:
    pass

REQUIRED_IDS = ("level1", "level2", "level3", "method", "explorer", "sec-findings",
                "sec-returns", "sec-categories", "sec-topics", "sec-gaps", "sec-resolution",
                "sec-promises", "sec-abandon", "sec-effort", "sec-quality", "sec-branches",
                "sec-metrics", "sec-stories", "sec-calls", "sec-atlas", "xc", "xs")
MAX_BYTES = 8_000_000
FORBIDDEN_WORD = "אצלכם"


class Page(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.ids: set[str] = set()
        self.hrefs: list[str] = []
        self.scripts = 0
        self.stories = 0

    def handle_starttag(self, tag, attrs):  # noqa: ANN001, ANN201
        d = {k: v or "" for k, v in attrs}
        if "id" in d:
            self.ids.add(d["id"])
        if tag == "a" and d.get("href"):
            self.hrefs.append(d["href"])
        if tag == "script":
            self.scripts += 1
        if tag == "details" and "story" in d.get("class", "").split():
            self.stories += 1


FLAG_TESTS = {  # a finding's filter -> which stories it should leave
    "failure": lambda s: s["f"] > 0, "promise": lambda s: s["pb"] > 0,
    "abandoned": lambda s: s["ab"] > 0, "retold": lambda s: s["rt"] > 0,
    "same_day": lambda s: s["sd"] >= 3, "min_bankers": lambda s: s["bk"] >= 3,
}


def expected_count(stories: list[dict], f: dict) -> int:
    out = stories
    if f.get("topic"):
        out = [s for s in out if s["t"] == f["topic"]]
    if f.get("status"):
        out = [s for s in out if s["st"] == f["status"]]
    if f.get("category"):
        out = [s for s in out if f["category"] in s["cats"]]
    if f.get("min_returns"):
        out = [s for s in out if s["r"] >= f["min_returns"]]
    for key, test in FLAG_TESTS.items():
        if f.get(key):
            out = [s for s in out if test(s)]
    return len(out)


def static_checks(path: Path, n_stories: int) -> tuple[list[str], dict]:
    problems: list[str] = []
    html = path.read_text(encoding="utf-8")
    size = path.stat().st_size
    page = Page()
    page.feed(html)
    missing = [i for i in REQUIRED_IDS if i not in page.ids]
    if missing:
        problems.append(f"sections missing: {missing}")
    if page.scripts != 2:
        problems.append(f"expected 2 <script> elements, found {page.scripts}")
    if size > MAX_BYTES:
        problems.append(f"report is {size:,} bytes (limit {MAX_BYTES:,})")
    if FORBIDDEN_WORD in html:
        problems.append(f"the word {FORBIDDEN_WORD!r} appears in the report")
    block = re.search(r'<script type="application/json" id="xr-data">(.*?)</script>', html, re.S)
    data: dict = {"stories": []}
    if not block:
        problems.append("no embedded data block")
    else:
        if re.search(r"\\u05[0-9a-fA-F]{2}", block.group(1)):
            problems.append("Hebrew is \\u-escaped in the data block")
        data = json.loads(block.group(1))
    if len(data["stories"]) != n_stories:
        problems.append(f"data covers {len(data['stories'])} stories, expected {n_stories}")
    if page.stories != n_stories:
        problems.append(f"{page.stories} story cards, expected {n_stories}")
    # the two analysis levels: a row per contact, a row per banker session
    n_contacts = sum(s["c"] for s in data["stories"])
    if len(data.get("contacts", [])) != n_contacts:
        problems.append(f"{len(data.get('contacts', []))} contact rows, expected {n_contacts}")
    if not data.get("sessions"):
        problems.append("no banker session rows")
    head = re.search(r'<table class="head-table">(.*?)</table>', html, re.S)
    if not head or head.group(1).count("<tr>") != 16:          # header + fifteen lines
        problems.append("the page of numbers does not have fifteen lines")
    root = path.parent
    for href in page.hrefs:
        if href.startswith("#"):
            if href[1:] not in page.ids:
                problems.append(f"anchor with no target: {href}")
            continue
        if not (root / href.split("?")[0]).is_file():
            problems.append(f"broken link: {href}")
    print(f"report: {size:,} bytes, {len(data['stories'])} stories, {len(page.hrefs)} links")
    return problems, data


def browser_checks(path: Path, data: dict, launch: dict, screenshot: Path | None) -> list[str]:
    from playwright.sync_api import sync_playwright

    stories = data["stories"]
    problems: list[str] = []
    errors: list[str] = []
    with sync_playwright() as p:
        browser = p.chromium.launch(**launch)
        print("browser:", launch or "chromium", browser.version)
        page = browser.new_context(viewport={"width": 1366, "height": 900},
                                   locale="he-IL").new_page()
        page.on("pageerror", lambda e: errors.append(f"pageerror: {e}"))
        page.on("console", lambda m: errors.append(f"console: {m.text}")
                if m.type == "error" else None)
        page.goto(path.resolve().as_uri())
        page.wait_for_timeout(500)
        if page.evaluate("document.documentElement.getAttribute('data-explorer')") != "ready":
            problems.append("the explorer script did not finish")
        rows = page.evaluate("document.querySelectorAll('#xp-body tr.row').length")
        if rows != min(50, len(stories)):
            problems.append(f"explorer drew {rows} rows, expected {min(50, len(stories))}")
        link = page.query_selector("#sec-findings a[data-filter]")
        if link is not None:
            f = json.loads(link.get_attribute("data-filter"))
            link.click()
            page.wait_for_timeout(200)
            want = expected_count(stories, f)
            visible = page.evaluate(
                "Array.from(document.querySelectorAll('#stories > details.story'))"
                ".filter(d => !d.hidden).length")
            if visible != want:
                problems.append(f"finding filter {f} leaves {visible} stories, expected {want}")
        if stories:
            page.click("#xp-body tr.row >> nth=0")
            page.wait_for_timeout(200)
            if not page.evaluate("document.querySelectorAll('details.story[open]').length"):
                problems.append("clicking a row did not open its story")
        mark = page.query_selector("details.story[open] .ch-timeline a[href]")
        if mark is not None:
            target = mark.get_attribute("href")[1:]
            mark.click()
            page.wait_for_timeout(200)
            if not page.evaluate(f"!!document.getElementById({json.dumps(target)})"):
                problems.append(f"timeline mark points at a missing contact: {target}")
        # the analysis levels: each tab draws its explorer, a row opens its story
        for tab, body, prefix in (("xc", "#xc-body", "-c"), ("xs", "#xs-body", "-x")):
            page.click(f".lvl-tabs .tab[data-panel={tab}]")
            page.wait_for_timeout(150)
            if page.evaluate(f"document.getElementById('{tab}').hidden"):
                problems.append(f"the {tab} tab did not show its explorer")
            n = page.evaluate(f"document.querySelectorAll('{body} tr.row').length")
            if not n:
                problems.append(f"the {tab} explorer drew no rows")
                continue
            page.click(f"{body} tr.row >> nth=0")
            page.wait_for_timeout(200)
            where = page.evaluate("location.hash").lstrip("#")
            if prefix not in where:
                problems.append(f"a {tab} row did not point at its contact or session ({where})")
            elif not page.evaluate(
                    f"(e => !!e && e.getClientRects().length > 0)(document.getElementById("
                    f"{json.dumps(where)}))"):
                problems.append(f"a {tab} row opened no visible target: {where}")
        page.click(".lvl-tabs .tab[data-panel=explorer]")
        page.click("#btn-theme")
        page.emulate_media(media="print")
        page.wait_for_timeout(200)
        page.emulate_media(media="screen")
        if screenshot is not None:
            page.screenshot(path=str(screenshot), full_page=False)
        browser.close()
    problems.extend(errors)
    return problems


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("report", type=Path)
    ap.add_argument("--stories", type=int, required=True, help="stories expected in the batch")
    ap.add_argument("--channel", default=None, help="msedge | chrome (installed browser)")
    ap.add_argument("--executable", default=None, help="or a browser executable path")
    ap.add_argument("--screenshot", type=Path, default=None)
    args = ap.parse_args()

    problems, data = static_checks(args.report, args.stories)
    if args.channel or args.executable:
        launch = {"executable_path": args.executable} if args.executable else \
            {"channel": args.channel}
        problems += browser_checks(args.report, data, launch, args.screenshot)
    if problems:
        print("\nJOURNEY REPORT CHECK FAILED:")
        for p in problems:
            print("  -", p)
        return 1
    print("journey report: OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
