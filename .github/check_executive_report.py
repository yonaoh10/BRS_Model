"""CI assertions for the management report (not shipped logic).

    python .github/check_executive_report.py REPORT.html --calls N [--dom DOM.html]

Checks the file the bank would open: the three levels and the methodology are
there, the embedded data parses and covers the batch, every relative link
points at a file that exists, Hebrew is literal (not \\u-escaped), the page is
of a size Edge opens comfortably, and - with --dom, a DOM dump from a real
browser (Edge headless on Windows) - that the script ran and drew the
explorer's first page of calls.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from html.parser import HTMLParser
from pathlib import Path

try:  # Hebrew in this script's own messages, on a cp1252 CI stream
    from callqa.portable import configure_stdio

    configure_stdio()
except ImportError:
    pass

REQUIRED_IDS = ("level1", "level2", "level3", "method", "explorer", "sec-findings",
                "sec-dims", "sec-dist", "sec-trend", "sec-segments", "sec-bankers",
                "sec-drivers", "sec-risk", "sec-coverage", "sec-cases", "sec-attention")
MAX_BYTES = 8_000_000


class Page(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.ids: set[str] = set()
        self.hrefs: list[str] = []
        self.scripts = 0
        self.rows = 0
        self.root_attrs: dict[str, str] = {}

    def handle_starttag(self, tag, attrs):  # noqa: ANN001, ANN201
        d = {k: v or "" for k, v in attrs}
        if tag == "html":
            self.root_attrs = d
        if "id" in d:
            self.ids.add(d["id"])
        if tag == "a" and d.get("href"):
            self.hrefs.append(d["href"])
        if tag == "script":
            self.scripts += 1
        if tag == "tr" and "row" in d.get("class", "").split():
            self.rows += 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("report", type=Path)
    ap.add_argument("--calls", type=int, required=True, help="calls expected in the batch")
    ap.add_argument("--dom", type=Path, default=None, help="DOM dumped by a real browser (optional)")
    args = ap.parse_args()

    problems: list[str] = []
    html = args.report.read_text(encoding="utf-8")
    size = args.report.stat().st_size
    page = Page()
    page.feed(html)
    missing = [i for i in REQUIRED_IDS if i not in page.ids]
    if missing:
        problems.append(f"sections missing: {missing}")
    if page.scripts != 2:
        problems.append(f"expected 2 <script> elements, found {page.scripts}")
    if size > MAX_BYTES:
        problems.append(f"report is {size:,} bytes (limit {MAX_BYTES:,})")
    block = re.search(r'<script type="application/json" id="xr-data">(.*?)</script>', html, re.S)
    if not block:
        problems.append("no embedded data block")
        data = {"calls": []}
    else:
        if re.search(r"\\u05[0-9a-fA-F]{2}", block.group(1)):
            problems.append("Hebrew is \\u-escaped in the data block")
        data = json.loads(block.group(1))
    if len(data["calls"]) != args.calls:
        problems.append(f"data covers {len(data['calls'])} calls, expected {args.calls}")
    root = args.report.parent
    for href in page.hrefs:
        if href.startswith("#"):
            if href[1:] not in page.ids:
                problems.append(f"anchor with no target: {href}")
            continue
        if not (root / href.split("?")[0]).is_file():
            problems.append(f"broken link: {href}")
    for call in data["calls"]:
        if call.get("rep") and not (root / call["rep"]).is_file():
            problems.append(f"broken call link: {call['rep']}")
    print(f"report: {size:,} bytes, {len(data['calls'])} calls, {len(page.hrefs)} links")

    if args.dom is not None:
        dom = Page()
        dom.feed(args.dom.read_text(encoding="utf-8", errors="replace"))
        if dom.root_attrs.get("data-explorer") != "ready":
            problems.append("in the browser, the explorer script did not finish")
        scored = sum(1 for c in data["calls"] if c.get("tot") is not None)
        if dom.rows != min(50, scored):
            problems.append(f"in the browser, the explorer drew {dom.rows} rows, "
                            f"expected {min(50, scored)}")
        print(f"browser DOM: explorer={dom.root_attrs.get('data-explorer')}, rows={dom.rows}")

    if problems:
        print("\nMANAGEMENT REPORT CHECK FAILED:")
        for p in problems:
            print("  -", p)
        return 1
    print("management report: OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
