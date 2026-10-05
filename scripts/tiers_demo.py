#!/usr/bin/env python
"""Run the input tiers over the synthetic demo and check that every report
says what it stands on.

    .venv\\Scripts\\python scripts\\tiers_demo.py [--stories 30] [--workspace DIR]

For each tier the demo batch is imported with fewer layers, the report is
rendered, and its sources table is read back: a layer that was left out
must be marked absent there, and the figures that need it must say
"cannot be computed" instead of showing a dash. Nothing here needs a
model; the content stage is not run.

    T0  recordings only
    T1  the contact list and the recordings (the handoff workbook)
    T3  T1 + the Atlas export
    T5  T3 + a units table (a contract folder derived from the workbook)

Exit code 0 when every expectation holds, 1 otherwise, with the failing
tier and expectation printed.
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

SOURCES_ROW = re.compile(r'<tr class="src-(\w+)"><td><b>([^<]+)</b></td>')
UNAVAILABLE = "לא ניתן לחשב"


def run(args: list[str], env: dict[str, str]) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, "-m", "callqa", *args], env=env,
                          capture_output=True, text=True, encoding="utf-8", errors="replace")


def sources_of(report: Path) -> dict[str, str]:
    html = report.read_text(encoding="utf-8")
    start = html.find("מקורות הדוח")
    return {label: status for status, label in SOURCES_ROW.findall(html[start:start + 30000])}


def contract_from_workbook(ws: Path, out: Path) -> None:
    """A contract folder with a units table, built from the demo workbook
    through the tool's own reader (the demo ships a workbook, not a contract)."""
    from callqa.journey.importers.workbook import read_workbook_rows  # noqa: PLC0415
    from callqa.journey.models import ImportReport  # noqa: PLC0415
    from callqa.journey.xlsx import read_xlsx  # noqa: PLC0415

    interactions, segments, messages = read_workbook_rows(
        read_xlsx(ws / "input" / "handoff.xlsx"), ImportReport(source="tiers-demo"))
    out.mkdir(parents=True, exist_ok=True)
    with (out / "interactions.csv").open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["interaction_id", "account_ref", "started_at", "channel", "direction",
                    "status", "call_key", "correspondence_id", "talk_seconds", "recorded"])
        for r in interactions:
            w.writerow([r.source_id or f"row{r.row}", f"{r.branch}/{r.account}",
                        r.at.isoformat(sep=" "), r.channel, r.direction,
                        r.answer if r.answer != "unknown" else "",
                        r.source_id if r.channel == "call" else "",
                        r.source_id if r.channel == "message" else "",
                        r.talk_seconds if r.talk_seconds is not None else "",
                        "1" if r.file_name else ""])
    with (out / "call_segments.csv").open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["call_key", "seq", "file_name"])
        for s in segments:
            w.writerow([s.call_key, s.seq or "", s.file_name])
    with (out / "messages.csv").open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["correspondence_id", "message_id", "sent_at", "direction", "subject", "body"])
        for m in messages:
            w.writerow([m.correspondence_id, m.message_id, m.at.isoformat(sep=" "), m.direction,
                        m.subject, m.body])
    (out / "units.csv").write_text(
        "unit_key,code_space,snif_id,org_unit_code,name,kind,crosswalk_basis\n"
        "s109,T1604,109,,מרכז הבנקאות,center,none\n"
        "s136,T1604,136,,תפעול עורפי,back_office,none\n"
        "o40012,T1017,,40012,צוות ב - אשכול נעמן,team,hypothesis\n", encoding="utf-8")
    (out / "manifest.yaml").write_text(
        "contract_version: 2\nangles: [vendor, atlas]\ndate_order: day_first\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--stories", type=int, default=30)
    ap.add_argument("--workspace", type=Path, default=None)
    args = ap.parse_args(argv)
    ws = args.workspace or Path(tempfile.mkdtemp(prefix="callqa-tiers-"))
    gen = subprocess.run([sys.executable, str(ROOT / "scripts" / "generate_journey_demo.py"),
                          "--stories", str(args.stories), "--workspace", str(ws), "--audio",
                          "--seconds", "2", "--no-report"], capture_output=True, text=True)
    if gen.returncode:
        print(gen.stdout, gen.stderr, file=sys.stderr)
        return 1
    inp = ws / "input"
    contract = ws / "contract"
    contract_from_workbook(ws, contract)
    tiers = {
        "T0": (["--audio", str(inp / "recordings.zip")],
               {"רשימת פניות": "absent", "אטלס": "absent"}, True),
        "T1": (["--xlsx", str(inp / "handoff.xlsx"), "--audio", str(inp / "recordings.zip")],
               {"רשימת פניות": "present", "אטלס": "absent"}, True),
        "T3": (["--xlsx", str(inp / "handoff.xlsx"), "--audio", str(inp / "recordings.zip"),
                "--atlas", str(inp / "atlas")],
               {"רשימת פניות": "present"}, False),
        "T5": (["--contract", str(contract), "--audio", str(inp / "recordings.zip"),
                "--atlas", str(inp / "atlas")],
               {"רשימת פניות": "present", "טבלת יחידות": "present"}, False),
    }
    failures: list[str] = []
    for tier, (import_args, expect, expect_unavailable) in tiers.items():
        out = ws / f"out-{tier}"
        env = dict(os.environ, CALLQA_PATHS__INPUT_DIR=str(inp), CALLQA_PATHS__OUTPUT_DIR=str(out),
                   PYTHONIOENCODING="utf-8")
        imp = run(["journey", "import", *import_args], env)
        if imp.returncode:
            failures.append(f"{tier}: import failed\n{imp.stdout}\n{imp.stderr}")
            continue
        rep = run(["journey", "report"], env)
        if rep.returncode:
            failures.append(f"{tier}: report failed\n{rep.stdout}\n{rep.stderr}")
            continue
        report = out / "reports" / "journey.html"
        sources = sources_of(report)
        for label, status in expect.items():
            if sources.get(label) != status:
                failures.append(f"{tier}: sources table says {label} = {sources.get(label)}, "
                                f"expected {status}")
        html = report.read_text(encoding="utf-8")
        has_unavailable = UNAVAILABLE in html
        if expect_unavailable and not has_unavailable:
            failures.append(f"{tier}: no figure says it cannot be computed, although a layer "
                            "is missing")
        atlas_absent = sources.get("אטלס") == "absent"
        if atlas_absent and "אין ייצוא אטלס" not in html:
            failures.append(f"{tier}: Atlas is absent but no figure says so")
        print(f"{tier}: ok - " + ", ".join(f"{k}={v}" for k, v in sources.items()))
    if failures:
        print("\n".join(failures), file=sys.stderr)
        return 1
    print(f"all tiers ok (workspace {ws})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
