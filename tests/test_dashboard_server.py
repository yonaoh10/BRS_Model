"""Tests for the local operator dashboard server.

Covers the data contract, the security controls (this console can start
billable cloud machines, so it must not be drivable by a page you happen to
visit), and the project-wide rule that raw PII never leaves the redaction
stage.
"""

from __future__ import annotations

import json
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "dashboard"))

from server import Handler, collect_state  # noqa: E402

RAW_ID = "123456782"          # seeded into the mock dialog fixture
RAW_PHONE = "052-1234567"


@pytest.fixture(scope="module")
def pipeline_output(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Run the real pipeline in mock mode and hand back its output dir."""
    work = tmp_path_factory.mktemp("dash")
    inp, out = work / "input", work / "output"
    subprocess.run([sys.executable, str(REPO_ROOT / "scripts" / "generate_sample_data.py"),
                    "--input-dir", str(inp)], check=True, capture_output=True)
    cfg = work / "config.yaml"
    cfg.write_text(
        f"paths: {{input_dir: {inp}, output_dir: {out}, state_db: {work}/state.db,"
        f" models_dir: {work}/models}}\n"
        "run: {mock: true}\naudio: {vad: energy}\nasr: {engine: mock}\n"
        "judge: {engine: mock, model: mock}\n", encoding="utf-8")
    for cmd in (["run"], ["report"], ["calibrate"]):
        subprocess.run([sys.executable, "-m", "callqa", *cmd, "--config", str(cfg), "--mock"],
                       cwd=REPO_ROOT, check=True, capture_output=True)
    return out


# ---------------------------------------------------------------- data shape

def test_collect_state_reports_every_call(pipeline_output: Path) -> None:
    state = collect_state(pipeline_output)
    assert state["summary"]["nCalls"] == 6
    assert {c["id"] for c in state["calls"]} == {f"CALL00{i}" for i in range(1, 7)}
    # highest score first, so the table reads as a ranking
    totals = [c["total"] for c in state["calls"]]
    assert totals == sorted(totals, reverse=True)


def test_collect_state_surfaces_gate_failures(pipeline_output: Path) -> None:
    state = collect_state(pipeline_output)
    failed = [c for c in state["calls"] if c["gate"]]
    assert failed, "the sample set contains gate failures; the dashboard must show them"
    for c in failed:
        assert c["failed_gates"], "a gate failure must name which gate failed"
        assert c["total"] <= 59.0, "the gate cap must be visible in the reported total"
    assert state["summary"]["needsReview"] == len(failed)


def test_collect_state_has_all_rubric_dimensions(pipeline_output: Path) -> None:
    state = collect_state(pipeline_output)
    assert len(state["dims"]) == 8
    gates = [d for d in state["dims"] if d["gate"]]
    assert {d["id"] for d in gates} == {"identification", "compliance"}
    for d in state["dims"]:
        assert 1.0 <= d["avg"] <= 5.0


def test_collect_state_includes_audit_trail(pipeline_output: Path) -> None:
    audit = collect_state(pipeline_output)["audit"]
    assert audit["promptVersion"] and audit["promptSha"] and audit["engine"]


def test_collect_state_survives_an_empty_output_dir(tmp_path: Path) -> None:
    """Day one: nothing processed yet. The dashboard must still render."""
    state = collect_state(tmp_path)
    assert state["calls"] == []
    assert state["summary"]["nCalls"] == 0
    assert state["summary"]["meanTotal"] == 0


# ---------------------------------------------------------------- no PII

def test_no_raw_pii_reaches_the_dashboard(pipeline_output: Path) -> None:
    """Only redacted text may leave the pipeline - the API is no exception."""
    blob = json.dumps(collect_state(pipeline_output), ensure_ascii=False)
    assert RAW_ID not in blob
    assert RAW_PHONE not in blob


def test_redaction_actually_ran_on_this_sample(pipeline_output: Path) -> None:
    """Guards the test above from passing vacuously: the sample set really does
    contain PII, and the redacted artifacts really do carry the masks. (Which
    quotes the judge happens to cite is not something to assert on.)"""
    redacted = list((pipeline_output / "redacted").glob("*.json"))
    assert redacted
    masked = [p for p in redacted if "████" in p.read_text(encoding="utf-8")]
    assert masked, "the sample set seeds PII, so some transcript must be masked"
    for path in redacted:
        text = path.read_text(encoding="utf-8")
        assert RAW_ID not in text and RAW_PHONE not in text


# ---------------------------------------------------------------- security

@pytest.fixture()
def live_server(pipeline_output: Path):
    from http.server import ThreadingHTTPServer

    Handler.token = "test-token-value"
    Handler.output_dir = pipeline_output
    srv = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    time.sleep(0.2)
    yield f"http://127.0.0.1:{srv.server_port}"
    srv.shutdown()


def _status(url: str, headers: dict | None = None) -> int:
    req = urllib.request.Request(url, headers=headers or {})
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return resp.status
    except urllib.error.HTTPError as exc:
        return exc.code


def test_requires_token(live_server: str) -> None:
    assert _status(f"{live_server}/api/state") == 403
    assert _status(f"{live_server}/api/state?t=wrong") == 403
    assert _status(f"{live_server}/api/state?t=test-token-value") == 200


def test_rejects_foreign_origin(live_server: str) -> None:
    """A page you visit must not be able to drive this console."""
    assert _status(f"{live_server}/api/state?t=test-token-value",
                   {"Origin": "https://evil.example"}) == 403


def test_rejects_spoofed_host(live_server: str) -> None:
    """DNS rebinding resolves an attacker domain to 127.0.0.1; the Host header
    still names that domain, so it is refused."""
    assert _status(f"{live_server}/api/state?t=test-token-value",
                   {"Host": "evil.example"}) == 403


def test_report_path_traversal_is_refused(live_server: str) -> None:
    assert _status(f"{live_server}/reports/../../../etc/passwd?t=test-token-value") == 404


def test_serves_generated_reports(live_server: str) -> None:
    assert _status(f"{live_server}/reports/index.html?t=test-token-value") == 200


def test_page_is_served_with_its_token(live_server: str) -> None:
    with urllib.request.urlopen(f"{live_server}/?t=test-token-value", timeout=10) as resp:
        html = resp.read().decode("utf-8")
    assert 'window.__CALLQA_TOKEN__="test-token-value"' in html
    assert 'dir="rtl"' in html or "direction:rtl" in html
