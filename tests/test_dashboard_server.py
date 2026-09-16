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


def _chrome() -> str | None:
    """The same Chromium lookup the QA harness uses, without importing it
    (that module pulls in playwright, which a bank machine will not have)."""
    import glob
    import shutil

    for pattern in ("/opt/pw-browsers/chromium*/chrome-linux/chrome",
                    str(Path.home() / ".cache/ms-playwright/chromium*/chrome-linux/chrome")):
        matches = sorted(glob.glob(pattern))
        if matches:
            return matches[-1]
    return shutil.which("chromium") or shutil.which("google-chrome")


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


def test_state_does_not_carry_transcripts(pipeline_output: Path) -> None:
    """/api/state is polled and must stay small: the full transcript has its
    own per-call endpoint, fetched only when the detail drawer opens. A
    hundred calls of transcript text in the state blob would choke the page."""
    state = collect_state(pipeline_output)
    for call in state["calls"]:
        assert "turns" not in call and "transcript" not in call


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


def test_sibling_directory_is_not_served(live_server: str, pipeline_output: Path) -> None:
    """A string-prefix guard also accepts a SIBLING directory whose name merely
    starts with the allowed one. reports_backup/ must not be reachable."""
    sibling = pipeline_output / "reports_backup"
    sibling.mkdir(exist_ok=True)
    (sibling / "secret.html").write_text("<p>not for the browser</p>", encoding="utf-8")
    assert _status(f"{live_server}/reports/../reports_backup/secret.html"
                   f"?t=test-token-value") == 404


def test_serves_generated_reports(live_server: str) -> None:
    assert _status(f"{live_server}/reports/index.html?t=test-token-value") == 200


# ------------------------------------------------------- transcript endpoint

def _get_json(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=10) as resp:
        return json.loads(resp.read().decode("utf-8"))


def test_transcript_endpoint_serves_redacted_turns(live_server: str) -> None:
    body = _get_json(f"{live_server}/api/transcript/CALL001?t=test-token-value")
    assert body["call_id"] == "CALL001"
    assert body["turns"], "the mock run produced a dialog for CALL001"
    for turn in body["turns"]:
        assert turn["speaker"] in ("banker", "customer")
        assert isinstance(turn["start"], (int, float))
    joined = " ".join(t["text"] for t in body["turns"])
    # the sample set seeds PII into dialog 0, so the masks must be VISIBLE:
    # showing the reviewer what was removed is the point of the view
    assert "████" in joined


def test_transcript_endpoint_never_serves_raw_pii(live_server: str, pipeline_output: Path) -> None:
    """The per-call transcript endpoint is a second door out of the pipeline;
    the raw-PII rule applies to it exactly as it does to /api/state."""
    for path in sorted((pipeline_output / "redacted").glob("*.json")):
        call_id = path.stem
        blob = json.dumps(
            _get_json(f"{live_server}/api/transcript/{call_id}?t=test-token-value"),
            ensure_ascii=False)
        assert RAW_ID not in blob
        assert RAW_PHONE not in blob


def test_transcript_endpoint_requires_token(live_server: str) -> None:
    assert _status(f"{live_server}/api/transcript/CALL001") == 403
    assert _status(f"{live_server}/api/transcript/CALL001?t=wrong") == 403


def test_transcript_endpoint_rejects_bad_call_ids(live_server: str) -> None:
    """The call_id charset check is the traversal guard for this endpoint."""
    for bad in ("..%2F..%2Fresults%2FCALL001", "a%2Fb", ".hidden", "x" * 80):
        assert _status(f"{live_server}/api/transcript/{bad}?t=test-token-value") == 404
    assert _status(f"{live_server}/api/transcript/NOSUCH?t=test-token-value") == 404


def test_transcript_endpoint_refuses_disabled_redaction(live_server: str,
                                                        pipeline_output: Path) -> None:
    """A disabled-redaction artifact carries RAW text for the pipeline's own
    consumers (loudly, by design). The browser must get the fact, not the text."""
    artifact = {
        "call_id": "RAWLEAK1", "engine": "regex:DISABLED", "enabled": False,
        "redaction_counts": {},
        "turns": [{"speaker": "customer", "start": 0.0, "end": 2.0,
                   "text": f"תעודת הזהות שלי {RAW_ID}"}],
    }
    path = pipeline_output / "redacted" / "RAWLEAK1.json"
    path.write_text(json.dumps(artifact, ensure_ascii=False), encoding="utf-8")
    try:
        body = _get_json(f"{live_server}/api/transcript/RAWLEAK1?t=test-token-value")
        assert body["enabled"] is False
        assert body["turns"] == []
        assert RAW_ID not in json.dumps(body)
    finally:
        path.unlink()


def test_page_is_served_with_its_token(live_server: str) -> None:
    with urllib.request.urlopen(f"{live_server}/?t=test-token-value", timeout=10) as resp:
        html = resp.read().decode("utf-8")
    assert 'window.__CALLQA_TOKEN__="test-token-value"' in html
    assert 'dir="rtl"' in html or "direction:rtl" in html


# ------------------------------------------------- live page, real browser

@pytest.mark.skipif(_chrome() is None, reason="no Chromium available")
def test_page_applies_live_data_without_js_errors(live_server: str) -> None:
    """The page must actually consume /api/state.

    Two real bugs hid here: helper functions ended up scoped inside the render
    function, so the boot block threw ReferenceError and every update after it
    was silently skipped. The page still looked fine because it fell back to
    its embedded sample figures. Only a browser check catches that.
    """
    playwright = pytest.importorskip("playwright.sync_api")
    chrome = _chrome()
    with playwright.sync_playwright() as p:
        browser = p.chromium.launch(executable_path=chrome)
        page = browser.new_context(viewport={"width": 1280, "height": 900},
                                   locale="he-IL").new_page()
        errors: list[str] = []
        page.on("pageerror", lambda e: errors.append(str(e)))
        page.goto(f"{live_server}/?t=test-token-value")
        page.wait_for_timeout(2000)

        assert not errors, f"page threw: {errors}"

        # the calibration panel must show the real verdict, not the placeholder
        flagged = page.evaluate("() => document.getElementById('calFlagged').textContent")
        assert flagged.isdigit()

        # a call row must open the drawer and offer its generated report
        page.click(".navlink[data-view='calls']")
        page.wait_for_timeout(250)
        page.click("#allTable tbody tr")
        page.wait_for_timeout(350)
        assert not page.evaluate("() => document.getElementById('drawer').hidden"), \
            "clicking a call row must open the detail drawer"
        assert not page.evaluate("() => document.getElementById('openReport').hidden"), \
            "a served call must link to its generated report"
        href = page.evaluate("() => document.getElementById('openReport').getAttribute('href')")
        assert page.request.get(live_server + href).status == 200
        browser.close()


def test_transcript_endpoint_bounds_a_pathological_artifact(live_server: str,
                                                            pipeline_output: Path) -> None:
    """A corrupt/huge redacted artifact must not serve an unbounded body:
    the endpoint caps turn count and per-turn length and flags truncation."""
    from server import MAX_TRANSCRIPT_TURNS
    artifact = {
        "call_id": "HUGE1", "engine": "regex", "enabled": True,
        "redaction_counts": {},
        "turns": [{"speaker": "banker", "start": float(i), "end": float(i) + 1,
                   "text": "x" * 9000} for i in range(MAX_TRANSCRIPT_TURNS + 500)],
    }
    path = pipeline_output / "redacted" / "HUGE1.json"
    path.write_text(json.dumps(artifact), encoding="utf-8")
    try:
        body = _get_json(f"{live_server}/api/transcript/HUGE1?t=test-token-value")
        assert body["truncated"] is True
        assert len(body["turns"]) == MAX_TRANSCRIPT_TURNS
        assert all(len(t["text"]) <= 4000 for t in body["turns"])
    finally:
        path.unlink()
