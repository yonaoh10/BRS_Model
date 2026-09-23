#!/usr/bin/env python3
"""Local operator dashboard for the Call-QA pipeline.

Runs on the operator's own machine and reads the artifacts the pipeline has
already written. Start it with:

    python dashboard/server.py

Deliberately NOT wired into `python -m callqa`: keeping it out of the core CLI
is what lets `rm -rf dashboard/` remove this feature without touching the
package the bank receives.

DESIGN NOTES
- stdlib only. The project has no JavaScript toolchain and a hard offline
  constraint, and this is a single-user localhost console, so FastAPI/uvicorn
  would add dependencies for nothing. The same reasoning produced
  a single-purpose stdlib server.
- Read-only by default. `--allow-actions` is required before any endpoint can
  spend money or mutate state, and even then each action is explicit.
- Bound to 127.0.0.1. Never 0.0.0.0: it serves call material, so it must not
  be reachable from the network.
- A random session token is required on every request and is embedded in the
  URL printed at startup. Together with the Origin/Host checks below this
  blocks DNS-rebinding: a malicious page you happen to visit can otherwise
  reach a localhost server and drive it.

SCOPE: this whole directory is removable. The bank deliverable stays CLI plus
static HTML (see dashboard/DESIGN.md).
"""

from __future__ import annotations

import argparse
import hmac
import json
import logging
import os
import re
import secrets
import statistics
import sys
import threading
import webbrowser
from collections import defaultdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from callqa.dotenv import load_dotenv  # noqa: E402
from callqa.ingestion import CALL_ID_RE  # noqa: E402

logger = logging.getLogger("callqa.dashboard")

PAGE = Path(__file__).parent / "prototype.html"
ALLOWED_HOSTS = {"127.0.0.1", "localhost", "[::1]"}
# Bounds on one /api/transcript response - a real call is far under these; the
# caps stop a corrupt or pathological artifact from serving an unbounded body.
MAX_TRANSCRIPT_TURNS = 5000
MAX_TURN_CHARS = 4000


# ---------------------------------------------------------------- data layer

def _load_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def collect_state(output_dir: Path, config_path: Path | None = None) -> dict:
    """Assemble everything the dashboard shows, straight off disk.

    Deliberately tolerant: a half-finished run must still render. Anything
    missing simply does not appear.
    """
    from callqa.rubric import load_rubric

    rubric_path = REPO_ROOT / "config" / "rubric.yaml"
    dims_meta: list[dict] = []
    gate_ids: set[str] = set()
    if rubric_path.exists():
        rubric = load_rubric(rubric_path)
        gate_ids = set(rubric.gate_ids)
        dims_meta = [{"id": d.id, "he": d.name_he, "gate": d.gate, "weight": d.weight}
                     for d in rubric.dimensions]

    # Tolerant by design: one unreadable scorecard must not take the whole
    # console down, it must simply be absent from it.
    cards = [c for c in (_load_json(p) for p in sorted((output_dir / "scores").glob("*.json")))
             if isinstance(c, dict) and c.get("call_id")]
    results = [r for r in (_load_json(p) for p in sorted((output_dir / "results").glob("*.json"))) if r]
    result_by_id = {r["call_id"]: r for r in results}

    calls = []
    for c in cards:
        res = result_by_id.get(c["call_id"], {})
        calls.append({
            "id": c["call_id"],
            "banker": c.get("banker_id", "—"),
            "total": c.get("weighted_total", 0),
            "gate": bool(c.get("gate_failed")),
            "failed_gates": c.get("failed_gates", []),
            "status": res.get("status", "success"),
            "date": (c.get("timestamp") or "")[:10],
            "summary": c.get("summary_he", ""),
            "scores": {k: v.get("score") for k, v in (c.get("scores") or {}).items()},
            "evidence": [
                {"dim": k, "t": e.get("timestamp"), "sp": e.get("speaker"), "q": e.get("quote")}
                for k, v in (c.get("scores") or {}).items()
                for e in (v.get("evidence") or [])[:1]
            ][:4],
            "report": f"reports/calls/{c['call_id']}.html",
        })
    calls.sort(key=lambda x: x["total"], reverse=True)

    # per-dimension averages
    dims = []
    for meta in dims_meta:
        vals = [c["scores"].get(meta["id"]) for c in calls if c["scores"].get(meta["id"]) is not None]
        if vals:
            dims.append({"id": meta["id"], "he": meta["he"], "gate": meta["gate"],
                         "avg": round(statistics.mean(vals), 2)})

    # per banker
    by_banker: dict[str, list] = defaultdict(list)
    for c in calls:
        by_banker[c["banker"]].append(c)
    bankers = [{"id": b, "n": len(cs),
                "mean": round(statistics.mean(x["total"] for x in cs), 1)}
               for b, cs in sorted(by_banker.items())]
    group_median = round(statistics.median(c["total"] for c in calls), 1) if calls else 0

    calibration = _load_json(output_dir / "reports" / "calibration.json") or {}

    # in-flight work, straight from the state DB
    running = []
    try:
        from callqa.pipeline import STAGES
        from callqa.state import StateDB

        db_path = REPO_ROOT / "data" / "callqa_state.db"
        if db_path.exists():
            state = StateDB(db_path)
            done_ids = {c["id"] for c in calls}
            import sqlite3
            with sqlite3.connect(db_path) as conn:
                # "in flight" = currently holds a processing lock. Stage-count
                # rows persist after a call finishes, so counting done stages
                # showed every completed call as still running; a live lock is
                # the honest "being processed right now" signal.
                locked = [r[0] for r in conn.execute("SELECT call_id FROM locks").fetchall()]
                done = dict(conn.execute(
                    "SELECT call_id, COUNT(*) FROM stages WHERE status='done' GROUP BY call_id"
                ).fetchall())
            for call_id in locked:
                if call_id in done_ids:
                    continue                      # already has a scorecard: finished
                n_done = done.get(call_id, 0)
                running.append({"id": call_id, "done": n_done,
                                "stage": STAGES[min(n_done, len(STAGES) - 1)]})
            _ = state
    except Exception as exc:  # noqa: BLE001 - the dashboard must never crash on this
        logger.debug("could not read run state: %s", exc)

    needs_review = sum(1 for c in calls if c["gate"] or c["status"] == "needs_human_review")
    return {
        "calls": calls,
        "dims": dims,
        "bankers": bankers,
        "groupMedian": group_median,
        "stages": list(__import__("callqa.pipeline", fromlist=["STAGES"]).STAGES),
        "running": running,
        "summary": {
            "nCalls": len(calls),
            "meanTotal": round(statistics.mean(c["total"] for c in calls), 1) if calls else 0,
            "needsReview": needs_review,
            "gateIds": sorted(gate_ids),
        },
        "calibration": {
            "qwk": calibration.get("overall_qwk"),
            "pass": calibration.get("overall_pass"),
            "nCalls": calibration.get("n_calls"),
            "nDouble": (calibration.get("human_vs_human") or {}).get("n_doubly_rated_calls"),
            "flagged": calibration.get("flagged_dimensions", []),
            # The dashboard verdict distinguishes "QWK passes but too few calls"
            # from "QWK below threshold"; both need the real thresholds rather
            # than a 0.70/20 baseline guessed in the page.
            "threshold": calibration.get("qwk_pass_threshold"),
            "minCalls": calibration.get("min_calls_for_pass"),
        },
        "audit": {
            "promptVersion": cards[0].get("prompt_version") if cards else None,
            "promptSha": (cards[0].get("prompt_sha256") or "")[:16] if cards else None,
            "model": cards[0].get("model") if cards else None,
            "engine": cards[0].get("judge_engine") if cards else None,
            "lastRun": cards[0].get("timestamp") if cards else None,
        },
    }


# ---------------------------------------------------------------- http layer

def _audio_is_current(output_dir: Path, call_id: str) -> bool:
    """Is the silenced WAV at least as new as the redacted transcript?

    The WAV is silenced where the transcript's mask was when it was made. When
    the mask is recomputed the pipeline regenerates the WAV, and if that fails
    it deletes the old one - but a file another program holds open (a media
    player on Windows) cannot always be deleted. An older WAV may leave audible
    an identifier the new mask covers, so it is never served.
    """
    wav = output_dir / "redacted_audio" / f"{call_id}.wav"
    transcript = output_dir / "redacted" / f"{call_id}.json"
    try:
        return wav.is_file() and wav.stat().st_mtime >= transcript.stat().st_mtime
    except OSError:
        return False


class Handler(BaseHTTPRequestHandler):
    token = ""
    output_dir = Path("data/output")
    allow_actions = False

    server_version = "callqa-dashboard/1.0"

    def _authorized(self) -> bool:
        """Token in the query string, plus Origin/Host checks.

        The header checks are what stop a web page you visit from driving this
        server through your browser (DNS rebinding); the token stops anything
        that never saw the startup URL.
        """
        host = (self.headers.get("Host") or "").rsplit(":", 1)[0]
        if host not in ALLOWED_HOSTS:
            return False
        origin = self.headers.get("Origin")
        if origin:
            origin_host = urlparse(origin).hostname or ""
            if origin_host not in ALLOWED_HOSTS:
                return False
        supplied = parse_qs(urlparse(self.path).query).get("t", [""])[0]
        return hmac.compare_digest(supplied, type(self).token)

    def _send(self, code: int, body: bytes, ctype: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        # This console never belongs in a frame on another page.
        self.send_header("X-Frame-Options", "DENY")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("X-Content-Type-Options", "nosniff")
        # The page is self-contained; nothing here should ever reach the network.
        self.send_header(
            "Content-Security-Policy",
            "default-src 'self'; script-src 'self' 'unsafe-inline'; "
            "style-src 'self' 'unsafe-inline'; img-src 'self' data:; "
            # media-src 'self': the page plays its own redacted audio, served
            # same-origin from /api/audio. It would fall back to default-src
            # 'self' anyway; naming it is the explicit statement of intent.
            "media-src 'self'; "
            "connect-src 'self'; form-action 'none'; frame-ancestors 'none'; "
            "base-uri 'none'",
        )
        self.end_headers()
        self.wfile.write(body)

    def _send_audio(self, path: Path) -> None:
        """Serve a WAV, honouring a single HTTP Range request.

        A browser's <audio> element issues Range requests to seek; without a
        206 the whole file is re-fetched on every scrub. Only the simple
        ``bytes=start-end`` / ``bytes=start-`` / ``bytes=-suffix`` forms are
        supported - enough for playback, and anything else falls back to the
        full body.
        """
        size = path.stat().st_size
        start, end, partial = 0, size - 1, False
        raw = self.headers.get("Range")
        if raw:
            m = re.fullmatch(r"\s*bytes=(\d*)-(\d*)\s*", raw)
            if m and (m.group(1) or m.group(2)):
                if m.group(1):
                    start = int(m.group(1))
                    end = int(m.group(2)) if m.group(2) else size - 1
                else:  # suffix range: the last N bytes
                    start = max(0, size - int(m.group(2)))
                    end = size - 1
                if start >= size or start > end:
                    self.send_response(416)
                    self.send_header("Content-Range", f"bytes */{size}")
                    self.send_header("Content-Length", "0")
                    self.send_header("Accept-Ranges", "bytes")
                    self.end_headers()
                    return
                end = min(end, size - 1)
                partial = True
        with path.open("rb") as fh:
            fh.seek(start)
            body = fh.read(end - start + 1)
        self.send_response(206 if partial else 200)
        self.send_header("Content-Type", "audio/wav")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Accept-Ranges", "bytes")
        if partial:
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Frame-Options", "DENY")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            # A browser aborts the audio request on every seek and on close;
            # a dropped client is normal here, not an error to log.
            pass

    def log_message(self, fmt: str, *a) -> None:
        # The session token rides in the query string, so the raw request line
        # must never be logged: it would leave a working credential for this
        # console in the terminal scrollback and in any captured log.
        message = fmt % a
        logger.info("%s", re.sub(r"\?t=[^\s\"]+", "?t=<redacted>", message))

    def do_GET(self) -> None:  # noqa: N802
        route = urlparse(self.path).path
        if not self._authorized():
            self._send(403, b'{"error":"forbidden"}', "application/json")
            return
        if route in ("/", "/index.html"):
            html = PAGE.read_text(encoding="utf-8")
            # Hand the page its own token. This must land BEFORE the page's own
            # script: that script calls loadLive() as it runs, and loadLive
            # reads the token immediately.
            token_tag = f'<script>window.__CALLQA_TOKEN__="{type(self).token}";</script>\n'
            if "<script>" in html:
                html = html.replace("<script>", token_tag + "<script>", 1)
            else:
                html = token_tag + html
            self._send(200, html.encode("utf-8"), "text/html; charset=utf-8")
            return
        if route == "/api/state":
            try:
                data = collect_state(type(self).output_dir)
            except Exception:  # noqa: BLE001
                # Deliberately not echoed: the exception text can carry file
                # paths and artifact content, and this body is rendered in a
                # browser.
                logger.exception("failed to collect state")
                self._send(500, b'{"error":"could not read the pipeline output"}',
                           "application/json")
                return
            self._send(200, json.dumps(data, ensure_ascii=False).encode("utf-8"),
                       "application/json; charset=utf-8")
            return
        if route.startswith("/api/transcript/"):
            # The full REDACTED transcript for one call, on its own endpoint:
            # /api/state must stay small enough to poll with a hundred calls
            # on disk, so transcripts are fetched one at a time when the
            # detail drawer opens. Reads ONLY redacted/ - the raw transcripts
            # under transcripts/ have no route to the browser, and the
            # call_id charset check (no separators beyond ._-, first char
            # alphanumeric) means the id cannot traverse out of the directory.
            call_id = unquote(route[len("/api/transcript/"):])
            if not CALL_ID_RE.fullmatch(call_id):
                self._send(404, b'{"error":"not found"}', "application/json")
                return
            redacted = _load_json(type(self).output_dir / "redacted" / f"{call_id}.json")
            if not isinstance(redacted, dict) or "turns" not in redacted:
                self._send(404, b'{"error":"not found"}', "application/json")
                return
            card = _load_json(type(self).output_dir / "scores" / f"{call_id}.json") or {}
            evidence = [
                {"dim": dim, "quote": e.get("quote"),
                 "speaker": e.get("speaker"), "t": e.get("timestamp")}
                for dim, v in (card.get("scores") or {}).items()
                for e in (v.get("evidence") or [])
                if isinstance(e, dict)
            ]
            # A disabled-redaction artifact carries the RAW text (loudly, by
            # design, for the pipeline's own consumers) - it must not reach a
            # browser. Serve the fact, never the turns.
            enabled = bool(redacted.get("enabled", True))
            all_turns = [t for t in (redacted.get("turns") or []) if isinstance(t, dict)]
            # Bound the response: a real call is a few hundred short turns; an
            # artifact with tens of thousands (corrupt, or a very long call)
            # would otherwise be read and serialised whole into one body. The
            # cap keeps a single request cheap; a truncation flag tells the UI.
            truncated = len(all_turns) > MAX_TRANSCRIPT_TURNS
            # Redacted-audio metadata for the drawer player. Present only when
            # redaction is enabled AND both the silenced WAV and its sidecar
            # exist - so a call with no playable audio (or a disabled-redaction
            # call, which must never expose audio) reports available:false and
            # the page shows the transcript alone.
            audio = {"available": False}
            audio_meta = _load_json(
                type(self).output_dir / "redacted_audio" / f"{call_id}.json")
            wav_present = _audio_is_current(type(self).output_dir, call_id)
            if enabled and wav_present and isinstance(audio_meta, dict):
                audio = {
                    "available": True,
                    "duration": audio_meta.get("duration"),
                    "silences": audio_meta.get("silences") or [],
                }
            body = {
                "call_id": call_id,
                "enabled": enabled,
                "truncated": truncated,
                "turns": [
                    {"speaker": t.get("speaker"), "start": t.get("start"),
                     "end": t.get("end"),
                     "text": str(t.get("text") or "")[:MAX_TURN_CHARS]}
                    for t in all_turns[:MAX_TRANSCRIPT_TURNS]
                ] if enabled else [],
                "evidence": evidence if enabled else [],
                "audio": audio,
            }
            self._send(200, json.dumps(body, ensure_ascii=False).encode("utf-8"),
                       "application/json; charset=utf-8")
            return
        if route.startswith("/api/audio/"):
            # The REDACTED audio for one call - silenced wherever the transcript
            # was masked (see callqa.audio_redaction). Serves ONLY from
            # redacted_audio/; the raw recordings under audio/wav/ have no route
            # to the browser. The call_id charset check plus the containment
            # check below are the traversal guards, exactly as for transcripts.
            call_id = unquote(route[len("/api/audio/"):])
            if not CALL_ID_RE.fullmatch(call_id):
                self._send(404, b'{"error":"not found"}', "application/json")
                return
            audio_root = (type(self).output_dir / "redacted_audio").resolve()
            target = (audio_root / f"{call_id}.wav").resolve()
            if not target.is_relative_to(audio_root) or not target.is_file() \
                    or not _audio_is_current(type(self).output_dir, call_id):
                self._send(404, b'{"error":"not found"}', "application/json")
                return
            # Defence in depth: the pipeline never writes audio for a
            # disabled-redaction call, but if a stale silenced WAV from an
            # earlier enabled run is still on disk, do not serve it once
            # redaction has been turned off - mirror the transcript route.
            redacted = _load_json(type(self).output_dir / "redacted" / f"{call_id}.json")
            if isinstance(redacted, dict) and redacted.get("enabled") is False:
                self._send(404, b'{"error":"not found"}', "application/json")
                return
            self._send_audio(target)
            return
        if route.startswith("/reports/"):
            # Serve the generated per-call and per-banker reports.
            # Percent-decoded, so a report whose id contains a space or a
            # non-ASCII character is reachable; the containment check below is
            # what keeps that safe.
            rel = unquote(route[len("/reports/"):])
            target = (type(self).output_dir / "reports" / rel).resolve()
            root = (type(self).output_dir / "reports").resolve()
            # is_relative_to, not startswith: a string prefix also accepts a
            # SIBLING directory whose name merely starts with "reports"
            # (reports_backup, reports-old), which would serve files outside it.
            if not target.is_relative_to(root) or not target.is_file():
                self._send(404, b'{"error":"not found"}', "application/json")
                return
            self._send(200, target.read_bytes(), "text/html; charset=utf-8")
            return
        self._send(404, b'{"error":"not found"}', "application/json")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "data" / "output")
    parser.add_argument("--allow-actions", action="store_true",
                        help="permit endpoints that mutate state or spend money (not yet implemented)")
    parser.add_argument("--no-browser", action="store_true")
    parser.add_argument("--bind", default="127.0.0.1",
                        help="interface to listen on. Only ever change this inside a "
                             "container whose port is published to 127.0.0.1 on the host.")
    args = parser.parse_args()
    load_dotenv()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    # A fixed token from .env keeps the URL stable across restarts, which is
    # what makes the dashboard usable from a container; otherwise a fresh one.
    Handler.token = os.environ.get("CALLQA_DASHBOARD_TOKEN") or secrets.token_urlsafe(24)
    Handler.output_dir = args.output_dir
    Handler.allow_actions = args.allow_actions

    if not PAGE.exists():
        print(f"ERROR: dashboard page missing at {PAGE}", file=sys.stderr)
        return 2

    # 127.0.0.1 only - this console serves call material. The
    # container is the one legitimate exception: it listens on all of ITS
    # interfaces when a container runtime publishes the port to the host.
    if args.bind != "127.0.0.1":
        logger.warning("listening on %s: make sure this port is only reachable from "
                       "this machine", args.bind)
    server = ThreadingHTTPServer((args.bind, args.port), Handler)
    url = f"http://127.0.0.1:{args.port}/?t={Handler.token}"
    print("\n  Call-QA dashboard is running.")
    print(f"  Open: {url}")
    print("  (the token in the link is required"
          + ("; set CALLQA_DASHBOARD_TOKEN in .env to keep it stable)"
             if not os.environ.get("CALLQA_DASHBOARD_TOKEN") else ")"))
    print("  Press Ctrl+C to stop.\n")
    if not args.no_browser:
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n  stopped.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
