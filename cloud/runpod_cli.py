#!/usr/bin/env python3
"""Control the rented GPU machine on RunPod (DEV PHASE ONLY).

Everything here talks to https://rest.runpod.io/v1 over HTTPS with a single
API key - no SSH, no browser. Delete the `cloud/` directory when the project
moves to the bank's own servers and nothing else changes.

    export RUNPOD_API_KEY=...
    python cloud/runpod_cli.py up          # create/start the pod, print URLs
    python cloud/runpod_cli.py status      # is it running? what does it cost?
    python cloud/runpod_cli.py urls        # print the two endpoint URLs
    python cloud/runpod_cli.py down        # STOP it (billing for GPU ends)
    python cloud/runpod_cli.py destroy     # terminate + delete the volume

COST SAFETY: a running pod bills every second, whether or not you use it.
`down` is the command that stops the meter. `status` warns when a pod has
been running for longer than --warn-hours.
"""

from __future__ import annotations

import argparse
import json
import os
import secrets
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from callqa.dotenv import load_dotenv, write_env_values  # noqa: E402

API_BASE = "https://rest.runpod.io/v1"
STATE_FILE = Path(__file__).parent / ".runpod_state.json"

# Image with CUDA + PyTorch preinstalled; the bootstrap script layers our code on top.
DEFAULT_IMAGE = "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04"
ASR_PORT = 8001
JUDGE_PORT = 8000

# Runs as the container's start command, so `up` is the whole bring-up: no
# web terminal, no manual exports. The repo URL, model id and both service
# keys arrive through the pod's env (set at creation below); the volume at
# /workspace keeps the clone and the models across stop/start, and the
# trailing sleep keeps the container alive after bootstrap has launched the
# two services in the background. All output lands on the volume so a failed
# boot can be read later.
POD_START_CMD = (
    # NO `set -x` here: the command tests $HF_TOKEN and $PUBLIC_KEY, and -x
    # would trace the secret values into bootstrap.log.
    "mkdir -p /workspace/logs; "
    "exec > >(tee -a /workspace/logs/bootstrap.log) 2>&1; "
    # SSH first, so a failed bootstrap can still be reached and read.
    'if [ -n "${PUBLIC_KEY:-}" ]; then mkdir -p /root/.ssh; '
    'echo "$PUBLIC_KEY" > /root/.ssh/authorized_keys; '
    "chmod 700 /root/.ssh; chmod 600 /root/.ssh/authorized_keys; "
    "command -v sshd >/dev/null || (apt-get update -qq && "
    "apt-get install -y -qq openssh-server); "
    # A fresh container has no SSH host keys and sshd refuses to start
    # without them; -A generates any that are missing.
    "ssh-keygen -A; mkdir -p /run/sshd; "
    "service ssh start || /usr/sbin/sshd || true; fi; "
    "command -v git >/dev/null || (apt-get update -qq && apt-get install -y -qq git); "
    "cd /workspace; "
    "if [ ! -d BRS_Model/.git ]; then "
    'GIT_TERMINAL_PROMPT=0 git clone "$CALLQA_REPO_URL" BRS_Model '
    "|| echo 'ERROR: clone failed. A private repo cannot be cloned from the "
    "pod; rsync the working tree to /workspace/BRS_Model over SSH instead "
    "and run cloud/bootstrap_pod.sh by hand.'; fi; "
    "cd BRS_Model 2>/dev/null && { git pull --ff-only || true; "
    "bash cloud/bootstrap_pod.sh; }; "
    # Diarization weights are gated; fetched only when the operator put
    # HF_TOKEN in .env. Harmless no-op otherwise.
    'if [ -n "${HF_TOKEN:-}" ]; then '
    "python scripts/download_models.py --diarization --models-dir models || true; fi; "
    "sleep infinity"
)


class RunPodError(RuntimeError):
    pass


def _request(method: str, path: str, payload: dict | None = None) -> dict:
    api_key = os.environ.get("RUNPOD_API_KEY", "")
    if not api_key:
        raise RunPodError(
            "RUNPOD_API_KEY is not set. Create a key at "
            "https://console.runpod.io/user/settings (API Keys), then paste it "
            f"into {REPO_ROOT / '.env'} on the line RUNPOD_API_KEY= "
            "(copy .env.example to .env if the file does not exist yet)."
        )
    url = f"{API_BASE}{path}"
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    req = urllib.request.Request(
        url,
        data=data,
        method=method,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            # Cloudflare in front of the RunPod API rejects urllib's default
            # Python-urllib/x.y User-Agent with a 403 (error code 1010).
            "User-Agent": "callqa-runpod-cli/1.0",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            body = resp.read().decode("utf-8")
            return json.loads(body) if body.strip() else {}
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:500]
        raise RunPodError(f"RunPod API {method} {path} failed (HTTP {exc.code}): {detail}") from exc
    except (urllib.error.URLError, OSError) as exc:
        raise RunPodError(f"cannot reach the RunPod API: {exc}") from exc


def load_state() -> dict:
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text(encoding="utf-8"))
    return {}


def save_state(state: dict) -> None:
    # Opened 0600 rather than written and then chmod'ed: between those two
    # calls the file holding the pod's shared secrets existed at the process
    # umask, which on a default machine is world-readable.
    fd = os.open(STATE_FILE, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        fh.write(json.dumps(state, indent=2))
    STATE_FILE.chmod(0o600)


def proxy_url(pod_id: str, port: int) -> str:
    return f"https://{pod_id}-{port}.proxy.runpod.net"


# -- commands ---------------------------------------------------------------

def cmd_up(args: argparse.Namespace) -> int:
    state = load_state()
    pod_id = state.get("pod_id")

    if pod_id:
        pod = _request("GET", f"/pods/{pod_id}")
        status = pod.get("desiredStatus") or pod.get("status")
        if status == "RUNNING":
            print(f"Pod {pod_id} is already running.")
            return _print_urls(state)
        print(f"Pod {pod_id} exists (status {status}); starting it...")
        _request("POST", f"/pods/{pod_id}/start")
    else:
        # Secrets are generated once and reused; both services check them.
        state.setdefault("asr_api_key", secrets.token_urlsafe(32))
        state.setdefault("judge_api_key", secrets.token_urlsafe(32))
        payload = {
            "name": args.name,
            "imageName": args.image,
            "gpuTypeIds": [args.gpu],
            "gpuCount": 1,
            "containerDiskInGb": args.disk,
            "volumeInGb": args.volume,
            "volumeMountPath": "/workspace",
            "ports": [f"{ASR_PORT}/http", f"{JUDGE_PORT}/http", "22/tcp"],
            "dockerStartCmd": ["bash", "-c", POD_START_CMD],
            "env": {
                "CALLQA_ASR_API_KEY": state["asr_api_key"],
                "VLLM_API_KEY": state["judge_api_key"],
                "CALLQA_REPO_URL": args.repo_url,
                "CALLQA_LLM_MODEL": args.llm_model,
                **({"PUBLIC_KEY": args.ssh_public_key} if args.ssh_public_key else {}),
                **({"HF_TOKEN": os.environ["HF_TOKEN"]}
                   if os.environ.get("HF_TOKEN") else {}),
            },
        }
        print(f"Creating pod ({args.gpu})...")
        pod = _request("POST", "/pods", payload)
        pod_id = pod.get("id") or pod.get("podId")
        if not pod_id:
            raise RunPodError(f"pod created but no id in response: {pod}")
        state["pod_id"] = pod_id
        state["gpu"] = args.gpu
        state["llm_model"] = args.llm_model

    state["last_started_at"] = time.time()
    save_state(state)
    print(f"Pod {pod_id} starting. It bootstraps itself (clone + deps + models + serve);")
    print("first boot downloads the models, expect 10-20 minutes. Progress is written to")
    print("/workspace/logs/bootstrap.log on the pod. Poll readiness with:")
    print("  python cloud/runpod_cli.py status")
    return _print_urls(state)


def _print_urls(state: dict) -> int:
    """Hand the pod's endpoints to the pipeline by writing them into .env.

    Printing shell exports for the operator to paste was the one step in the
    cloud round trip that could not be automated away by a launcher, and a
    console that prints two API keys leaves them in the scrollback. The
    pipeline loads .env itself, so after this the next `callqa run` just works.
    """
    pod_id = state.get("pod_id")
    if not pod_id:
        print("No pod recorded yet. Run: python cloud/runpod_cli.py up")
        return 2
    values = {
        "CALLQA_ASR__BASE_URL": proxy_url(pod_id, ASR_PORT),
        "CALLQA_ASR__API_KEY": state.get("asr_api_key", ""),
        "CALLQA_JUDGE__BASE_URL": proxy_url(pod_id, JUDGE_PORT) + "/v1",
        "CALLQA_JUDGE__API_KEY": state.get("judge_api_key", ""),
    }
    target = write_env_values(values)
    print()
    print(f"Endpoints written to {target} (keys not shown):")
    print(f"  ASR   : {values['CALLQA_ASR__BASE_URL']}")
    print(f"  judge : {values['CALLQA_JUDGE__BASE_URL']}")
    print()
    print("Next:  python -m callqa run --config config/config.cloud.yaml")
    print("Stop paying when done:  python cloud/runpod_cli.py down")
    return 0


def cmd_urls(args: argparse.Namespace) -> int:
    return _print_urls(load_state())


def cmd_status(args: argparse.Namespace) -> int:
    state = load_state()
    pod_id = state.get("pod_id")
    if not pod_id:
        print("No pod recorded. Nothing is running, nothing is billing.")
        return 0
    pod = _request("GET", f"/pods/{pod_id}")
    status = pod.get("desiredStatus") or pod.get("status") or "UNKNOWN"
    cost = pod.get("costPerHr")
    print(f"pod {pod_id}: {status}" + (f"  (${cost}/hr while running)" if cost else ""))
    if status == "RUNNING":
        started = state.get("last_started_at")
        if started:
            hours = (time.time() - started) / 3600
            print(f"running for ~{hours:.1f}h" + (f"  (~${hours * float(cost):.2f})" if cost else ""))
            if hours > args.warn_hours:
                print(f"WARNING: running longer than {args.warn_hours}h and still billing.")
                print("Stop it with: python cloud/runpod_cli.py down")
    else:
        print("GPU billing is stopped. Storage for the volume may still be charged.")
    return 0


def cmd_down(args: argparse.Namespace) -> int:
    state = load_state()
    pod_id = state.get("pod_id")
    if not pod_id:
        print("No pod recorded; nothing to stop.")
        return 0
    _request("POST", f"/pods/{pod_id}/stop")
    state["last_stopped_at"] = time.time()
    save_state(state)
    print(f"Pod {pod_id} stopped. GPU billing has ended; the volume is kept so the")
    print("next `up` skips the model download. Use `destroy` to remove it entirely.")
    return 0


def cmd_destroy(args: argparse.Namespace) -> int:
    state = load_state()
    pod_id = state.get("pod_id")
    if not pod_id:
        print("No pod recorded; nothing to destroy.")
        return 0
    if not args.yes:
        print(f"This permanently deletes pod {pod_id} and its volume (models included).")
        print("Re-run with --yes to confirm.")
        return 1
    _request("DELETE", f"/pods/{pod_id}")
    STATE_FILE.unlink(missing_ok=True)
    print(f"Pod {pod_id} destroyed and local state cleared. Nothing is billing.")
    return 0


def main() -> int:
    load_dotenv()
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("up", help="create or start the GPU pod")
    p.add_argument("--gpu", default="NVIDIA GeForce RTX 4090",
                   help="GPU type id, e.g. 'NVIDIA GeForce RTX 4090' or 'NVIDIA RTX A6000'")
    p.add_argument("--image", default=DEFAULT_IMAGE)
    p.add_argument("--name", default="callqa-dev")
    p.add_argument("--disk", type=int, default=40, help="container disk GB")
    p.add_argument("--volume", type=int, default=60, help="persistent volume GB (holds models)")
    p.add_argument("--llm-model", default="dicta-il/dictalm2.0-instruct")
    p.add_argument("--repo-url", default="https://github.com/yonaoh10/BRS_Model.git")
    p.add_argument("--ssh-public-key", default=None,
                   help="OpenSSH public key line; when set, the pod runs sshd on 22/tcp")
    p.set_defaults(func=cmd_up)

    p = sub.add_parser("status", help="show pod status and running cost")
    p.add_argument("--warn-hours", type=float, default=4.0)
    p.set_defaults(func=cmd_status)

    p = sub.add_parser("urls", help="print endpoint URLs and keys")
    p.set_defaults(func=cmd_urls)

    p = sub.add_parser("down", help="STOP the pod (ends GPU billing)")
    p.set_defaults(func=cmd_down)

    p = sub.add_parser("destroy", help="terminate the pod and delete its volume")
    p.add_argument("--yes", action="store_true")
    p.set_defaults(func=cmd_destroy)

    args = parser.parse_args()
    try:
        return args.func(args)
    except RunPodError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
