#!/usr/bin/env python3
"""Control the rented GPU machine on RunPod (DEV PHASE ONLY).

Everything here talks to https://rest.runpod.io/v1 over HTTPS with a single
API key - no SSH, no browser. Delete the `cloud/` directory when the project
moves to the bank's own servers and nothing else changes.

    export RUNPOD_API_KEY=...
    python cloud/runpod_cli.py up          # create the network volume (once) + pod, print URLs
    python cloud/runpod_cli.py status      # is it running? what does it cost?
    python cloud/runpod_cli.py urls        # print the two endpoint URLs
    python cloud/runpod_cli.py down        # STOP the pod (billing for GPU ends; volume kept)
    python cloud/runpod_cli.py destroy     # terminate the pod, KEEP the volume
    python cloud/runpod_cli.py destroy --volume   # also delete the volume (loses the models)

Storage lives on a DATA-CENTRE network volume, not on the pod's host, so a
destroyed pod can be recreated on any host in that DC with a free GPU - the
"not enough free GPUs on the host machine" wall a pod-local volume hits does
not apply. The volume persists across pod destroy and bills for storage until
`destroy --volume`.

COST SAFETY: a running pod bills every second, whether or not you use it.
`down` is the command that stops the meter. `status` warns when a pod has
been running for longer than --warn-hours and always prints the standing
volume-storage cost.
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

# A NETWORK volume, not a pod-local one. A pod-local volume lives on the pod's
# physical host, so a stopped pod can only restart where there is a free GPU ON
# THAT HOST - and when the host is saturated the pod is stranded (exactly the
# HTTP 500 "not enough free GPUs on the host machine" wall we hit). A network
# volume lives in the data centre, not on a host, so a fresh pod attaches it on
# ANY host in that DC that has a free GPU. Volume storage bills whether or not a
# pod is attached (see `status`), so `destroy` keeps it and only `destroy
# --volume` removes it.
DEFAULT_VOLUME_SIZE_GB = 80          # gemma-27b (~17GB) + ASR + diarization + headroom
# Data centres in the INTERSECTION of two RunPod enums: those that support
# network volumes (POST /networkvolumes) AND those that accept pod placement
# (pods.dataCenterIds). A DC in only the first enum creates a volume fine but
# then 400s on pod create, so it must not be walked. `up` tries each until one
# has a free GPU, so a single DC being full is no longer a dead end. Re-derive
# each enum by sending an invalid dataCenterId to the respective endpoint.
CANDIDATE_DATA_CENTERS = ["US-IL-1", "US-GA-2", "US-TX-3", "EU-RO-1",
                          "US-CA-2", "CA-MTL-3", "EUR-IS-1"]

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


# -- network volume ----------------------------------------------------------

def _volume_exists(vol_id: str) -> dict | None:
    try:
        vol = _request("GET", f"/networkvolumes/{vol_id}")
        return vol if vol.get("id") else None
    except RunPodError:
        return None


def _create_volume(name: str, size: int, dc: str) -> str:
    vol = _request("POST", "/networkvolumes",
                   {"name": name, "size": size, "dataCenterId": dc})
    vol_id = vol.get("id")
    if not vol_id:
        raise RunPodError(f"volume created but no id in response: {vol}")
    return vol_id


def _delete_volume(vol_id: str) -> None:
    try:
        _request("DELETE", f"/networkvolumes/{vol_id}")
    except RunPodError as exc:                       # best effort
        print(f"  warning: could not delete volume {vol_id}: {exc}")


def _is_not_found(exc: RunPodError) -> bool:
    return "http 404" in str(exc).lower()


def _forget_pod(state: dict) -> None:
    """The pod no longer exists on RunPod (terminated in the console, expired).
    Clear only the pod fields so the CLI is not deadlocked GET-404ing a ghost;
    the network volume and keys are kept."""
    for key in ("pod_id", "gpu", "last_started_at", "last_stopped_at"):
        state.pop(key, None)
    save_state(state)


def _is_capacity_error(exc: RunPodError) -> bool:
    # RunPod phrases "this DC has no free GPU of this type right now" several
    # ways; all of them mean "try another DC", not "abort".
    msg = str(exc).lower()
    return any(s in msg for s in (
        "no instances currently available",
        "not enough free gpus",
        "could not find any pods with required specifications",
        "no longer any instances available",
    ))


# -- commands ---------------------------------------------------------------

def cmd_up(args: argparse.Namespace) -> int:
    state = load_state()
    pod_id = state.get("pod_id")

    if pod_id:
        try:
            pod = _request("GET", f"/pods/{pod_id}")
            status = pod.get("desiredStatus") or pod.get("status")
            if status == "RUNNING":
                print(f"Pod {pod_id} is already running.")
                return _print_urls(state)
            print(f"Pod {pod_id} exists (status {status}); starting it...")
            _request("POST", f"/pods/{pod_id}/start")
        except RunPodError as exc:
            if not _is_not_found(exc):
                raise
            # The recorded pod is gone (console-terminated / expired). Don't
            # deadlock GET-404ing it: forget it and create a fresh one on the
            # kept network volume.
            print(f"Recorded pod {pod_id} no longer exists; creating a new one.")
            _forget_pod(state)
            pod_id = _create_pod_on_volume(state, args)
    else:
        # Secrets are generated once and reused; both services check them.
        state.setdefault("asr_api_key", secrets.token_urlsafe(32))
        state.setdefault("judge_api_key", secrets.token_urlsafe(32))
        pod_id = _create_pod_on_volume(state, args)

    state["last_started_at"] = time.time()
    save_state(state)
    print(f"Pod {pod_id} starting. It bootstraps itself (clone + deps + models + serve);")
    print("first boot downloads the models, expect 10-20 minutes. Progress is written to")
    print("/workspace/logs/bootstrap.log on the pod. Poll readiness with:")
    print("  python cloud/runpod_cli.py status")
    return _print_urls(state)


def _gpu_ids(gpu: str) -> list[str]:
    # A comma-separated --gpu becomes a list of ACCEPTABLE GPU types; RunPod
    # places the pod on whichever is free. This is what lets a fresh pod land
    # when the preferred card (the cheap 4090) is momentarily sold out across
    # every DC - any listed card that runs the judge is fine, and the network
    # volume attaches to it identically.
    return [g.strip() for g in gpu.split(",") if g.strip()]


def _pod_payload(state: dict, args: argparse.Namespace, vol_id: str, dc: str) -> dict:
    return {
        "name": args.name,
        "imageName": args.image,
        "gpuTypeIds": _gpu_ids(args.gpu),
        "gpuCount": 1,
        "containerDiskInGb": args.disk,
        # NETWORK volume, mounted where the pod-local volume used to be, so the
        # bootstrap (/workspace clone + models) is unchanged. The pod is pinned
        # to the volume's DC but free to land on any host in it.
        "networkVolumeId": vol_id,
        "dataCenterIds": [dc],
        "volumeMountPath": "/workspace",
        "ports": [f"{ASR_PORT}/http", f"{JUDGE_PORT}/http", "22/tcp"],
        "dockerStartCmd": ["bash", "-c", POD_START_CMD],
        "env": {
            "CALLQA_ASR_API_KEY": state["asr_api_key"],
            "VLLM_API_KEY": state["judge_api_key"],
            "CALLQA_REPO_URL": args.repo_url,
            "CALLQA_LLM_MODEL": args.llm_model,
            **({"PUBLIC_KEY": args.ssh_public_key} if args.ssh_public_key else {}),
            **({"HF_TOKEN": os.environ["HF_TOKEN"]} if os.environ.get("HF_TOKEN") else {}),
        },
    }

def _create_pod_on_volume(state: dict, args: argparse.Namespace) -> str:
    """Create a pod attached to the network volume, and return its id.

    A network volume is DC-scoped: any DC accepts a volume, but only some have
    a free GPU right now. So volume creation and pod placement must be tried as
    a UNIT - if the pod cannot be placed in a DC, the volume there is useless
    and is deleted before trying the next. An already-existing volume (in
    state) pins its DC, so we only retry placement there.
    """
    existing = state.get("network_volume_id")
    if existing and _volume_exists(existing):
        dc = state["data_center_id"]
        print(f"Reusing network volume {existing} in {dc}; placing a pod...")
        pod = _request("POST", "/pods", _pod_payload(state, args, existing, dc))
        pod_id = pod.get("id") or pod.get("podId")
        if not pod_id:
            raise RunPodError(f"pod created but no id in response: {pod}")
        _record_pod(state, args, pod_id)
        return pod_id

    dcs = [args.data_center] if args.data_center else list(CANDIDATE_DATA_CENTERS)

    def _drop_volume(vid: str) -> None:
        _delete_volume(vid)
        state.pop("network_volume_id", None)
        state.pop("data_center_id", None)
        save_state(state)

    last_error = ""
    for dc in dcs:
        try:
            vol_id = _create_volume(f"{args.name}-vol", args.volume_size, dc)
        except RunPodError as exc:
            last_error = f"{dc}: volume create failed: {exc}"
            print(f"  {last_error}; next DC")
            continue
        # Track the volume the instant it exists, so a crash before the pod is
        # created leaves a RECORDED volume (status/destroy can find it), never a
        # silent storage charge.
        state["network_volume_id"] = vol_id
        state["data_center_id"] = dc
        save_state(state)
        try:
            pod = _request("POST", "/pods", _pod_payload(state, args, vol_id, dc))
        except RunPodError as exc:
            if _is_capacity_error(exc):
                print(f"  {dc}: no free {args.gpu} right now; deleting volume, next DC")
                _drop_volume(vol_id)
                last_error = f"{dc}: {exc}"
                continue
            _drop_volume(vol_id)                 # unknown error: don't strand the volume
            raise
        pod_id = pod.get("id") or pod.get("podId")
        if not pod_id:
            _drop_volume(vol_id)
            raise RunPodError(f"pod created but no id in response: {pod}")
        print(f"Created {args.volume_size} GB network volume {vol_id} and a pod in {dc}.")
        _record_pod(state, args, pod_id)         # persists pod_id immediately
        return pod_id
    raise RunPodError(f"no candidate data centre had a free {args.gpu}. "
                      f"Last error: {last_error}")


def _record_pod(state: dict, args: argparse.Namespace, pod_id: str) -> None:
    # Persist IMMEDIATELY: the pod is billing the instant POST /pods returns, so
    # if anything crashes between here and the caller's save_state the pod would
    # be a live, billing machine with no local record - and `down`/`destroy`
    # would then say "nothing to stop". Saving here shrinks that window to a
    # single statement. The network_volume_id was already saved before this.
    state["pod_id"] = pod_id
    state["gpu"] = args.gpu
    state["llm_model"] = args.llm_model
    save_state(state)

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
        return _volume_status_line(state)
    try:
        pod = _request("GET", f"/pods/{pod_id}")
    except RunPodError as exc:
        if not _is_not_found(exc):
            raise
        print(f"Recorded pod {pod_id} no longer exists; clearing the record.")
        _forget_pod(state)
        return _volume_status_line(state)
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
        print("GPU billing is stopped.")
    return _volume_status_line(state)


def _volume_status_line(state: dict) -> int:
    """Print the standing network-volume storage cost, so it is never a
    surprise the way the runaway pod was. Returns 0."""
    vol_id = state.get("network_volume_id")
    if vol_id:
        vol = _volume_exists(vol_id)
        size = vol.get("size") if vol else None
        note = (f"  (~${size * 0.07:.2f}/mo at $0.07/GB)" if isinstance(size, (int, float))
                else "")
        print(f"network volume {vol_id} in {state.get('data_center_id')}: "
              f"{size} GB{note} - bills whether or not a pod is attached")
    return 0


def cmd_down(args: argparse.Namespace) -> int:
    state = load_state()
    pod_id = state.get("pod_id")
    if not pod_id:
        print("No pod recorded; nothing to stop.")
        return 0
    try:
        _request("POST", f"/pods/{pod_id}/stop")
    except RunPodError as exc:
        if not _is_not_found(exc):
            raise
        print(f"Pod {pod_id} no longer exists; nothing to stop. Clearing record.")
        _forget_pod(state)
        return 0
    state["last_stopped_at"] = time.time()
    save_state(state)
    # Confirm the stop actually took effect rather than trusting the 200: a
    # queued-but-incomplete stop would otherwise print "billing ended" while
    # the pod keeps running.
    status = "UNKNOWN"
    try:
        status = (_request("GET", f"/pods/{pod_id}").get("desiredStatus") or "UNKNOWN")
    except RunPodError:
        pass
    if status == "RUNNING":
        print(f"WARNING: asked pod {pod_id} to stop but it still reports RUNNING.")
        print("Re-run `python cloud/runpod_cli.py status` and `down` to confirm.")
        return 1
    print(f"Pod {pod_id} stopped ({status}). GPU billing has ended; the network volume")
    print("persists (and still bills for storage) so the next `up` skips the download.")
    return 0


def cmd_destroy(args: argparse.Namespace) -> int:
    state = load_state()
    pod_id = state.get("pod_id")
    vol_id = state.get("network_volume_id")
    if not pod_id and not (args.volume and vol_id):
        print("No pod recorded; nothing to destroy.")
        return 0
    if not args.yes:
        print(f"This permanently deletes pod {pod_id}.")
        if args.volume and vol_id:
            print(f"AND network volume {vol_id} - the gemma weights and all models "
                  "go with it (a fresh pod re-downloads ~17 GB).")
        else:
            print("The network volume is KEPT (models preserved); add --volume to remove it.")
        print("Re-run with --yes to confirm.")
        return 1
    if pod_id:
        try:
            _request("DELETE", f"/pods/{pod_id}")
            print(f"Pod {pod_id} destroyed. GPU billing stopped.")
        except RunPodError as exc:
            if not _is_not_found(exc):
                raise
            print(f"Pod {pod_id} was already gone; clearing the record.")
        # Keep the volume and the shared keys; only the pod is disposable. The
        # next `up` attaches the SAME volume on whatever host has a free GPU.
        _forget_pod(state)
    if args.volume and vol_id:
        _request("DELETE", f"/networkvolumes/{vol_id}")
        STATE_FILE.unlink(missing_ok=True)
        print(f"Network volume {vol_id} deleted and local state cleared. Nothing is billing.")
    elif vol_id:
        print(f"Network volume {vol_id} kept (still bills for storage). "
              "`up` reuses it; `destroy --volume` removes it.")
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
    p.add_argument("--volume-size", type=int, default=DEFAULT_VOLUME_SIZE_GB,
                   help="network volume GB, created once and reused (holds models)")
    p.add_argument("--data-center", default=None,
                   help="force a data centre id; default walks CANDIDATE_DATA_CENTERS")
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

    p = sub.add_parser("destroy",
                       help="terminate the pod; KEEPS the network volume unless --volume")
    p.add_argument("--yes", action="store_true")
    p.add_argument("--volume", action="store_true",
                   help="also delete the network volume (loses the models; ~17 GB re-download)")
    p.set_defaults(func=cmd_destroy)

    args = parser.parse_args()
    try:
        return args.func(args)
    except RunPodError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
