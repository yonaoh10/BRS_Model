r"""Serve the judge LLM with llama.cpp - the way to run it on Windows.

    Windows:  .venv\Scripts\python scripts\start_llama_server.py models\<...>\<file>.gguf
    Linux:    .venv/bin/python scripts/start_llama_server.py models/<...>/<file>.gguf

    --port N        default 8000 (judge.base_url is http://localhost:8000/v1)
    --threads N     CPU threads for generation (default: llama.cpp's choice)
    --ctx-size N    context window in tokens (default 16384: a long call's
                    transcript, the rubric and a full scorecard must all fit)
    --server PATH   llama-server executable, if it is not under tools/ or on PATH

vLLM, the server scripts/start_vllm.sh starts, runs only on Linux with an
NVIDIA GPU. llama.cpp's `llama-server` runs on Windows, on the CPU, from a zip
file that needs no installation and no admin rights: download the Windows CPU
build (llama-<version>-bin-win-cpu-x64.zip) from
https://github.com/ggml-org/llama.cpp/releases and unzip it into tools/ inside
this project. It speaks the same OpenAI protocol the pipeline uses, including
the JSON-schema structured output the judge relies on.

The API key is required, exactly as for vLLM: the server answers with text built
from call transcripts. It is read from CALLQA_JUDGE__API_KEY in .env - the same
variable the pipeline reads - so the two cannot disagree. The server listens on
127.0.0.1 only.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from callqa.dotenv import load_dotenv  # noqa: E402
from callqa.portable import configure_stdio, find_executable  # noqa: E402


def _find_server(explicit: Path | None) -> str | None:
    if explicit:
        return str(explicit) if explicit.is_file() else None
    exe = "llama-server.exe" if os.name == "nt" else "llama-server"
    tools = ROOT / "tools"
    if tools.is_dir():
        found = sorted(tools.rglob(exe))
        if found:
            return str(found[0])
    return find_executable("llama-server")


def main(argv: list[str] | None = None) -> int:
    configure_stdio()
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("model", type=Path, help="the .gguf file")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--threads", type=int, default=None)
    parser.add_argument("--ctx-size", type=int, default=16384)
    parser.add_argument("--server", type=Path, default=None)
    args = parser.parse_args(argv)

    load_dotenv()
    key = os.environ.get("CALLQA_JUDGE__API_KEY") or os.environ.get("CALLQA_JUDGE_API_KEY")
    if not key:
        print("ERROR: no API key. Create one with\n\n"
              '    python -c "import secrets; print(secrets.token_hex(32))"\n\n'
              "and put it in .env as   CALLQA_JUDGE__API_KEY=<the value>", file=sys.stderr)
        return 2
    if not args.model.is_file():
        print(f"ERROR: {args.model} is not a file. Download one with "
              "scripts/download_models.py --llm --llm-model <repo> --llm-gguf <file>",
              file=sys.stderr)
        return 2
    server = _find_server(args.server)
    if server is None:
        print("ERROR: llama-server was not found. Unzip the llama.cpp Windows build into "
              "tools/ (see the top of this script), or pass --server.", file=sys.stderr)
        return 2

    # Refuse a port someone else holds. On a shared (multi-session) host,
    # llama-server would fail to bind and exit, while a colleague's server on
    # that port kept answering - and the pipeline's model check is then the
    # only thing between this user's calls and it.
    import socket

    probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        if hasattr(socket, "SO_EXCLUSIVEADDRUSE"):
            probe.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
        probe.bind(("127.0.0.1", args.port))
    except OSError:
        print(f"ERROR: port {args.port} is already in use on this machine - perhaps by "
              "another user's judge. Start with --port <another>, and set "
              f"CALLQA_JUDGE__BASE_URL=http://127.0.0.1:<that port>/v1 in .env.",
              file=sys.stderr)
        return 2
    finally:
        probe.close()

    cmd = [server, "--model", str(args.model), "--host", "127.0.0.1",
           "--port", str(args.port), "--ctx-size", str(args.ctx_size),
           # One request at a time gets the WHOLE context; with the default
           # slots the window is divided between parallel requests.
           "--parallel", "1", "--jinja"]
    if args.threads:
        cmd += ["--threads", str(args.threads)]
    print(f"Serving {args.model.name} at http://127.0.0.1:{args.port}/v1 (Ctrl+C stops it)",
          flush=True)
    # The key goes through the environment, not argv, where any user of the
    # machine could read it in the process list.
    env = dict(os.environ, LLAMA_API_KEY=key)
    try:
        return subprocess.call(cmd, env=env)
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
