#!/usr/bin/env python3
"""Download model weights - RUN ON THE BANK SERVER ONLY.

This is the ONLY component of the system that downloads anything. It requires
internet access (or pre-staged files) and is meant to run once on the target
server. The pipeline itself never downloads models and should run with
HF_HUB_OFFLINE=1 / TRANSFORMERS_OFFLINE=1.

Usage:
    python scripts/download_models.py --all --llm-model dicta-il/dictalm2.0-instruct
    python scripts/download_models.py --asr
    python scripts/download_models.py --diarization      # requires HF_TOKEN (gated)
    python scripts/download_models.py --llm --llm-model <hf-model-id>
    python scripts/download_models.py --ner              # optional

    # Windows / no GPU: ONE quantized file for llama.cpp instead of the full model
    python scripts/download_models.py --llm --llm-model dicta-il/dictalm2.0-instruct-GGUF \
        --llm-gguf dictalm2.0-instruct.Q4_K_M.gguf

LLM candidates by VRAM budget (pass the chosen id via --llm-model):
  ~16-24 GB : dicta-il/dictalm2.0-instruct          (7B, fp16 ~16GB; Hebrew-tuned)
  ~24 GB    : a 12-27B instruct model quantized AWQ/GPTQ
              (e.g. google/gemma-3-27b-it with AWQ quantization)
  >=48 GB   : meta-llama/Llama-3.3-70B-Instruct AWQ (gated; needs HF_TOKEN)
Pick the largest model that fits; use --max-model-len 8192 in vLLM either way.
No NVIDIA GPU (a Windows VDI desktop): a 4-bit GGUF of the 7B model (~4.4 GB)
served by llama.cpp on the CPU - see scripts/start_llama_server.py. Slow
(minutes per call), but it runs on any machine with 16 GB of memory.

After each download a MODELS_MANIFEST.json is written into the models dir;
the pipeline uses it (and the model directories) to fail fast with a clear
error when a required model is missing.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from callqa.ops.provenance import dir_sha256  # noqa: E402 - the one canonical hasher
from callqa.portable import configure_stdio  # noqa: E402

ASR_MODEL_ID = "ivrit-ai/whisper-large-v3-turbo-ct2"  # Apache-2.0
DIARIZATION_MODEL_ID = "pyannote/speaker-diarization-community-1"  # CC-BY-4.0; accept terms on HF
NER_MODEL_ID = "dicta-il/dictabert-ner"  # optional; licence recorded from its card at download


def _banner() -> None:
    print("=" * 72)
    print("callqa model downloader - BANK SERVER USE")
    print("Requires internet access (or pre-staged files in HF_HOME).")
    print("The runtime pipeline never downloads anything.")
    print("=" * 72)


def _snapshot(model_id: str, target: Path, token: str | None = None,
              allow_patterns: list[str] | None = None) -> tuple[str, Path]:
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        # Deliberately not in requirements.txt: it is the one library that
        # exists to download things, and the runtime must never be able to.
        # That is a good decision with a bad failure mode, so it says so here
        # rather than surfacing as a bare ImportError from a nested call.
        raise SystemExit(
            "huggingface_hub is not installed.\n"
            "It is kept out of the runtime dependencies on purpose - the "
            "pipeline must never be able to download anything - so install it "
            "for this step only:\n\n"
            "    pip install huggingface_hub\n"
        ) from None

    print(f"-> downloading {model_id} into {target} ...")
    path = snapshot_download(model_id, local_dir=target, token=token,
                             allow_patterns=allow_patterns)
    return model_id, Path(path)


def _warm_diarization_cache(model_id: str, token: str | None) -> Path | None:
    """Instantiate the pipeline once so the whole dependency tree is cached.

    A diarization pipeline is not one file: its config names a segmentation
    model and an embedding model that are fetched separately. Downloading only
    the pipeline repo leaves those missing, and the failure surfaces later on
    the offline machine, which is the worst possible moment. Building the
    pipeline here pulls everything into the Hugging Face cache, and that cache
    directory is what gets copied to the air-gapped server.
    """
    try:
        from pyannote.audio import Pipeline
    except ImportError:
        print("   note: pyannote.audio is not installed here, so only the pipeline")
        print("         repo is fetched. Its segmentation and embedding models will")
        print("         be downloaded on first use, which an offline machine cannot do.")
        return None

    print(f"-> building {model_id} once to cache its dependencies ...")
    try:
        Pipeline.from_pretrained(model_id, token=token)
    except TypeError:
        Pipeline.from_pretrained(model_id, use_auth_token=token)

    from huggingface_hub.constants import HF_HUB_CACHE

    cache = Path(HF_HUB_CACHE)
    print(f"   cached under {cache}")
    print("   copy that directory to the offline machine, point HF_HOME at it,")
    print("   and set HF_HUB_OFFLINE=1")
    return cache


def _dir_size(path: Path) -> int:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


def _update_manifest(models_dir: Path, entry: dict) -> None:
    manifest_path = models_dir / "MODELS_MANIFEST.json"
    manifest = {}
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest[entry["role"]] = entry
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"   manifest updated: {manifest_path}")


def _licence_of(model_id: str, local_path: Path) -> str:
    """The licence the model's own card declares, recorded at download time.

    A bank's compliance review asks what every model is licensed under, and the
    honest answer is whatever the publisher says TODAY - not whatever a table in
    a document said when it was written. Model cards get relicensed, and a
    judge model is chosen at deployment time and cannot be listed in advance at
    all. So this reads it from the card that came with the weights.

    Best-effort by design: an unreadable card records "unknown" rather than
    failing the download, and "unknown" is itself the useful answer - it tells
    the reviewer to go and look.
    """
    for name in ("README.md", "LICENSE", "LICENSE.txt"):
        card = local_path / name
        if not card.exists():
            continue
        try:
            head = card.read_text(encoding="utf-8", errors="replace")[:4000]
        except OSError:                               # pragma: no cover - defensive
            continue
        # YAML front matter on a HF model card: `license: apache-2.0`
        for line in head.splitlines():
            stripped = line.strip()
            if stripped.lower().startswith("license:"):
                value = stripped.split(":", 1)[1].strip().strip("\"'")
                if value:
                    return value
    return "unknown - read the model card at https://huggingface.co/" + model_id


def _record(models_dir: Path, role: str, model_id: str, local_path: Path) -> None:
    # Content hash of the weights, computed once here at download time. Preflight
    # re-hashes with --deep to verify the loaded weights are the ones that
    # produced earlier results; verify compares it across runs. A name is not a
    # fingerprint (a model can be re-trained under the same id), so this is.
    print("   hashing weights (once) ...", flush=True)
    try:
        weight_sha256 = dir_sha256(local_path)
    except OSError as exc:  # pragma: no cover - defensive
        print(f"   note: could not hash weights ({exc}); sha256 left blank")
        weight_sha256 = ""
    _update_manifest(
        models_dir,
        {
            "role": role,
            "model_id": model_id,
            "local_path": str(local_path),
            "size_bytes": _dir_size(local_path),
            "sha256": weight_sha256,
            "license": _licence_of(model_id, local_path),
            "downloaded_at": datetime.now(UTC).isoformat(timespec="seconds"),
        },
    )


def _write_llm_into_config(model_id: str) -> None:
    """Point judge.model in config/config.yaml at the downloaded LLM.

    Edits the one line rather than reserialising the file. Round-tripping it
    through yaml.safe_dump deleted every comment in it, and those comments are
    how the bank's operators know what the settings mean.

    The path arrives with forward slashes (Path.as_posix), which Windows
    accepts: inside a double-quoted YAML string a Windows backslash is an
    escape character, and "models\\Qwen..." made config.yaml unreadable.
    """
    config_path = Path("config/config.yaml")
    if not config_path.exists():
        print("   note: config/config.yaml not found; set judge.model manually")
        return
    lines = config_path.read_text(encoding="utf-8").splitlines(keepends=True)
    in_judge = False
    for i, line in enumerate(lines):
        if re.match(r"^judge:\s*$", line):
            in_judge = True
            continue
        if in_judge and re.match(r"^\S", line):
            break                                    # left the judge block
        if in_judge and re.match(r"^\s+model:\s", line):
            indent = line[: len(line) - len(line.lstrip())]
            comment = line.split("#", 1)
            trailing = f"  #{comment[1].rstrip()}" if len(comment) > 1 else ""
            lines[i] = f'{indent}model: "{model_id}"{trailing}\n'
            config_path.write_text("".join(lines), encoding="utf-8")
            print(f"   config/config.yaml updated: judge.model = {model_id}")
            return
    print("   note: judge.model not found in config/config.yaml; set it manually")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--all", action="store_true", help="download asr + diarization + llm")
    parser.add_argument("--asr", action="store_true")
    parser.add_argument("--diarization", action="store_true",
                        help="gated: requires HF_TOKEN and accepted terms; only needed for mono recordings")
    parser.add_argument("--llm", action="store_true")
    parser.add_argument("--ner", action="store_true", help="optional DictaBERT-NER")
    parser.add_argument("--llm-model", default=None,
                        help="HF model id for the judge LLM (see candidates in --help)")
    parser.add_argument("--llm-gguf", default=None, metavar="FILE",
                        help="download only this .gguf file from --llm-model (for llama.cpp)")
    parser.add_argument("--models-dir", default="models", type=Path)
    args = parser.parse_args()
    configure_stdio()

    if not any([args.all, args.asr, args.diarization, args.llm, args.ner]):
        parser.error("nothing selected: pass --all or one of --asr/--diarization/--llm/--ner")
    # Usage errors are caught BEFORE any download. `--all` without
    # `--llm-model` used to be discovered only after the 1.6 GB ASR model had
    # already come down.
    if (args.all or args.llm) and not args.llm_model:
        parser.error("--all/--llm need --llm-model <hf-model-id> (candidates in --help)")

    _banner()
    models_dir: Path = args.models_dir
    models_dir.mkdir(parents=True, exist_ok=True)
    token = os.environ.get("HF_TOKEN")

    # Each model is its own step, and one failing does not stop the others.
    # With `--all` and no HF_TOKEN, the gated diarization step used to
    # `return 1` - silently skipping the JUDGE model after it, the one model
    # without which the pipeline produces no scores at all, while its message
    # named only diarization as outstanding.
    steps = []
    if args.all or args.asr:
        steps.append(("asr", lambda: _download_asr(models_dir)))
    if args.all or args.diarization:
        steps.append(("diarization", lambda: _download_diarization(models_dir, token)))
    if args.all or args.llm:
        steps.append(("llm", lambda: _download_llm(models_dir, args.llm_model, token,
                                                   args.llm_gguf)))
    if args.ner:
        steps.append(("ner", lambda: _download_ner(models_dir, token)))

    outcomes: list[tuple[str, str | None]] = []
    for role, run in steps:
        try:
            run()
            outcomes.append((role, None))
        except SystemExit:
            raise                        # huggingface_hub missing: nothing else can work
        except Exception as exc:  # noqa: BLE001 - reported per model, below
            outcomes.append((role, _explain(exc, role)))
            print(f"   FAILED: {role}")

    print()
    print("=" * 72)
    for role, problem in outcomes:
        print(f"  {'OK    ' if problem is None else 'FAILED'}  {role}")
        if problem:
            for line in problem.splitlines():
                print(f"          {line}")
    print("=" * 72)
    if any(problem for _, problem in outcomes):
        print("Some models did not download. Fix the above and re-run the failed ones;")
        print("models that succeeded are recorded and will not be fetched again.")
        return 1
    print("Done. Set HF_HUB_OFFLINE=1 and TRANSFORMERS_OFFLINE=1 before running the pipeline.")
    return 0


def _explain(exc: Exception, role: str) -> str:
    """What went wrong, in terms of what to DO about it.

    Only ImportError used to be handled, and it always printed "huggingface_hub
    is not installed" - a gated 401, a mistyped model id, a full disk and a
    dropped connection all escaped as raw tracebacks instead.
    """
    import errno

    name = type(exc).__name__
    text = str(exc)
    if name == "GatedRepoError" or " 401" in text or " 403" in text:
        return ("the model is GATED: accept its conditions once on huggingface.co with\n"
                "the account behind HF_TOKEN, then re-run. (A 401/403 here never means\n"
                "the model is missing.)")
    if name == "RepositoryNotFoundError" or " 404" in text:
        return "no such model id on Hugging Face - check the spelling."
    if isinstance(exc, OSError) and exc.errno == errno.ENOSPC:
        return "the disk is full. The models need ~25 GB in total."
    return f"{name}: {text[:300]}"


def _download_asr(models_dir: Path) -> None:
    model_id, path = _snapshot(ASR_MODEL_ID, models_dir / "ivrit-whisper-large-v3-turbo-ct2")
    _record(models_dir, "asr", model_id, path)


def _download_diarization(models_dir: Path, token: str | None) -> None:
    if not token:
        raise RuntimeError(
            "needs HF_TOKEN. This model is gated: accept its conditions once on\n"
            f"https://huggingface.co/{DIARIZATION_MODEL_ID} then set HF_TOKEN in .env.\n"
            "Only mono recordings need it; stereo ones run without it.")
    model_id, path = _snapshot(
        DIARIZATION_MODEL_ID, models_dir / DIARIZATION_MODEL_ID.replace("/", "--"), token)
    _record(models_dir, "diarization", model_id, path)
    cache = _warm_diarization_cache(DIARIZATION_MODEL_ID, token)
    if cache:
        _update_manifest(models_dir, {
            "role": "diarization_cache",
            "model_id": DIARIZATION_MODEL_ID,
            "local_path": str(cache),
            "size_bytes": _dir_size(cache),
            "downloaded_at": datetime.now(UTC).isoformat(timespec="seconds"),
        })


def _download_llm(models_dir: Path, llm_model: str, token: str | None,
                  gguf: str | None = None) -> None:
    target = models_dir / llm_model.replace("/", "--")
    if gguf:
        # One quantized file, not the repository: a GGUF repo holds a dozen
        # quantizations of the same model, 3-15 GB each.
        _snapshot(llm_model, target, token, allow_patterns=[gguf])
        path = target / gguf
        if not path.is_file():
            raise RuntimeError(f"{llm_model} has no file named {gguf}. The .gguf file "
                               f"names are listed at https://huggingface.co/{llm_model}")
        _record(models_dir, "llm", llm_model, target)
        _write_llm_into_config(path.as_posix())
        return
    model_id, path = _snapshot(llm_model, target, token)
    _record(models_dir, "llm", model_id, path)
    # The LOCAL PATH, not the repo id: snapshot_download(local_dir=...)
    # deliberately bypasses the Hugging Face cache, so an offline machine asked
    # to serve the repo id has nowhere to resolve it from and vLLM fails.
    _write_llm_into_config(path.as_posix())


def _download_ner(models_dir: Path, token: str | None) -> None:
    model_id, path = _snapshot(NER_MODEL_ID, models_dir / "dictabert-ner", token)
    _record(models_dir, "ner", model_id, path)


if __name__ == "__main__":
    sys.exit(main())
