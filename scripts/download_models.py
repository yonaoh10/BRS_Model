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

LLM candidates by VRAM budget (pass the chosen id via --llm-model):
  ~16-24 GB : dicta-il/dictalm2.0-instruct          (7B, fp16 ~16GB; Hebrew-tuned)
  ~24 GB    : a 12-27B instruct model quantized AWQ/GPTQ
              (e.g. google/gemma-3-27b-it with AWQ quantization)
  >=48 GB   : meta-llama/Llama-3.3-70B-Instruct AWQ (gated; needs HF_TOKEN)
Pick the largest model that fits; use --max-model-len 8192 in vLLM either way.

After each download a MODELS_MANIFEST.json is written into the models dir;
the pipeline uses it (and the model directories) to fail fast with a clear
error when a required model is missing.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path

ASR_MODEL_ID = "ivrit-ai/whisper-large-v3-turbo-ct2"  # Apache-2.0
DIARIZATION_MODEL_ID = "pyannote/speaker-diarization-3.1"  # GATED - accept terms on HF
DIARIZATION_DEP_ID = "pyannote/segmentation-3.0"  # GATED - dependency of the above
NER_MODEL_ID = "dicta-il/dictabert-ner"  # optional


def _banner() -> None:
    print("=" * 72)
    print("callqa model downloader - BANK SERVER USE")
    print("Requires internet access (or pre-staged files in HF_HOME).")
    print("The runtime pipeline never downloads anything.")
    print("=" * 72)


def _snapshot(model_id: str, target: Path, token: str | None = None) -> tuple[str, Path]:
    from huggingface_hub import snapshot_download

    print(f"-> downloading {model_id} into {target} ...")
    path = snapshot_download(model_id, local_dir=target, token=token)
    return model_id, Path(path)


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


def _record(models_dir: Path, role: str, model_id: str, local_path: Path) -> None:
    _update_manifest(
        models_dir,
        {
            "role": role,
            "model_id": model_id,
            "local_path": str(local_path),
            "size_bytes": _dir_size(local_path),
            "downloaded_at": datetime.now(UTC).isoformat(timespec="seconds"),
        },
    )


def _write_llm_into_config(model_id: str) -> None:
    """Point judge.model in config/config.yaml at the downloaded LLM."""
    config_path = Path("config/config.yaml")
    if not config_path.exists():
        print("   note: config/config.yaml not found; set judge.model manually")
        return
    import yaml

    data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    data.setdefault("judge", {})["model"] = model_id
    config_path.write_text(yaml.safe_dump(data, allow_unicode=True, sort_keys=False),
                           encoding="utf-8")
    print(f"   config/config.yaml updated: judge.model = {model_id}")


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
    parser.add_argument("--models-dir", default="models", type=Path)
    args = parser.parse_args()

    if not any([args.all, args.asr, args.diarization, args.llm, args.ner]):
        parser.error("nothing selected: pass --all or one of --asr/--diarization/--llm/--ner")

    _banner()
    models_dir: Path = args.models_dir
    models_dir.mkdir(parents=True, exist_ok=True)
    token = os.environ.get("HF_TOKEN")

    try:
        if args.all or args.asr:
            model_id, path = _snapshot(
                ASR_MODEL_ID, models_dir / "ivrit-whisper-large-v3-turbo-ct2"
            )
            _record(models_dir, "asr", model_id, path)

        if args.all or args.diarization:
            if not token:
                print("ERROR: --diarization requires the HF_TOKEN environment variable.")
                print("The pyannote models are gated: accept their terms on huggingface.co")
                print(f"  {DIARIZATION_MODEL_ID}")
                print(f"  {DIARIZATION_DEP_ID}")
                return 1
            for role, mid in (("diarization", DIARIZATION_MODEL_ID),
                              ("diarization_dep", DIARIZATION_DEP_ID)):
                model_id, path = _snapshot(mid, models_dir / mid.replace("/", "--"), token)
                _record(models_dir, role, model_id, path)

        if args.all or args.llm:
            if not args.llm_model:
                print("ERROR: --llm requires --llm-model <hf-model-id>.")
                print("Candidates by VRAM budget are listed in --help.")
                return 1
            model_id, path = _snapshot(
                args.llm_model, models_dir / args.llm_model.replace("/", "--"), token
            )
            _record(models_dir, "llm", model_id, path)
            _write_llm_into_config(args.llm_model)

        if args.ner:
            model_id, path = _snapshot(NER_MODEL_ID, models_dir / "dictabert-ner", token)
            _record(models_dir, "ner", model_id, path)
    except ImportError:
        print("ERROR: huggingface_hub is not installed (pip install huggingface_hub).")
        return 1

    print("Done. Set HF_HUB_OFFLINE=1 and TRANSFORMERS_OFFLINE=1 before running the pipeline.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
