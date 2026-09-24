"""Where a journey dataset and everything derived from it live on disk.

    <output_dir>/journey/
        latest.txt                  the most recently imported dataset id
        <dataset_id>/
            dataset.json            the normalised dataset (no account numbers)
            import_report.json      counts and issues, safe to share
            private/accounts.csv    story number -> account; owner-only folder
            content.json            what was read from the contents (cards, judgements,
                                    story verdicts), when the content stage has run
"""

from __future__ import annotations

import re
from pathlib import Path

from callqa.config import Config
from callqa.journey.models import ContentLayer, JourneyDataset
from callqa.state import atomic_write_model, atomic_write_text

_DATASET_ID = re.compile(r"^ds-\d{8}-[0-9a-f]{8}$")


def journey_root(config: Config) -> Path:
    root = config.journey.root
    return root if root.is_absolute() else config.paths.output_dir / root


def dataset_dir(config: Config, dataset_id: str) -> Path:
    if not _DATASET_ID.match(dataset_id):
        raise ValueError(f"not a dataset id: {dataset_id!r}")
    return journey_root(config) / dataset_id


def private_dir(config: Config, dataset_id: str) -> Path:
    return dataset_dir(config, dataset_id) / "private"


def save_dataset(config: Config, dataset: JourneyDataset) -> Path:
    folder = dataset_dir(config, dataset.dataset_id)
    folder.mkdir(parents=True, exist_ok=True)
    atomic_write_model(folder / "dataset.json", dataset)
    atomic_write_model(folder / "import_report.json", dataset.report)
    atomic_write_text(journey_root(config) / "latest.txt", dataset.dataset_id + "\n")
    return folder


def resolve_dataset_id(config: Config, dataset_id: str | None) -> str:
    """An explicit id, or the latest import."""
    if dataset_id:
        return dataset_id
    latest = journey_root(config) / "latest.txt"
    if not latest.exists():
        raise FileNotFoundError("no journey dataset has been imported yet "
                                "(callqa journey import ...)")
    return latest.read_text(encoding="utf-8").strip()


def load_dataset(config: Config, dataset_id: str | None = None) -> JourneyDataset:
    ds = resolve_dataset_id(config, dataset_id)
    path = dataset_dir(config, ds) / "dataset.json"
    return JourneyDataset.model_validate_json(path.read_text(encoding="utf-8"))


def save_content(config: Config, dataset_id: str, content: ContentLayer) -> Path:
    path = dataset_dir(config, dataset_id) / "content.json"
    atomic_write_model(path, content)
    return path


def load_content(config: Config, dataset_id: str) -> ContentLayer | None:
    path = dataset_dir(config, dataset_id) / "content.json"
    if not path.exists():
        return None
    return ContentLayer.model_validate_json(path.read_text(encoding="utf-8"))
