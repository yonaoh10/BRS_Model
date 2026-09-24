"""The bank-editable vocabularies: units, topics, return categories."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

from callqa.resources import find_config, load_yaml

UNIT_KINDS = ("center", "back_office", "own_branch", "branch", "other")


@dataclass(frozen=True)
class Units:
    by_code: dict[str, tuple[str, str]]      # code -> (label, kind)
    kind_labels: dict[str, str]

    def kind(self, code: str | None, own_branch: str | None) -> str:
        code = (code or "").lstrip("0")
        if code in self.by_code:
            return self.by_code[code][1]
        if own_branch and code == str(own_branch).lstrip("0"):
            return "own_branch"
        return "branch" if code else "other"

    def label(self, code: str | None, own_branch: str | None) -> str:
        return self.kind_labels.get(self.kind(code, own_branch), "יחידה")


@dataclass(frozen=True)
class Taxonomy:
    topics: dict[str, dict]
    categories: dict[str, dict]
    sha256: str

    @property
    def failure_categories(self) -> set[str]:
        return {k for k, v in self.categories.items() if v.get("failure")}

    def topic_label(self, key: str | None) -> str:
        return self.topics.get(key or "", {}).get("label", "לא סווג")

    def category_label(self, key: str | None) -> str:
        return self.categories.get(key or "", {}).get("label", "לא סווג")


def _sha(path: Path) -> str:
    raw = path.read_bytes().removeprefix(b"\xef\xbb\xbf").replace(b"\r\n", b"\n")
    return hashlib.sha256(raw).hexdigest()[:16]


def load_units(name: str = "journey_units.yaml") -> Units:
    path = find_config(name)
    data = load_yaml(path) or {}
    units = {str(k).lstrip("0"): (str(v.get("label", "")), str(v.get("kind", "branch")))
             for k, v in (data.get("units") or {}).items()}
    for _label, kind in units.values():
        if kind not in UNIT_KINDS:
            raise ValueError(f"{path.name}: unknown unit kind {kind!r} (use {', '.join(UNIT_KINDS)})")
    labels = {k: str(v) for k, v in (data.get("kind_labels") or {}).items()}
    return Units(by_code=units, kind_labels=labels)


REQUIRED_CATEGORIES = {"unclosed_loop", "excessive_runaround", "legit_return", "bank_initiated",
                       "new_topic", "unclassifiable"}


def load_taxonomy(name: str = "journey_taxonomy.yaml") -> Taxonomy:
    path = find_config(name)
    data = load_yaml(path) or {}
    topics = dict(data.get("topics") or {})
    cats = dict(data.get("return_categories") or {})
    missing = REQUIRED_CATEGORIES - set(cats)
    if missing:
        raise ValueError(f"{path.name}: return_categories must include {', '.join(sorted(missing))}")
    if "other" not in topics:
        raise ValueError(f"{path.name}: topics must include 'other'")
    return Taxonomy(topics=topics, categories=cats, sha256=_sha(path))


ATLAS_CODE_KINDS = ("open", "info", "execute", "not_customer")


def normalise_op_code(value: object) -> str:
    """Atlas codes are text with leading zeros ('035'); decode tables drop
    them (35). Both sides are compared without them, as ATL_03 does."""
    v = str(value if value is not None else "").strip()
    if v.endswith(".0") and v[:-2].isdigit():
        v = v[:-2]
    return v.lstrip("0") or ("0" if v else "")


def load_atlas_codes(name: str = "journey_atlas_codes.yaml") -> dict[str, str]:
    """Operation code -> open / info / execute / not_customer (ATL_R02)."""
    path = find_config(name)
    data = load_yaml(path) or {}
    codes: dict[str, str] = {}
    for kind, values in data.items():
        if kind not in ATLAS_CODE_KINDS:
            raise ValueError(f"{path.name}: unknown kind {kind!r} "
                             f"(use {', '.join(ATLAS_CODE_KINDS)})")
        for v in values or []:
            code = normalise_op_code(v)
            if code in codes and codes[code] != kind:
                raise ValueError(f"{path.name}: code {v} is listed as both {codes[code]} and {kind}")
            codes[code] = kind
    return codes
