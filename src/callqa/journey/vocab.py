"""The bank-editable vocabularies: units, topics, return categories."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

from callqa.resources import find_config, load_yaml

# The kinds a unit can be. The first five are what the editable YAML list
# knew; the rest are the bank's own unit types (T1604 / T1017, R4 §A) and
# arrive with a batch's units.csv. "unknown" is a code no table lists.
UNIT_KINDS = ("center", "back_office", "own_branch", "branch", "other",
              "team", "cluster", "live", "ipb", "business", "mortgage", "digital_support",
              "credit", "region", "hq", "unknown")
KIND_LABELS_HE = {
    "center": "מרכז הבנקאות", "back_office": "תפעול עורפי", "own_branch": "סניף החשבון",
    "branch": "סניף אחר", "other": "יחידת מטה", "team": "צוות במרכז הבנקאות",
    "cluster": "אשכול במרכז הבנקאות", "live": "סניף LIVE", "ipb": "בנקאות פרטית",
    "business": "בנקאות עסקית", "mortgage": "משכנתאות", "digital_support": "תמיכה דיגיטלית",
    "credit": "אשראי צרכני", "region": "מרחב", "hq": "מטה", "unknown": "יחידה לא מזוהה",
}


@dataclass(frozen=True)
class Units:
    by_code: dict[str, tuple[str, str]]      # code (SNIF_ID space) -> (label, kind)
    kind_labels: dict[str, str]
    # True when a batch's units.csv is the source: a code it does not list is
    # then "unknown", never assumed to be a branch. The editable YAML list
    # keeps the old assumption (it names the few non-branch units only).
    strict: bool = False

    def kind(self, code: str | None, own_branch: str | None) -> str:
        code = (code or "").lstrip("0")
        if code in self.by_code:
            kind = self.by_code[code][1]
            if kind in ("branch", "live") and own_branch and code == str(own_branch).lstrip("0"):
                return "own_branch"
            return kind
        if own_branch and code == str(own_branch).lstrip("0"):
            return "own_branch"
        if not code:
            return "other"
        return "unknown" if self.strict else "branch"

    def label(self, code: str | None, own_branch: str | None) -> str:
        return self.kind_labels.get(self.kind(code, own_branch), "יחידה")

    def name(self, code: str | None) -> str:
        """The unit's own name when a table gives one."""
        return self.by_code.get((code or "").lstrip("0"), ("", ""))[0]

    def known(self, code: str | None) -> bool:
        return (code or "").lstrip("0") in self.by_code

    @classmethod
    def from_rows(cls, rows, fallback: Units) -> Units:  # noqa: ANN001 - list[UnitRow]
        """The units of a batch (its units.csv, SNIF_ID space only - the CRM's
        T1017 codes are another space and are not merged in), over the
        editable list's kind labels."""
        by_code = dict(fallback.by_code)
        for r in rows:
            if r.code_space != "T1604" or not r.snif_id:
                continue
            by_code[str(r.snif_id).lstrip("0")] = (r.name, r.kind)
        labels = dict(KIND_LABELS_HE)
        labels.update(fallback.kind_labels)
        return cls(by_code=by_code, kind_labels=labels, strict=bool(rows))


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
    labels = dict(KIND_LABELS_HE)
    labels.update({k: str(v) for k, v in (data.get("kind_labels") or {}).items()})
    return Units(by_code=units, kind_labels=labels)


def units_of(dataset, name: str = "journey_units.yaml") -> Units:  # noqa: ANN001 - JourneyDataset
    """The units a batch is analysed with: its own units.csv when it carries
    one (over the editable list's labels), else the editable list alone."""
    base = load_units(name)
    rows = getattr(dataset, "units", None) or []
    return Units.from_rows(rows, base) if rows else base


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
