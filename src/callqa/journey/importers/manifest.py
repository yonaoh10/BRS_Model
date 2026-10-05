"""What a batch folder says about itself: manifest.yaml, holidays.yaml,
export_manifest.csv.

A manifest is the batch's contract with the tool: which layers it carries,
how its dates are written, the first day its Atlas log holds, the last
moment its sources cover, the matching constants the bank used. Everything
the analysis would otherwise have to guess from the data is read from here
first. A batch without one is contract v1 with nothing declared.

What the manifest declares is checked, never trusted blindly: a layer it
claims and the folder does not hold is an error (a report must not read as
"passed" over a missing layer, the bank's rule in file 24 §7), and a row
count it states for a file is compared with the rows read.
"""

from __future__ import annotations

from datetime import date, datetime
from pathlib import Path

from pydantic import ValidationError

from callqa.ingestion import _csv_reader, _read_text_any_encoding, _visible
from callqa.journey.models import AtlasRules, ImportReport, Manifest
from callqa.resources import load_yaml

MANIFEST = "manifest.yaml"
HOLIDAYS = "holidays.yaml"
EXPORT_MANIFEST = "export_manifest.csv"
CHECKS = "checks.csv"
SUPPORTED_VERSIONS = (1, 2)


def load_manifest(folder: Path, report: ImportReport) -> Manifest | None:
    """The batch's manifest, or None when it has none. An unreadable or
    unknown one is an error in the report (and None), never a guess."""
    path = folder / MANIFEST
    if not path.exists():
        return None
    try:
        data = load_yaml(path)
    except ValueError as exc:
        report.add("manifest_invalid", "error", f"{MANIFEST}: {exc}")
        return None
    if data is None:
        return Manifest()
    if not isinstance(data, dict):
        report.add("manifest_invalid", "error", f"{MANIFEST}: expected a mapping of fields")
        return None
    try:
        manifest = Manifest.model_validate(data)
    except ValidationError as exc:
        fields = sorted({".".join(str(p) for p in e["loc"]) for e in exc.errors()})
        report.add("manifest_invalid", "error",
                   f"{MANIFEST}: field(s) not understood or of the wrong type: "
                   f"{', '.join(fields)}")
        return None
    if manifest.contract_version not in SUPPORTED_VERSIONS:
        report.add("contract_version", "error",
                   f"contract version {manifest.contract_version} is not supported "
                   f"(this reads {', '.join(map(str, SUPPORTED_VERSIONS))})")
        return None
    return manifest


def load_holidays(folder: Path, report: ImportReport) -> list[date]:
    """holidays.yaml: a list of dates, or a mapping of year -> list of dates.
    Half days are whole days here (a promise due on one is due the next
    business day). Unreadable entries are reported, the rest kept."""
    path = folder / HOLIDAYS
    if not path.exists():
        return []
    try:
        data = load_yaml(path)
    except ValueError as exc:
        report.add("holidays_invalid", "error", f"{HOLIDAYS}: {exc}")
        return []
    items: list[object] = []
    if isinstance(data, dict):
        for value in data.values():
            items.extend(value if isinstance(value, list) else [value])
    elif isinstance(data, list):
        items = data
    elif data is not None:
        report.add("holidays_invalid", "error", f"{HOLIDAYS}: expected a list of dates")
        return []
    out: list[date] = []
    bad = 0
    for item in items:
        if isinstance(item, datetime):
            out.append(item.date())
        elif isinstance(item, date):
            out.append(item)
        else:
            try:
                out.append(date.fromisoformat(str(item).strip()[:10]))
            except ValueError:
                bad += 1
    if bad:
        report.add("holidays_invalid", "warning",
                   f"{HOLIDAYS}: entries that are not dates (YYYY-MM-DD); skipped", count=bad)
    return sorted(set(out))


def declared_row_counts(folder: Path, report: ImportReport) -> dict[str, int]:
    """export_manifest.csv: `file, rows` (other columns ride along), the
    counts the exporting program wrote, to be compared with what was read."""
    path = folder / EXPORT_MANIFEST
    if not path.exists():
        return {}
    out: dict[str, int] = {}
    for raw in _csv_reader(_read_text_any_encoding(path)):
        raw.pop(None, None)
        row = {_visible(k or "").strip().lower(): _visible(v or "").strip()
               for k, v in raw.items() if k}
        name = row.get("file", "")
        if not name:
            continue
        try:
            out[Path(name).name.lower()] = int(float(row.get("rows", "")))
        except ValueError:
            report.add("export_manifest_invalid", "warning",
                       f"{EXPORT_MANIFEST}: a row count that is not a number",
                       examples=[Path(name).name[:40]])
    return out


def check_row_counts(declared: dict[str, int], read: dict[str, int], report: ImportReport
                     ) -> None:
    """A file the exporter counted differently from what arrived is a
    truncated transfer or a stale manifest: an error either way."""
    differ = sorted(name for name, n in declared.items()
                    if name in read and read[name] != n)
    if differ:
        report.add("row_count_mismatch", "error",
                   f"files whose row count differs from {EXPORT_MANIFEST} (a truncated "
                   "transfer, or a stale manifest)", count=len(differ),
                   examples=[f"{n}: read {read[n]:,}, declared {declared[n]:,}" for n in differ[:5]])


def check_declared_layers(manifest: Manifest | None, present: set[str], report: ImportReport
                          ) -> None:
    """Every angle the manifest claims must be in the folder: a report over a
    batch that lost its Atlas or CRM layer on the way would otherwise read
    as a batch where nothing happened behind the contacts."""
    if manifest is None:
        return
    missing = sorted(a for a in manifest.angles if a not in present)
    if missing:
        report.add("declared_layer_missing", "error",
                   f"{MANIFEST} declares layer(s) the batch does not hold: "
                   f"{', '.join(missing)}")
    undeclared = sorted(present - set(manifest.angles) - {"vendor"})
    if undeclared and manifest.angles:
        report.add("layer_not_declared", "info",
                   f"layer(s) present but not declared in {MANIFEST}: "
                   f"{', '.join(undeclared)}")


def rules_from_manifest(manifest: Manifest | None, base: AtlasRules) -> AtlasRules:
    """The Atlas rules of a batch: the config's, overridden by what the
    manifest declares (its windows and the first day its log holds)."""
    rules = base.model_copy()
    if manifest is None:
        return rules
    if manifest.coverage_from is not None:
        c = manifest.coverage_from
        rules.coverage_from = datetime(c.year, c.month, c.day)
    w = manifest.windows
    if w is not None:
        rules.call_out_before = w.out_before_min
        rules.call_in_after = w.in_after_min
        rules.msg_out_before = w.msg_out_before_min
        rules.msg_in_after = w.msg_in_after_min
        rules.unk_before = w.unk_before_min
        rules.unk_after = w.unk_after_min
    return rules


def day_first_of(manifest: Manifest | None) -> bool | None:
    if manifest is None or manifest.date_order is None:
        return None
    return manifest.date_order == "day_first"
