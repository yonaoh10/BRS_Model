"""units.csv: the organisational units of a batch, as the bank's tables name
them (contract v2).

    unit_key          required; any text unique in the file
    code_space        T1604 (SNIF_ID: branches, the centre, back office - what
                      Atlas and the account carry) or T1017 (SAP ORG_UNIT_CODE:
                      teams and clusters - what the CRM carries)
    snif_id / org_unit_code   the code in that space
    name, kind        kind: center | team | cluster | back_office | branch | live |
                      ipb | business | mortgage | digital_support | credit | region |
                      hq | other
    parent_unit_key, cluster, team, region, in_closure, merged_into,
    valid_from, valid_to, crosswalk_basis (measured | hypothesis | none)

The two code spaces are never merged here: the crosswalk between a SAP team
and a branch number has not been measured at the bank, and a row's
`crosswalk_basis` says what its parent link rests on.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

from pydantic import ValidationError

from callqa.ingestion import _csv_reader, _read_text_any_encoding, _visible
from callqa.journey.models import ImportReport, UnitRow
from callqa.journey.timeparse import try_parse_dt

UNITS_FILE = "units.csv"
_TRUE = {"1", "1.0", "yes", "y", "true", "כן"}
_KINDS = {"center", "team", "cluster", "back_office", "branch", "live", "ipb", "business",
          "mortgage", "digital_support", "credit", "region", "hq", "other"}
# Spellings the bank's own tables use for the same kinds.
_KIND_ALIASES = {"centre": "center", "call_centre": "center", "call_center": "center",
                 "backoffice": "back_office", "back-office": "back_office",
                 "live_branch": "live", "headquarters": "hq", "head_office": "hq"}


def _date(value: str) -> date | None:
    dt = try_parse_dt(value) if value else None
    return dt.date() if dt else None


def _code(value: str) -> str | None:
    text = value.strip()
    if text.endswith(".0"):
        text = text[:-2]
    return text.lstrip("0") or ("0" if text else None)


def read_units(folder: Path, report: ImportReport) -> list[UnitRow]:
    """The rows of units.csv, or [] when the batch has none. Rows that
    cannot be read are counted, the rest kept; an unknown kind becomes
    "unknown" and is counted, never silently a branch."""
    path = folder / UNITS_FILE
    if not path.exists():
        return []
    rows: list[UnitRow] = []
    bad = 0
    unknown_kinds: list[str] = []
    seen: set[str] = set()
    for raw in _csv_reader(_read_text_any_encoding(path)):
        raw.pop(None, None)
        row = {_visible(k or "").strip().lower(): _visible(v or "").strip()
               for k, v in raw.items() if k}
        key = row.get("unit_key", "")
        space = row.get("code_space", "").upper()
        if not key or space not in ("T1604", "T1017"):
            bad += 1
            continue
        if key in seen:
            bad += 1
            continue
        seen.add(key)
        kind = row.get("kind", "").strip().lower()
        kind = _KIND_ALIASES.get(kind, kind)
        if kind not in _KINDS:
            if kind:
                unknown_kinds.append(kind)
            kind = "unknown"
        try:
            rows.append(UnitRow(
                unit_key=key, code_space=space, snif_id=_code(row.get("snif_id", "")),
                org_unit_code=_code(row.get("org_unit_code", "")), name=row.get("name", ""),
                kind=kind, parent_unit_key=row.get("parent_unit_key") or None,
                cluster=row.get("cluster") or None, team=row.get("team") or None,
                region=row.get("region") or None,
                in_closure=row.get("in_closure", "").lower() in _TRUE,
                merged_into=row.get("merged_into") or None,
                valid_from=_date(row.get("valid_from", "")),
                valid_to=_date(row.get("valid_to", "")),
                crosswalk_basis=(row.get("crosswalk_basis", "").lower() or "none")
                if row.get("crosswalk_basis", "").lower() in ("measured", "hypothesis", "none")
                else "none"))
        except ValidationError:
            bad += 1
    if bad:
        report.add("units_rows_skipped", "warning",
                   f"{UNITS_FILE}: rows without a unit_key, with a code space other than "
                   "T1604 / T1017, or repeating a key; skipped", count=bad)
    if unknown_kinds:
        report.add("units_kind_unknown", "warning",
                   f"{UNITS_FILE}: unit kinds the tool does not know (kept as 'unknown')",
                   count=len(unknown_kinds), examples=sorted(set(unknown_kinds))[:5])
    report.counts.units = len(rows)
    return rows
