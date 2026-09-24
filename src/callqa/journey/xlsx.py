"""Read an .xlsx workbook with the standard library only.

An .xlsx is a ZIP of XML files. This reads what a data export needs - every
sheet's cell values, with dates recognised from the cell's number format -
and nothing else (no formulas are evaluated; the cached value is used, which
is what Excel shows). No new dependency, so nothing to add to the offline
bundle, and nothing that executes content from the file.

Guarded against a hostile or broken file: member sizes and compression
ratios are capped (a "zip bomb" is refused), external entities cannot be
resolved (ElementTree does not fetch them), and a sheet is streamed.
"""

from __future__ import annotations

import posixpath
import re
import zipfile
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import IO, Any
from xml.etree import ElementTree as ET

from callqa.journey.timeparse import TimeParseError, from_number

NS_MAIN = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
NS_REL = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
NS_PKG = "http://schemas.openxmlformats.org/package/2006/relationships"
_M = f"{{{NS_MAIN}}}"

MAX_MEMBER_BYTES = 400 * 1024 * 1024
MAX_RATIO = 200          # uncompressed / compressed; text XML compresses ~10-30x
MAX_CELLS = 5_000_000

# Built-in number formats that are dates or times (ECMA-376 18.8.30).
_BUILTIN_DATE_IDS = set(range(14, 23)) | set(range(27, 37)) | {45, 46, 47} | set(range(50, 59))
_QUOTED = re.compile(r'"[^"]*"|\\.|\[[^\]]*\]')
_COL = re.compile(r"^([A-Z]+)")


class XlsxError(ValueError):
    pass


@dataclass
class Sheet:
    name: str
    rows: list[list[Any]] = field(default_factory=list)

    def cell(self, row: int, col: int) -> Any:
        values = self.rows[row] if 0 <= row < len(self.rows) else []
        return values[col] if 0 <= col < len(values) else None


@dataclass
class Workbook:
    sheets: list[Sheet]
    date1904: bool = False

    def by_name(self, name: str) -> Sheet | None:
        for s in self.sheets:
            if s.name.strip() == name.strip():
                return s
        return None


def column_index(ref: str) -> int:
    """'A' -> 0, 'Z' -> 25, 'AA' -> 26."""
    m = _COL.match(ref or "")
    if not m:
        raise XlsxError(f"bad cell reference {ref!r}")
    n = 0
    for ch in m.group(1):
        n = n * 26 + (ord(ch) - 64)
    return n - 1


def _is_date_format(code: str) -> bool:
    body = _QUOTED.sub("", code or "").lower()
    if not body or body == "general" or "@" in body:
        return False
    # the first section decides (positive numbers)
    body = body.split(";", 1)[0]
    return any(ch in body for ch in "dyhs") or ("m" in body and "0" not in body)


def _check_members(zf: zipfile.ZipFile) -> None:
    for info in zf.infolist():
        if info.file_size > MAX_MEMBER_BYTES:
            raise XlsxError(f"member {info.filename} is too large")
        if info.compress_size and info.file_size / info.compress_size > MAX_RATIO:
            raise XlsxError(f"member {info.filename} has an implausible compression ratio")


def _read(zf: zipfile.ZipFile, name: str) -> bytes | None:
    try:
        return zf.read(name)
    except KeyError:
        return None


def _text_of(elem: ET.Element) -> str:
    """The visible text of a shared or inline string: its <t>, or every <r><t>
    run; phonetic hints (<rPh>) are not text."""
    direct = elem.find(f"{_M}t")
    if direct is not None:
        return direct.text or ""
    return "".join((t.text or "") for r in elem.findall(f"{_M}r") for t in r.findall(f"{_M}t"))


def _shared_strings(zf: zipfile.ZipFile, path: str) -> list[str]:
    raw = _read(zf, path)
    if raw is None:
        return []
    root = ET.fromstring(raw)
    return [_text_of(si) for si in root.findall(f"{_M}si")]


def _date_styles(zf: zipfile.ZipFile, path: str) -> set[int]:
    raw = _read(zf, path)
    if raw is None:
        return set()
    root = ET.fromstring(raw)
    custom: dict[int, str] = {}
    fmts = root.find(f"{_M}numFmts")
    if fmts is not None:
        for nf in fmts.findall(f"{_M}numFmt"):
            try:
                custom[int(nf.get("numFmtId", "-1"))] = nf.get("formatCode", "")
            except ValueError:
                continue
    dates: set[int] = set()
    xfs = root.find(f"{_M}cellXfs")
    if xfs is not None:
        for idx, xf in enumerate(xfs.findall(f"{_M}xf")):
            try:
                fid = int(xf.get("numFmtId", "0"))
            except ValueError:
                continue
            if fid in _BUILTIN_DATE_IDS or (fid in custom and _is_date_format(custom[fid])):
                dates.add(idx)
    return dates


def _relationships(zf: zipfile.ZipFile) -> dict[str, str]:
    raw = _read(zf, "xl/_rels/workbook.xml.rels")
    if raw is None:
        return {}
    root = ET.fromstring(raw)
    out = {}
    for rel in root.findall(f"{{{NS_PKG}}}Relationship"):
        target = rel.get("Target", "")
        if target.startswith("/"):
            target = target[1:]
        else:
            target = posixpath.normpath(posixpath.join("xl", target))
        out[rel.get("Id", "")] = target
    return out


def _cell_value(c: ET.Element, strings: list[str], date_styles: set[int], date1904: bool
                ) -> Any:
    kind = c.get("t", "n")
    if kind == "inlineStr":
        is_ = c.find(f"{_M}is")
        return _text_of(is_) if is_ is not None else ""
    v = c.find(f"{_M}v")
    text = v.text if v is not None else None
    if text is None:
        return None
    if kind == "s":
        try:
            return strings[int(text)]
        except (ValueError, IndexError):
            return None
    if kind in ("str", "e"):
        return text
    if kind == "b":
        return text.strip() == "1"
    if kind == "d":
        try:
            return datetime.fromisoformat(text.replace("Z", ""))
        except ValueError:
            return text
    try:
        number = float(text)
    except ValueError:
        return text
    try:
        style = int(c.get("s", "0"))
    except ValueError:
        style = 0
    if style in date_styles:
        try:
            return from_number(number, date1904=date1904)
        except TimeParseError:
            return number
    return number


def _sheet_rows(stream: IO[bytes], strings: list[str], date_styles: set[int],
                date1904: bool) -> list[list[Any]]:
    rows: dict[int, list[Any]] = {}
    cells = 0
    next_row = 0
    for _event, elem in ET.iterparse(stream, events=("end",)):
        if elem.tag != f"{_M}row":
            continue
        try:
            r = int(elem.get("r", "0")) - 1
        except ValueError:
            r = -1
        if r < 0:
            r = next_row
        next_row = r + 1
        values: list[Any] = []
        col_cursor = 0
        for c in elem.findall(f"{_M}c"):
            ref = c.get("r")
            col = column_index(ref) if ref else col_cursor
            col_cursor = col + 1
            cells += 1
            if cells > MAX_CELLS:
                raise XlsxError("the sheet has too many cells")
            value = _cell_value(c, strings, date_styles, date1904)
            if col >= len(values):
                values.extend([None] * (col + 1 - len(values)))
            values[col] = value
        rows[r] = values
        elem.clear()
    if not rows:
        return []
    return [rows.get(i, []) for i in range(max(rows) + 1)]


def read_xlsx(path: str | Path) -> Workbook:
    """Every sheet of the workbook, in the workbook's own order."""
    try:
        zf = zipfile.ZipFile(path)
    except (zipfile.BadZipFile, OSError) as exc:
        raise XlsxError(f"not a readable .xlsx workbook: {exc}") from None
    with zf:
        _check_members(zf)
        raw = _read(zf, "xl/workbook.xml")
        if raw is None:
            raise XlsxError("no xl/workbook.xml - not an .xlsx workbook")
        root = ET.fromstring(raw)
        pr = root.find(f"{_M}workbookPr")
        date1904 = pr is not None and pr.get("date1904", "0") in ("1", "true")
        rels = _relationships(zf)
        strings = _shared_strings(zf, "xl/sharedStrings.xml")
        date_styles = _date_styles(zf, "xl/styles.xml")
        sheets: list[Sheet] = []
        container = root.find(f"{_M}sheets")
        for s in (container.findall(f"{_M}sheet") if container is not None else []):
            rid = s.get(f"{{{NS_REL}}}id", "")
            target = rels.get(rid) or f"xl/worksheets/sheet{s.get('sheetId', '')}.xml"
            try:
                with zf.open(target) as fh:
                    rows = _sheet_rows(fh, strings, date_styles, date1904)
            except KeyError:
                raise XlsxError(f"sheet {s.get('name')!r} is missing from the file") from None
            except ET.ParseError as exc:
                raise XlsxError(f"sheet {s.get('name')!r} is not valid XML: {exc}") from None
            sheets.append(Sheet(name=s.get("name", ""), rows=rows))
    return Workbook(sheets=sheets, date1904=date1904)


def write_xlsx(path: str | Path, sheets: list[tuple[str, list[list[Any]]]]) -> None:
    """A minimal writer for tests and the demo generator: strings are inline,
    numbers are numbers, datetimes are Excel serials with a date style."""
    from xml.sax.saxutils import escape

    def cell(ref: str, value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, bool):
            return f'<c r="{ref}" t="b"><v>{int(value)}</v></c>'
        if isinstance(value, datetime):
            serial = (value - datetime(1899, 12, 30)).total_seconds() / 86400.0
            return f'<c r="{ref}" s="1"><v>{serial!r}</v></c>'
        if isinstance(value, int | float):
            return f'<c r="{ref}"><v>{value!r}</v></c>'
        return (f'<c r="{ref}" t="inlineStr"><is><t xml:space="preserve">'
                f"{escape(str(value))}</t></is></c>")

    def col_name(i: int) -> str:
        name = ""
        i += 1
        while i:
            i, rem = divmod(i - 1, 26)
            name = chr(65 + rem) + name
        return name

    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("[Content_Types].xml",
                    '<?xml version="1.0" encoding="UTF-8"?><Types xmlns="http://schemas.'
                    'openxmlformats.org/package/2006/content-types"><Default Extension="xml" '
                    'ContentType="application/xml"/><Default Extension="rels" ContentType='
                    '"application/vnd.openxmlformats-package.relationships+xml"/></Types>')
        sheet_tags = "".join(
            f'<sheet name="{escape(name)}" sheetId="{i + 1}" r:id="rId{i + 1}"/>'
            for i, (name, _rows) in enumerate(sheets))
        zf.writestr("xl/workbook.xml",
                    f'<?xml version="1.0" encoding="UTF-8"?><workbook xmlns="{NS_MAIN}" '
                    f'xmlns:r="{NS_REL}"><sheets>{sheet_tags}</sheets></workbook>')
        rels = "".join(
            f'<Relationship Id="rId{i + 1}" Type="{NS_REL}/worksheet" '
            f'Target="worksheets/sheet{i + 1}.xml"/>' for i in range(len(sheets)))
        zf.writestr("xl/_rels/workbook.xml.rels",
                    f'<?xml version="1.0" encoding="UTF-8"?><Relationships xmlns="{NS_PKG}">'
                    f"{rels}</Relationships>")
        zf.writestr("xl/styles.xml",
                    f'<?xml version="1.0" encoding="UTF-8"?><styleSheet xmlns="{NS_MAIN}">'
                    '<numFmts count="1"><numFmt numFmtId="164" formatCode="ddmmmyyyy hh:mm:ss"/>'
                    '</numFmts><cellXfs count="2"><xf numFmtId="0"/><xf numFmtId="164"/>'
                    "</cellXfs></styleSheet>")
        for i, (_name, rows) in enumerate(sheets):
            body = []
            for r, values in enumerate(rows, start=1):
                cells = "".join(cell(f"{col_name(c)}{r}", v) for c, v in enumerate(values))
                body.append(f'<row r="{r}">{cells}</row>')
            zf.writestr(f"xl/worksheets/sheet{i + 1}.xml",
                        f'<?xml version="1.0" encoding="UTF-8"?><worksheet xmlns="{NS_MAIN}">'
                        f"<sheetData>{''.join(body)}</sheetData></worksheet>")
