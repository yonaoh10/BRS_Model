"""Atlas: what the bankers did on each account, beside what the customer did.

Reads the tables of the Atlas repeat-contact project (SAS: ATL_R01/R02),
exported as CSV files or as sheets of one workbook:

    ATLR_INT      one row per contact: ACC_KEY, INT_SEQ, INT_ID, INT_DT,
                  INT_TYPE (REC/ABN/TALK/UM/UNK), DIR1 (I/O), TALK_N, CHURN_N
    ATLR_SESS     one row per banker session: ACC_KEY, SESS_NO, S_START, S_END,
                  S_BANKER (a running code, never a name), S_UNIT, N_OPS, INT_SEQ
                  - optional: without it the sessions are built from ATLR_ROWS
                  by the ATL_R01 rules (sessions.py)
    ATLR_ROWS     one row per Atlas operation: ACC_KEY, TS_DT, UNIT_NO,
                  BANKER_CODE, OP_KEY, OP_DESC - in the export's own order
    ATLR_CODECAT  CAT (OPEN/LOOK/DO/NONC/OTHER), OP_KEY - optional, over the
                  shipped journey_atlas_codes.yaml

The raw log (STG_ATLAS_LOG_357) is never read here: it names the customer by
national id and the banker by user name, and the step from it to an account
goes through id numbers - that stays in the bank's SAS program (ATL_R01).

The generic names (atlas_interactions / atlas_sessions / atlas_ops /
atlas_codes, with interaction_id, at, session_id, start, end, banker_code,
unit_code, op_code, op_category...) are read too. Account numbers that ride
along (SNIF_ID, CHESHBON_ID) are never read: an Atlas account (ACC_KEY) is
tied to a story through the contacts both sides share - the same call or
correspondence id at the same time.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

from callqa.ingestion import _csv_reader, _read_text_any_encoding, _visible
from callqa.journey.importers.common import text
from callqa.journey.models import AtlasOp, AtlasRules, AtlasSession, JourneyDataset
from callqa.journey.sessions import AtlasRow, build_sessions
from callqa.journey.timeparse import try_parse_dt
from callqa.journey.vocab import load_atlas_codes, normalise_op_code
from callqa.journey.xlsx import read_xlsx

TABLES = {
    "interactions": ("atlr_int", "atlas_interactions"),
    "sessions": ("atlr_sess", "atlr_sess2", "atlas_sessions"),
    "ops": ("atlr_rows", "atlas_ops"),
    "codes": ("atlr_codecat", "atlas_codes"),
}
COLUMNS = {
    "acc": ("acc_key", "atlas_account"),
    "int_seq": ("int_seq",),
    "int_id": ("int_id", "interaction_id", "source_id"),
    "int_at": ("int_dt", "at", "started_at"),
    "int_type": ("int_type",),
    "direction": ("dir1", "direction"),
    "talk": ("talk_n", "talk_seconds"),
    "abandoned": ("churn_n", "abandoned"),
    "session": ("sess_no", "session_id"),
    "start": ("s_start", "start"),
    "end": ("s_end", "end"),
    "banker": ("s_banker", "banker_code", "banker_txt"),
    "unit": ("s_unit", "unit_no", "unit_code"),
    "n_ops": ("n_ops",),
    "peek": ("peek",),
    "op_at": ("ts_dt", "at"),
    "op": ("op_key", "op_code"),
    "op_desc": ("op_desc", "description"),
    "cat": ("cat", "op_category"),
}
CATEGORIES = {"open": "open", "look": "info", "info": "info", "do": "execute",
              "execute": "execute", "nonc": "not_customer", "not_customer": "not_customer",
              "other": "unclassified", "unclassified": "unclassified"}


def _get(row: dict[str, str], key: str) -> str:
    for name in COLUMNS[key]:
        if row.get(name):
            return row[name]
    return ""


def _code(value: str) -> str:
    """Atlas codes are text with leading zeros ('035'); decode tables drop them."""
    return normalise_op_code(text(value))


def _read_tables(source: Path) -> dict[str, list[dict[str, str]]]:
    tables: dict[str, list[dict[str, str]]] = {}

    def keep(kind: str, rows: list[dict[str, str]]) -> None:
        tables.setdefault(kind, rows)

    def kind_of(name: str) -> str | None:
        stem = Path(name).stem.lower()
        for kind, names in TABLES.items():
            if stem in names:
                return kind
        return None

    if source.is_file() and source.suffix.lower() == ".xlsx":
        for sheet in read_xlsx(source).sheets:
            kind = kind_of(sheet.name)
            if not kind or not sheet.rows:
                continue
            header = [text(h).lower() for h in sheet.rows[0]]
            rows = [{h: text(v) for h, v in zip(header, r, strict=False) if h}
                    for r in sheet.rows[1:]]
            keep(kind, [r for r in rows if any(r.values())])
    elif source.is_dir():
        for path in sorted(source.glob("*.csv")):
            kind = kind_of(path.name)
            if not kind:
                continue
            reader = _csv_reader(_read_text_any_encoding(path))
            rows = []
            for raw in reader:
                raw.pop(None, None)
                rows.append({_visible(k or "").strip().lower(): _visible(v or "").strip()
                             for k, v in raw.items() if k})
            keep(kind, rows)
    else:
        raise FileNotFoundError(f"Atlas source not found: {source}")
    return tables


def attach_atlas(dataset: JourneyDataset, source: str | Path, *,
                 join_tolerance_sec: float = 60.0,
                 coverage_start: datetime | None = None,
                 rules: AtlasRules | None = None,
                 codes: dict[str, str] | None = None) -> None:
    """Add Atlas facts to the dataset in place: contact direction, answer and
    talk time; banker sessions with their operations; per-story coverage.

    Sessions come from the export's own table (ATLR_SESS) when it has one -
    checked against the sessions its log rows make - or are built from the
    log rows (ATLR_ROWS / atlas_ops) by the ATL_R01 rules."""
    report = dataset.report
    tables = _read_tables(Path(source))
    rules = (rules or AtlasRules()).model_copy()
    if coverage_start is not None:
        rules.coverage_from = coverage_start
    if "sessions" not in tables and "ops" not in tables:
        report.add("atlas_no_sessions", "error",
                   "no Atlas sessions table and no Atlas log rows were found")
        return

    # 1. which Atlas account is which story: through shared contacts
    by_ref: dict[str, list] = defaultdict(list)
    for i in dataset.interactions:
        ref = (i.call_key or i.correspondence_id or "").lower()
        if ref:
            by_ref[ref].append(i)
    acc_votes: dict[str, Counter] = defaultdict(Counter)
    seq_to_iid: dict[tuple[str, str], str] = {}
    matched = 0
    for row in tables.get("interactions", []):
        acc = _get(row, "acc")
        ref = _get(row, "int_id").lower()
        at = try_parse_dt(_get(row, "int_at"))
        candidates = by_ref.get(ref, [])
        if at is not None:
            candidates = [c for c in candidates
                          if abs((c.at - at).total_seconds()) <= join_tolerance_sec] or candidates
        if not candidates:
            continue
        target = min(candidates, key=lambda c: abs((c.at - at).total_seconds()) if at else 0)
        matched += 1
        acc_votes[acc][target.story_key] += 1
        seq = _get(row, "int_seq")
        if seq:
            seq_to_iid[(acc, seq)] = target.interaction_id
        # facts the contact table did not have
        direction = _get(row, "direction").upper()[:1]
        if target.direction == "unknown" and direction in ("I", "O"):
            target.direction = "inbound" if direction == "I" else "outbound"
        itype = _get(row, "int_type").upper()
        talk = _get(row, "talk")
        abandoned = _get(row, "abandoned")
        if target.channel == "call" and target.answer == "unknown":
            if itype == "ABN" or abandoned in ("1", "1.0"):
                target.answer = "abandoned"
            elif itype in ("REC", "TALK"):
                target.answer = "answered"
        if target.talk_seconds is None and talk:
            try:
                target.talk_seconds = float(talk)
            except ValueError:
                pass
    story_of_acc = {acc: votes.most_common(1)[0][0] for acc, votes in acc_votes.items()}
    total_int = len(tables.get("interactions", []))
    if total_int and matched < total_int:
        report.add("atlas_contacts_unmatched", "warning",
                   "Atlas contact rows with no matching contact in the dataset "
                   f"(tolerance {join_tolerance_sec:.0f} s)", count=total_int - matched)

    # 2. operation categories: the shipped lists (ATL_R02), then the export's own
    cats = dict(codes if codes is not None else load_atlas_codes())
    for r in tables.get("codes", []):
        if _get(r, "op"):
            cats[_code(_get(r, "op"))] = CATEGORIES.get(_get(r, "cat").lower(), "unclassified")

    # 3. the log rows of the accounts that are stories, in export order
    rows: list[AtlasRow] = []
    orphan_rows = 0
    for n, row in enumerate(tables.get("ops", [])):
        acc = _get(row, "acc")
        at = try_parse_dt(_get(row, "op_at"))
        if at is None or acc not in story_of_acc:
            orphan_rows += 1
            continue
        rows.append(AtlasRow(account=acc, at=at, banker=_get(row, "banker") or "?",
                             unit=_code(_get(row, "unit")) or "?", op=_code(_get(row, "op")),
                             description=_get(row, "op_desc")[:120], order=n))
    if orphan_rows:
        report.add("atlas_rows_unmapped", "warning",
                   "Atlas log rows of accounts that could not be tied to a story, "
                   "or with no time", count=orphan_rows)
    built = build_sessions(rows, gap_min=rules.gap_min, start_op=rules.start_op)

    def op_of(r: AtlasRow) -> AtlasOp:
        return AtlasOp(at=r.at, op_code=r.op, op_category=cats.get(r.op, "unclassified"),
                       description=r.description)

    # 4. sessions: the export's own (ATLR_SESS) when given, else built from the rows
    sessions: list[AtlasSession] = []
    if "sessions" in tables:
        rules.sessions_from = "export"
        by_key: dict[tuple[str, str], AtlasSession] = {}
        unmapped = 0
        for row in tables["sessions"]:
            acc = _get(row, "acc")
            skey = story_of_acc.get(acc)
            start, end = try_parse_dt(_get(row, "start")), try_parse_dt(_get(row, "end"))
            if skey is None or start is None:
                unmapped += 1
                continue
            sid = _get(row, "session") or f"{len(by_key) + 1}"
            try:
                n_ops = int(float(_get(row, "n_ops") or 0))
            except ValueError:
                n_ops = 0
            by_key[(acc, sid)] = AtlasSession(
                session_id=f"{skey[:8]}-{sid}", story_key=skey,
                banker_code=_get(row, "banker") or "?", unit_code=_code(_get(row, "unit")) or "?",
                start=start, end=end or start, n_ops=n_ops,
                matched_interaction_id=seq_to_iid.get((acc, _get(row, "int_seq"))),
                peek=_get(row, "peek") in ("1", "1.0"))
        if unmapped:
            report.add("atlas_sessions_unmapped", "warning",
                       "Atlas sessions of accounts that could not be tied to a story",
                       count=unmapped)
        # a row belongs to the session of its account and banker whose span holds
        # it; on a boundary shared by two, the earlier one (ATL_R02 B3)
        by_acc_banker: dict[tuple[str, str], list[AtlasSession]] = defaultdict(list)
        for (acc, _sid), sess in by_key.items():
            by_acc_banker[(acc, sess.banker_code)].append(sess)
        for group in by_acc_banker.values():
            group.sort(key=lambda x: x.start)
        orphan_ops = 0
        for r in rows:
            home = next((x for x in by_acc_banker.get((r.account, r.banker), [])
                         if x.start <= r.at <= x.end), None)
            if home is None:
                orphan_ops += 1
                continue
            home.ops.append(op_of(r))
        if orphan_ops:
            report.add("atlas_ops_outside_sessions", "info",
                       "Atlas operations that fall in no session of their banker",
                       count=orphan_ops)
        _compare(report, by_key, built)
        sessions = list(by_key.values())
    elif built:
        rules.sessions_from = "rows"
        numbers: Counter = Counter()
        for group in built:
            acc = group[0].account
            numbers[acc] += 1
            skey = story_of_acc[acc]
            sessions.append(AtlasSession(
                session_id=f"{skey[:8]}-{numbers[acc]}", story_key=skey,
                banker_code=group[0].banker, unit_code=group[0].unit,
                start=group[0].at, end=group[-1].at, n_ops=len(group),
                ops=[op_of(r) for r in group]))
    else:
        report.add("atlas_no_sessions", "error",
                   "no Atlas sessions table and no Atlas log rows were found")
        return
    for sess in sessions:
        sess.ops.sort(key=lambda op: op.at)
        if not sess.n_ops:
            sess.n_ops = len(sess.ops)
        if sess.ops and not sess.first_op:
            sess.first_op = sess.ops[0].op_code
        if rules.sessions_from == "rows" or not sess.peek:
            sess.peek = sess.n_ops == 1 and sess.first_op == rules.start_op
    dataset.atlas_sessions = sorted(sessions, key=lambda x: (x.story_key, x.start))
    unclassified = {op.op_code for x in sessions for op in x.ops
                    if op.op_category == "unclassified"}
    if unclassified:
        report.add("atlas_codes_unclassified", "info",
                   "Atlas operation codes in no list of journey_atlas_codes.yaml "
                   "(shown as unclassified)", count=len(unclassified),
                   examples=sorted(unclassified)[:5])

    # 5. coverage per story: the log keeps ~90 days, so a story that began
    # before the first day it holds is only partly visible. ATL_R02: covered =
    # the day of the story's first contact is on or after that first day.
    first_seen = rules.coverage_from or min(
        (r.at for r in rows),
        default=min((x.start for x in dataset.atlas_sessions), default=None))
    if first_seen is not None:
        first_seen = first_seen.replace(hour=0, minute=0, second=0, microsecond=0)
    rules.coverage_from = first_seen
    for story in dataset.stories:
        if first_seen is None:
            story.atlas_coverage = "none"
        else:
            story.atlas_coverage = ("full" if story.first_at.date() >= first_seen.date()
                                    else "partial")
    dataset.atlas_rules = rules
    report.counts.atlas_sessions = len(dataset.atlas_sessions)
    report.counts.atlas_ops = sum(len(x.ops) for x in dataset.atlas_sessions)


def _compare(report, exported: dict[tuple[str, str], AtlasSession],
             built: list[list[AtlasRow]]) -> None:
    """The sessions the export carries against the ones its own rows make by
    the ATL_R01 rules: the same count, starts, ends and bankers, account by
    account. A difference means the export and these rules disagree - it is
    reported, never silently resolved."""
    if not built:
        return
    mine: dict[str, list[tuple]] = defaultdict(list)
    for group in built:
        mine[group[0].account].append((group[0].at, group[-1].at, group[0].banker))
    theirs: dict[str, list[tuple]] = defaultdict(list)
    for (acc, _sid), x in exported.items():
        theirs[acc].append((x.start, x.end, x.banker_code))
    differ = [acc for acc in set(mine) | set(theirs)
              if sorted(mine.get(acc, [])) != sorted(theirs.get(acc, []))]
    n_mine = sum(len(v) for v in mine.values())
    n_theirs = sum(len(v) for v in theirs.values())
    if differ:
        report.add("atlas_sessions_differ", "warning",
                   f"accounts whose exported sessions ({n_theirs:,}) differ from the ones "
                   f"their log rows make by the session rules ({n_mine:,})", count=len(differ))
    else:
        report.add("atlas_sessions_rebuilt", "info",
                   "exported sessions that the log rows rebuild exactly by the session rules",
                   count=n_theirs)
