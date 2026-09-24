"""Synthetic journey inputs shaped like the bank's real exports.

Account numbers here are invented; tests check that they never reach any
output outside the private map.
"""

from __future__ import annotations

import zipfile
from datetime import datetime, timedelta
from pathlib import Path

from callqa.journey.xlsx import write_xlsx

ACCOUNTS = [("83", "335145"), ("67", "302556"), ("49", "225723")]
RAW_ACCOUNT_NUMBERS = [a for _b, a in ACCOUNTS]


def fmt(dt: datetime) -> str:
    return dt.strftime("%d%b%Y %H:%M:%S")   # 02Aug2026 16:04:34 (C locale in tests)


def build_workbook(folder: Path, *, audio: bool = True) -> tuple[Path, Path | None]:
    """Three stories:
       1: call (2 files) -> abandoned-looking unrecorded call -> call (1 file) -> UM correspondence
       2: call (3 files, ids in mixed case) -> unrecorded call
       3: UM correspondence -> call (1 file); declared repeat count is wrong (5)
    Sheets are in the order an EG import produced (files, messages, interactions)
    and the files sheet has no header row."""
    t0 = datetime(2026, 7, 2, 10, 8)
    rows = [["מועד אינטראקציה ראשונה", "סניף", "חשבון", "מועד השיחה/התכתבות",
             "מזהה שיחה/התכתבות", "סגמנטציית שירות", "כמות אינטראקציות חוזרות", "שם קובץ שיחה"]]
    files = []
    msgs = []
    # story 1
    b, a = ACCOUNTS[0]
    s1 = [
        (t0, "007203b62c453f86", "1_1111111111111111111_2222222222222222201"),
        (t0 + timedelta(minutes=19), "007203b62c453f87", ""),
        (t0 + timedelta(hours=5), "007203b62c453f88", "1_1111111111111111112_2222222222222222301"),
        (t0 + timedelta(days=1), "UM-10598898", ""),
    ]
    for at, rid, f in s1:
        rows.append([fmt(t0), b, a, fmt(at), rid, 1, 3, f])
    files += [["007203B62C453F86", "1_1111111111111111111_2222222222222222202"],
              ["007203B62C453F86", "1_1111111111111111111_2222222222222222201"],
              ["007203B62C453F88", "1_1111111111111111112_2222222222222222301"]]
    msgs += [["UM-10598898", "COR-5081090", fmt(t0 + timedelta(days=1)), "Inbound", "ערבות",
              "שלום, זו הפעם השלישית שאני פונה. ת.ז. 123456782 והטלפון 050-1234567"],
             ["UM-10598898", "COR-5088152", fmt(t0 + timedelta(days=1, hours=5)), "Outbound",
              "ערבות", "שלום רב, נבדוק ונחזור אליך"]]
    # story 2
    b, a = ACCOUNTS[1]
    t1 = datetime(2026, 7, 5, 9, 0)
    rows.append([fmt(t1), b, a, fmt(t1), "007203b059935cda", 2, 1,
                 "1_3333333333333333333_4444444444444444401"])
    rows.append([fmt(t1), b, a, fmt(t1 + timedelta(hours=2)), "007203b059935cdb", 2, 1, ""])
    files += [["007203B059935CDA", f"1_3333333333333333333_44444444444444444{n:02d}"]
              for n in (3, 1, 2)]
    # story 3
    b, a = ACCOUNTS[2]
    t2 = datetime(2026, 7, 9, 12, 0)
    rows.append([fmt(t2), b, a, fmt(t2), "UM-10099059", 3, 5, ""])
    rows.append([fmt(t2), b, a, fmt(t2 + timedelta(days=2)), "007203b0599bbe2c", 3, 5,
                 "1_5555555555555555555_6666666666666666601"])
    files.append(["007203B0599BBE2C", "1_5555555555555555555_6666666666666666601"])
    msgs.append(["UM-10099059", "COR-5081435", fmt(t2), "Inbound", "המלצות קנייה",
                 "היי רעות, אשמח לקבל המלצות קנייה מעודכנות"])
    msg_rows = [["מזהה תהליך התכתבות", "מזהה הודעה", "מועד שליחת ההודעה", "כיוון ההודעה",
                 "נושא ההתכתבות", "תוכן ההודעה"]] + msgs
    folder.mkdir(parents=True, exist_ok=True)
    xlsx = folder / "handoff.xlsx"
    write_xlsx(xlsx, [("ATL_REPEAT_0000", files), ("ATL_REPEAT_0001", msg_rows),
                      ("ATL_REPEAT_0002", rows)])
    zip_path = None
    if audio:
        zip_path = folder / "recordings.zip"
        with zipfile.ZipFile(zip_path, "w") as zf:
            for _key, name in files:
                zf.writestr(f"הקלטות/{name}.NMF", b"\x00" * 64)
            zf.writestr("הקלטות/1_9999999999999999999_1111111111111111101.NMF", b"\x00")  # orphan
    return xlsx, zip_path


def build_atlas(folder: Path) -> Path:
    """ATLR_INT / ATLR_SESS / ATLR_ROWS / ATLR_CODECAT as CSV exports (SAS
    DATETIME format), for story 1 only. Account numbers ride along in
    SNIF_ID/CHESHBON_ID and must be ignored."""
    folder.mkdir(parents=True, exist_ok=True)
    t0 = datetime(2026, 7, 2, 10, 8)

    def sas(dt: datetime) -> str:
        return dt.strftime("%d%b%Y:%H:%M:%S").upper()

    (folder / "ATLR_INT.csv").write_text(
        "ACC_KEY,SNIF_ID,CHESHBON_ID,INT_SEQ,INT_ID,INT_DT,INT_TYPE,KIND,DIR1,TALK_N,CHURN_N\n"
        f"1,83,335145,1,007203b62c453f86,{sas(t0)},REC,CALL,I,240,0\n"
        f"1,83,335145,2,007203b62c453f87,{sas(t0 + timedelta(minutes=19))},ABN,CALL,I,0,1\n"
        f"1,83,335145,3,007203b62c453f88,{sas(t0 + timedelta(hours=5))},REC,CALL,O,300,0\n"
        f"1,83,335145,4,UM-10598898,{sas(t0 + timedelta(days=1))},UM,UM,,,\n",
        encoding="utf-8")
    (folder / "ATLR_SESS.csv").write_text(
        "ACC_KEY,SESS_NO,S_START,S_END,S_BANKER,S_UNIT,N_OPS,INT_SEQ,LINKED\n"
        f"1,1,{sas(t0 + timedelta(minutes=1))},{sas(t0 + timedelta(minutes=7))},B0001,109,3,1,1\n"
        f"1,2,{sas(t0 + timedelta(hours=2))},{sas(t0 + timedelta(hours=2, minutes=20))},B0002,83,2,,0\n",
        encoding="utf-8")
    (folder / "ATLR_ROWS.csv").write_text(
        "ROW_ID,ACC_KEY,TS_DT,UNIT_NO,BANKER_CODE,OP_KEY,OP_DESC\n"
        f"1,1,{sas(t0 + timedelta(minutes=1))},109,B0001,201,מצב לקוח\n"
        f"2,1,{sas(t0 + timedelta(minutes=3))},109,B0001,011,תנועות אחרונות\n"
        f"3,1,{sas(t0 + timedelta(minutes=7))},109,B0001,332,פרטי חשבון\n"
        f"4,1,{sas(t0 + timedelta(hours=2))},083,B0002,201,מצב לקוח\n"
        f"5,1,{sas(t0 + timedelta(hours=2, minutes=20))},083,B0002,970,הזמנת כרטיסים\n",
        encoding="utf-8")
    (folder / "ATLR_CODECAT.csv").write_text(
        "CAT,OP_KEY,OP_DESC,N_ROWS\nOPEN,201,,2\nLOOK,011,,1\nLOOK,332,,1\nDO,970,,1\n",
        encoding="utf-8")
    return folder
