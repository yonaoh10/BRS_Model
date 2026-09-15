import sys, json
sys.path.insert(0, "/Users/yonatanohayon/Desktop/BRS_Model-main/src")
from callqa.models import RedactedTranscript
from callqa.judge.validation import (
    validate_judge_output, normalize_for_match, JudgeValidationError,
    _same_speaker_blocks, _snap_quote_to_turn, MIN_QUOTE_CHARS,
)
from callqa.judge.prompts import mmss

red = RedactedTranscript.model_validate(json.load(open(
    "/Users/yonatanohayon/Desktop/BRS_Model-main/data/output/redacted/REAL002.json")))
DIMS = ["identification","compliance","empathy","listening","clarity",
        "resolution","suitability","closure"]

# ---- exact 60-normalized-char threshold ----
long_turn = "אני עוצר אותך כאן, אין צורך שתמסור לי את מספר הכרטיס המלא בטלפון ארבע הספרות האחרונות מספיקות."
# cut a prefix whose normalized length is exactly 60
pref = ""
for ch in long_turn:
    pref += ch
    if len(normalize_for_match(pref)) == 60:
        break
print("prefix normalized len:", len(normalize_for_match(pref)))
def typo(q, n, seed=42):
    import random
    chars = list(q); letters=[i for i,c in enumerate(chars) if 'א'<=c<='ת']
    rng=random.Random(seed)
    for i in rng.sample(letters, n):
        chars[i]='ק' if chars[i]!='ק' else 'ר'
    return "".join(chars)
for k in range(0,10):
    mq = typo(pref,k) if k else pref
    snapped = _snap_quote_to_turn(mq, long_turn)
    print(f"  60-char quote, subs={k}: snap={'OK' if snapped else 'REJECT'}")

# ---- typo tolerance on a SHORT quote (~9-11 normalized chars) ----
print("\nshort-quote typo tolerance:")
for src, turn in [("לא, זה נפרד.", "לא, זה נפרד."), ("בשמחה אבדוק.", "בשמחה אבדוק.")]:
    L = len(normalize_for_match(src))
    for k in (1,2):
        mq = typo(src,k)
        snapped = _snap_quote_to_turn(mq, turn)
        print(f"  {src!r} norm_len={L} subs={k}: snap={'OK' if snapped else 'REJECT'}")

# ---- timestamp not cross-checked against quote location ----
def build_one(dim_ev):
    scores = {}
    default = ("הוא עולה 10 שקלים בחודש וכולל את פעולות הערוץ הישיר.", "01:35", "banker")
    for d in DIMS:
        q,t,s = dim_ev.get(d, default)
        scores[d]={"score":4,"reasoning_he":"ניתוח","evidence":[{"quote":q,"timestamp":t,"speaker":s}]}
    return json.dumps({"scores":scores,"strengths_he":["x"],"development_area_he":"y","summary_he":"z"}, ensure_ascii=False)

def check(label, raw):
    try:
        validate_judge_output(raw, DIMS, red)
        print(f"[PASS] {label}")
    except JudgeValidationError as e:
        print(f"[FAIL] {label}: {str(e)[:200]}")

print("\ntimestamp/speaker checks:")
check("wrong-but-well-formed timestamp (00:00 for a 02:39 quote)",
      build_one({"clarity": ("בוצע, עברת למסלול החודשי.", "00:00", "banker")}))
check("timestamp past call end +61s (04:05, call ends 03:03)",
      build_one({"clarity": ("בוצע, עברת למסלול החודשי.", "04:05", "banker")}))
check("timestamp at call end +59s (04:02)",
      build_one({"clarity": ("בוצע, עברת למסלול החודשי.", "04:02", "banker")}))
check("banker quote attributed to customer",
      build_one({"clarity": ("בוצע, עברת למסלול החודשי.", "02:39", "customer")}))
check("quote said by BOTH speakers? ('תודה רבה' banker turn 50; customer says 'תודה רבה שותף')",
      build_one({"closure": ("תודה רבה שותף.", "02:56", "banker")}))

# ---- evidence relevance is not checked ----
check("irrelevant-but-verbatim evidence for every dimension (greeting quoted 8x)",
      json.dumps({"scores":{d:{"score":5,"reasoning_he":"ניתוח",
        "evidence":[{"quote":"לידיעתך השיחה מוקלטת ומתומללת לצורכי בקרת איכות.","timestamp":"00:11","speaker":"banker"}]}
        for d in DIMS},"strengths_he":[],"development_area_he":"","summary_he":""}, ensure_ascii=False))

# ---- Q4: evidence supply per dimension ----
print("\n=== evidence supply (REAL002) ===")
blocks = _same_speaker_blocks(red)
def quotable(units):
    return [(sp, txt) for sp, txt in units if len(normalize_for_match(txt)) >= MIN_QUOTE_CHARS]
turns_units = [(t.speaker, t.text) for t in red.turns]
qt = quotable(turns_units); qb = quotable(blocks)
print(f"turns: {len(red.turns)} total, {len(qt)} quotable ({sum(1 for s,_ in qt if s=='banker')} banker, {sum(1 for s,_ in qt if s=='customer')} customer)")
print(f"blocks: {len(blocks)} total, {len(qb)} quotable ({sum(1 for s,_ in qb if s=='banker')} banker, {sum(1 for s,_ in qb if s=='customer')} customer)")

# per-dimension relevant windows (by call phase / speaker)
call_end = max(t.end for t in red.turns)
def count_window(t0, t1, speaker=None):
    n = 0; items=[]
    for t in red.turns:
        if t.start >= t0 and t.start < t1 and (speaker is None or t.speaker==speaker):
            if len(normalize_for_match(t.text)) >= MIN_QUOTE_CHARS:
                n += 1; items.append((mmss(t.start), t.speaker, t.text[:60]))
    return n, items

n_id, items_id = count_window(17.9, 70.0, None)  # identification exchange
print(f"\nidentification exchange (17.9s-70s): {n_id} quotable turns")
for ts, sp, tx in items_id: print(f"   [{ts}] {sp}: {tx}")
n_close, items_close = count_window(159.0, call_end+1, "banker")
print(f"closure window (159s-end) banker: {n_close} quotable turns")
for ts, sp, tx in items_close: print(f"   [{ts}] {sp}: {tx}")

# empathy: customer emotional statements + banker responses; here: how many customer turns express difficulty?
print(f"\ncall duration: {call_end:.0f}s ({mmss(call_end)})")

# ---- redaction leak check (digits left in customer turns) ----
import re
print("\nunmasked digit runs in redacted transcript:")
for i, t in enumerate(red.turns):
    for mtch in re.finditer(r"\d[\d\s\-,\.]{2,}", t.text):
        print(f"  turn {i:2d} [{mmss(t.start)}] {t.speaker}: ...{t.text!r}")
        break
