import sys, json, random
sys.path.insert(0, "/Users/yonatanohayon/Desktop/BRS_Model-main/src")
from callqa.models import RedactedTranscript
from callqa.judge.validation import (
    validate_judge_output, verify_evidence, parse_judge_response,
    normalize_for_match, JudgeValidationError, MIN_QUOTE_CHARS,
    _same_speaker_blocks, _snap_quote_to_turn, SNAP_MIN_SIMILARITY,
)
from callqa.judge.prompts import mmss

red = RedactedTranscript.model_validate(json.load(open(
    "/Users/yonatanohayon/Desktop/BRS_Model-main/data/output/redacted/REAL002.json")))
DIMS = ["identification","compliance","empathy","listening","clarity",
        "resolution","suitability","closure"]

def build(evmap):
    """evmap: dim -> (quote, ts, speaker)"""
    return json.dumps({
        "scores": {d: {"score": 4, "reasoning_he": "ניתוח",
                       "evidence": [{"quote": q, "timestamp": t, "speaker": s}]}
                   for d, (q, t, s) in evmap.items()},
        "strengths_he": ["x"], "development_area_he": "y", "summary_he": "z",
    }, ensure_ascii=False)

def try_validate(label, raw):
    try:
        resp = validate_judge_output(raw, DIMS, red)
        print(f"[PASS] {label}")
        return resp
    except JudgeValidationError as e:
        print(f"[FAIL] {label}: {str(e)[:300]}")
        return None

# ---------- (a) perfect judge: verbatim per-turn quotes, correct speaker/ts ----------
verbatim = {
    "identification": ("לפני כן אני חייב לזהות אותך. מה מספר תעודת הזהות שלך?", mmss(23.71), "banker"),
    "compliance": ("הוא עולה 10 שקלים בחודש וכולל את פעולות הערוץ הישיר.", mmss(94.96), "banker"),
    "empathy": ("בסדר גמור. אגב, שאלת קודם על החיסכון.", mmss(146.91), "banker"),
    "listening": ("כמה פעולות כאלה אתה עושה בחודש בממוצע?", mmss(77.42), "banker"),
    "clarity": ("לסיכום, העמלה שראית היא עמלה פעולה בסניף.", mmss(161.94), "banker"),
    "resolution": ("בוצע, עברת למסלול החודשי.", mmss(159.16), "banker"),
    "suitability": ("אז המסלול החודשי משתלם לך,", mmss(92.92), "banker"),
    "closure": ("אשלח לך אישור בהודעה. יש עוד משהו שאוכל לעזור בו?", mmss(172.06), "banker"),
}
print("=== (a) perfect judge, verbatim quotes ===")
try_validate("perfect judge", build(verbatim))

# ---------- (b) good-faith typos: 1-2 char letter typos per quote ----------
print("\n=== (b) 1-2 char typos per quote (letter substitutions, not punctuation) ===")
def typo(q, n):
    # substitute n Hebrew letters (deterministic positions) with a different letter
    chars = list(q)
    letters = [i for i, c in enumerate(chars) if 'א' <= c <= 'ת']
    rng = random.Random(42)
    for i in rng.sample(letters, n):
        chars[i] = 'ק' if chars[i] != 'ק' else 'ר'
    return "".join(chars)

typo_map = {d: (typo(q, 2), t, s) for d, (q, t, s) in verbatim.items()}
resp = try_validate("2 letter typos in every quote", build(typo_map))
if resp:
    print("  snapped quotes (should be verbatim transcript spans):")
    for d, ds in resp.scores.items():
        orig = typo_map[d][0]
        got = ds.evidence[0].quote
        print(f"   {d:15s} snapped_back_to_verbatim={got != orig and normalize_for_match(got) in [normalize_for_match(b[1]) for b in _same_speaker_blocks(red)] or 'check'}")
        print(f"      sent: {orig!r}")
        print(f"      kept: {got!r}")

# ---------- (b2) empirical similarity threshold on a 60-char quote ----------
print("\n=== (b2) edit-distance threshold for 0.90 on a ~60-char quote ===")
q60 = "אני עוצר אותך כאן, אין צורך שתמסור לי את מספר הכרטיס המלא בטלפון ארבע הספרות האחרונות מספיקות."
# take exactly first 60 normalized chars worth: use a real turn
turn60 = "אני עוצר אותך כאן, אין צורך שתמסור לי את מספר הכרטיס המלא בטלפון ארבע הספרות האחרונות מספיקות."
base = "מצוין, הזיהוי הושלם. אני רואה את החשבון שמסתיים ב -<חשבון:████> בסניף <חשבון:████>."
# use the long banker turn, normalized length reported
for src in [turn60]:
    norm_len = len(normalize_for_match(src))
    print(f"quote: {src[:50]}... normalized_len={norm_len}")
    from difflib import SequenceMatcher
    for k in range(0, 12):
        mq = typo(src, k) if k else src
        snapped = _snap_quote_to_turn(mq, src)
        ratio = SequenceMatcher(None, normalize_for_match(mq), normalize_for_match(src)).ratio()
        print(f"  substitutions={k:2d} direct_ratio={ratio:.3f} snap={'OK' if snapped else 'REJECT'}")

# pure-theory check for exactly 60 normalized chars
print("theory: ratio for k subs on L=60 vs 60 = (60-k)/60; k<=6 keeps >=0.90")

# ---------- (c) splice across a speaker change ----------
print("\n=== (c) quote spliced across speaker change ===")
# banker: "...תהיה" (turn 1) + customer: "בנקאי." (turn 2)
splice = dict(verbatim)
splice["identification"] = ("כן כן, נו, תהיה בנקאי.", mmss(3.24), "banker")
try_validate("splice banker->customer", build(splice))
# splice customer end + banker start
splice["identification"] = ("נכון, זה החשבון. רגע אחד, אני מושך את פירוט החיובים.", mmss(59.9), "customer")
try_validate("splice customer->banker", build(splice))

# ---------- (d) same-speaker block spanning consecutive turns ----------
print("\n=== (d) quote spanning consecutive same-speaker turns ===")
block = dict(verbatim)
block["compliance"] = ("עמלת פעולה בסניף על הפקדה שביצעת ב -10 לחודש, 30 שקלים.", mmss(73.58), "banker")
try_validate("cross-turn same-speaker (banker turns @73.58+77.42)", build(block))
block["compliance"] = ("חשוב שאציין שהעמדה נכבד גם בחודש שבו לא תבצע פעולות, ושאפשר לבצע את המסלול בכל עת ללא קנס. האם תרצה שאעביר אותך?", mmss(99.12), "banker")
try_validate("3-turn same-speaker span", build(block))

# ---------- (e) unquotable turns (normalized < 8 chars) ----------
print("\n=== (e) turns whose whole text normalizes below MIN_QUOTE_CHARS=8 ===")
short = []
for i, t in enumerate(red.turns):
    n = normalize_for_match(t.text)
    if len(n) < MIN_QUOTE_CHARS:
        short.append((i, t.speaker, mmss(t.start), t.text, len(n)))
print(f"{len(short)} of {len(red.turns)} turns unquotable as single turns:")
for i, sp, ts, txt, ln in short:
    print(f"  turn {i:2d} [{ts}] {sp:8s} norm_len={ln} text={txt!r}")

# same at block level
blocks = _same_speaker_blocks(red)
short_blocks = [(i, sp, txt) for i, (sp, txt) in enumerate(blocks)
                if len(normalize_for_match(txt)) < MIN_QUOTE_CHARS]
print(f"\n{len(short_blocks)} of {len(blocks)} same-speaker blocks unquotable:")
for i, sp, txt in short_blocks:
    print(f"  block {i:2d} {sp:8s} text={txt!r}")

# ---------- (e2) masked tokens inside quotes ----------
print("\n=== (e2) mask tokens in quotes ===")
m = dict(verbatim)
m["identification"] = ("מצוין, הזיהוי הושלם. אני רואה את החשבון שמסתיים ב -<חשבון:████> בסניף <חשבון:████>.", mmss(48.01), "banker")
try_validate("quote containing <חשבון:████> masks, verbatim", build(m))
m["identification"] = ("<טלפון:████>.", mmss(46.29), "customer")
print("  normalized mask-only turn:", repr(normalize_for_match("<טלפון:████>.")), "len:", len(normalize_for_match("<טלפון:████>.")))
try_validate("customer mask-only turn as quote", build(m))
# judge writes the quote but paraphrases the mask away
m["identification"] = ("מצוין, הזיהוי הושלם. אני רואה את החשבון שמסתיים ב בסניף", mmss(48.01), "banker")
try_validate("mask tokens omitted by judge (near-quote)", build(m))
# judge fills in a plausible number instead of the mask
m["identification"] = ("מצוין, הזיהוי הושלם. אני רואה את החשבון שמסתיים ב-9265 בסניף 812.", mmss(48.01), "banker")
try_validate("mask replaced with digits by judge", build(m))
