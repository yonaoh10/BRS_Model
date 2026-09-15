"""Prototype of the proposed design, measured on REAL002. NOT a fix - lives outside the repo.

Pass B (fold_number_words) sits between normalize_for_detection and _spans_in_normalized,
exactly as the design proposes. Classification gains one context-gated branch (gap B)
and a repeated-fragment rule (gap A tail).
"""
import json, re, sys, time
sys.path.insert(0, "/Users/yonatanohayon/Desktop/BRS_Model-main/src")

from callqa import redaction as R

# --- Pass B: Hebrew digit-word folding --------------------------------------
DIGIT_WORDS = {
    "אפס": "0",
    "אחת": "1", "אחד": "1",
    "שתיים": "2", "שניים": "2", "שתי": "2", "שני": "2",
    "שלוש": "3", "שלושה": "3",
    "ארבע": "4", "ארבעה": "4",
    "חמש": "5", "חמישה": "5",
    "שש": "6", "שישה": "6",
    "שבע": "7", "שבעה": "7",
    "שמונה": "8",
    "תשע": "9", "תשעה": "9",
}
MIN_SEQ = 4  # >=4 consecutive digit-words to fold

WORD_RE = re.compile(r"[֐-׿]+|\S+")

def _word_digit(token: str) -> str | None:
    core = token.strip(".,?!:;\"'")
    if core in DIGIT_WORDS:
        return DIGIT_WORDS[core]
    # conjunction prefix ו- ("וחמש"); NOT ה- (הארבע is not dictation)
    if len(core) > 1 and core[0] == "ו" and core[1:] in DIGIT_WORDS:
        return DIGIT_WORDS[core[1:]]
    return None

def fold_number_words(text: str):
    """Return (folded_text, spans) where spans[i] = (orig_start, orig_end) per output char."""
    out = []
    spans = []
    tokens = [(m.group(), m.start(), m.end()) for m in WORD_RE.finditer(text)]
    # find maximal runs of digit-words separated only by whitespace
    i = 0
    cursor = 0
    n = len(tokens)
    while i < n:
        d = _word_digit(tokens[i][0])
        if d is None:
            i += 1
            continue
        j = i
        digits = []
        while j < n:
            dj = _word_digit(tokens[j][0])
            if dj is None:
                break
            # tokens must be separated by whitespace only (they are, by tokenization,
            # unless punctuation token intervened -- WORD_RE keeps punctuation attached)
            digits.append(dj)
            j += 1
            # stop the run if the token carried terminal punctuation ("אפס.")
            if tokens[j-1][0] != tokens[j-1][0].strip(".,?!:;"):
                break
        if len(digits) >= MIN_SEQ:
            run_start, run_end = tokens[i][1], tokens[j-1][2]
            # pass-through up to run
            for k in range(cursor, run_start):
                out.append(text[k]); spans.append((k, k+1))
            # contiguous digits, each mapping to the word run's full span share
            out.append("".join(digits))
            first = True
            for _ in digits:
                # every folded char maps to the whole run; start uses first char, end last
                spans.append((run_start, run_end))
                first = False
            cursor = run_end
            i = j
        else:
            i = j if j > i else i + 1
    for k in range(cursor, len(text)):
        out.append(text[k]); spans.append((k, k+1))
    return "".join(out), spans

# fix spans list: out list mixes single chars and digit-chunks; flatten properly
def fold_number_words2(text: str):
    tokens = [(m.group(), m.start(), m.end()) for m in WORD_RE.finditer(text)]
    folded_chars = []
    spans = []
    i = 0
    cursor = 0
    n = len(tokens)
    def passthrough(upto):
        nonlocal cursor
        for k in range(cursor, upto):
            folded_chars.append(text[k]); spans.append((k, k+1))
        cursor = upto
    while i < n:
        if _word_digit(tokens[i][0]) is None:
            i += 1
            continue
        j = i
        digits = []
        while j < n and _word_digit(tokens[j][0]) is not None:
            digits.append(_word_digit(tokens[j][0]))
            broke = tokens[j][0] != tokens[j][0].rstrip(".,?!:;")
            j += 1
            if broke:
                break
        if len(digits) >= MIN_SEQ:
            run_start, run_end = tokens[i][1], tokens[j-1][2]
            passthrough(run_start)
            for d in digits:
                folded_chars.append(d); spans.append((run_start, run_end))
            cursor = run_end
        i = j
    passthrough(len(text))
    return "".join(folded_chars), spans

# --- classification tweak (gap B) --------------------------------------------
ORIG_CLASSIFY = R._classify_run
def classify_v2(text, start, end):
    got = ORIG_CLASSIFY(text, start, end)
    if got:
        return got
    raw = text[start:end]
    digits = re.sub(r"\D", "", raw)
    before = R._context_before(text, start)
    id_hint = any(w in before for w in R.ID_CONTEXT)
    acct_hint = any(w in before for w in R.ACCOUNT_CONTEXT)
    if R._looks_like_amount(text, end):
        return None
    # NEW BRANCH: dictation context lifts the structural_only requirement
    if 6 <= len(digits) <= 20 and (id_hint or acct_hint):
        return "ISRAELI_ID" if id_hint else "ACCOUNT_LIKE"
    return None

def find_pii_v2(text):
    norm1, map1 = R.normalize_for_detection(text)
    norm2, map2 = fold_number_words2(norm1)
    old = R._classify_run
    R._classify_run = classify_v2
    # patch module-level reference used inside _spans_in_normalized
    spans = R._spans_in_normalized.__wrapped__(norm2) if hasattr(R._spans_in_normalized, "__wrapped__") else None
    R._classify_run = classify_v2
    try:
        spans = R._spans_in_normalized(norm2)
    finally:
        R._classify_run = old
    # repeated-fragment rule: a 4-5 digit run equal to a substring of a masked >=6 run
    masked_digits = [re.sub(r"\D", "", norm2[s.start:s.end]) for s in spans]
    extra = []
    for m in R.DIGIT_RUN_RE.finditer(norm2):
        ms, me = m.start(), m.end()
        if any(ms < s.end and s.start < me for s in spans):
            continue
        d = re.sub(r"\D", "", norm2[ms:me])
        if 4 <= len(d) <= 5 and any(d in md for md in masked_digits if len(md) >= 6):
            extra.append(R.PIIMatch("ACCOUNT_LIKE", ms, me))
    spans = sorted(spans + extra, key=lambda s: s.start)
    out = []
    for s in spans:
        if s.start >= len(map2) or s.end - 1 >= len(map2):
            continue
        s1 = map2[s.start][0]
        e1 = map2[s.end - 1][1]
        out.append(R.PIIMatch(s.entity_type, map1[s1], map1[e1 - 1] + 1))
    return out

# --- run on REAL002 -----------------------------------------------------------
dialog = json.load(open(
    "/Users/yonatanohayon/Desktop/BRS_Model-main/data/output/transcripts/REAL002.dialog.json"))
texts = [t["text"] for t in dialog["turns"]]
document = R.TURN_SEPARATOR.join(texts)

print("== baseline masks ==")
base = R.find_pii(document)
for p in base:
    print("  ", p.entity_type, repr(document[p.start:p.end]))

print("== v2 masks ==")
v2 = find_pii_v2(document)
for p in v2:
    print("  ", p.entity_type, repr(document[p.start:p.end]))

base_set = {(p.start, p.end) for p in base}
added = [p for p in v2 if (p.start, p.end) not in base_set]
print("\nadded spans: %d" % len(added))
tok_added = 0
for p in added:
    seg = document[p.start:p.end]
    tok_added += len(seg.split())
    print("   +", p.entity_type, repr(seg))
print("additional original tokens masked:", tok_added)
total_tokens = len(document.split())
print("document tokens:", total_tokens)

# check false-positive sentinels stayed unmasked
for sentinel in ["רגע אחד", "ארבע הספרות", "30 שקלים", "10 שקלים", "120 ,000 שקל", "3 -4"]:
    hit = any(document[p.start:p.end].find(sentinel) >= 0 for p in v2)
    covered = False
    idx = document.find(sentinel)
    while idx >= 0:
        if any(p.start <= idx < p.end for p in v2):
            covered = True
        idx = document.find(sentinel, idx + 1)
    print("sentinel %-16r masked=%s" % (sentinel, covered))

# --- latency ------------------------------------------------------------------
N = 200
t0 = time.perf_counter()
for _ in range(N):
    R.find_pii(document)
t_base = (time.perf_counter() - t0) / N * 1000
t0 = time.perf_counter()
for _ in range(N):
    find_pii_v2(document)
t_v2 = (time.perf_counter() - t0) / N * 1000
t0 = time.perf_counter()
for _ in range(N):
    fold_number_words2(document)
t_fold = (time.perf_counter() - t0) / N * 1000
print("\nlatency per whole-dialog find_pii (51 turns, %d chars):" % len(document))
print("  baseline : %.3f ms" % t_base)
print("  v2 total : %.3f ms" % t_v2)
print("  fold pass: %.3f ms" % t_fold)
