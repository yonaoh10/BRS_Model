"""Read-only trace of why the cross-turn fragments 415 / 926 -9265 / -9265 escape."""
import json, re, sys
sys.path.insert(0, "/Users/yonatanohayon/Desktop/BRS_Model-main/src")

from callqa.redaction import (
    DIGIT_RUN_RE, TURN_SEPARATOR, _classify_run, _context_before,
    normalize_for_detection, find_pii, is_valid_israeli_id, is_valid_luhn,
    is_israeli_phone, STRUCTURAL_SEPARATORS,
)

dialog = json.load(open(
    "/Users/yonatanohayon/Desktop/BRS_Model-main/data/output/transcripts/REAL002.dialog.json"))
texts = [t["text"] for t in dialog["turns"]]
print("n_turns =", len(texts))
document = TURN_SEPARATOR.join(texts)
normalized, index = normalize_for_detection(document)
assert normalized == document, "no invisible chars in this call" if normalized != document else ""

for m in DIGIT_RUN_RE.finditer(normalized):
    raw = normalized[m.start():m.end()]
    digits = re.sub(r"\D", "", raw)
    gaps = [g for g in re.split(r"\d+", raw) if g]
    structural_only = all(any(ch in STRUCTURAL_SEPARATORS for ch in g) for g in gaps)
    cls = _classify_run(normalized, m.start(), m.end())
    before = _context_before(normalized, m.start())
    print("-" * 60)
    print("RUN  %r" % raw)
    print("  digits=%s len=%d structural_only=%s" % (digits, len(digits), structural_only))
    print("  gaps=%r" % gaps)
    print("  id_ok=%s luhn=%s phone=%s" % (
        is_valid_israeli_id(digits), is_valid_luhn(digits), is_israeli_phone(digits)))
    print("  context_before=%r" % before[-40:])
    print("  CLASSIFIED AS: %s" % cls)

print("=" * 60)
print("find_pii matches on the whole document:")
for p in find_pii(document):
    print(" ", p.entity_type, repr(document[p.start:p.end]))
