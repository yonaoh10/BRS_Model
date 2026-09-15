import sys, json
sys.path.insert(0, "/Users/yonatanohayon/Desktop/BRS_Model-main/src")
from callqa.rubric import load_rubric, weighted_total, GATE_CAP
from callqa.models import DimensionScore

rubric = load_rubric("/Users/yonatanohayon/Desktop/BRS_Model-main/config/rubric.yaml")
print("weights:", {d.id: d.weight for d in rubric.dimensions})
print("gate ids:", rubric.gate_ids)

def mk(scores):
    return {k: DimensionScore(score=v, reasoning_he="x") for k, v in scores.items()}

ids = [d.id for d in rubric.dimensions]

for label, s in [
    ("all 3s (average)", {i: 3 for i in ids}),
    ("all 4s (good)", {i: 4 for i in ids}),
    ("all 5s (excellent)", {i: 5 for i in ids}),
    ("all 5s but identification=2 (gate)", {**{i: 5 for i in ids}, "identification": 2}),
    ("all 5s but compliance=2 (gate)", {**{i: 5 for i in ids}, "compliance": 2}),
    ("all 3s but identification=2", {**{i: 3 for i in ids}, "identification": 2}),
    ("gate=3 boundary (all 5s, identification=3)", {**{i: 5 for i in ids}, "identification": 3}),
    ("all 2s", {i: 2 for i in ids}),
    ("all 1s", {i: 1 for i in ids}),
]:
    total, gf, fg = weighted_total(rubric, mk(s))
    print(f"{label:48s} total={total:6.1f} gate_failed={gf} failed={fg}")

# REAL002 replication
real = json.load(open("/Users/yonatanohayon/Desktop/BRS_Model-main/data/output/scores/REAL002.json"))
scores = {k: v["score"] for k, v in real["scores"].items()}
print("\nREAL002 scores:", scores)
total, gf, fg = weighted_total(rubric, mk(scores))
print("recomputed:", total, gf, fg, "| artifact:", real["weighted_total"], real["gate_failed"])
# hand arithmetic
acc = 0.0
for d in rubric.dimensions:
    c = d.weight * (scores[d.id] - 1) / 4.0 * 100.0
    acc += c
    print(f"  {d.id:15s} w={d.weight:.2f} score={scores[d.id]} -> {c:6.2f}")
print("  sum =", acc)
