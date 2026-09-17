"""Component metrics for the evaluation harness — stdlib only.

WER/CER use a plain Levenshtein (adding `jiwer` for ~30 lines is not justified
air-gapped). Redaction precision/recall run the real `redaction.find_pii` over
the reference and score its spans against hand-labelled gold identifiers, so the
recall number is the one that matters for a leak.
"""

from __future__ import annotations


def _levenshtein(a: list, b: list) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def wer(reference: str, hypothesis: str) -> float:
    """Word error rate: edit distance over whitespace tokens / reference length."""
    ref, hyp = reference.split(), hypothesis.split()
    if not ref:
        return 0.0 if not hyp else 1.0
    return round(_levenshtein(ref, hyp) / len(ref), 4)


def cer(reference: str, hypothesis: str) -> float:
    """Character error rate: edit distance over characters (spaces collapsed)."""
    ref = list("".join(reference.split()))
    hyp = list("".join(hypothesis.split()))
    if not ref:
        return 0.0 if not hyp else 1.0
    return round(_levenshtein(ref, hyp) / len(ref), 4)


def _overlap(a: tuple[int, int], b: tuple[int, int]) -> bool:
    return a[0] < b[1] and b[0] < a[1]


def redaction_prf(reference_text: str, gold_literals: list[str]) -> dict:
    """Precision/recall/F1 of `find_pii` on `reference_text` vs gold identifiers.

    Gold spans are located by string search for the KNOWN seeded identifiers,
    independent of find_pii — otherwise recall would be 100% by construction.
    A gold span counts as recalled if any detected span overlaps it; a detected
    span counts as precise if it overlaps some gold span.
    """
    from callqa.redaction import find_pii

    detected = [(m.start, m.end) for m in find_pii(reference_text)]
    gold: list[tuple[int, int]] = []
    for lit in gold_literals:
        idx = reference_text.find(lit)
        if idx >= 0:
            gold.append((idx, idx + len(lit)))

    recalled = sum(1 for g in gold if any(_overlap(g, d) for d in detected))
    precise = sum(1 for d in detected if any(_overlap(g, d) for g in gold))
    recall = round(recalled / len(gold), 4) if gold else 1.0
    precision = round(precise / len(detected), 4) if detected else 1.0
    denom = precision + recall
    f1 = round(2 * precision * recall / denom, 4) if denom else 0.0
    return {"recall": recall, "precision": precision, "f1": f1,
            "gold": len(gold), "detected": len(detected)}


def role_accuracy(reference_turns: list[dict], produced_turns: list[dict]) -> float:
    """Fraction of turns whose speaker matches, aligned by order. A coarse but
    sufficient speaker-separation check on the synthetic set; a real golden set
    with RTTM references uses the full DER scorer in scripts/eval_diarization.py."""
    n = min(len(reference_turns), len(produced_turns))
    if n == 0:
        return 1.0 if not reference_turns and not produced_turns else 0.0
    matches = sum(1 for i in range(n)
                  if reference_turns[i].get("speaker") == produced_turns[i].get("speaker"))
    # penalise a length mismatch so a truncated/expanded dialog cannot score 1.0
    return round(matches / max(len(reference_turns), len(produced_turns)), 4)
