"""How right is the reading? Human labels against the content layer.

    journey label-sample   a stratified sample of returns with content, as an
                           offline HTML form. The form is blind: it shows the
                           contact's text and the story around it, never the
                           model's answer. It saves labels as CSV.
    journey eval           the labels against content.json: accuracy, macro-F1
                           and Cohen's kappa on the return category, the same
                           on failure / not failure, agreement on "told it
                           again", a confusion matrix, and 95% intervals by
                           story (returns of one customer are not independent).
                           With a baseline, a drop beyond the tolerance fails.

Labels are keyed by story number and contact number, as the report shows
them: `story_no,contact_no,category[,retold][,break_point]`.
"""

from __future__ import annotations

import csv
import json
import random
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path

from callqa.journey.analysis import PENDING, shown_category
from callqa.journey.models import ContentLayer, JourneyDataset
from callqa.journey.stats import cluster_bootstrap_share
from callqa.journey.timeline import build_timelines
from callqa.journey.vocab import Taxonomy

TOLERANCE = 0.02


@dataclass
class Label:
    story_no: int
    contact_no: int
    category: str
    retold: str = ""
    break_point: str = ""


def read_labels(path: Path, taxonomy: Taxonomy) -> list[Label]:
    text = path.read_text(encoding="utf-8-sig")
    rows = list(csv.DictReader(text.splitlines()))
    out, bad = [], []
    for n, r in enumerate(rows, start=2):
        try:
            label = Label(int(r["story_no"]), int(r["contact_no"]), (r.get("category") or "").strip(),
                          (r.get("retold") or "").strip(), (r.get("break_point") or "").strip())
        except (KeyError, ValueError):
            bad.append(n)
            continue
        if label.category and label.category not in taxonomy.categories:
            bad.append(n)
            continue
        out.append(label)
    if bad:
        raise ValueError(f"{path.name}: unreadable rows {bad[:10]} (need story_no, contact_no, "
                         f"category from {', '.join(taxonomy.categories)})")
    return out


# -- metrics ---------------------------------------------------------------------------------

def cohen_kappa(pairs: list[tuple[str, str]]) -> float | None:
    n = len(pairs)
    if not n:
        return None
    po = sum(a == b for a, b in pairs) / n
    ca, cb = Counter(a for a, _ in pairs), Counter(b for _, b in pairs)
    pe = sum(ca[k] * cb.get(k, 0) for k in ca) / (n * n)
    return None if pe == 1 else (po - pe) / (1 - pe)


def macro_f1(pairs: list[tuple[str, str]]) -> float | None:
    """pairs: (truth, predicted); over the classes that occur in the truth."""
    classes = sorted({t for t, _ in pairs})
    if not classes:
        return None
    f1s = []
    for c in classes:
        tp = sum(1 for t, p in pairs if t == c and p == c)
        fp = sum(1 for t, p in pairs if t != c and p == c)
        fn = sum(1 for t, p in pairs if t == c and p != c)
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(2 * prec * rec / (prec + rec) if prec + rec else 0.0)
    return sum(f1s) / len(f1s)


@dataclass
class Agreement:
    n: int
    accuracy: float | None
    low: float | None
    high: float | None
    kappa: float | None
    macro_f1: float | None


def agreement(items: list[tuple[int, str, str]]) -> Agreement:
    """items: (story_no, truth, predicted)."""
    pairs = [(t, p) for _s, t, p in items]
    by_story: dict[int, list[int]] = defaultdict(lambda: [0, 0])
    for s, t, p in items:
        by_story[s][0] += int(t == p)
        by_story[s][1] += 1
    share = cluster_bootstrap_share([tuple(v) for v in by_story.values()])
    return Agreement(len(items), share.p, share.low, share.high, cohen_kappa(pairs),
                     macro_f1(pairs))


@dataclass
class EvalResult:
    labels: int
    matched: int
    pending: int
    category: Agreement
    failure: Agreement
    retold: Agreement | None
    confusion: dict[str, dict[str, int]] = field(default_factory=dict)
    engine: str = ""
    model: str = ""

    def to_json(self) -> dict:
        return asdict(self)

    def lines(self) -> list[str]:
        def a(x: Agreement | None, name: str) -> str:
            if x is None or x.accuracy is None:
                return f"{name}: no labels"
            ci = f" (95%: {x.low:.2f}-{x.high:.2f})" if x.low is not None else ""
            k = f", kappa {x.kappa:.2f}" if x.kappa is not None else ""
            f = f", macro-F1 {x.macro_f1:.2f}" if x.macro_f1 is not None else ""
            return f"{name}: accuracy {x.accuracy:.2f}{ci}{k}{f} on {x.n}"
        out = [f"labels {self.labels}, matched {self.matched}, still pending {self.pending} "
               f"(engine {self.engine} / {self.model})",
               a(self.category, "category"), a(self.failure, "failure yes/no"),
               a(self.retold, "told it again")]
        cats = sorted(self.confusion)
        if cats:
            out.append("confusion (rows = labels, columns = model):")
            out.append("  " + " ".join(f"{c[:10]:>10}" for c in [""] + cats))
            for t in cats:
                out.append("  " + f"{t[:10]:>10} " + " ".join(
                    f"{self.confusion[t].get(p, 0):>10}" for p in cats))
        return out


def evaluate(dataset: JourneyDataset, content: ContentLayer, labels: list[Label],
             taxonomy: Taxonomy) -> EvalResult:
    by_pos: dict[tuple[int, int], str] = {}
    for tl in build_timelines(dataset):
        for c in tl.contacts:
            by_pos[(tl.story.story_no, c.index + 1)] = c.interaction.interaction_id
    failure = taxonomy.failure_categories
    cat_items, fail_items, retold_items = [], [], []
    confusion: dict[str, Counter] = defaultdict(Counter)
    pending = matched = 0
    for lab in labels:
        iid = by_pos.get((lab.story_no, lab.contact_no))
        if iid is None:
            continue
        j = content.judgements.get(iid)
        if lab.category:
            if j is None or shown_category(j) == PENDING:
                pending += 1
            else:
                matched += 1
                pred = j.category
                cat_items.append((lab.story_no, lab.category, pred))
                fail_items.append((lab.story_no, str(lab.category in failure), str(pred in failure)))
                confusion[lab.category][pred] += 1
        card = content.cards.get(iid)
        if lab.retold and card is not None:
            yes = {"yes", "partial"}
            retold_items.append((lab.story_no, str(lab.retold in yes), str(card.retold in yes)))
    return EvalResult(labels=len(labels), matched=matched, pending=pending,
                      category=agreement(cat_items), failure=agreement(fail_items),
                      retold=agreement(retold_items) if retold_items else None,
                      confusion={k: dict(v) for k, v in confusion.items()},
                      engine=content.engine, model=content.model)


def compare_baseline(result: EvalResult, baseline: dict, tolerance: float = TOLERANCE
                     ) -> list[str]:
    """Regressions against a stored result (the same keys as to_json)."""
    problems = []
    for part in ("category", "failure", "retold"):
        now = (result.to_json().get(part) or {})
        was = baseline.get(part) or {}
        for key in ("accuracy", "kappa"):
            if was.get(key) is not None and now.get(key) is not None \
                    and now[key] < was[key] - tolerance:
                problems.append(f"{part} {key} fell from {was[key]:.3f} to {now[key]:.3f}")
    return problems


# -- the labelling form ------------------------------------------------------------------------

def sample_returns(dataset: JourneyDataset, content: ContentLayer, n: int = 60,
                   seed: int = 7) -> list[tuple[int, int, str]]:
    """(story_no, contact_no, interaction_id) of up to n returns with content,
    stratified by the model's category so rare ones are represented."""
    strata: dict[str, list[tuple[int, int, str]]] = defaultdict(list)
    for tl in build_timelines(dataset):
        for c in tl.returns:
            j = content.judgements.get(c.interaction.interaction_id)
            if j is not None and j.objective_class == "content" and j.decided_by != "none":
                strata[j.category].append((tl.story.story_no, c.index + 1,
                                           c.interaction.interaction_id))
    rng = random.Random(seed)
    for items in strata.values():
        rng.shuffle(items)
    picked: list[tuple[int, int, str]] = []
    keys = sorted(strata)
    while len(picked) < n and any(strata[k] for k in keys):
        for k in keys:
            if strata[k] and len(picked) < n:
                picked.append(strata[k].pop())
    return sorted(picked)


def write_label_form(path: Path, dataset: JourneyDataset, content: ContentLayer,
                     taxonomy: Taxonomy, sample: list[tuple[int, int, str]],
                     views: dict[str, list[dict]]) -> Path:
    """The blind form: text and context only; labels saved by the browser."""
    from markupsafe import Markup

    from callqa.reporting.common import jinja_env
    from callqa.reporting.executive.render import script_json
    from callqa.state import atomic_write_text

    timelines = {tl.story.story_no: tl for tl in build_timelines(dataset)}
    items = []
    for story_no, contact_no, iid in sample:
        tl = timelines[story_no]
        context = [{"no": c.index + 1, "when": c.at.strftime("%d.%m %H:%M"),
                    "kind": c.kind, "direction": c.direction,
                    "here": c.index + 1 == contact_no} for c in tl.contacts]
        prev = tl.contacts[contact_no - 2].interaction.interaction_id if contact_no > 1 else None
        items.append({"story_no": story_no, "contact_no": contact_no, "context": context,
                      "lines": views.get(iid, []), "prev_lines": views.get(prev, []) if prev else []})
    cats = [{"id": k, "label": v.get("label", k), "definition": v.get("definition", "")}
            for k, v in taxonomy.categories.items() if k != "bank_initiated"]
    import hashlib
    sample_key = hashlib.sha256(json.dumps(sample).encode("utf-8")).hexdigest()[:12]
    html = jinja_env().get_template("journey_labels.html.j2").render(
        items=items, categories=cats, dataset_id=dataset.dataset_id, sample_key=sample_key,
        data_json=Markup(script_json({"n": len(items)})))
    atomic_write_text(path, html)
    return path


def save_eval(path: Path, result: EvalResult) -> None:
    from callqa.state import atomic_write_text
    atomic_write_text(path, json.dumps(result.to_json(), ensure_ascii=False, indent=2))
