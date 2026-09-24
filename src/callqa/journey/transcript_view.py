"""What the language model reads: a contact's text as numbered lines.

Every line has a number, so the model can point at a line instead of making up
a quote, and the verifier can check the quote against that exact line.

A line is marked ⚠ when its words are uncertain: low recognition confidence,
a turn that crosses the join between two recorded parts, or roles that were
guessed with little confidence. The model is told never to quote a ⚠ line, and
the verifier refuses to accept one.

The cpu profile shortens a long conversation: the opening, the closing, and a
few lines around every line that holds a cue phrase (journey_lexicon_he.yaml).
What was left out is said in place ("[... 14 שורות הושמטו ...]").
"""

from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

from callqa.journey.models import Message
from callqa.models import RedactedTranscript
from callqa.resources import find_config, load_yaml

WHO_HE = {"bank": "בנק", "customer": "לקוח", "unknown": "?"}
_NIQQUD_RE = re.compile(r"[֑-ׇ]")
_PUNCT_RE = re.compile(r"[^\w\s]", re.UNICODE)


def fold(text: str) -> str:
    """Lower-noise form for cue matching: no niqqud, no punctuation, one space."""
    text = _NIQQUD_RE.sub("", unicodedata.normalize("NFC", text or ""))
    return " ".join(_PUNCT_RE.sub(" ", text).split())


@dataclass(frozen=True)
class Lexicon:
    cues: dict[str, list[str]]
    topics: dict[str, list[str]]

    def hits(self, text: str) -> set[str]:
        t = fold(text)
        return {group for group, phrases in self.cues.items()
                if any(fold(p) and fold(p) in t for p in phrases)}

    def topic_votes(self, text: str) -> dict[str, int]:
        t = fold(text)
        votes = {}
        for topic, words in self.topics.items():
            n = sum(t.count(fold(w)) for w in words if fold(w))
            if n:
                votes[topic] = n
        return votes


def load_lexicon(name: str = "journey_lexicon_he.yaml") -> Lexicon:
    data = load_yaml(find_config(name)) or {}
    cues = {str(k): [str(p) for p in (v or [])] for k, v in (data.get("cues") or {}).items()}
    topics = {str(k): [str(p) for p in (v or [])] for k, v in (data.get("topics") or {}).items()}
    return Lexicon(cues=cues, topics=topics)


@dataclass
class Line:
    no: int                      # 1-based, as the model sees it
    who: str                     # bank / customer / unknown
    text: str
    uncertain: bool = False
    part: int | None = None      # recorded part (1-based) of a multi-file call


@dataclass
class ContentView:
    interaction_id: str
    kind: str                    # call / message
    lines: list[Line]
    shown: list[int] = field(default_factory=list)   # line numbers kept after compression

    def line(self, no: int) -> Line | None:
        if 1 <= no <= len(self.lines):
            return self.lines[no - 1]
        return None

    def render(self) -> str:
        keep = set(self.shown) if self.shown else {ln.no for ln in self.lines}
        out: list[str] = []
        skipped = 0
        last_part = None
        for ln in self.lines:
            if ln.no not in keep:
                skipped += 1
                continue
            if skipped:
                out.append(f"[... {skipped} שורות הושמטו ...]")
                skipped = 0
            if ln.part is not None and ln.part != last_part and last_part is not None:
                out.append(f"— קטע הקלטה {ln.part} —")
            last_part = ln.part
            mark = " ⚠" if ln.uncertain else ""
            out.append(f"L{ln.no} [{WHO_HE.get(ln.who, '?')}]{mark} {ln.text}")
        if skipped:
            out.append(f"[... {skipped} שורות הושמטו ...]")
        return "\n".join(out)

    @property
    def chars(self) -> int:
        return len(self.render())


# -- building views -----------------------------------------------------------------

def _segment_bounds(segmap_path: Path | None) -> list[tuple[float, float]]:
    """(start, end) of each recorded part on the assembled call's clock."""
    if segmap_path is None or not segmap_path.exists():
        return []
    try:
        data = json.loads(segmap_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    out = []
    for s in data.get("segments") or []:
        try:
            out.append((float(s["start_in_call"]), float(s["end_in_call"])))
        except (KeyError, TypeError, ValueError):
            continue
    return out


def _uncertain_turns(dialog_path: Path | None, threshold: float, n_turns: int) -> set[int]:
    """Indices of turns whose words were recognised with low confidence. Only
    the probabilities are read from the raw dialog - never its text - and only
    when it lines up turn for turn with the redacted transcript."""
    if dialog_path is None or not dialog_path.exists():
        return set()
    try:
        data = json.loads(dialog_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return set()
    turns = data.get("turns") or []
    if len(turns) != n_turns:
        return set()
    out = set()
    for i, t in enumerate(turns):
        probs = [w.get("probability") for w in t.get("words") or []
                 if isinstance(w.get("probability"), (int, float))]
        if probs and sum(1 for p in probs if p < threshold) / len(probs) > 0.3:
            out.add(i)
    return out


def call_view(interaction_id: str, transcript: RedactedTranscript, *,
              segmap_path: Path | None = None, dialog_path: Path | None = None,
              role_confidence: float | None = None, uncertain_prob: float = 0.5) -> ContentView:
    bounds = _segment_bounds(segmap_path)
    low_conf = _uncertain_turns(dialog_path, uncertain_prob, len(transcript.turns))
    weak_roles = role_confidence is not None and role_confidence < 0.3
    lines = []
    for i, t in enumerate(transcript.turns):
        part = None
        crosses = False
        for n, (a, b) in enumerate(bounds, start=1):
            if a <= t.start < b:
                part = n
            if t.start < b < t.end and n < len(bounds):
                crosses = True
        who = {"banker": "bank", "customer": "customer"}.get(t.speaker, "unknown")
        lines.append(Line(no=i + 1, who=who, text=" ".join(t.text.split()),
                          uncertain=(i in low_conf) or crosses or weak_roles, part=part))
    return ContentView(interaction_id=interaction_id, kind="call", lines=lines)


def message_view(interaction_id: str, messages: list[Message]) -> ContentView:
    lines = []
    for m in sorted(messages, key=lambda m: m.at):
        who = {"inbound": "customer", "outbound": "bank"}.get(m.direction, "unknown")
        text = " ".join(f"{m.subject}: {m.body}".split()) if m.subject else " ".join(m.body.split())
        lines.append(Line(no=len(lines) + 1, who=who, text=text))
    return ContentView(interaction_id=interaction_id, kind="message", lines=lines)


def compress(view: ContentView, lexicon: Lexicon, max_chars: int, *, head: int = 6,
             tail: int = 6, around: int = 2) -> ContentView:
    """Keep the opening, the closing and the lines around cue phrases, within
    `max_chars`. A view that already fits is returned whole. Linear in the
    number of lines: each line's rendered length is computed once, and cue
    windows are added in order while they fit."""
    lengths = [len(f"L{ln.no} [?] ⚠ {ln.text}") + 1 for ln in view.lines]
    if sum(lengths) <= max_chars or len(view.lines) <= head + tail:
        return view
    n = len(view.lines)
    gap_cost = 32                       # "[... N שורות הושמטו ...]" and a part marker
    keep: set[int] = set()
    used = 0

    def add(nos: list[int]) -> bool:
        nonlocal used
        new = [x for x in nos if x not in keep]
        cost = sum(lengths[x - 1] for x in new) + gap_cost
        if keep and used + cost > max_chars:
            return False
        keep.update(new)
        used += cost
        return True

    add(list(range(1, min(head, n) + 1)) + list(range(max(1, n - tail + 1), n + 1)))
    for ln in view.lines:
        if head < ln.no <= n - tail and lexicon.hits(ln.text):
            window = list(range(max(1, ln.no - around), min(n, ln.no + around) + 1))
            if not add(window):
                break
    return ContentView(view.interaction_id, view.kind, view.lines, sorted(keep))
