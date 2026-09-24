"""Answers by the content of the question.

The key is the engine, the model, the task version and the full prompt text:
a changed task version invalidates that task only, and a new export of the
same calls costs nothing. Answers are the model's raw JSON, re-parsed and
re-verified on every use, so a stricter verifier also applies to old answers.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from callqa.journey.llm.tasks import Prompt
from callqa.state import atomic_write_text


class AnswerCache:
    def __init__(self, folder: Path) -> None:
        self.folder = folder
        self.hits = 0
        self.misses = 0

    def key(self, engine: str, model: str, prompt: Prompt, feedback: str = "") -> str:
        raw = json.dumps([engine, model, prompt.task, prompt.version, prompt.system,
                          prompt.user, feedback], ensure_ascii=False)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def _path(self, prompt: Prompt, key: str) -> Path:
        return self.folder / prompt.task / f"{key[:40]}.json"

    def get(self, prompt: Prompt, key: str) -> str | None:
        path = self._path(prompt, key)
        try:
            text = json.loads(path.read_text(encoding="utf-8"))["answer"]
        except (OSError, json.JSONDecodeError, KeyError, TypeError):
            self.misses += 1
            return None
        self.hits += 1
        return text

    def put(self, prompt: Prompt, key: str, answer: str) -> None:
        atomic_write_text(self._path(prompt, key),
                          json.dumps({"task": prompt.task, "version": prompt.version,
                                      "answer": answer}, ensure_ascii=False))
