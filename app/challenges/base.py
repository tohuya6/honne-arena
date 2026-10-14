"""The challenge interface. Each kind of challenge is one subclass.

A challenge row (from the `challenges` table) carries a `kind` and a JSON `config`.
`app.challenges.load()` picks the subclass by kind and hands it the row. Subclasses
only decide two things: how to normalize an answer, and how to judge it.

There are two modes:
- "solve": an answer is right or wrong (a CTF flag). Points come from solve counts.
- "score": an answer gets a number and the best one wins (steering). Points come from rank.
"""

from __future__ import annotations

import json
import re
from typing import ClassVar, NamedTuple


class Outcome(NamedTuple):
    correct: bool | None   # set by "solve" challenges, None for "score" ones
    score: float | None    # set by "score" challenges, None for "solve" ones
    detail: dict


class Challenge:
    kind: ClassVar[str] = ""
    mode: ClassVar[str] = "solve"
    # True when judging an answer makes a remote model call. Those submissions go
    # through the NDIF budget and the bounded queue.
    uses_ndif: ClassVar[bool] = False

    def __init__(self, row: dict):
        self.row = row
        cfg = row.get("config") or {}
        self.config: dict = json.loads(cfg) if isinstance(cfg, str) else dict(cfg)
        self.validate_config()

    @property
    def id(self) -> int:
        return self.row["id"]

    @property
    def slug(self) -> str:
        return self.row["slug"]

    @property
    def higher_is_better(self) -> bool:
        """For "score" challenges, which end of the board wins."""
        return True

    def validate_config(self) -> None:
        """Raise ValueError when the config can't work. Called on load and on create."""

    def normalize(self, answer: str) -> str:
        return answer.strip()

    def answer_key(self, answer: str) -> str:
        """Dedup key. Two answers with the same key are the same answer."""
        return re.sub(r"\s+", " ", self.normalize(answer))

    def evaluate(self, answer: str) -> Outcome:
        raise NotImplementedError

    def public_config(self) -> dict:
        """Config fields safe to show players. Never include anything that gives away
        the answer."""
        return {}

    def public(self) -> dict:
        return {
            "slug": self.slug,
            "title": self.row["title"],
            "kind": self.kind,
            "mode": self.mode,
            "track": self.row.get("track", ""),
            "tier": self.row.get("tier", "compete"),
            "description": self.row.get("description", ""),
            "config": self.public_config(),
        }
