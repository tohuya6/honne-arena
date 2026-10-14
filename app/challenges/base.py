"""The challenge interface. Each kind of challenge is one subclass.

A challenge row (from the `challenges` table) carries a `kind` and a JSON `config`.
`app.challenges.load()` picks the subclass by kind and hands it the row. Subclasses
only decide two things: how to normalize an answer, and how to judge it.

There are two modes:
- "solve": an answer is right or wrong (a CTF flag). Points come from solve counts.
- "score": an answer gets a number and the best one wins (steering). Points come from rank.

And three places judging can run (`compute_mode()`):
- "inline": a cheap check (hash compare, AUROC). Runs in the request thread.
- "local":  heavy work on this server, such as a sandboxed code run or a small local
            model. Goes through the local job gate so a burst can't starve the server.
- "ndif":   a remote model call on NDIF. Goes through the NDIF budget and gate.
"""

from __future__ import annotations

import json
import re
from typing import Callable, ClassVar, NamedTuple


class Outcome(NamedTuple):
    correct: bool | None   # set by "solve" challenges, None for "score" ones
    score: float | None    # set by "score" challenges, None for "solve" ones
    detail: dict


class Challenge:
    kind: ClassVar[str] = ""
    mode: ClassVar[str] = "solve"
    compute: ClassVar[str] = "inline"     # default for compute_mode()

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

    def compute_mode(self) -> str:
        """Where judging runs. An instance method so config can choose (a steering
        challenge can score on NDIF or on a small local model)."""
        return self.compute

    @property
    def uses_ndif(self) -> bool:
        return self.compute_mode() == "ndif"

    def prepare(self, settings, on_remote_call: Callable[[], None]) -> None:
        """Called before evaluate() for "local" and "ndif" challenges, inside the job
        gate. Build heavy state here (a model, a cached baseline) and keep it in a
        module-level cache keyed by challenge id, because a Challenge instance lives for
        one request. Must be cheap after the first call. Call `on_remote_call()` once per
        remote NDIF forward so the budget ledger sees it."""

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
