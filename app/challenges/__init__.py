"""Challenge kinds. Add a new kind by subclassing Challenge and registering it here."""

from __future__ import annotations

from app.challenges.base import Challenge, Outcome
from app.challenges.code import CodeChallenge
from app.challenges.flag import FlagChallenge, hash_flag
from app.challenges.predictions import PredictionsChallenge
from app.challenges.steering import SteeringChallenge

KINDS: dict[str, type[Challenge]] = {
    CodeChallenge.kind: CodeChallenge,
    FlagChallenge.kind: FlagChallenge,
    PredictionsChallenge.kind: PredictionsChallenge,
    SteeringChallenge.kind: SteeringChallenge,
}


def load(row: dict) -> Challenge:
    cls = KINDS.get(row["kind"])
    if cls is None:
        raise ValueError(f"unknown challenge kind {row['kind']!r}")
    return cls(row)


__all__ = ["Challenge", "Outcome", "KINDS", "load", "hash_flag"]
