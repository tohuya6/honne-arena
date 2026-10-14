"""How challenges turn into points on the season board.

Flag challenges use CTFd's dynamic scoring. A flag starts at `initial` points and falls
along a parabola to `minimum` once `decay` teams have solved it. Every solver holds the
current value, so a flag everyone finds ends up worth little to everyone, and a flag
only one team finds stays expensive. The first solve is worth full value.

Scored challenges (steering) have no right answer, only a better one. Teams are ranked
by their best entry and points fall linearly from `initial` (rank 1) to `minimum` (last).
"""

from __future__ import annotations

import math


def dynamic_value(initial: int, minimum: int, decay: int, solves: int) -> int:
    if decay <= 0 or solves <= 1:
        return initial
    n = solves - 1
    value = ((minimum - initial) / (decay ** 2)) * (n ** 2) + initial
    return max(minimum, math.ceil(value))


def rank_value(initial: int, minimum: int, rank: int, entrants: int) -> int:
    if entrants <= 1:
        return initial
    frac = (rank - 1) / (entrants - 1)
    return round(initial - frac * (initial - minimum))


def best_per(rows: list[dict], key: str, higher_is_better: bool) -> list[dict]:
    """Keep each `key`'s best-scoring row, sorted best first. Ties go to the earlier entry."""
    best: dict = {}
    for r in sorted(rows, key=lambda r: (r["created_at"], r["id"])):
        cur = best.get(r[key])
        better = cur is None or (r["score"] > cur["score"] if higher_is_better
                                 else r["score"] < cur["score"])
        if better:
            best[r[key]] = r
    out = list(best.values())
    out.sort(key=lambda r: ((-r["score"] if higher_is_better else r["score"]),
                            r["created_at"], r["id"]))
    return out
