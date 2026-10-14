"""The NDIF budget. NDIF is free for research use but it is a small shared cluster, so
this app spends it on purpose, never by accident.

Checked before any submission that would make a remote call. The counts come from the
`ndif_calls` ledger, so they survive restarts. The baseline build for a challenge is
logged there too (with no player), so it counts against the global cap.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from app.errors import RateLimited, ScoringUnavailable


def check_ndif_budget(db, player_id: int, settings, now: datetime | None = None) -> None:
    if not settings.ndif_enabled:
        raise ScoringUnavailable("Model scoring is switched off right now. "
                                 "Non-model challenges still work.")
    now = now or datetime.now(timezone.utc)
    minute_ago = (now - timedelta(seconds=60)).isoformat()
    day_ago = (now - timedelta(days=1)).isoformat()

    if db.count_ndif_since(day_ago) >= settings.ndif_calls_per_day:
        raise RateLimited("The arena has used today's model budget. Model challenges "
                          "reopen tomorrow.")
    if db.count_ndif_since(minute_ago, player_id) >= settings.ndif_player_per_min:
        raise RateLimited("One model submission at a time, please wait a minute.")
    if db.count_ndif_since(day_ago, player_id) >= settings.ndif_player_per_day:
        raise RateLimited("You have used your model submissions for today.")
