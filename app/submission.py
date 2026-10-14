"""The /submit pipeline and team membership, as plain functions over an injected db.
Everything here is testable without HTTP, NDIF, or a network.

Identity is a handle per season, with no accounts. That fits a club where people know
each other; see DECISIONS.md "Identity" for when that stops being enough.
"""

from __future__ import annotations

import logging
import re
import sqlite3
from datetime import datetime, timezone
from typing import Callable

from app import challenges
from app.budget import check_ndif_budget
from app.errors import (DuplicateError, NotFound, ScoringUnavailable, SubmitError,
                        ValidationError)
from app.ratelimit import check_rate_limits

_log = logging.getLogger("honne")

HANDLE_RE = re.compile(r"^[\w \-.]{1,32}$")   # letters, digits, _, space, -, .
TEAM_RE = re.compile(r"^[\w \-.']{1,40}$")
MAX_ANSWER_CHARS = 200_000                    # room for a predictions file


def validate_handle(handle: str) -> str:
    handle = (handle or "").strip()
    if not HANDLE_RE.match(handle):
        raise ValidationError("Handle must be 1-32 characters: letters, digits, spaces, "
                              "_ - . only.")
    return handle


def validate_answer(answer: str) -> str:
    if answer is None or not answer.strip():
        raise ValidationError("Answer cannot be empty.")
    if any(ord(c) < 32 and c not in "\t\n\r" for c in answer):
        raise ValidationError("Answer contains invalid control characters.")
    if len(answer) > MAX_ANSWER_CHARS:
        raise ValidationError(f"Answer too long (max {MAX_ANSWER_CHARS} characters).")
    return answer


def active_season(db) -> dict:
    season = db.get_active_season()
    if not season:
        raise ScoringUnavailable("No season is open right now.")
    return season


def get_or_create_player(db, season_id: int, handle: str) -> dict:
    """A new handle gets a solo team named after it, so nobody has to set up a team
    before playing."""
    player = db.get_player(season_id, handle)
    if player:
        return player
    team = db.get_team_by_name(season_id, handle) or db.create_team(season_id, handle)
    return db.create_player(season_id, handle, team["id"])


def create_team(db, *, handle: str, name: str) -> dict:
    season = active_season(db)
    handle = validate_handle(handle)
    name = (name or "").strip()
    if not TEAM_RE.match(name):
        raise ValidationError("Team name must be 1-40 characters.")
    if db.get_team_by_name(season["id"], name):
        raise DuplicateError("That team name is taken.", existing={}, rank=0)
    player = get_or_create_player(db, season["id"], handle)
    _ensure_can_switch(db, player)
    team = db.create_team(season["id"], name)
    db.set_player_team(player["id"], team["id"])
    return team


def join_team(db, *, handle: str, join_code: str) -> dict:
    season = active_season(db)
    handle = validate_handle(handle)
    team = db.get_team_by_code(season["id"], (join_code or "").strip())
    if not team:
        raise NotFound("No team has that join code.")
    player = get_or_create_player(db, season["id"], handle)
    if player["team_id"] == team["id"]:
        return team
    _ensure_can_switch(db, player)
    db.set_player_team(player["id"], team["id"])
    return team


def _ensure_can_switch(db, player: dict) -> None:
    # Switching after submitting would let one player's solves count for two teams.
    if db.player_submission_count(player["id"]) > 0:
        raise ValidationError("You can only change teams before your first submission "
                              "this season.")


def process_submission(
    *,
    handle: str,
    slug: str,
    answer: str,
    ip_hash: str,
    db,
    settings,
    run_remote: Callable | None = None,
    run_local: Callable | None = None,
    now: datetime | None = None,
) -> dict:
    """Validate, dedup, rate-limit, judge, record.

    `run_remote(fn)` runs a callable behind the bounded NDIF queue, `run_local(fn)`
    behind the local job queue (both ScoringGate.run). A challenge's compute_mode()
    picks which one, and only "ndif" challenges touch the NDIF budget.
    """
    now = now or datetime.now(timezone.utc)
    season = active_season(db)
    handle = validate_handle(handle)
    answer = validate_answer(answer)

    row = db.get_challenge(season["id"], slug)
    if not row or row["hidden"]:
        raise NotFound("No such challenge.")
    challenge = challenges.load(row)
    key = challenge.answer_key(answer)

    player = get_or_create_player(db, season["id"], handle)
    team_id = player["team_id"]

    if challenge.mode == "solve" and db.team_solved(challenge.id, team_id):
        raise DuplicateError("Your team already solved this one.", existing={}, rank=0)
    if challenge.mode == "score":
        existing = db.find_scored(challenge.id, key)
        if existing:
            raise DuplicateError(
                f"That answer is already on the board (submitted by {existing['handle']}).",
                existing={"score": existing["score"], "handle": existing["handle"]}, rank=0)

    check_rate_limits(db, ip_hash, settings, now)

    compute = challenge.compute_mode()
    if compute == "ndif":
        check_ndif_budget(db, player["id"], settings, now)
        outcome = _evaluate_gated(challenge, answer, db, player, settings, run_remote, now,
                                  log_ndif=True)
    elif compute == "local":
        outcome = _evaluate_gated(challenge, answer, db, player, settings, run_local, now,
                                  log_ndif=False)
    else:
        outcome = challenge.evaluate(answer)

    try:
        db.insert_submission({
            "challenge_id": challenge.id, "player_id": player["id"], "team_id": team_id,
            "answer": answer, "answer_key": key,
            "correct": None if outcome.correct is None else int(outcome.correct),
            "score": outcome.score, "detail": outcome.detail, "ip_hash": ip_hash,
            "created_at": now.isoformat(),
        })
    except sqlite3.IntegrityError:
        # A teammate solved it, or someone scored the same answer, a moment earlier.
        raise DuplicateError("Someone just beat you to that, try again.",
                             existing={}, rank=0) from None

    result = {"challenge": challenge.slug, "mode": challenge.mode}
    if challenge.mode == "solve":
        result["correct"] = bool(outcome.correct)
    else:
        result["score"] = outcome.score
        result.update(outcome.detail)
    return result


def _evaluate_gated(challenge, answer, db, player, settings, run, now, *, log_ndif: bool):
    """Judge behind a job gate. For NDIF challenges, log each remote forward to the
    budget ledger.

    Calls made while building the scorer (the one-time probe baseline) are logged with
    no player, so they count against the global cap but not this player's."""
    calls = {"prepare": 0, "evaluate": 0}
    phase = {"now": "prepare"}

    def on_remote_call():
        calls[phase["now"]] += 1

    def work():
        challenge.prepare(settings, on_remote_call)
        phase["now"] = "evaluate"
        return challenge.evaluate(answer)

    ok = False
    try:
        outcome = run(work) if run else work()
        ok = True
        return outcome
    except SubmitError:
        raise
    except Exception as exc:  # noqa: BLE001, detail goes to the server log
        _log.exception("judging failed for %s", challenge.slug)
        raise ScoringUnavailable("Judging failed, please try again later.") from exc
    finally:
        if log_ndif:
            stamp = now.isoformat()
            for _ in range(calls["prepare"]):
                db.log_ndif_call(player_id=None, challenge_id=challenge.id, ok=ok,
                                 created_at=stamp)
            for _ in range(calls["evaluate"]):
                db.log_ndif_call(player_id=player["id"], challenge_id=challenge.id, ok=ok,
                                 created_at=stamp)
