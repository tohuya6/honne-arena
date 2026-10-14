"""The /submit pipeline, team membership, hints, and writeups, as plain functions over
an injected db. Everything here is testable without HTTP, NDIF, or a network.

Identity is a signed-in user (NU email). A user becomes a player in a season on their
first action there (a submission, a hint, a team), and gets a solo team named after
their handle so nobody has to set up a team before playing.
"""

from __future__ import annotations

import logging
import re
import sqlite3
from datetime import datetime, timezone
from typing import Callable

from app import challenges
from app.budget import check_ndif_budget
from app.errors import (DuplicateError, Forbidden, NotFound, ScoringUnavailable,
                        SubmitError, Unauthorized, ValidationError)
from app.ratelimit import check_rate_limits

_log = logging.getLogger("honne")

HANDLE_RE = re.compile(r"^[\w \-.]{1,32}$")   # letters, digits, _, space, -, .
TEAM_RE = re.compile(r"^[\w \-.']{1,40}$")
MAX_ANSWER_CHARS = 200_000                    # room for a predictions file
MAX_WRITEUP_CHARS = 20_000


def validate_handle(handle: str) -> str:
    handle = (handle or "").strip()
    if not HANDLE_RE.match(handle):
        raise ValidationError("Handle must be 1-32 characters: letters, digits, spaces, "
                              "_ - . only.")
    return handle


def validate_answer(answer: str) -> str:
    if answer is None or not answer.strip():
        raise ValidationError("Answer cannot be empty.")
    if _has_control_chars(answer):
        raise ValidationError("Answer contains invalid control characters.")
    if len(answer) > MAX_ANSWER_CHARS:
        raise ValidationError(f"Answer too long (max {MAX_ANSWER_CHARS} characters).")
    return answer


def _has_control_chars(text: str) -> bool:
    return any(ord(c) < 32 and c not in "\t\n\r" for c in text)


def parse_time(value: str) -> datetime:
    """An ISO timestamp as an aware UTC datetime. A value with no offset is taken as UTC."""
    dt = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def is_closed(row: dict, now: datetime | None = None) -> bool:
    closes = row.get("closes_at")
    if not closes:
        return False
    return (now or datetime.now(timezone.utc)) >= parse_time(closes)


def active_season(db) -> dict:
    season = db.get_active_season()
    if not season:
        raise ScoringUnavailable("No season is open right now.")
    return season


# ── handles and players ──

def set_handle(db, user: dict, handle: str) -> dict:
    handle = validate_handle(handle)
    other = db.get_user_by_handle(handle)
    if other and other["id"] != user["id"]:
        raise DuplicateError("That handle is taken.", existing={}, rank=0)
    try:
        db.set_user_handle(user["id"], handle)
    except sqlite3.IntegrityError:
        raise DuplicateError("That handle is taken.", existing={}, rank=0) from None
    return db.get_user(user["id"])


def _require_handle(user: dict | None) -> dict:
    if user is None:
        raise Unauthorized("Sign in first.")
    if not user.get("handle"):
        raise ValidationError("Choose a handle first.")
    return user


def get_or_create_player(db, season_id: int, user: dict) -> dict:
    """The user's player row this season, created with a solo team on first use."""
    user = _require_handle(user)
    player = db.get_player_by_user(season_id, user["id"])
    if player:
        return player
    team = _create_solo_team(db, season_id, user["handle"])
    try:
        return db.create_player(season_id, user["id"], team["id"])
    except sqlite3.IntegrityError:
        # A second request from the same user got there first. Use its row.
        db.delete_team_if_unused(team["id"])
        return db.get_player_by_user(season_id, user["id"])


def _create_solo_team(db, season_id: int, handle: str) -> dict:
    """A team named after the handle, with " 2", " 3", ... if the name is taken."""
    for n in range(1, 1000):
        name = handle if n == 1 else f"{handle} {n}"
        if db.get_team_by_name(season_id, name):
            continue
        try:
            return db.create_team(season_id, name)
        except sqlite3.IntegrityError:
            continue
    raise ScoringUnavailable("Could not create a team, please try again.")


# ── teams ──

def create_team(db, *, user: dict, name: str) -> dict:
    season = active_season(db)
    _require_handle(user)
    name = (name or "").strip()
    if not TEAM_RE.match(name):
        raise ValidationError("Team name must be 1-40 characters.")
    if db.get_team_by_name(season["id"], name):
        raise DuplicateError("That team name is taken.", existing={}, rank=0)
    player = get_or_create_player(db, season["id"], user)
    _ensure_can_switch(db, player)
    try:
        team = db.create_team(season["id"], name)
    except sqlite3.IntegrityError:
        raise DuplicateError("That team name is taken.", existing={}, rank=0) from None
    _move(db, player, team["id"])
    return team


def join_team(db, *, user: dict, join_code: str) -> dict:
    season = active_season(db)
    _require_handle(user)
    team = db.get_team_by_code(season["id"], (join_code or "").strip())
    if not team:
        raise NotFound("No team has that join code.")
    player = get_or_create_player(db, season["id"], user)
    if player["team_id"] == team["id"]:
        return team
    _ensure_can_switch(db, player)
    _move(db, player, team["id"])
    return team


def _move(db, player: dict, team_id: int) -> None:
    old = player["team_id"]
    db.set_player_team(player["id"], team_id)
    db.delete_team_if_unused(old)


def _ensure_can_switch(db, player: dict) -> None:
    # Switching after submitting would let one player's solves count for two teams.
    if db.player_submission_count(player["id"]) > 0:
        raise ValidationError("You can only change teams before your first submission "
                              "this season.")


# ── submit ──

def process_submission(
    *,
    user: dict | None,
    slug: str,
    answer: str,
    ip_hash: str,
    db,
    settings,
    run_remote: Callable | None = None,
    run_local: Callable | None = None,
    publish: Callable[[dict], None] | None = None,
    now: datetime | None = None,
) -> dict:
    """Validate, dedup, rate-limit, judge, record, announce.

    `run_remote(fn)` runs a callable behind the bounded NDIF queue, `run_local(fn)`
    behind the local job queue (both ScoringGate.run). A challenge's compute_mode()
    picks which one, and only "ndif" challenges touch the NDIF budget. `publish(event)`
    gets the live-feed event after a successful insert. Wrong answers are never
    published.
    """
    now = now or datetime.now(timezone.utc)
    user = _require_handle(user)
    season = active_season(db)
    answer = validate_answer(answer)

    row = db.get_challenge(season["id"], slug)
    if not row or row["hidden"]:
        raise NotFound("No such challenge.")
    if is_closed(row, now):
        raise Forbidden("This challenge is closed.")
    challenge = challenges.load(row)
    key = challenge.answer_key(answer)

    player = get_or_create_player(db, season["id"], user)
    team_id = player["team_id"]

    if challenge.mode == "solve" and db.player_solved(challenge.id, player["id"]):
        raise DuplicateError("You already solved this one.", existing={}, rank=0)
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
        inserted = db.insert_submission({
            "challenge_id": challenge.id, "player_id": player["id"], "team_id": team_id,
            "answer": answer, "answer_key": key,
            "correct": None if outcome.correct is None else int(outcome.correct),
            "score": outcome.score, "detail": outcome.detail, "ip_hash": ip_hash,
            "created_at": now.isoformat(),
        })
    except sqlite3.IntegrityError:
        # The same player solved it, or someone scored the same answer, a moment earlier.
        raise DuplicateError("Someone just beat you to that, try again.",
                             existing={}, rank=0) from None

    team = db.get_team(team_id)
    result = {"challenge": challenge.slug, "mode": challenge.mode}
    if challenge.mode == "solve":
        correct = bool(outcome.correct)
        # The very first correct submission on a challenge is also its first team solve.
        first_blood = correct and db.first_solve_id(challenge.id) == inserted["id"]
        result.update(outcome.detail)
        result["correct"], result["first_blood"] = correct, first_blood
        if correct and publish:
            publish({"type": "solve", "challenge": challenge.slug, "handle": user["handle"],
                     "team": team["name"], "first_blood": first_blood,
                     "at": now.isoformat()})
    else:
        result["score"] = outcome.score
        result.update(outcome.detail)
        if publish:
            publish({"type": "entry", "challenge": challenge.slug, "handle": user["handle"],
                     "team": team["name"], "score": outcome.score, "at": now.isoformat()})
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


# ── hints ──

def hint_cost(row: dict, stored_cost: int) -> int:
    """Hints on learn-tier challenges are free, whatever cost is stored."""
    return 0 if row.get("tier") == "learn" else int(stored_cost)


def hint_view(row: dict, hint: dict, unlocked: bool) -> dict:
    # The body goes only to someone who unlocked it.
    return {"position": hint["position"], "cost": hint_cost(row, hint["cost"]),
            "unlocked": unlocked, "body": hint["body"] if unlocked else None}


def unlock_hint(db, *, user: dict | None, slug: str, position: int) -> dict:
    user = _require_handle(user)
    season = active_season(db)
    row = db.get_challenge(season["id"], slug)
    if not row or row["hidden"]:
        raise NotFound("No such challenge.")
    hint = db.get_hint(row["id"], position)
    if not hint:
        raise NotFound("No such hint.")
    player = get_or_create_player(db, season["id"], user)
    db.unlock_hint(hint["id"], player["id"])     # unlocking twice costs nothing extra
    return hint_view(row, hint, unlocked=True)


# ── writeups ──

def post_writeup(db, *, user: dict | None, slug: str, body: str,
                 now: datetime | None = None) -> None:
    user = _require_handle(user)
    season = active_season(db)
    row = db.get_challenge(season["id"], slug)
    if not row or row["hidden"]:
        raise NotFound("No such challenge.")
    if not is_closed(row, now):
        raise Forbidden("Writeups open once the challenge closes.")
    if not db.user_submitted(row["id"], user["id"]):
        raise Forbidden("Only people who submitted to this challenge can post a writeup.")
    body = body or ""
    if not body.strip():
        raise ValidationError("Writeup cannot be empty.")
    if len(body) > MAX_WRITEUP_CHARS:
        raise ValidationError(f"Writeup too long (max {MAX_WRITEUP_CHARS} characters).")
    if _has_control_chars(body):
        raise ValidationError("Writeup contains invalid control characters.")
    db.upsert_writeup(row["id"], user["id"], body)
