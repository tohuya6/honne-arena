"""Writing challenges: the validation shared by the admin API and scripts/seed.py.

Every new challenge goes through its kind's constructor, so a config that cannot work
is refused before it is stored. Admin views never include secrets: flag hashes, inline
labels, and inline tests are stripped, while `*_file` paths are kept as paths.
"""

from __future__ import annotations

import re

from pydantic import BaseModel, Field

from app import challenges
from app.errors import DuplicateError, ValidationError
from app.submission import parse_time

# Config keys that look like answer material. A key ending in "_file" is only a path
# and is kept.
_SECRET_KEY = re.compile(r"sha256|label|test|answer|solution|secret|gold|hidden",
                         re.IGNORECASE)


class HintIn(BaseModel):
    body: str = Field(min_length=1, max_length=5000)
    cost: int = Field(default=0, ge=0, le=100_000)


class ChallengeIn(BaseModel):
    slug: str = Field(pattern=r"^[a-z0-9][a-z0-9\-]{0,63}$")
    title: str = Field(min_length=1, max_length=120)
    kind: str
    track: str = Field(default="", pattern=r"^[a-z0-9\-]{0,32}$")
    tier: str = Field(default="compete", pattern=r"^(learn|compete)$")
    description: str = Field(default="", max_length=50_000)
    config: dict = {}
    # Plain-text flag, for kind "flag" only. Hashed here and never stored.
    flag: str | None = None
    points_initial: int = Field(default=500, ge=0, le=100_000)
    points_min: int = Field(default=100, ge=0, le=100_000)
    points_decay: int = Field(default=10, ge=0, le=100_000)
    hidden: bool = False
    hints: list[HintIn] = []
    closes_at: str | None = None


class ChallengePatch(BaseModel):
    title: str | None = Field(default=None, min_length=1, max_length=120)
    description: str | None = Field(default=None, max_length=50_000)
    hidden: bool | None = None
    closes_at: str | None = None        # null reopens a closed challenge
    points_initial: int | None = Field(default=None, ge=0, le=100_000)
    points_min: int | None = Field(default=None, ge=0, le=100_000)
    points_decay: int | None = Field(default=None, ge=0, le=100_000)


def normalize_closes_at(value: str | None) -> str | None:
    if value is None or not str(value).strip():
        return None
    try:
        return parse_time(str(value)).isoformat()
    except ValueError:
        raise ValidationError("closes_at must be an ISO date and time, "
                              "e.g. 2026-11-01T23:59:00Z.") from None


def _check_points(initial: int, minimum: int) -> None:
    if minimum > initial:
        raise ValidationError("points_min cannot be more than points_initial.")


def create_challenge(db, season_id: int, body: ChallengeIn) -> dict:
    """Validate and store a challenge with its hints. Returns the new row."""
    if body.kind not in challenges.KINDS:
        raise ValidationError(f"Unknown kind. Known: {sorted(challenges.KINDS)}")
    _check_points(body.points_initial, body.points_min)
    closes_at = normalize_closes_at(body.closes_at)
    config = dict(body.config)
    if body.kind == "flag":
        if not body.flag:
            raise ValidationError("A flag challenge needs `flag`.")
        config["flag_sha256"] = challenges.hash_flag(
            body.flag, bool(config.get("case_sensitive", False)))
    elif body.flag:
        raise ValidationError("`flag` is only for flag challenges.")
    try:
        challenges.KINDS[body.kind]({"id": 0, "slug": body.slug, "title": body.title,
                                     "kind": body.kind, "config": config})
    except ValueError as exc:
        raise ValidationError(str(exc)) from None
    if db.get_challenge(season_id, body.slug):
        raise DuplicateError("That slug is taken this season.", existing={}, rank=0)
    row = db.create_challenge(
        season_id, body.slug, body.title, body.kind, track=body.track, tier=body.tier,
        description=body.description, config=config,
        points_initial=body.points_initial, points_min=body.points_min,
        points_decay=body.points_decay, hidden=body.hidden, closes_at=closes_at)
    for h in body.hints:
        db.add_hint(row["id"], h.body, h.cost)
    return row


def patch_challenge(db, row: dict, patch: ChallengePatch) -> None:
    fields = {k: getattr(patch, k) for k in patch.model_fields_set}
    for k in ("title", "hidden", "points_initial", "points_min", "points_decay"):
        if k in fields and fields[k] is None:
            raise ValidationError(f"{k} cannot be null.")
    if "description" in fields and fields["description"] is None:
        fields["description"] = ""
    if "closes_at" in fields:
        fields["closes_at"] = normalize_closes_at(fields["closes_at"])
    _check_points(fields.get("points_initial", row["points_initial"]),
                  fields.get("points_min", row["points_min"]))
    db.update_challenge(row["id"], fields)


def safe_config(ch) -> dict:
    """Full config minus anything that gives an answer away."""
    public = ch.public_config()
    out = {}
    for k, v in ch.config.items():
        if k.endswith("_file") or k in public or not _SECRET_KEY.search(k):
            out[k] = v
    return out


def admin_view(db, row: dict) -> dict:
    from app.standings import challenge_view    # standings imports submission, not us

    ch = challenges.load(row)
    out = challenge_view(db, row, detail=False)
    out.update({"hidden": bool(row["hidden"]), "points_initial": row["points_initial"],
                "points_min": row["points_min"], "points_decay": row["points_decay"],
                "config": safe_config(ch),
                "hints": [{"position": h["position"], "body": h["body"], "cost": h["cost"]}
                          for h in db.list_hints(row["id"])]})
    return out
