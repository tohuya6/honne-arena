"""HTTP layer. Thin: parse, call app.submission / app.standings, map errors to codes.

Run with the factory so importing this module has no side effects:
    uvicorn app.main:create_app --factory --port 7860
"""

from __future__ import annotations

import logging
from pathlib import Path

from fastapi import FastAPI, Header, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from app import captcha, challenges, standings, submission
from app.config import Settings, settings
from app.db import Database
from app.errors import Forbidden, NotFound, SubmitError, ValidationError
from app.queue import ScoringGate
from app.ratelimit import hash_ip

_log = logging.getLogger("honne")
WEB_DIR = Path(__file__).resolve().parent.parent / "web"


def _client_ip(request: Request) -> str:
    """Client IP for rate limiting, resistant to X-Forwarded-For spoofing.

    A client can prepend fake XFF entries but cannot control the one the outermost
    trusted proxy appends. With `trusted_proxy_hops = N` that is the Nth from the right.
    """
    hops = settings.trusted_proxy_hops
    xff = request.headers.get("x-forwarded-for")
    if xff and hops >= 1:
        parts = [p.strip() for p in xff.split(",") if p.strip()]
        if len(parts) >= hops:
            return parts[-hops]
    return request.client.host if request.client else "unknown"


class SubmitIn(BaseModel):
    handle: str
    challenge: str
    answer: str
    turnstile_token: str = ""


class TeamIn(BaseModel):
    handle: str
    name: str


class JoinIn(BaseModel):
    handle: str
    join_code: str


class SeasonIn(BaseModel):
    name: str
    active: bool = True


class ChallengeIn(BaseModel):
    slug: str = Field(pattern=r"^[a-z0-9][a-z0-9\-]{0,63}$")
    title: str
    kind: str
    track: str = ""
    tier: str = Field(default="compete", pattern=r"^(learn|compete)$")
    description: str = ""
    config: dict = {}
    # Plain-text flag, for kind "flag" only. Hashed here and never stored.
    flag: str | None = None
    points_initial: int = 500
    points_min: int = 100
    points_decay: int = 10
    hidden: bool = False


def create_app(cfg: Settings | None = None, db: Database | None = None,
               gate: ScoringGate | None = None) -> FastAPI:
    cfg = cfg or settings
    if db is None:
        db = Database(cfg.db_path)
    db.migrate()
    gate = gate or ScoringGate(cfg.ndif_concurrency, call_timeout_s=cfg.ndif_timeout_s)

    app = FastAPI(title="Honne Arena", docs_url="/api/docs", redoc_url=None)
    app.state.db, app.state.settings, app.state.gate = db, cfg, gate

    if cfg.allowed_origin:
        app.add_middleware(CORSMiddleware, allow_origins=[cfg.allowed_origin],
                           allow_methods=["GET", "POST"], allow_headers=["*"])

    @app.exception_handler(SubmitError)
    def _submit_error(_request, exc: SubmitError):
        body = {"error": exc.message}
        if getattr(exc, "existing", None):
            body["existing"] = exc.existing
        return JSONResponse(body, status_code=exc.http_status)

    def require_admin(token: str | None) -> None:
        if not cfg.admin_token or token != cfg.admin_token:
            raise Forbidden("Admin token required.")

    # ── public ──
    @app.get("/health")
    def health() -> dict:
        season = db.get_active_season()
        return {"ok": True, "season": season["name"] if season else None,
                "model_scoring": cfg.ndif_enabled}

    @app.get("/api/season")
    def season() -> dict:
        s = submission.active_season(db)
        return {"id": s["id"], "name": s["name"],
                "captcha_sitekey": cfg.turnstile_sitekey or None}

    @app.get("/api/challenges")
    def list_challenges(track: str = "", tier: str = "") -> dict:
        s = submission.active_season(db)
        rows = db.list_challenges(s["id"])
        if track:
            rows = [r for r in rows if r["track"] == track]
        if tier:
            rows = [r for r in rows if r["tier"] == tier]
        return {"challenges": [standings.challenge_view(db, r, limit=0) for r in rows]}

    @app.get("/api/challenges/{slug}")
    def get_challenge(slug: str, limit: int = 50) -> dict:
        s = submission.active_season(db)
        row = db.get_challenge(s["id"], slug)
        if not row or row["hidden"]:
            raise NotFound("No such challenge.")
        return standings.challenge_view(db, row, limit=min(limit, cfg.leaderboard_max))

    @app.get("/api/leaderboard")
    def leaderboard() -> dict:
        s = submission.active_season(db)
        return {"season": s["name"], "teams": standings.season_board(db, s["id"])}

    @app.post("/api/teams")
    def create_team(body: TeamIn) -> dict:
        team = submission.create_team(db, handle=body.handle, name=body.name)
        # The join code goes only to the creator, who shares it with teammates.
        return {"team": team["name"], "join_code": team["join_code"]}

    @app.post("/api/teams/join")
    def join_team(body: JoinIn) -> dict:
        team = submission.join_team(db, handle=body.handle, join_code=body.join_code)
        return {"team": team["name"], "members": db.team_members(team["id"])}

    @app.post("/api/submit")
    def submit(body: SubmitIn, request: Request) -> dict:
        ip = _client_ip(request)
        if not captcha.verify(body.turnstile_token, cfg.turnstile_secret, ip):
            raise ValidationError("CAPTCHA check failed, please retry.")
        return submission.process_submission(
            handle=body.handle, slug=body.challenge, answer=body.answer,
            ip_hash=hash_ip(ip, cfg.ip_hash_salt), db=db, settings=cfg,
            run_remote=gate.run)

    # ── admin ──
    @app.post("/api/admin/seasons")
    def admin_season(body: SeasonIn, x_admin_token: str | None = Header(default=None)) -> dict:
        require_admin(x_admin_token)
        return db.create_season(body.name, body.active)

    @app.post("/api/admin/challenges")
    def admin_challenge(body: ChallengeIn,
                        x_admin_token: str | None = Header(default=None)) -> dict:
        require_admin(x_admin_token)
        s = submission.active_season(db)
        if body.kind not in challenges.KINDS:
            raise ValidationError(f"Unknown kind. Known: {sorted(challenges.KINDS)}")
        config = dict(body.config)
        if body.kind == "flag":
            if not body.flag:
                raise ValidationError("A flag challenge needs `flag`.")
            config["flag_sha256"] = challenges.hash_flag(
                body.flag, bool(config.get("case_sensitive", False)))
        try:
            challenges.KINDS[body.kind]({"id": 0, "slug": body.slug, "title": body.title,
                                         "kind": body.kind, "config": config})
        except ValueError as exc:
            raise ValidationError(str(exc)) from None
        if db.get_challenge(s["id"], body.slug):
            raise ValidationError("That slug is taken this season.")
        row = db.create_challenge(
            s["id"], body.slug, body.title, body.kind, track=body.track, tier=body.tier,
            description=body.description, config=config,
            points_initial=body.points_initial, points_min=body.points_min,
            points_decay=body.points_decay, hidden=body.hidden)
        return {"id": row["id"], "slug": body.slug}

    @app.post("/api/admin/challenges/{slug}/hidden")
    def admin_hide(slug: str, hidden: bool = True,
                   x_admin_token: str | None = Header(default=None)) -> dict:
        require_admin(x_admin_token)
        s = submission.active_season(db)
        row = db.get_challenge(s["id"], slug)
        if not row:
            raise NotFound("No such challenge.")
        db.set_challenge_hidden(row["id"], hidden)
        return {"slug": slug, "hidden": hidden}

    if WEB_DIR.is_dir():
        app.mount("/", StaticFiles(directory=WEB_DIR, html=True), name="web")
    return app
