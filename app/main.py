"""HTTP layer. Thin: parse, call app.auth / app.submission / app.standings, map errors
to codes. The routes and shapes are the ones in API.md.

Run with the factory so importing this module has no side effects:
    uvicorn app.main:create_app --factory --port 7860
"""

from __future__ import annotations

import asyncio
import hmac
import logging
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path

from fastapi import Depends, FastAPI, Request, Response, WebSocket
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.websockets import WebSocketDisconnect

from app import admin, auth, captcha, standings, submission
from app.config import Settings, settings
from app.db import Database
from app.email_send import make_sender
from app.errors import Forbidden, NotFound, SubmitError, ValidationError
from app.live import Broadcaster
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


class InFlight:
    """Counts jobs inside a gate (waiting or running), for /api/admin/usage."""

    def __init__(self):
        self._n = 0
        self._lock = threading.Lock()

    @property
    def count(self) -> int:
        with self._lock:
            return self._n

    def wrap(self, run):
        def counted(fn, *args, **kwargs):
            with self._lock:
                self._n += 1
            try:
                return run(fn, *args, **kwargs)
            finally:
                with self._lock:
                    self._n -= 1
        return counted


class EmailIn(BaseModel):
    email: str = Field(max_length=320)


class TokenIn(BaseModel):
    token: str = Field(max_length=200)


class EmptyIn(BaseModel):
    pass


class HandleIn(BaseModel):
    handle: str


class SubmitIn(BaseModel):
    challenge: str
    answer: str
    turnstile_token: str = ""


class TeamIn(BaseModel):
    name: str


class JoinIn(BaseModel):
    join_code: str


class WriteupIn(BaseModel):
    body: str


class SeasonIn(BaseModel):
    name: str = Field(min_length=1, max_length=60)
    active: bool = True


def create_app(cfg: Settings | None = None, db: Database | None = None,
               gate: ScoringGate | None = None,
               local_gate: ScoringGate | None = None,
               mailer=None) -> FastAPI:
    cfg = cfg or settings
    if db is None:
        db = Database(cfg.db_path)
    db.migrate()
    # Two separate queues: a burst of code runs must not delay NDIF scoring, and the
    # other way round.
    gate = gate or ScoringGate(cfg.ndif_concurrency, call_timeout_s=cfg.ndif_timeout_s)
    local_gate = local_gate or ScoringGate(cfg.local_concurrency,
                                           call_timeout_s=cfg.local_timeout_s)
    local_jobs = InFlight()
    run_local = local_jobs.wrap(local_gate.run)
    mailer = mailer or make_sender(cfg)
    # Submissions run in the threadpool. The broadcaster hands their events to the
    # event loop that each socket lives on.
    live = Broadcaster(queue_size=cfg.ws_queue_size, max_clients=cfg.ws_max_clients)

    app = FastAPI(title="Honne Arena", docs_url="/api/docs", redoc_url=None)
    app.state.db, app.state.settings = db, cfg
    app.state.gate, app.state.local_gate = gate, local_gate
    app.state.mailer, app.state.live, app.state.local_jobs = mailer, live, local_jobs

    if cfg.allowed_origin:
        app.add_middleware(CORSMiddleware, allow_origins=[cfg.allowed_origin],
                           allow_credentials=True,
                           allow_methods=["GET", "POST", "PATCH"], allow_headers=["*"])

    @app.exception_handler(SubmitError)
    def _submit_error(_request, exc: SubmitError):
        body = {"error": exc.message}
        if getattr(exc, "existing", None):
            body["existing"] = exc.existing
        return JSONResponse(body, status_code=exc.http_status)

    @app.exception_handler(RequestValidationError)
    def _bad_request(_request, exc: RequestValidationError):
        errs = exc.errors()
        loc = ".".join(str(x) for x in errs[0]["loc"] if x != "body") if errs else ""
        msg = f"Invalid value for {loc}." if loc else "Invalid request body."
        return JSONResponse({"error": msg}, status_code=400)

    @app.exception_handler(StarletteHTTPException)
    def _http_error(_request, exc: StarletteHTTPException):
        return JSONResponse({"error": str(exc.detail)}, status_code=exc.status_code,
                            headers=getattr(exc, "headers", None))

    def current_user(request: Request) -> dict | None:
        return auth.session_user(db, request.cookies.get(auth.COOKIE_NAME))

    def is_admin_request(request: Request, user: dict | None) -> bool:
        token = request.headers.get("x-admin-token")
        if cfg.admin_token and token and hmac.compare_digest(token.encode(),
                                                             cfg.admin_token.encode()):
            return True
        return auth.is_admin(user, cfg)

    def require_admin(request: Request) -> None:
        if not is_admin_request(request, current_user(request)):
            raise Forbidden("Admin only.")

    def player_for(user: dict | None) -> dict | None:
        season = db.get_active_season()
        if not user or not season:
            return None
        return db.get_player_by_user(season["id"], user["id"])

    # ── public ──
    @app.get("/health")
    def health() -> dict:
        season = db.get_active_season()
        return {"ok": True, "season": season["name"] if season else None,
                "model_scoring": cfg.ndif_enabled}

    # ── auth ──
    @app.post("/api/auth/request")
    def auth_request(body: EmailIn, request: Request) -> dict:
        link = auth.request_link(db, mailer, cfg, email=body.email,
                                 ip_hash=hash_ip(_client_ip(request), cfg.ip_hash_salt))
        out = {"ok": True}
        if cfg.dev_expose_links and cfg.email_backend.lower() == "console":
            out["dev_link"] = link
        return out

    @app.post("/api/auth/verify")
    def auth_verify(body: TokenIn, request: Request, response: Response) -> dict:
        user, session = auth.verify_link(db, cfg, token=body.token)
        # Replace any session this browser already had.
        auth.end_session(db, request.cookies.get(auth.COOKIE_NAME))
        auth.set_cookie(response, session, cfg)
        return {"user": auth.user_view(user, cfg)}

    @app.post("/api/auth/logout")
    def auth_logout(request: Request, response: Response,
                    body: EmptyIn | None = None) -> dict:
        auth.end_session(db, request.cookies.get(auth.COOKIE_NAME))
        auth.clear_cookie(response, cfg)
        return {"ok": True}

    @app.get("/api/me")
    def me(user: dict | None = Depends(current_user)) -> dict:
        if not user:
            return {"user": None, "team": None, "tiers": {}}
        player = player_for(user)
        team = None
        if player:
            t = db.get_team(player["team_id"])
            team = {"name": t["name"], "join_code": t["join_code"],
                    "members": db.team_members(t["id"])}
        return {"user": auth.user_view(user, cfg), "team": team,
                "tiers": standings.user_tiers(db, user["id"])}

    @app.post("/api/me/handle")
    def set_handle(body: HandleIn, user: dict | None = Depends(current_user)) -> dict:
        user = submission.set_handle(db, auth.require_user(user), body.handle)
        return {"user": auth.user_view(user, cfg)}

    # ── play ──
    @app.get("/api/season")
    def season() -> dict:
        s = db.get_active_season()
        return {"id": s["id"] if s else None, "name": s["name"] if s else None,
                "captcha_sitekey": cfg.turnstile_sitekey or None,
                "seasons": [{"id": r["id"], "name": r["name"], "active": bool(r["active"])}
                            for r in db.list_seasons()]}

    @app.get("/api/challenges")
    def list_challenges(track: str = "", tier: str = "",
                        user: dict | None = Depends(current_user)) -> dict:
        s = submission.active_season(db)
        rows = db.list_challenges(s["id"])
        if track:
            rows = [r for r in rows if r["track"] == track]
        if tier:
            rows = [r for r in rows if r["tier"] == tier]
        player = player_for(user)
        return {"challenges": [standings.challenge_view(db, r, player=player, detail=False)
                               for r in rows]}

    @app.get("/api/challenges/{slug}")
    def get_challenge(slug: str, request: Request, limit: int = 50,
                      user: dict | None = Depends(current_user)) -> dict:
        s = submission.active_season(db)
        row = db.get_challenge(s["id"], slug)
        # Admins can preview hidden challenges. Everyone else gets a plain 404.
        if not row or (row["hidden"] and not is_admin_request(request, user)):
            raise NotFound("No such challenge.")
        return standings.challenge_view(db, row, player=player_for(user),
                                        limit=max(0, min(limit, cfg.leaderboard_max)))

    @app.post("/api/submit")
    def submit(body: SubmitIn, request: Request,
               user: dict | None = Depends(current_user)) -> dict:
        user = auth.require_user(user)
        ip = _client_ip(request)
        if not captcha.verify(body.turnstile_token, cfg.turnstile_secret, ip):
            raise ValidationError("CAPTCHA check failed, please retry.")
        return submission.process_submission(
            user=user, slug=body.challenge, answer=body.answer,
            ip_hash=hash_ip(ip, cfg.ip_hash_salt), db=db, settings=cfg,
            run_remote=gate.run, run_local=run_local, publish=live.publish)

    @app.post("/api/challenges/{slug}/hints/{position}/unlock")
    def unlock_hint(slug: str, position: int, body: EmptyIn | None = None,
                    user: dict | None = Depends(current_user)) -> dict:
        return submission.unlock_hint(db, user=auth.require_user(user), slug=slug,
                                      position=position)

    @app.post("/api/challenges/{slug}/writeup")
    def writeup(slug: str, body: WriteupIn, user: dict | None = Depends(current_user)) -> dict:
        submission.post_writeup(db, user=auth.require_user(user), slug=slug, body=body.body)
        return {"ok": True}

    # ── teams ──
    @app.post("/api/teams")
    def create_team(body: TeamIn, user: dict | None = Depends(current_user)) -> dict:
        team = submission.create_team(db, user=auth.require_user(user), name=body.name)
        return {"team": team["name"], "join_code": team["join_code"]}

    @app.post("/api/teams/join")
    def join_team(body: JoinIn, user: dict | None = Depends(current_user)) -> dict:
        team = submission.join_team(db, user=auth.require_user(user), join_code=body.join_code)
        return {"team": team["name"], "members": db.team_members(team["id"])}

    # ── boards ──
    @app.get("/api/leaderboard")
    def leaderboard(season: int | None = None) -> dict:
        s = db.get_season(season) if season is not None else submission.active_season(db)
        if not s:
            raise NotFound("No such season.")
        return {"season": {"id": s["id"], "name": s["name"]},
                "teams": standings.season_board(db, s["id"])}

    @app.get("/api/tracks")
    def tracks() -> dict:
        return {"tracks": standings.tracks_view(db)}

    @app.websocket("/ws")
    async def ws(websocket: WebSocket):
        # Subscribe before accepting, so nothing published after the handshake is missed.
        sub = live.subscribe()
        if sub is None:
            await websocket.close(code=1013)        # full, try again later
            return
        await websocket.accept()

        async def pump():
            while True:
                event = await sub.next()
                if event is None:                   # dropped for falling behind
                    await websocket.close(code=1013)
                    return
                await websocket.send_json(event)

        async def drain():
            # Clients have nothing to say. Reading is how a disconnect shows up.
            try:
                while True:
                    await websocket.receive_text()
            except WebSocketDisconnect:
                return

        tasks = [asyncio.create_task(pump()), asyncio.create_task(drain())]
        try:
            await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        finally:
            for t in tasks:
                t.cancel()
            live.unsubscribe(sub)

    # ── admin ──
    guard = [Depends(require_admin)]

    def admin_row(slug: str) -> dict:
        s = submission.active_season(db)
        row = db.get_challenge(s["id"], slug)
        if not row:
            raise NotFound("No such challenge.")
        return row

    @app.post("/api/admin/seasons", dependencies=guard)
    def admin_season(body: SeasonIn) -> dict:
        row = db.create_season(body.name.strip(), body.active)
        row["active"] = bool(row["active"])
        return row

    @app.get("/api/admin/challenges", dependencies=guard)
    def admin_list() -> dict:
        s = submission.active_season(db)
        return {"challenges": [admin.admin_view(db, r)
                               for r in db.list_challenges(s["id"], include_hidden=True)]}

    @app.post("/api/admin/challenges", dependencies=guard)
    def admin_challenge(body: admin.ChallengeIn) -> dict:
        s = submission.active_season(db)
        row = admin.create_challenge(db, s["id"], body)
        return {"id": row["id"], "slug": body.slug}

    @app.patch("/api/admin/challenges/{slug}", dependencies=guard)
    def admin_patch(slug: str, body: admin.ChallengePatch) -> dict:
        admin.patch_challenge(db, admin_row(slug), body)
        return {"slug": slug}

    @app.post("/api/admin/challenges/{slug}/hints", dependencies=guard)
    def admin_hint(slug: str, body: admin.HintIn) -> dict:
        return {"position": db.add_hint(admin_row(slug)["id"], body.body, body.cost)}

    @app.get("/api/admin/usage", dependencies=guard)
    def admin_usage() -> dict:
        # "Today" is the rolling 24 hours, the same window the budgets use.
        day_ago = (datetime.now(timezone.utc) - timedelta(days=1)).isoformat()
        return {"ndif": {"enabled": cfg.ndif_enabled,
                         "calls_today": db.count_ndif_since(day_ago),
                         "cap_per_day": cfg.ndif_calls_per_day,
                         "cap_per_player_day": cfg.ndif_player_per_day},
                "submissions_today": db.count_global_since(day_ago),
                "users": db.count_users(),
                "local_jobs_in_flight": local_jobs.count}

    # ── static ── /datasets first, so the catch-all web mount cannot shadow it.
    public_dir = Path(cfg.public_dir)
    public_dir.mkdir(parents=True, exist_ok=True)
    app.mount("/datasets", StaticFiles(directory=public_dir), name="datasets")
    if WEB_DIR.is_dir():
        app.mount("/", StaticFiles(directory=WEB_DIR, html=True), name="web")
    return app
