"""Data access over SQLite.

One small file database is enough for a club (tens of players, a few hundred
submissions a week), needs no hosted service, and `:memory:` makes tests self-contained.
Methods stay narrow and query-shaped so a Postgres backend could slot in later.

FastAPI runs sync endpoints in a threadpool, so one connection is shared behind a lock.
"""

from __future__ import annotations

import json
import secrets
import sqlite3
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path

SCHEMA = Path(__file__).resolve().parent.parent / "db" / "schema.sql"


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class Database:
    def __init__(self, path: str = ":memory:"):
        if path != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(path, check_same_thread=False, isolation_level=None)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("pragma foreign_keys = on")
        self._lock = threading.Lock()

    def migrate(self) -> None:
        with self._lock:
            self._conn.executescript(SCHEMA.read_text(encoding="utf-8"))

    # ── helpers ──
    def _one(self, sql: str, args=()) -> dict | None:
        with self._lock:
            row = self._conn.execute(sql, args).fetchone()
        return dict(row) if row else None

    def _all(self, sql: str, args=()) -> list[dict]:
        with self._lock:
            return [dict(r) for r in self._conn.execute(sql, args).fetchall()]

    def _insert(self, table: str, row: dict) -> dict:
        cols = ", ".join(row)
        marks = ", ".join("?" for _ in row)
        with self._lock:
            cur = self._conn.execute(f"insert into {table} ({cols}) values ({marks})",
                                     tuple(row.values()))
            new_id = cur.lastrowid
        return {"id": new_id, **row}

    def _exec(self, sql: str, args=()) -> int:
        """Run a write and return the number of rows it touched."""
        with self._lock:
            return self._conn.execute(sql, args).rowcount

    def _count(self, sql: str, args=()) -> int:
        with self._lock:
            return int(self._conn.execute(sql, args).fetchone()[0])

    # ── users ──
    def get_user(self, user_id: int) -> dict | None:
        return self._one("select * from users where id = ?", (user_id,))

    def get_user_by_email(self, email: str) -> dict | None:
        return self._one("select * from users where email = ?", (email,))

    def get_or_create_user(self, email: str) -> dict:
        with self._lock:
            self._conn.execute("insert or ignore into users (email, created_at) values (?, ?)",
                               (email, now_iso()))
        return self.get_user_by_email(email)

    def get_user_by_handle(self, handle: str) -> dict | None:
        return self._one("select * from users where handle = ?", (handle,))  # nocase column

    def set_user_handle(self, user_id: int, handle: str) -> None:
        """Raises sqlite3.IntegrityError when another user has it (any case)."""
        self._exec("update users set handle = ? where id = ?", (handle, user_id))

    def count_users(self) -> int:
        return self._count("select count(*) from users")

    # ── sign-in tokens and sessions ──
    def create_login_token(self, *, token_sha256: str, email: str, ip_hash: str,
                           expires_at: str, created_at: str) -> None:
        self._insert("login_tokens", {"token_sha256": token_sha256, "email": email,
                                      "ip_hash": ip_hash, "expires_at": expires_at,
                                      "created_at": created_at})

    def consume_login_token(self, token_sha256: str, now: str) -> str | None:
        """Mark the token used and return its email, or None if it is unknown, used,
        or expired. One UPDATE, so two concurrent clicks cannot both succeed."""
        with self._lock:
            cur = self._conn.execute(
                "update login_tokens set used_at = ? where token_sha256 = ? "
                "and used_at is null and expires_at > ?", (now, token_sha256, now))
            if cur.rowcount != 1:
                return None
            row = self._conn.execute("select email from login_tokens where token_sha256 = ?",
                                     (token_sha256,)).fetchone()
        return row["email"]

    def count_login_tokens_since(self, since_iso: str, *, email: str | None = None,
                                 ip_hash: str | None = None) -> int:
        if email is not None:
            return self._count("select count(*) from login_tokens where email = ? "
                               "and created_at >= ?", (email, since_iso))
        return self._count("select count(*) from login_tokens where ip_hash = ? "
                           "and created_at >= ?", (ip_hash, since_iso))

    def create_session(self, *, token_sha256: str, user_id: int, expires_at: str,
                       created_at: str) -> None:
        self._insert("sessions", {"token_sha256": token_sha256, "user_id": user_id,
                                  "expires_at": expires_at, "created_at": created_at})

    def session_user(self, token_sha256: str, now: str) -> dict | None:
        return self._one(
            "select u.* from sessions s join users u on u.id = s.user_id "
            "where s.token_sha256 = ? and s.expires_at > ?", (token_sha256, now))

    def delete_session(self, token_sha256: str) -> None:
        self._exec("delete from sessions where token_sha256 = ?", (token_sha256,))

    def purge_expired_auth(self, now: str) -> None:
        self._exec("delete from sessions where expires_at <= ?", (now,))
        # Keep a day of spent tokens so the sign-in rate limits still see them.
        self._exec("delete from login_tokens where expires_at <= ? and created_at < ?",
                   (now, _day_before(now)))

    # ── seasons ──
    def create_season(self, name: str, active: bool = True) -> dict:
        if active:
            with self._lock:
                self._conn.execute("update seasons set active = 0 where active = 1")
        return self._insert("seasons", {"name": name, "active": int(active),
                                        "created_at": now_iso()})

    def get_active_season(self) -> dict | None:
        return self._one("select * from seasons where active = 1")

    def get_season(self, season_id: int) -> dict | None:
        return self._one("select * from seasons where id = ?", (season_id,))

    def get_season_by_name(self, name: str) -> dict | None:
        return self._one("select * from seasons where name = ? order by id limit 1", (name,))

    def activate_season(self, season_id: int) -> None:
        with self._lock:
            self._conn.execute("update seasons set active = 0 where active = 1")
            self._conn.execute("update seasons set active = 1 where id = ?", (season_id,))

    def list_seasons(self) -> list[dict]:
        return self._all("select id, name, active from seasons order by id")

    # ── teams / players ──
    def create_team(self, season_id: int, name: str) -> dict:
        """Raises sqlite3.IntegrityError when the name is taken this season (any case)."""
        return self._insert("teams", {"season_id": season_id, "name": name,
                                      "join_code": secrets.token_urlsafe(6),
                                      "created_at": now_iso()})

    def get_team(self, team_id: int) -> dict | None:
        return self._one("select * from teams where id = ?", (team_id,))

    def get_team_by_name(self, season_id: int, name: str) -> dict | None:
        return self._one("select * from teams where season_id = ? and name = ?", (season_id, name))

    def get_team_by_code(self, season_id: int, code: str) -> dict | None:
        return self._one("select * from teams where season_id = ? and join_code = ?",
                         (season_id, code))

    def list_teams(self, season_id: int) -> list[dict]:
        return self._all("select id, name from teams where season_id = ? order by id", (season_id,))

    def delete_team_if_unused(self, team_id: int) -> None:
        """Drop a team nobody is in and nothing was submitted under, such as the solo
        team left behind when its only member joins another team."""
        self._exec("delete from teams where id = ? "
                   "and not exists (select 1 from players where team_id = ?) "
                   "and not exists (select 1 from submissions where team_id = ?)",
                   (team_id, team_id, team_id))

    def get_player(self, player_id: int) -> dict | None:
        return self._one("select p.*, u.handle from players p join users u on u.id = p.user_id "
                         "where p.id = ?", (player_id,))

    def get_player_by_user(self, season_id: int, user_id: int) -> dict | None:
        return self._one("select p.*, u.handle from players p join users u on u.id = p.user_id "
                         "where p.season_id = ? and p.user_id = ?", (season_id, user_id))

    def create_player(self, season_id: int, user_id: int, team_id: int) -> dict:
        """Raises sqlite3.IntegrityError if the user already plays this season."""
        row = self._insert("players", {"season_id": season_id, "user_id": user_id,
                                       "team_id": team_id, "created_at": now_iso()})
        return self.get_player(row["id"])

    def set_player_team(self, player_id: int, team_id: int) -> None:
        self._exec("update players set team_id = ? where id = ?", (team_id, player_id))

    def player_submission_count(self, player_id: int) -> int:
        return self._count("select count(*) from submissions where player_id = ?", (player_id,))

    def team_members(self, team_id: int) -> list[str]:
        return [r["handle"] for r in
                self._all("select u.handle from players p join users u on u.id = p.user_id "
                          "where p.team_id = ? order by p.id", (team_id,))]

    def players_users(self) -> dict[int, dict]:
        """player id -> {user_id, handle}, every season."""
        return {r["id"]: r for r in self._all(
            "select p.id, p.user_id, u.handle from players p join users u on u.id = p.user_id")}

    # ── challenges ──
    def create_challenge(self, season_id: int, slug: str, title: str, kind: str, *,
                         track: str = "", tier: str = "compete",
                         description: str = "", config: dict | None = None,
                         points_initial: int = 500, points_min: int = 100,
                         points_decay: int = 10, hidden: bool = False,
                         closes_at: str | None = None) -> dict:
        return self._insert("challenges", {
            "season_id": season_id, "slug": slug, "title": title, "kind": kind,
            "track": track, "tier": tier,
            "description": description, "config": json.dumps(config or {}),
            "points_initial": points_initial, "points_min": points_min,
            "points_decay": points_decay, "hidden": int(hidden), "closes_at": closes_at,
            "created_at": now_iso(),
        })

    def get_challenge(self, season_id: int, slug: str) -> dict | None:
        return self._one("select * from challenges where season_id = ? and slug = ?",
                         (season_id, slug))

    def list_challenges(self, season_id: int, include_hidden: bool = False) -> list[dict]:
        sql = "select * from challenges where season_id = ?"
        if not include_hidden:
            sql += " and hidden = 0"
        return self._all(sql + " order by id", (season_id,))

    def all_visible_challenges(self) -> list[dict]:
        """Every season's playable challenges, for the all-time track boards."""
        return self._all("select * from challenges where hidden = 0 order by id")

    # Columns PATCH may change. Anything else stays fixed after creation.
    PATCHABLE = ("title", "description", "hidden", "closes_at", "points_initial",
                 "points_min", "points_decay")

    def update_challenge(self, challenge_id: int, fields: dict) -> None:
        fields = {k: v for k, v in fields.items() if k in self.PATCHABLE}
        if not fields:
            return
        if "hidden" in fields:
            fields["hidden"] = int(fields["hidden"])
        sets = ", ".join(f"{k} = ?" for k in fields)
        self._exec(f"update challenges set {sets} where id = ?",
                   (*fields.values(), challenge_id))

    # ── hints ──
    def add_hint(self, challenge_id: int, body: str, cost: int) -> int:
        """Append a hint and return its position (1-based)."""
        with self._lock:
            pos = self._conn.execute(
                "select coalesce(max(position), 0) + 1 from hints where challenge_id = ?",
                (challenge_id,)).fetchone()[0]
            self._conn.execute(
                "insert into hints (challenge_id, position, body, cost, created_at) "
                "values (?, ?, ?, ?, ?)", (challenge_id, pos, body, cost, now_iso()))
        return int(pos)

    def list_hints(self, challenge_id: int) -> list[dict]:
        return self._all("select * from hints where challenge_id = ? order by position",
                         (challenge_id,))

    def get_hint(self, challenge_id: int, position: int) -> dict | None:
        return self._one("select * from hints where challenge_id = ? and position = ?",
                         (challenge_id, position))

    def unlock_hint(self, hint_id: int, player_id: int) -> None:
        with self._lock:
            self._conn.execute("insert or ignore into hint_unlocks (hint_id, player_id, "
                               "created_at) values (?, ?, ?)", (hint_id, player_id, now_iso()))

    def unlocked_hint_ids(self, challenge_id: int, player_id: int) -> set[int]:
        return {r["hint_id"] for r in self._all(
            "select hu.hint_id from hint_unlocks hu join hints h on h.id = hu.hint_id "
            "where h.challenge_id = ? and hu.player_id = ?", (challenge_id, player_id))}

    def hint_cost_by_player(self, challenge_id: int) -> dict[int, int]:
        """player id -> total stored cost of the hints they unlocked on this challenge."""
        return {r["player_id"]: int(r["cost"]) for r in self._all(
            "select hu.player_id, sum(h.cost) as cost from hint_unlocks hu "
            "join hints h on h.id = hu.hint_id where h.challenge_id = ? "
            "group by hu.player_id", (challenge_id,))}

    def hint_cost_by_team(self, challenge_id: int) -> dict[int, int]:
        """team id -> total stored cost of the DISTINCT hints any member unlocked, so two
        teammates opening the same hint pay for it once."""
        return {r["team_id"]: int(r["cost"]) for r in self._all(
            "select x.team_id, sum(h.cost) as cost from ("
            "  select distinct hu.hint_id, p.team_id from hint_unlocks hu "
            "  join players p on p.id = hu.player_id "
            "  join hints h on h.id = hu.hint_id where h.challenge_id = ?"
            ") x join hints h on h.id = x.hint_id group by x.team_id", (challenge_id,))}

    # ── writeups ──
    def upsert_writeup(self, challenge_id: int, user_id: int, body: str) -> None:
        stamp = now_iso()
        with self._lock:
            self._conn.execute(
                "insert into writeups (challenge_id, user_id, body, created_at, updated_at) "
                "values (?, ?, ?, ?, ?) on conflict (challenge_id, user_id) "
                "do update set body = excluded.body, updated_at = excluded.updated_at",
                (challenge_id, user_id, body, stamp, stamp))

    def list_writeups(self, challenge_id: int) -> list[dict]:
        return self._all(
            "select u.handle, w.body, w.created_at, w.updated_at from writeups w "
            "join users u on u.id = w.user_id where w.challenge_id = ? order by w.created_at, w.id",
            (challenge_id,))

    def user_submitted(self, challenge_id: int, user_id: int) -> bool:
        return self._count(
            "select count(*) from submissions s join players p on p.id = s.player_id "
            "where s.challenge_id = ? and p.user_id = ?", (challenge_id, user_id)) > 0

    # ── submissions ──
    def insert_submission(self, row: dict) -> dict:
        """Raises sqlite3.IntegrityError when a uniqueness rule is hit (player already
        solved, answer already scored). The caller maps that to DuplicateError."""
        row = dict(row)
        if isinstance(row.get("detail"), dict):
            row["detail"] = json.dumps(row["detail"])
        row.setdefault("created_at", now_iso())
        return self._insert("submissions", row)

    def player_solved(self, challenge_id: int, player_id: int) -> bool:
        return self._count("select count(*) from submissions where challenge_id = ? "
                           "and player_id = ? and correct = 1", (challenge_id, player_id)) > 0

    def player_submitted(self, challenge_id: int, player_id: int) -> bool:
        return self._count("select count(*) from submissions where challenge_id = ? "
                           "and player_id = ?", (challenge_id, player_id)) > 0

    def first_solve_id(self, challenge_id: int) -> int | None:
        row = self._one("select id from submissions where challenge_id = ? and correct = 1 "
                        "order by created_at, id limit 1", (challenge_id,))
        return row["id"] if row else None

    def solves(self, challenge_id: int) -> list[dict]:
        """Every correct solve, earliest first, with its player and team."""
        return self._all(
            "select s.id, s.created_at, s.player_id, s.team_id, t.name as team, u.handle "
            "from submissions s join teams t on t.id = s.team_id "
            "join players p on p.id = s.player_id join users u on u.id = p.user_id "
            "where s.challenge_id = ? and s.correct = 1 order by s.created_at, s.id",
            (challenge_id,))

    def find_scored(self, challenge_id: int, answer_key: str) -> dict | None:
        return self._one(
            "select s.*, u.handle from submissions s join players p on p.id = s.player_id "
            "join users u on u.id = p.user_id "
            "where s.challenge_id = ? and s.answer_key = ? and s.correct is null",
            (challenge_id, answer_key))

    def scored_entries(self, challenge_id: int) -> list[dict]:
        """Every scored answer, with its player and team. Ranking happens in Python."""
        return self._all(
            "select s.id, s.answer, s.score, s.created_at, s.player_id, s.team_id, "
            "u.handle, t.name as team from submissions s "
            "join players p on p.id = s.player_id join users u on u.id = p.user_id "
            "join teams t on t.id = s.team_id "
            "where s.challenge_id = ? and s.correct is null", (challenge_id,))

    def count_ip_since(self, ip_hash: str, since_iso: str) -> int:
        return self._count("select count(*) from submissions where ip_hash = ? "
                           "and created_at >= ?", (ip_hash, since_iso))

    def count_global_since(self, since_iso: str) -> int:
        return self._count("select count(*) from submissions where created_at >= ?",
                           (since_iso,))

    # ── NDIF budget ledger ──
    def log_ndif_call(self, *, player_id: int | None, challenge_id: int | None, ok: bool,
                      created_at: str | None = None) -> None:
        self._insert("ndif_calls", {"player_id": player_id, "challenge_id": challenge_id,
                                    "ok": int(ok), "created_at": created_at or now_iso()})

    def count_ndif_since(self, since_iso: str, player_id: int | None = None) -> int:
        if player_id is None:
            return self._count("select count(*) from ndif_calls where created_at >= ?",
                               (since_iso,))
        return self._count("select count(*) from ndif_calls where player_id = ? "
                           "and created_at >= ?", (player_id, since_iso))


def _day_before(iso: str) -> str:
    return (datetime.fromisoformat(iso) - timedelta(days=1)).isoformat()
