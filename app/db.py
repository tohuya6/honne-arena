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
from datetime import datetime, timezone
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

    def _count(self, sql: str, args=()) -> int:
        with self._lock:
            return int(self._conn.execute(sql, args).fetchone()[0])

    # ── seasons ──
    def create_season(self, name: str, active: bool = True) -> dict:
        if active:
            with self._lock:
                self._conn.execute("update seasons set active = 0 where active = 1")
        return self._insert("seasons", {"name": name, "active": int(active),
                                        "created_at": now_iso()})

    def get_active_season(self) -> dict | None:
        return self._one("select * from seasons where active = 1")

    # ── teams / players ──
    def create_team(self, season_id: int, name: str) -> dict:
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

    def get_player(self, season_id: int, handle: str) -> dict | None:
        return self._one("select * from players where season_id = ? and handle = ?",
                         (season_id, handle))

    def create_player(self, season_id: int, handle: str, team_id: int) -> dict:
        return self._insert("players", {"season_id": season_id, "handle": handle,
                                        "team_id": team_id, "created_at": now_iso()})

    def set_player_team(self, player_id: int, team_id: int) -> None:
        with self._lock:
            self._conn.execute("update players set team_id = ? where id = ?", (team_id, player_id))

    def player_submission_count(self, player_id: int) -> int:
        return self._count("select count(*) from submissions where player_id = ?", (player_id,))

    def team_members(self, team_id: int) -> list[str]:
        return [r["handle"] for r in
                self._all("select handle from players where team_id = ? order by id", (team_id,))]

    # ── challenges ──
    def create_challenge(self, season_id: int, slug: str, title: str, kind: str, *,
                         track: str = "", tier: str = "compete",
                         description: str = "", config: dict | None = None,
                         points_initial: int = 500, points_min: int = 100,
                         points_decay: int = 10, hidden: bool = False) -> dict:
        return self._insert("challenges", {
            "season_id": season_id, "slug": slug, "title": title, "kind": kind,
            "track": track, "tier": tier,
            "description": description, "config": json.dumps(config or {}),
            "points_initial": points_initial, "points_min": points_min,
            "points_decay": points_decay, "hidden": int(hidden), "created_at": now_iso(),
        })

    def get_challenge(self, season_id: int, slug: str) -> dict | None:
        return self._one("select * from challenges where season_id = ? and slug = ?",
                         (season_id, slug))

    def list_challenges(self, season_id: int, include_hidden: bool = False) -> list[dict]:
        sql = "select * from challenges where season_id = ?"
        if not include_hidden:
            sql += " and hidden = 0"
        return self._all(sql + " order by id", (season_id,))

    def set_challenge_hidden(self, challenge_id: int, hidden: bool) -> None:
        with self._lock:
            self._conn.execute("update challenges set hidden = ? where id = ?",
                               (int(hidden), challenge_id))

    # ── submissions ──
    def insert_submission(self, row: dict) -> dict:
        """Raises sqlite3.IntegrityError when a uniqueness rule is hit (team already
        solved, answer already scored). The caller maps that to DuplicateError."""
        row = dict(row)
        if isinstance(row.get("detail"), dict):
            row["detail"] = json.dumps(row["detail"])
        row.setdefault("created_at", now_iso())
        return self._insert("submissions", row)

    def team_solved(self, challenge_id: int, team_id: int) -> bool:
        return self._count("select count(*) from submissions where challenge_id = ? "
                           "and team_id = ? and correct = 1", (challenge_id, team_id)) > 0

    def solve_count(self, challenge_id: int) -> int:
        return self._count("select count(*) from submissions where challenge_id = ? "
                           "and correct = 1", (challenge_id,))

    def solves(self, challenge_id: int) -> list[dict]:
        """First-blood order: earliest solve first."""
        return self._all(
            "select s.created_at, t.name as team, p.handle from submissions s "
            "join teams t on t.id = s.team_id join players p on p.id = s.player_id "
            "where s.challenge_id = ? and s.correct = 1 order by s.created_at, s.id",
            (challenge_id,))

    def find_scored(self, challenge_id: int, answer_key: str) -> dict | None:
        return self._one(
            "select s.*, p.handle from submissions s join players p on p.id = s.player_id "
            "where s.challenge_id = ? and s.answer_key = ? and s.correct is null",
            (challenge_id, answer_key))

    def scored_entries(self, challenge_id: int) -> list[dict]:
        """Every scored answer, with its player and team. Ranking happens in Python."""
        return self._all(
            "select s.id, s.answer, s.score, s.created_at, s.player_id, s.team_id, "
            "p.handle, t.name as team from submissions s "
            "join players p on p.id = s.player_id join teams t on t.id = s.team_id "
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
