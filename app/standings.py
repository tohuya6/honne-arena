"""Read-side views: one challenge with its board, the season team board, and the
all-time track boards that tiers come from.

Two kinds of points:
- Team (season board). A flag pays its current value once per team, however many
  members solved it. A scored challenge pays by the team's rank on its best entry.
  Either way, minus the costs of the DISTINCT hints any member unlocked.
- Personal (track boards). A solved flag pays its current value, a scored challenge
  pays by the player's rank among players on their best entry. Either way, minus the
  player's own hint costs. Summed per user over every season.
"""

from __future__ import annotations

from app import challenges
from app.points import best_per, dynamic_value, net, rank_value
from app.submission import hint_view, is_closed

KNOWN_TRACKS = ("ctf", "lie-detection", "steering", "circuits", "cp")


def first_per_team(solves: list[dict]) -> list[dict]:
    """Each team's first solve, earliest first. `solves` must already be in time order."""
    seen, out = set(), []
    for s in solves:
        if s["team_id"] not in seen:
            seen.add(s["team_id"])
            out.append(s)
    return out


def flag_value(row: dict, solving_teams: int) -> int:
    return dynamic_value(row["points_initial"], row["points_min"], row["points_decay"],
                         solving_teams)


def _hint_costs(db, row: dict, by: str) -> dict[int, int]:
    if row["tier"] == "learn":
        return {}           # learn-tier hints are free
    return db.hint_cost_by_team(row["id"]) if by == "team" else db.hint_cost_by_player(row["id"])


def challenge_view(db, row: dict, *, player: dict | None = None, limit: int = 50,
                   detail: bool = True) -> dict:
    """The Challenge object. With `detail`, also its board, hints, and writeups.
    `player` is the viewer's player row this season, if any."""
    ch = challenges.load(row)
    out = ch.public()
    closed = is_closed(row)
    pid = player["id"] if player else None
    out.update({"closes_at": row.get("closes_at"), "closed": closed,
                "solves": None, "entries": None, "solved_by_me": False,
                "solved_by_team": False, "my_best": None,
                "submitted_by_me": bool(pid) and db.player_submitted(ch.id, pid)})
    if ch.mode == "solve":
        solves = db.solves(ch.id)
        teams = first_per_team(solves)
        out["solves"] = len(teams)
        out["points"] = flag_value(row, len(teams))
        # solved_by_me: this player solved it. solved_by_team: anyone on their team did.
        out["solved_by_me"] = any(s["player_id"] == pid for s in solves)
        out["solved_by_team"] = bool(player) and any(s["team_id"] == player["team_id"]
                                                     for s in solves)
        board = [{"team": s["team"], "handle": s["handle"], "created_at": s["created_at"]}
                 for s in teams[:limit]]
    else:
        best = best_per(db.scored_entries(ch.id), "player_id", ch.higher_is_better)
        out["entries"] = len(best)
        out["points"] = row["points_initial"]
        mine = [r for r in best if r["player_id"] == pid]
        out["my_best"] = mine[0]["score"] if mine else None
        board = [{"rank": i + 1, "handle": r["handle"], "team": r["team"],
                  "score": r["score"], "answer": r["answer"] if ch.kind == "steering"
                  else None, "created_at": r["created_at"]}
                 for i, r in enumerate(best[:limit])]
    if detail:
        unlocked = db.unlocked_hint_ids(ch.id, pid) if pid else set()
        out["board"] = board
        out["hints"] = [hint_view(row, h, h["id"] in unlocked) for h in db.list_hints(ch.id)]
        out["writeups"] = ([{"handle": w["handle"], "body": w["body"],
                             "created_at": w["created_at"]}
                            for w in db.list_writeups(ch.id)] if closed else [])
    return out


def season_board(db, season_id: int) -> list[dict]:
    """Team totals for one season, best first. Teams with no solve or entry are left out."""
    totals: dict[int, dict] = {t["id"]: {"team": t["name"], "points": 0, "solves": 0,
                                         "on_board": False}
                               for t in db.list_teams(season_id)}
    for row in db.list_challenges(season_id):
        ch = challenges.load(row)
        costs = _hint_costs(db, row, "team")
        if ch.mode == "solve":
            teams = first_per_team(db.solves(ch.id))
            value = flag_value(row, len(teams))
            for s in teams:
                t = totals[s["team_id"]]
                t["points"] += net(value, costs.get(s["team_id"], 0))
                t["solves"] += 1
                t["on_board"] = True
        else:
            best = best_per(db.scored_entries(ch.id), "team_id", ch.higher_is_better)
            for i, r in enumerate(best):
                t = totals[r["team_id"]]
                value = rank_value(row["points_initial"], row["points_min"], i + 1, len(best))
                t["points"] += net(value, costs.get(r["team_id"], 0))
                t["on_board"] = True
    board = [v for v in totals.values() if v.pop("on_board")]
    board.sort(key=lambda v: (-v["points"], v["team"].lower()))
    return [{"rank": i + 1, **v} for i, v in enumerate(board)]


def tier_for_rank(rank: int | None) -> int:
    if rank is None:
        return 0
    if rank <= 5:
        return 3
    if rank <= 15:
        return 2
    return 1


def track_boards(db) -> dict[str, list[dict]]:
    """All-time personal boards, one per track, across every season.

    A user is on a track's board once they have a solve or a scored entry there, even
    if hint costs bring it to 0 points. Ties go to whoever reached their total first
    (the earlier of each user's latest counted entry), then to the lower user id, so
    ranks are fixed and every rank number is used once."""
    who = db.players_users()
    agg: dict[str, dict[int, dict]] = {t: {} for t in KNOWN_TRACKS}

    def add(track, player_id, points, at):
        u = who[player_id]
        cur = agg.setdefault(track, {}).setdefault(
            u["user_id"], {"user_id": u["user_id"], "handle": u["handle"], "points": 0,
                           "last_at": ""})
        cur["points"] += points
        cur["last_at"] = max(cur["last_at"], at)

    for row in db.all_visible_challenges():
        if not row["track"]:
            continue
        ch = challenges.load(row)
        costs = _hint_costs(db, row, "player")
        if ch.mode == "solve":
            solves = db.solves(ch.id)
            value = flag_value(row, len(first_per_team(solves)))
            for s in solves:
                add(row["track"], s["player_id"], net(value, costs.get(s["player_id"], 0)),
                    s["created_at"])
        else:
            best = best_per(db.scored_entries(ch.id), "player_id", ch.higher_is_better)
            for i, r in enumerate(best):
                value = rank_value(row["points_initial"], row["points_min"], i + 1, len(best))
                add(row["track"], r["player_id"], net(value, costs.get(r["player_id"], 0)),
                    r["created_at"])

    boards = {}
    for track, users in agg.items():
        rows = sorted(users.values(), key=lambda u: (-u["points"], u["last_at"], u["user_id"]))
        boards[track] = [{"rank": i + 1, "user_id": u["user_id"], "handle": u["handle"],
                          "points": u["points"], "tier": tier_for_rank(i + 1)}
                         for i, u in enumerate(rows)]
    return boards


def ordered_tracks(boards: dict) -> list[str]:
    return list(KNOWN_TRACKS) + sorted(t for t in boards if t not in KNOWN_TRACKS)


def tracks_view(db) -> list[dict]:
    boards = track_boards(db)
    return [{"track": t, "board": [{k: r[k] for k in ("rank", "handle", "points", "tier")}
                                   for r in boards[t]]}
            for t in ordered_tracks(boards)]


def user_tiers(db, user_id: int) -> dict[str, dict]:
    """The Tier for every track. Tier 0, rank null, 0 points when not on a board yet."""
    boards = track_boards(db)
    out = {}
    for t in ordered_tracks(boards):
        mine = next((r for r in boards[t] if r["user_id"] == user_id), None)
        out[t] = ({"tier": mine["tier"], "rank": mine["rank"], "points": mine["points"]}
                  if mine else {"tier": 0, "rank": None, "points": 0})
    return out
