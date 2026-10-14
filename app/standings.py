"""Read-side views: one challenge's board, and the season board across challenges."""

from __future__ import annotations

from app import challenges
from app.points import best_per, dynamic_value, rank_value


def challenge_view(db, row: dict, limit: int = 50) -> dict:
    ch = challenges.load(row)
    out = ch.public()
    if ch.mode == "solve":
        solves = db.solves(ch.id)
        out["solves"] = len(solves)
        out["points"] = dynamic_value(row["points_initial"], row["points_min"],
                                      row["points_decay"], max(len(solves), 1))
        out["board"] = solves[:limit]
    else:
        best = best_per(db.scored_entries(ch.id), "player_id", ch.higher_is_better)
        out["entries"] = len(best)
        out["points"] = row["points_initial"]
        out["board"] = [{"rank": i + 1, "handle": r["handle"], "team": r["team"],
                         "score": r["score"], "answer": r["answer"] if ch.kind == "steering"
                         else None, "created_at": r["created_at"]}
                        for i, r in enumerate(best[:limit])]
    return out


def season_board(db, season_id: int) -> list[dict]:
    """Team totals. Flags pay their current dynamic value to every solving team.
    Scored challenges pay by the team's rank on its best entry."""
    totals: dict[int, dict] = {t["id"]: {"team": t["name"], "points": 0, "solves": 0}
                               for t in db.list_teams(season_id)}
    for row in db.list_challenges(season_id):
        ch = challenges.load(row)
        if ch.mode == "solve":
            solves = db.solves(ch.id)
            value = dynamic_value(row["points_initial"], row["points_min"],
                                  row["points_decay"], len(solves))
            for s in solves:
                team = db.get_team_by_name(season_id, s["team"])
                totals[team["id"]]["points"] += value
                totals[team["id"]]["solves"] += 1
        else:
            best = best_per(db.scored_entries(ch.id), "team_id", ch.higher_is_better)
            for i, r in enumerate(best):
                totals[r["team_id"]]["points"] += rank_value(
                    row["points_initial"], row["points_min"], i + 1, len(best))
    board = [v for v in totals.values() if v["points"] > 0 or v["solves"] > 0]
    board.sort(key=lambda v: (-v["points"], v["team"]))
    for i, v in enumerate(board):
        v["rank"] = i + 1
    return board
