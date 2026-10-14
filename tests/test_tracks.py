"""All-time track boards and tiers."""

import pytest

from app import standings
from app.standings import tier_for_rank
from tests.conftest import ADMIN, make_flag, submit


@pytest.mark.parametrize("rank,tier", [(1, 3), (5, 3), (6, 2), (15, 2), (16, 1), (99, 1),
                                       (None, 0)])
def test_tier_for_rank(rank, tier):
    assert tier_for_rank(rank) == tier


def _seed_scored(db, n):
    """n users with strictly falling scores on one predictions challenge, written
    straight to the db so the test does not need n sign-ins."""
    season = db.get_active_season()["id"]
    ch = db.create_challenge(season, "liar", "Liar", "predictions", track="lie-detection",
                             config={"labels": {"a": 1, "b": 0}},
                             points_initial=1600, points_min=100)
    for i in range(n):
        user = db.get_or_create_user(f"u{i:02d}@northeastern.edu")
        db.set_user_handle(user["id"], f"u{i:02d}")
        team = db.create_team(season, f"u{i:02d}")
        player = db.create_player(season, user["id"], team["id"])
        db.insert_submission({"challenge_id": ch["id"], "player_id": player["id"],
                              "team_id": team["id"], "answer": str(i), "answer_key": str(i),
                              "correct": None, "score": 1.0 - i / 100, "detail": {},
                              "ip_hash": "h", "created_at": f"2026-10-14T00:00:{i:02d}"})


def test_tier_boundaries_on_the_board(client, db):
    _seed_scored(db, 17)
    board = next(t["board"] for t in client.get("/api/tracks").json()["tracks"]
                 if t["track"] == "lie-detection")
    by_rank = {r["rank"]: r for r in board}
    assert [by_rank[k]["tier"] for k in (1, 5, 6, 15, 16, 17)] == [3, 3, 2, 2, 1, 1]
    assert by_rank[5]["handle"] == "u04" and by_rank[16]["handle"] == "u15"
    assert [r["points"] for r in board] == sorted((r["points"] for r in board), reverse=True)
    assert set(board[0]) == {"rank", "handle", "points", "tier"}     # no user ids or emails


def test_me_reports_tiers(app, db):
    from tests.conftest import sign_in
    _seed_scored(db, 6)
    c = sign_in(app, "u05@northeastern.edu")        # the sixth-best user
    tiers = c.get("/api/me").json()["tiers"]
    assert tiers["lie-detection"] == {"tier": 2, "rank": 6,
                                      "points": tiers["lie-detection"]["points"]}
    assert tiers["ctf"] == {"tier": 0, "rank": None, "points": 0}
    assert set(tiers) >= {"ctf", "lie-detection", "steering", "circuits", "cp"}


def test_all_tracks_listed_and_absent_users_left_out(client, login):
    make_flag(client)
    login("lurker")
    tracks = client.get("/api/tracks").json()["tracks"]
    assert [t["track"] for t in tracks] == ["ctf", "lie-detection", "steering", "circuits",
                                            "cp"]
    assert all(t["board"] == [] for t in tracks)


def test_wrong_answers_are_not_entries(client, login):
    make_flag(client)
    submit(login("alice"), "warmup", "nope")
    assert client.get("/api/tracks").json()["tracks"][0]["board"] == []


def test_sums_across_seasons(client, login, db):
    alice, bob = login("alice"), login("bob")
    make_flag(client, slug="s1-flag")
    submit(alice, "s1-flag", "nuais{hello}")
    db.create_season("Season 2")
    make_flag(client, slug="s2-flag")
    submit(alice, "s2-flag", "nuais{hello}")
    submit(bob, "s2-flag", "nuais{hello}")
    ctf = client.get("/api/tracks").json()["tracks"][0]["board"]
    # s1-flag: alice alone, 500. s2-flag: two teams with decay 2, worth 400 each.
    assert [(r["handle"], r["points"], r["rank"]) for r in ctf] == [("alice", 900, 1),
                                                                    ("bob", 400, 2)]


def test_personal_points_use_the_current_value(client, login):
    make_flag(client)                                # decay 2
    a, b, c = login("a"), login("b"), login("c")
    for p in (a, b, c):
        submit(p, "warmup", "nuais{hello}")
    ctf = client.get("/api/tracks").json()["tracks"][0]["board"]
    assert [r["points"] for r in ctf] == [100, 100, 100]


def test_ties_go_to_whoever_got_there_first(client, login):
    make_flag(client, slug="f1", points_decay=0)
    make_flag(client, slug="f2", points_decay=0)
    a, b = login("a"), login("b")
    submit(b, "f1", "nuais{hello}")
    submit(a, "f1", "nuais{hello}")
    submit(a, "f2", "nuais{hello}")
    submit(b, "f2", "nuais{hello}")                 # b finishes last
    ctf = client.get("/api/tracks").json()["tracks"][0]["board"]
    assert [(r["handle"], r["points"], r["rank"]) for r in ctf] == [("a", 1000, 1),
                                                                    ("b", 1000, 2)]


def test_scored_points_rank_among_players(client, login):
    body = {"slug": "liar", "title": "Liar", "kind": "predictions", "track": "lie-detection",
            "config": {"labels": {"t1": 1, "t2": 0}}, "points_initial": 500,
            "points_min": 100}
    assert client.post("/api/admin/challenges", json=body, headers=ADMIN).status_code == 200
    a, b = login("a"), login("b")
    code = a.post("/api/teams", json={"name": "duo"}).json()["join_code"]
    b.post("/api/teams/join", json={"join_code": code})
    submit(a, "liar", "t1,0.9\nt2,0.1")             # AUROC 1.0
    submit(b, "liar", "t1,0.1\nt2,0.9")             # AUROC 0.0
    submit(a, "liar", "t1,0.4\nt2,0.6")             # worse, and a's best still counts
    lie = next(t["board"] for t in client.get("/api/tracks").json()["tracks"]
               if t["track"] == "lie-detection")
    assert [(r["handle"], r["points"]) for r in lie] == [("a", 500), ("b", 100)]
    # The team board takes the team's best entry only.
    assert client.get("/api/leaderboard").json()["teams"] == [
        {"rank": 1, "team": "duo", "points": 500, "solves": 0}]
    view = a.get("/api/challenges/liar").json()
    assert view["entries"] == 2 and view["my_best"] == 1.0 and view["solves"] is None
    assert [r["handle"] for r in view["board"]] == ["a", "b"]


def test_hidden_challenges_do_not_count(client, login, db):
    make_flag(client)
    submit(login("alice"), "warmup", "nuais{hello}")
    client.patch("/api/admin/challenges/warmup", json={"hidden": True}, headers=ADMIN)
    assert standings.track_boards(db)["ctf"] == []
