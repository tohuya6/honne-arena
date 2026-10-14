"""Hints: bodies only for whoever unlocked them, costs per team and per player,
free on learn-tier challenges."""

from tests.conftest import ADMIN, make_flag, submit

HINTS = [{"body": "Look at the page source.", "cost": 100},
         {"body": "Base64, twice.", "cost": 50}]


def _tracks(client):
    return {t["track"]: {r["handle"]: r["points"] for r in t["board"]}
            for t in client.get("/api/tracks").json()["tracks"]}


def test_bodies_hidden_until_unlocked(client, login):
    make_flag(client, hints=HINTS)
    hints = client.get("/api/challenges/warmup").json()["hints"]
    assert hints == [{"position": 1, "cost": 100, "unlocked": False, "body": None},
                     {"position": 2, "cost": 50, "unlocked": False, "body": None}]
    alice, bob = login("alice"), login("bob")
    r = alice.post("/api/challenges/warmup/hints/1/unlock", json={})
    assert r.json() == {"position": 1, "cost": 100, "unlocked": True,
                        "body": "Look at the page source."}
    mine = alice.get("/api/challenges/warmup").json()["hints"]
    assert mine[0]["body"] == "Look at the page source." and mine[1]["body"] is None
    assert bob.get("/api/challenges/warmup").json()["hints"][0]["body"] is None
    assert "page source" not in str(client.get("/api/challenges/warmup").json())


def test_unlock_needs_sign_in_and_a_real_hint(client, login):
    make_flag(client, hints=HINTS)
    assert client.post("/api/challenges/warmup/hints/1/unlock", json={}).status_code == 401
    alice = login("alice")
    assert alice.post("/api/challenges/warmup/hints/9/unlock", json={}).status_code == 404
    assert alice.post("/api/challenges/nope/hints/1/unlock", json={}).status_code == 404


def test_hint_cost_comes_off_team_and_personal_points(client, login):
    make_flag(client, hints=HINTS)
    alice = login("alice")
    alice.post("/api/challenges/warmup/hints/1/unlock", json={})
    alice.post("/api/challenges/warmup/hints/1/unlock", json={})     # twice costs once
    submit(alice, "warmup", "nuais{hello}")
    assert client.get("/api/leaderboard").json()["teams"][0]["points"] == 400
    assert _tracks(client)["ctf"] == {"alice": 400}


def test_team_pays_for_each_distinct_hint_once(client, login):
    make_flag(client, hints=HINTS)
    alice, bob = login("alice"), login("bob")
    code = alice.post("/api/teams", json={"name": "probes"}).json()["join_code"]
    bob.post("/api/teams/join", json={"join_code": code})
    alice.post("/api/challenges/warmup/hints/1/unlock", json={})
    bob.post("/api/challenges/warmup/hints/1/unlock", json={})       # same hint
    bob.post("/api/challenges/warmup/hints/2/unlock", json={})
    submit(alice, "warmup", "nuais{hello}")
    submit(bob, "warmup", "nuais{hello}")
    # Team: 500 - (100 + 50). Personal: alice 500 - 100, bob 500 - 150.
    assert client.get("/api/leaderboard").json()["teams"] == [
        {"rank": 1, "team": "probes", "points": 350, "solves": 1}]
    assert _tracks(client)["ctf"] == {"alice": 400, "bob": 350}


def test_learn_tier_hints_are_free(client, login):
    make_flag(client, tier="learn", hints=HINTS)
    assert [h["cost"] for h in client.get("/api/challenges/warmup").json()["hints"]] == [0, 0]
    alice = login("alice")
    assert alice.post("/api/challenges/warmup/hints/1/unlock", json={}).json()["cost"] == 0
    alice.post("/api/challenges/warmup/hints/2/unlock", json={})
    submit(alice, "warmup", "nuais{hello}")
    assert client.get("/api/leaderboard").json()["teams"][0]["points"] == 500
    assert _tracks(client)["ctf"] == {"alice": 500}


def test_points_never_go_negative(client, login):
    make_flag(client, hints=[{"body": "big", "cost": 900}])
    alice = login("alice")
    alice.post("/api/challenges/warmup/hints/1/unlock", json={})
    submit(alice, "warmup", "nuais{hello}")
    board = client.get("/api/leaderboard").json()["teams"]
    assert board == [{"rank": 1, "team": "alice", "points": 0, "solves": 1}]
    assert _tracks(client)["ctf"] == {"alice": 0}           # still on the board


def test_unlocking_without_solving_scores_nothing(client, login):
    make_flag(client, hints=HINTS)
    login("alice").post("/api/challenges/warmup/hints/1/unlock", json={})
    assert client.get("/api/leaderboard").json()["teams"] == []


def test_admin_adds_hints_in_order(client):
    make_flag(client)
    assert client.post("/api/admin/challenges/warmup/hints", json={"body": "one", "cost": 10},
                       headers=ADMIN).json() == {"position": 1}
    assert client.post("/api/admin/challenges/warmup/hints", json={"body": "two"},
                       headers=ADMIN).json() == {"position": 2}
    assert [h["cost"] for h in client.get("/api/challenges/warmup").json()["hints"]] == [10, 0]
    assert client.post("/api/admin/challenges/warmup/hints", json={"body": "x", "cost": -1},
                       headers=ADMIN).status_code == 400
