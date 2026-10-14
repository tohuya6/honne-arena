"""CTF flags end to end over HTTP: create, solve per player, decay by teams, first blood."""

from tests.conftest import ADMIN, make_flag, submit


def test_admin_routes_need_token(client):
    r = client.post("/api/admin/challenges", json={"slug": "x", "title": "x", "kind": "flag",
                                                    "flag": "f"})
    assert r.status_code == 403


def test_flag_is_never_stored_in_plain_text(client, db):
    make_flag(client)
    row = db.get_challenge(db.get_active_season()["id"], "warmup")
    assert "nuais{hello}" not in row["config"]
    pub = client.get("/api/challenges/warmup").json()
    assert "flag_sha256" not in str(pub)
    assert pub["config"] == {"flag_format": "nuais{...}"}


def test_wrong_then_right(client, login):
    make_flag(client)
    alice = login("alice")
    assert submit(alice, "warmup", "nuais{nope}").json() == {
        "challenge": "warmup", "mode": "solve", "correct": False, "first_blood": False}
    r = submit(alice, "warmup", "  NUAIS{Hello} ").json()      # case-insensitive
    assert r["correct"] is True and r["first_blood"] is True


def test_player_solves_once(client, login):
    make_flag(client)
    alice = login("alice")
    assert submit(alice, "warmup", "nuais{hello}").json()["correct"] is True
    r = submit(alice, "warmup", "nuais{hello}")
    assert r.status_code == 409
    assert r.json()["error"] == "You already solved this one."


def test_teammates_each_solve_but_team_scores_once(client, login):
    make_flag(client)
    alice, bob = login("alice"), login("bob")
    code = alice.post("/api/teams", json={"name": "probes"}).json()["join_code"]
    assert bob.post("/api/teams/join", json={"join_code": code}).status_code == 200
    a = submit(alice, "warmup", "nuais{hello}").json()
    b = submit(bob, "warmup", "nuais{hello}").json()
    assert a["correct"] and a["first_blood"]
    assert b["correct"] and not b["first_blood"]
    view = client.get("/api/challenges/warmup").json()
    assert view["solves"] == 1                       # one team
    assert view["points"] == 500                     # no decay from a teammate
    assert [(r["team"], r["handle"]) for r in view["board"]] == [("probes", "alice")]
    teams = client.get("/api/leaderboard").json()["teams"]
    assert teams == [{"rank": 1, "team": "probes", "points": 500, "solves": 1}]


def test_points_decay_by_distinct_teams(client, login):
    make_flag(client)                                # decay = 2
    players = [login(h) for h in ("a", "b", "c")]
    for p in players:
        assert submit(p, "warmup", "nuais{hello}").json()["correct"] is True
    view = client.get("/api/challenges/warmup").json()
    assert view["solves"] == 3
    assert view["points"] == 100                     # the third team hits the floor
    assert [s["handle"] for s in view["board"]] == ["a", "b", "c"]   # first blood first
    teams = client.get("/api/leaderboard").json()["teams"]
    assert {t["team"]: t["points"] for t in teams} == {"a": 100, "b": 100, "c": 100}


def test_first_blood_only_once(client, login):
    make_flag(client)
    a, b = login("a"), login("b")
    assert submit(a, "warmup", "nuais{hello}").json()["first_blood"] is True
    assert submit(b, "warmup", "nuais{hello}").json()["first_blood"] is False


def test_solved_by_me_and_team(client, login):
    make_flag(client)
    a, b, c = login("a"), login("b"), login("c")
    code = a.post("/api/teams", json={"name": "duo"}).json()["join_code"]
    b.post("/api/teams/join", json={"join_code": code})
    submit(c, "warmup", "wrong")
    submit(a, "warmup", "nuais{hello}")
    va, vb, vc = (p.get("/api/challenges/warmup").json() for p in (a, b, c))
    assert (va["solved_by_me"], va["solved_by_team"], va["submitted_by_me"]) == (True, True, True)
    assert (vb["solved_by_me"], vb["solved_by_team"], vb["submitted_by_me"]) == (False, True, False)
    assert (vc["solved_by_me"], vc["solved_by_team"], vc["submitted_by_me"]) == (False, False, True)
    anon = client.get("/api/challenges/warmup").json()
    assert (anon["solved_by_me"], anon["solved_by_team"], anon["submitted_by_me"]) == \
        (False, False, False)
    listed = a.get("/api/challenges").json()["challenges"][0]
    assert listed["solved_by_me"] is True and listed["entries"] is None
    assert "board" not in listed


def test_hidden_challenge_is_not_playable(client, login):
    make_flag(client, hidden=True)
    alice = login("alice")
    assert client.get("/api/challenges").json()["challenges"] == []
    assert client.get("/api/challenges/warmup").status_code == 404
    assert alice.get("/api/challenges/warmup").status_code == 404
    assert submit(alice, "warmup", "nuais{hello}").status_code == 404


def test_admins_can_preview_hidden(client, admin_client):
    make_flag(client, hidden=True)
    assert client.get("/api/challenges/warmup", headers=ADMIN).status_code == 200
    assert admin_client.get("/api/challenges/warmup").json()["slug"] == "warmup"
    assert admin_client.get("/api/challenges").json()["challenges"] == []


def test_rate_limit(client, cfg, login):
    cfg.rate_per_min = 2
    make_flag(client)
    alice = login("alice")
    assert submit(alice, "warmup", "a").status_code == 200
    assert submit(alice, "warmup", "b").status_code == 200
    assert submit(alice, "warmup", "c").status_code == 429


def test_filters(client):
    make_flag(client)
    make_flag(client, slug="learn-one", tier="learn")
    assert [c["slug"] for c in client.get("/api/challenges?tier=learn").json()["challenges"]] \
        == ["learn-one"]
    assert client.get("/api/challenges?track=steering").json()["challenges"] == []


def test_validation_errors_use_the_error_shape(client):
    r = client.post("/api/admin/challenges", json={"slug": "Bad Slug", "title": "x",
                                                    "kind": "flag", "flag": "f"},
                    headers=ADMIN)
    assert r.status_code == 400
    assert "slug" in r.json()["error"]


def test_leaderboard_by_season(client, login, db):
    make_flag(client)
    first = db.get_active_season()["id"]
    submit(login("a"), "warmup", "nuais{hello}")
    db.create_season("Season 2")
    second = db.get_active_season()["id"]
    assert client.get("/api/leaderboard").json() == {
        "season": {"id": second, "name": "Season 2"}, "teams": []}
    old = client.get(f"/api/leaderboard?season={first}").json()
    assert old["season"] == {"id": first, "name": "Season 1"}
    assert old["teams"][0]["team"] == "a"
    assert client.get("/api/leaderboard?season=999").status_code == 404
    seasons = client.get("/api/season").json()["seasons"]
    assert [(s["name"], s["active"]) for s in seasons] == [("Season 1", False),
                                                           ("Season 2", True)]
