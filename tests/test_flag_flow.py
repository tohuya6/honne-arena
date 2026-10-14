"""CTF flags end to end over HTTP: create, solve, decay, teams."""

from tests.conftest import ADMIN


def _make_flag(client, slug="warmup", flag="nuais{hello}", **kw):
    body = {"slug": slug, "title": "Warmup", "kind": "flag", "flag": flag,
            "track": "ctf", "tier": "learn", "config": {"flag_format": "nuais{...}"},
            "points_initial": 500, "points_min": 100, "points_decay": 2, **kw}
    r = client.post("/api/admin/challenges", json=body, headers=ADMIN)
    assert r.status_code == 200, r.text


def _submit(client, handle, answer, slug="warmup"):
    return client.post("/api/submit", json={"handle": handle, "challenge": slug,
                                            "answer": answer})


def test_admin_routes_need_token(client):
    r = client.post("/api/admin/challenges", json={"slug": "x", "title": "x", "kind": "flag",
                                                    "flag": "f"})
    assert r.status_code == 403


def test_flag_is_never_stored_in_plain_text(client, db):
    _make_flag(client)
    row = db.get_challenge(db.get_active_season()["id"], "warmup")
    assert "nuais{hello}" not in row["config"]
    pub = client.get("/api/challenges/warmup").json()
    assert "flag_sha256" not in str(pub)
    assert pub["config"] == {"flag_format": "nuais{...}"}


def test_wrong_then_right(client):
    _make_flag(client)
    assert _submit(client, "alice", "nuais{nope}").json() == {
        "challenge": "warmup", "mode": "solve", "correct": False}
    assert _submit(client, "alice", "  NUAIS{Hello} ").json()["correct"] is True  # case-insensitive


def test_team_solves_once(client):
    _make_flag(client)
    code = client.post("/api/teams", json={"handle": "alice", "name": "probes"}).json()["join_code"]
    assert client.post("/api/teams/join", json={"handle": "bob", "join_code": code}).status_code == 200
    assert _submit(client, "alice", "nuais{hello}").json()["correct"] is True
    r = _submit(client, "bob", "nuais{hello}")
    assert r.status_code == 409


def test_points_decay_for_everyone(client):
    _make_flag(client)
    for h in ("a", "b", "c"):
        assert _submit(client, h, "nuais{hello}").json()["correct"] is True
    view = client.get("/api/challenges/warmup").json()
    assert view["solves"] == 3
    assert view["points"] == 100           # decay=2: the third solve hits the floor
    assert [s["handle"] for s in view["board"]] == ["a", "b", "c"]   # first blood first
    teams = client.get("/api/leaderboard").json()["teams"]
    assert {t["team"]: t["points"] for t in teams} == {"a": 100, "b": 100, "c": 100}


def test_cannot_switch_team_after_submitting(client):
    _make_flag(client)
    _submit(client, "alice", "nuais{nope}")
    code = client.post("/api/teams", json={"handle": "bob", "name": "t2"}).json()["join_code"]
    r = client.post("/api/teams/join", json={"handle": "alice", "join_code": code})
    assert r.status_code == 400


def test_hidden_challenge_is_not_playable(client):
    _make_flag(client, hidden=True)
    assert client.get("/api/challenges").json()["challenges"] == []
    assert _submit(client, "alice", "nuais{hello}").status_code == 404


def test_rate_limit(client, cfg):
    cfg.rate_per_min = 2
    _make_flag(client)
    assert _submit(client, "alice", "a").status_code == 200
    assert _submit(client, "alice", "b").status_code == 200
    assert _submit(client, "alice", "c").status_code == 429


def test_bad_handle(client):
    _make_flag(client)
    assert _submit(client, "<script>", "x").status_code == 400
