"""Teams: solo team on first action, create/join from the session, the switch rule."""

from tests.conftest import make_flag, submit


def test_first_action_makes_a_solo_team(client, login):
    make_flag(client)
    alice = login("alice")
    assert alice.get("/api/me").json()["team"] is None
    submit(alice, "warmup", "nuais{nope}")
    team = alice.get("/api/me").json()["team"]
    assert team["name"] == "alice" and team["members"] == ["alice"] and team["join_code"]


def test_solo_team_name_gets_a_suffix_when_taken(client, login):
    make_flag(client)
    owner = login("owner")
    owner.post("/api/teams", json={"name": "Alice"})       # a team called Alice exists
    alice = login("alice")
    submit(alice, "warmup", "x")
    assert alice.get("/api/me").json()["team"]["name"] == "alice 2"


def test_create_and_join(login):
    alice, bob = login("alice"), login("bob")
    r = alice.post("/api/teams", json={"name": "probes"}).json()
    assert r["team"] == "probes" and r["join_code"]
    joined = bob.post("/api/teams/join", json={"join_code": r["join_code"]}).json()
    assert joined == {"team": "probes", "members": ["alice", "bob"]}
    assert bob.get("/api/me").json()["team"]["members"] == ["alice", "bob"]


def test_team_names_unique_ignoring_case(login):
    login("alice").post("/api/teams", json={"name": "Probes"})
    assert login("bob").post("/api/teams", json={"name": "probes"}).status_code == 409


def test_teams_need_sign_in_and_handle(app, client):
    from tests.conftest import sign_in
    assert client.post("/api/teams", json={"name": "t"}).status_code == 401
    assert client.post("/api/teams/join", json={"join_code": "x"}).status_code == 401
    c = sign_in(app, "nohandle@northeastern.edu")
    assert c.post("/api/teams", json={"name": "t"}).status_code == 400


def test_bad_join_code(login):
    assert login("alice").post("/api/teams/join", json={"join_code": "nope"}).status_code == 404


def test_cannot_switch_team_after_submitting(client, login):
    make_flag(client)
    alice, bob = login("alice"), login("bob")
    submit(alice, "warmup", "nuais{nope}")              # even a wrong answer locks it
    code = bob.post("/api/teams", json={"name": "t2"}).json()["join_code"]
    assert alice.post("/api/teams/join", json={"join_code": code}).status_code == 400
    assert alice.post("/api/teams", json={"name": "t3"}).status_code == 400
    assert alice.get("/api/me").json()["team"]["name"] == "alice"


def test_can_switch_before_submitting(client, login):
    make_flag(client)
    alice, bob = login("alice"), login("bob")
    code1 = alice.post("/api/teams", json={"name": "one"}).json()["join_code"]
    code2 = bob.post("/api/teams", json={"name": "two"}).json()["join_code"]
    carol = login("carol")
    assert carol.post("/api/teams/join", json={"join_code": code1}).status_code == 200
    assert carol.post("/api/teams/join", json={"join_code": code2}).status_code == 200
    assert carol.get("/api/me").json()["team"]["name"] == "two"


def test_left_behind_solo_team_is_removed(client, login, db):
    alice, bob = login("alice"), login("bob")
    code = alice.post("/api/teams", json={"name": "probes"}).json()["join_code"]
    bob.post("/api/teams/join", json={"join_code": code})
    names = [t["name"] for t in db.list_teams(db.get_active_season()["id"])]
    assert names == ["probes"]


def test_new_season_starts_fresh(client, login, db):
    make_flag(client)
    alice = login("alice")
    alice.post("/api/teams", json={"name": "probes"})
    submit(alice, "warmup", "x")
    db.create_season("Season 2")
    assert alice.get("/api/me").json()["team"] is None
    alice.post("/api/teams", json={"name": "new crew"})
    assert alice.get("/api/me").json()["team"]["name"] == "new crew"
