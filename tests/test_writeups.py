"""Closing challenges and writeups."""

from tests.conftest import ADMIN, make_flag, submit

PAST = "2020-01-01T00:00:00Z"
FUTURE = "2999-01-01T00:00:00Z"


def _close(client, when=PAST, slug="warmup"):
    r = client.patch(f"/api/admin/challenges/{slug}", json={"closes_at": when}, headers=ADMIN)
    assert r.status_code == 200, r.text


def _post(c, body, slug="warmup"):
    return c.post(f"/api/challenges/{slug}/writeup", json={"body": body})


def test_submit_after_close_is_403(client, login):
    make_flag(client)
    _close(client)
    view = client.get("/api/challenges/warmup").json()
    assert view["closed"] is True and view["closes_at"] == "2020-01-01T00:00:00+00:00"
    r = submit(login("alice"), "warmup", "nuais{hello}")
    assert r.status_code == 403


def test_open_until_closes_at(client, login):
    make_flag(client, closes_at=FUTURE)
    view = client.get("/api/challenges/warmup").json()
    assert view["closed"] is False
    assert submit(login("alice"), "warmup", "nuais{hello}").json()["correct"] is True


def test_writeup_rules(client, login, db):
    make_flag(client)
    alice, bob = login("alice"), login("bob")
    submit(alice, "warmup", "nuais{nope}")            # a wrong answer counts as submitting
    assert _post(alice, "I read the source.").status_code == 403     # not closed yet
    _close(client)
    assert _post(alice, "I read the source.").json() == {"ok": True}
    assert _post(bob, "me too").status_code == 403                   # never submitted
    assert _post(client, "anon").status_code == 401
    assert _post(alice, "   ").status_code == 400
    assert _post(alice, "x" * 20_001).status_code == 400
    assert _post(alice, "x" * 20_000).status_code == 200
    assert _post(alice, "Second version.").status_code == 200        # replaces
    writeups = client.get("/api/challenges/warmup").json()["writeups"]
    assert [(w["handle"], w["body"]) for w in writeups] == [("alice", "Second version.")]
    assert db._count("select count(*) from writeups") == 1


def test_writeups_hidden_while_open(client, login):
    make_flag(client)
    alice = login("alice")
    submit(alice, "warmup", "x")
    _close(client)
    _post(alice, "how I did it")
    assert len(client.get("/api/challenges/warmup").json()["writeups"]) == 1
    # Reopening hides them again.
    client.patch("/api/admin/challenges/warmup", json={"closes_at": None}, headers=ADMIN)
    view = client.get("/api/challenges/warmup").json()
    assert view["closed"] is False and view["writeups"] == []


def test_closed_challenge_still_scores(client, login):
    make_flag(client)
    submit(login("alice"), "warmup", "nuais{hello}")
    _close(client)
    assert client.get("/api/leaderboard").json()["teams"][0]["points"] == 500
