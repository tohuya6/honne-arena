"""Admin API: who may use it, PATCH, listing without secrets, usage."""

from app.challenges.steering import SteeringChallenge
from tests.conftest import ADMIN, make_flag, submit

FLAG = {"slug": "warmup", "title": "Warmup", "kind": "flag", "flag": "nuais{hello}",
        "track": "ctf"}


def test_admin_by_token(client):
    assert client.post("/api/admin/challenges", json=FLAG, headers=ADMIN).status_code == 200


def test_admin_by_session(admin_client):
    assert admin_client.post("/api/admin/challenges", json=FLAG).status_code == 200
    assert admin_client.get("/api/admin/usage").status_code == 200


def test_not_admin(client, login):
    alice = login("alice")
    assert alice.post("/api/admin/challenges", json=FLAG).status_code == 403
    assert alice.get("/api/admin/challenges").status_code == 403
    assert client.get("/api/admin/usage").status_code == 403
    bad = {"X-Admin-Token": "wrong"}
    assert client.post("/api/admin/seasons", json={"name": "x"}, headers=bad).status_code == 403


def test_empty_admin_token_never_matches(client, cfg):
    cfg.admin_token = ""
    r = client.post("/api/admin/challenges", json=FLAG, headers={"X-Admin-Token": ""})
    assert r.status_code == 403


def test_create_season(client):
    r = client.post("/api/admin/seasons", json={"name": "Spring", "active": True},
                    headers=ADMIN).json()
    assert r["name"] == "Spring" and r["active"] is True
    assert client.get("/api/season").json()["name"] == "Spring"


def test_create_validates_through_the_kind(client):
    bad = {"slug": "liar", "title": "x", "kind": "predictions", "config": {}}
    r = client.post("/api/admin/challenges", json=bad, headers=ADMIN)
    assert r.status_code == 400 and "labels" in r.json()["error"]
    r = client.post("/api/admin/challenges", json={**FLAG, "kind": "nope"}, headers=ADMIN)
    assert r.status_code == 400
    r = client.post("/api/admin/challenges", json={**FLAG, "flag": None}, headers=ADMIN)
    assert r.status_code == 400
    r = client.post("/api/admin/challenges", json={**FLAG, "points_min": 900}, headers=ADMIN)
    assert r.status_code == 400
    r = client.post("/api/admin/challenges", json={**FLAG, "closes_at": "soon"}, headers=ADMIN)
    assert r.status_code == 400
    assert client.post("/api/admin/challenges", json=FLAG, headers=ADMIN).status_code == 200
    assert client.post("/api/admin/challenges", json=FLAG, headers=ADMIN).status_code == 409


def test_list_includes_hidden_and_never_secrets(client):
    make_flag(client, hidden=True, hints=[{"body": "think", "cost": 5}])
    preds = {"slug": "liar", "title": "Liar", "kind": "predictions", "track": "lie-detection",
             "config": {"labels": {"t1": 1, "t2": 0}, "metric": "auroc"}}
    client.post("/api/admin/challenges", json=preds, headers=ADMIN)
    on_disk = {"slug": "liar2", "title": "Liar 2", "kind": "predictions",
               "config": {"labels_file": "data/private/liar2.json",
                          "tests": [{"in": "1", "out": "2"}]}}
    client.post("/api/admin/challenges", json=on_disk, headers=ADMIN)
    rows = {r["slug"]: r for r in
            client.get("/api/admin/challenges", headers=ADMIN).json()["challenges"]}
    assert set(rows) == {"warmup", "liar", "liar2"}
    w = rows["warmup"]
    assert w["hidden"] is True and w["points_decay"] == 2 and w["points_min"] == 100
    assert w["hints"] == [{"position": 1, "body": "think", "cost": 5}]
    assert "flag_sha256" not in str(rows)
    assert "labels" not in rows["liar"]["config"] and rows["liar"]["config"]["metric"] == "auroc"
    assert rows["liar2"]["config"] == {"labels_file": "data/private/liar2.json"}
    assert "t1" not in str(rows)


def test_patch(client, login):
    make_flag(client)
    r = client.patch("/api/admin/challenges/warmup",
                     json={"title": "Warm-up", "description": "New text", "points_initial": 300,
                           "points_min": 50, "points_decay": 5}, headers=ADMIN)
    assert r.json() == {"slug": "warmup"}
    view = client.get("/api/challenges/warmup").json()
    assert view["title"] == "Warm-up" and view["description"] == "New text"
    assert view["points"] == 300
    client.patch("/api/admin/challenges/warmup", json={"hidden": True}, headers=ADMIN)
    assert client.get("/api/challenges/warmup").status_code == 404
    client.patch("/api/admin/challenges/warmup", json={"hidden": False}, headers=ADMIN)
    assert client.get("/api/challenges/warmup").status_code == 200


def test_patch_validation(client, admin_client):
    make_flag(client)
    url = "/api/admin/challenges/warmup"
    assert client.patch(url, json={"title": "x"}).status_code == 403
    assert admin_client.patch(url, json={"title": None}).status_code == 400
    assert admin_client.patch(url, json={"points_min": 9999}).status_code == 400
    assert admin_client.patch(url, json={"closes_at": "tomorrow"}).status_code == 400
    assert admin_client.patch("/api/admin/challenges/nope", json={}).status_code == 404
    assert admin_client.patch(url, json={"closes_at": "2026-11-01T23:59:00-04:00"}).status_code \
        == 200
    assert client.get("/api/challenges/warmup").json()["closes_at"] == \
        "2026-11-02T03:59:00+00:00"


def test_usage(client, login, cfg):
    make_flag(client)
    alice = login("alice")
    login("bob")
    submit(alice, "warmup", "nope")
    submit(alice, "warmup", "nuais{hello}")
    usage = client.get("/api/admin/usage", headers=ADMIN).json()
    assert usage == {"ndif": {"enabled": False, "calls_today": 0,
                              "cap_per_day": cfg.ndif_calls_per_day,
                              "cap_per_player_day": cfg.ndif_player_per_day},
                     "submissions_today": 2, "users": 2, "local_jobs_in_flight": 0}


def test_usage_counts_local_jobs_in_flight(client, app, login, monkeypatch):
    seen = []

    def factory(challenge, settings, on_remote_call):
        def score_fn(seq):
            seen.append(client.get("/api/admin/usage", headers=ADMIN)
                        .json()["local_jobs_in_flight"])
            return 1.0
        return (lambda text: 1), score_fn
    monkeypatch.setattr(SteeringChallenge, "scorer_factory", staticmethod(factory))
    body = {"slug": "steer-local", "title": "Steer", "kind": "steering", "track": "steering",
            "config": {"model_id": "gpt2", "layer": 6, "d_file": "x.npz",
                       "probe_file": "x.json", "backend": "local"}}
    assert client.post("/api/admin/challenges", json=body, headers=ADMIN).status_code == 200
    assert submit(login("alice"), "steer-local", "be kind").status_code == 200
    assert seen == [1]
    assert client.get("/api/admin/usage", headers=ADMIN).json()["local_jobs_in_flight"] == 0
