"""Steering challenges and the NDIF budget, with a fake scorer.

The point of these tests is the spend: no remote call happens when scoring is off,
over budget, or a repeat, and every call that does happen lands in the ledger."""

import pytest

from app.challenges.steering import SteeringChallenge
from tests.conftest import ADMIN

CONFIG = {"model_id": "meta-llama/Llama-3.1-8B", "layer": 16,
          "d_file": "unused.npz", "probe_file": "unused.json", "token_budget": 8}


def _fake_factory(calls):
    def factory(challenge, settings, on_remote_call):
        on_remote_call()                    # the one-time probe baseline
        calls["builds"] += 1

        def count_tokens(text):
            return len(text.split())

        def score_fn(seq):
            on_remote_call()                # one batched forward per submission
            calls["scores"] += 1
            return float(len(seq))
        return count_tokens, score_fn
    return factory


def _setup(client, monkeypatch):
    calls = {"builds": 0, "scores": 0}
    monkeypatch.setattr(SteeringChallenge, "scorer_factory", staticmethod(_fake_factory(calls)))
    body = {"slug": "steer", "title": "Steer it", "kind": "steering", "track": "steering",
            "config": CONFIG}
    assert client.post("/api/admin/challenges", json=body, headers=ADMIN).status_code == 200
    return calls


@pytest.fixture()
def as_player(login):
    """as_player("a") -> a signed-in client for handle a, made once per test."""
    clients = {}

    def get(handle):
        if handle not in clients:
            clients[handle] = login(handle)
        return clients[handle]
    return get


def _submit(as_player, handle, text, slug="steer"):
    return as_player(handle).post("/api/submit", json={"challenge": slug, "answer": text})


def test_off_by_default_makes_no_calls(client, as_player, monkeypatch, db):
    calls = _setup(client, monkeypatch)
    r = _submit(as_player, "a", "be kind")
    assert r.status_code == 503
    assert calls == {"builds": 0, "scores": 0}
    assert db.count_ndif_since("0") == 0


def test_scores_and_logs_every_call(client, as_player, cfg, monkeypatch, db):
    cfg.ndif_enabled = True
    calls = _setup(client, monkeypatch)
    assert _submit(as_player, "a", "be kind").json()["score"] == 7.0
    assert _submit(as_player, "b", "be very kind").json()["score"] == 12.0
    assert calls == {"builds": 1, "scores": 2}           # baseline built once
    assert db.count_ndif_since("0") == 3                 # baseline + 2 submissions
    assert db.count_ndif_since("0", player_id=None) == 3
    user = db.get_user_by_handle("a")
    a = db.get_player_by_user(db.get_active_season()["id"], user["id"])
    assert db.count_ndif_since("0", player_id=a["id"]) == 1   # baseline not charged to a


def test_repeat_answer_is_not_rescored(client, as_player, cfg, monkeypatch):
    cfg.ndif_enabled = True
    calls = _setup(client, monkeypatch)
    assert _submit(as_player, "a", "be kind").status_code == 200
    assert _submit(as_player, "b", "  BE   kind ").status_code == 409
    assert calls["scores"] == 1


def test_global_budget_stops_calls(client, as_player, cfg, monkeypatch):
    cfg.ndif_enabled = True
    cfg.ndif_calls_per_day = 2
    calls = _setup(client, monkeypatch)
    assert _submit(as_player, "a", "one").status_code == 200     # baseline + 1 = 2
    assert _submit(as_player, "b", "two").status_code == 429
    assert calls["scores"] == 1


def test_player_budget(client, as_player, cfg, monkeypatch):
    cfg.ndif_enabled = True
    cfg.ndif_player_per_min = 1
    _setup(client, monkeypatch)
    assert _submit(as_player, "a", "one").status_code == 200
    assert _submit(as_player, "a", "two").status_code == 429
    assert _submit(as_player, "b", "two").status_code == 200


def test_local_backend_never_touches_ndif(client, as_player, cfg, monkeypatch, db):
    # NDIF switched off, yet a "local" steering challenge still scores, through the
    # local gate, with nothing written to the NDIF ledger.
    calls = {"builds": 0, "scores": 0}
    monkeypatch.setattr(SteeringChallenge, "scorer_factory", staticmethod(_fake_factory(calls)))
    body = {"slug": "steer-local", "title": "Steer GPT-2", "kind": "steering",
            "track": "steering", "tier": "learn", "config": {**CONFIG, "backend": "local"}}
    assert client.post("/api/admin/challenges", json=body, headers=ADMIN).status_code == 200
    r = _submit(as_player, "a", "be kind", slug="steer-local")
    assert r.status_code == 200, r.text
    assert calls == {"builds": 1, "scores": 1}
    assert db.count_ndif_since("0") == 0


def test_token_budget(client, as_player, cfg, monkeypatch):
    cfg.ndif_enabled = True
    _setup(client, monkeypatch)
    r = _submit(as_player, "a", "w " * 9)
    assert r.status_code == 400
    assert "budget" in r.json()["error"]
