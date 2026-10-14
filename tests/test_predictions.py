"""Hidden-label prediction challenges (the lie-detection shape). No model calls."""

import numpy as np
import pytest

from app.challenges.predictions import auroc
from tests.conftest import ADMIN

LABELS = {"t1": 1, "t2": 1, "t3": 0, "t4": 0}


def test_auroc_matches_pairwise_definition():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 200)
    s = np.round(rng.random(200), 1)        # rounding forces ties
    pos, neg = s[y == 1], s[y == 0]
    pairs = (pos[:, None] > neg[None, :]).mean() + 0.5 * (pos[:, None] == neg[None, :]).mean()
    assert auroc(y, s) == pytest.approx(pairs)


def _make(client, **cfg):
    body = {"slug": "liar", "title": "Spot the liar", "kind": "predictions",
            "track": "lie-detection", "config": {"labels": LABELS, **cfg}}
    assert client.post("/api/admin/challenges", json=body, headers=ADMIN).status_code == 200


@pytest.fixture()
def as_player(login):
    clients = {}

    def get(handle):
        if handle not in clients:
            clients[handle] = login(handle)
        return clients[handle]
    return get


def _submit(as_player, handle, text):
    return as_player(handle).post("/api/submit", json={"challenge": "liar", "answer": text})


def test_perfect_and_inverted(client, as_player):
    _make(client)
    assert _submit(as_player, "a", "t1,0.9\nt2,0.8\nt3,0.1\nt4,0.2").json()["score"] == 1.0
    assert _submit(as_player, "b", "t1,0.1\nt2,0.2\nt3,0.9\nt4,0.8").json()["score"] == 0.0
    board = client.get("/api/challenges/liar").json()["board"]
    assert [r["handle"] for r in board] == ["a", "b"]
    assert board[0]["answer"] is None      # predictions are not shown on the board


def test_reordered_duplicate_is_rejected(client, as_player):
    _make(client)
    assert _submit(as_player, "a", "t1,0.9\nt2,0.8\nt3,0.1\nt4,0.2").status_code == 200
    assert _submit(as_player, "b", "# copied\nt4,0.2\nt3,0.1\nt2,0.8\nt1,0.9").status_code == 409


@pytest.mark.parametrize("text", [
    "t1,0.9\nt2,0.8\nt3,0.1",              # missing an id
    "t1,0.9\nt2,0.8\nt3,0.1\nt4,0.2\nt9,0.5",
    "t1,1.5\nt2,0.8\nt3,0.1\nt4,0.2",      # out of range
    "t1,0.9\nt1,0.8\nt3,0.1\nt4,0.2",      # repeated id
    "t1;0.9",
])
def test_malformed(client, as_player, text):
    _make(client)
    assert _submit(as_player, "a", text).status_code == 400


def test_accuracy_metric(client, as_player):
    _make(client, metric="accuracy")
    assert _submit(as_player, "a", "t1,0.9\nt2,0.4\nt3,0.1\nt4,0.2").json()["score"] == 0.75
