"""Solve-mode kinds can add detail to the submit result (the `code` kind adds verdict,
passed, total, failed_test, time_ms). A fake kind stands in, since real code runs
need Docker. It also checks that "local" judging goes through the local gate."""

import pytest

from app import challenges
from app.challenges.base import Challenge, Outcome
from app.email_send import ConsoleSender
from app.main import create_app
from app.queue import ScoringGate
from tests.conftest import ADMIN, submit


class RecordingGate(ScoringGate):
    def __init__(self):
        super().__init__(1)
        self.calls = 0

    def run(self, fn, *args, **kwargs):
        self.calls += 1
        return super().run(fn, *args, **kwargs)


class FakeCode(Challenge):
    kind = "fake-code"
    mode = "solve"
    compute = "local"
    runs = []

    def evaluate(self, answer):
        ok = answer.strip() == "print(42)"
        FakeCode.runs.append(answer)
        return Outcome(correct=ok, score=None, detail={
            "verdict": "accepted" if ok else "wrong answer", "passed": 3 if ok else 1,
            "total": 3, "failed_test": None if ok else 2, "time_ms": 12})


@pytest.fixture()
def app(cfg, db, outbox):
    """This module's app records every job sent through the local gate."""
    return create_app(cfg, db=db, mailer=ConsoleSender(outbox=outbox),
                      local_gate=RecordingGate())


@pytest.fixture()
def fake_kind(monkeypatch):
    FakeCode.runs = []
    monkeypatch.setitem(challenges.KINDS, FakeCode.kind, FakeCode)


def test_detail_is_merged_into_solve_results(app, client, login, fake_kind):
    body = {"slug": "add", "title": "Add", "kind": "fake-code", "track": "cp"}
    assert client.post("/api/admin/challenges", json=body, headers=ADMIN).status_code == 200
    alice = login("alice")
    with client.websocket_connect("/ws") as ws:
        wrong = submit(alice, "add", "print(41)").json()
        right = submit(alice, "add", "print(42)").json()
        event = ws.receive_json()
    assert wrong == {"challenge": "add", "mode": "solve", "correct": False,
                     "first_blood": False, "verdict": "wrong answer", "passed": 1,
                     "total": 3, "failed_test": 2, "time_ms": 12}
    assert right["correct"] is True and right["first_blood"] is True
    assert right["verdict"] == "accepted" and right["passed"] == 3
    assert event["type"] == "solve" and event["handle"] == "alice"
    assert FakeCode.runs == ["print(41)", "print(42)"]
    assert app.state.local_gate.calls == 2          # judged behind the local gate
    assert app.state.gate.abandoned == 0
