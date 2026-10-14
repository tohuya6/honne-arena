"""The live feed: solves and entries are pushed, wrong answers are not, slow clients
are dropped."""

import asyncio
import threading

from app.live import Broadcaster
from tests.conftest import ADMIN, make_flag, submit


def test_solve_is_broadcast_and_wrong_answer_is_not(client, login):
    make_flag(client)
    alice, bob = login("alice"), login("bob")
    with client.websocket_connect("/ws") as ws:
        assert submit(alice, "warmup", "nuais{nope}").json()["correct"] is False
        assert submit(alice, "warmup", "nuais{hello}").json()["correct"] is True
        submit(bob, "warmup", "nuais{hello}")
        # The wrong answer came first, so if it had been sent it would be first here.
        first = ws.receive_json()
        second = ws.receive_json()
    assert first["type"] == "solve" and first["challenge"] == "warmup"
    assert first["handle"] == "alice" and first["team"] == "alice"
    assert first["first_blood"] is True and first["at"]
    assert second["handle"] == "bob" and second["first_blood"] is False
    assert set(first) == {"type", "challenge", "handle", "team", "first_blood", "at"}


def test_scored_entry_is_broadcast(client, login):
    body = {"slug": "liar", "title": "Liar", "kind": "predictions", "track": "lie-detection",
            "config": {"labels": {"t1": 1, "t2": 0}}}
    assert client.post("/api/admin/challenges", json=body, headers=ADMIN).status_code == 200
    alice = login("alice")
    with client.websocket_connect("/ws") as ws:
        submit(alice, "liar", "t1,0.9\nt2,0.1")
        event = ws.receive_json()
    assert event["type"] == "entry" and event["score"] == 1.0
    assert event["handle"] == "alice" and event["challenge"] == "liar"
    assert "answer" not in event


def test_every_client_gets_the_event(client, login):
    make_flag(client)
    alice = login("alice")
    with client.websocket_connect("/ws") as one, client.websocket_connect("/ws") as two:
        submit(alice, "warmup", "nuais{hello}")
        assert one.receive_json()["handle"] == "alice"
        assert two.receive_json()["handle"] == "alice"


def test_publish_from_another_thread_and_drop_slow_clients():
    async def scenario():
        b = Broadcaster(queue_size=2)
        fast, slow = b.subscribe(), b.subscribe()
        got = []

        async def reader():
            while len(got) < 3:
                got.append(await fast.next())
        task = asyncio.create_task(reader())
        for i in range(3):
            # publish() is called from the threadpool in the real app.
            t = threading.Thread(target=b.publish, args=({"n": i},))
            t.start()
            t.join()
            await asyncio.sleep(0.01)
        await asyncio.wait_for(task, 1)
        return b, fast, slow, got

    b, fast, slow, got = asyncio.run(scenario())
    assert [e["n"] for e in got] == [0, 1, 2]
    assert b.clients == 1                           # the slow one was dropped
    assert slow.queue.get_nowait() is None          # and told to close


def test_publish_before_any_loop_is_a_noop():
    Broadcaster().publish({"type": "solve"})


def test_client_cap(client, cfg, db):
    from fastapi.testclient import TestClient
    from starlette.websockets import WebSocketDisconnect
    import pytest

    from app.main import create_app
    cfg.ws_max_clients = 1
    c = TestClient(create_app(cfg, db=db))
    with c.websocket_connect("/ws"):
        with pytest.raises(WebSocketDisconnect):
            with c.websocket_connect("/ws") as second:
                second.receive_json()
