"""ScoringGate call deadline — a stuck NDIF job must not hold a slot forever.

Regression for 2026-10-07/08: two NDIF jobs sat in RECEIVED for 20h, nnsight polled them
with no overall timeout, both gate slots stayed taken, and every /generate failed.
No NDIF, no network: the "stuck job" is a function blocked on an Event.
"""

import threading
import time

import pytest

from app.errors import BackendTimeout, QueueFull, ScoringUnavailable
from app.queue import ScoringGate


def test_normal_call_returns_and_frees_slot():
    g = ScoringGate(1, acquire_timeout_s=0.1, call_timeout_s=2)
    assert g.run(lambda x: x + 1, 1) == 2
    assert g.run(lambda: "again") == "again"   # slot was released
    assert g.abandoned == 0


def test_exception_propagates_and_frees_slot():
    g = ScoringGate(1, acquire_timeout_s=0.1, call_timeout_s=2)
    with pytest.raises(ValueError):
        g.run(lambda: (_ for _ in ()).throw(ValueError("ndif said no")))
    assert g.run(lambda: 7) == 7
    assert g.abandoned == 0


def test_stuck_call_times_out_frees_slot_and_is_counted():
    g = ScoringGate(1, acquire_timeout_s=0.1, call_timeout_s=0.2)
    release = threading.Event()
    t0 = time.monotonic()
    with pytest.raises(BackendTimeout) as ei:
        g.run(release.wait)
    assert time.monotonic() - t0 < 1.0
    assert ei.value.http_status == 504
    assert g.abandoned == 1
    assert g.run(lambda: "next") == "next"     # the only slot is free again
    release.set()                              # the stuck job finally returns
    for _ in range(100):
        if g.abandoned == 0:
            break
        time.sleep(0.01)
    assert g.abandoned == 0


def test_too_many_abandoned_fails_fast_without_calling():
    g = ScoringGate(1, acquire_timeout_s=0.1, call_timeout_s=0.05, max_abandoned=2)
    release = threading.Event()
    for _ in range(2):
        with pytest.raises(BackendTimeout):
            g.run(release.wait)
    called = []
    t0 = time.monotonic()
    with pytest.raises(ScoringUnavailable):
        g.run(lambda: called.append(1))
    assert time.monotonic() - t0 < 0.05 and not called
    release.set()


def test_queue_full_still_raised_while_a_call_is_running():
    g = ScoringGate(1, acquire_timeout_s=0.05, call_timeout_s=5)
    release = threading.Event()
    th = threading.Thread(target=lambda: g.run(release.wait))
    th.start()
    time.sleep(0.05)
    with pytest.raises(QueueFull):
        g.run(lambda: None)
    release.set()
    th.join(2)


def test_no_deadline_keeps_old_behaviour():
    g = ScoringGate(1, acquire_timeout_s=0.1)
    assert g.run(lambda: 3) == 3
