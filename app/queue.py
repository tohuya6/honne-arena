"""Bounded scoring gate — caps concurrent NDIF forward passes so a burst of
submissions can't exceed NDIF's limits or stampede the Space (PROJECT_SPEC.md §9).

FastAPI runs sync endpoints in a threadpool, so a threading semaphore is enough.

Every call also gets a deadline. nnsight 0.7.0's remote backend polls a job in a
`while True` loop with no overall timeout, so a job NDIF never schedules (stuck in
RECEIVED — seen 2026-10-07/08, two such jobs held both slots for 20h and every
request failed with QueueFull) would otherwise hold its slot forever. Past the
deadline the slot is released and the caller gets BackendTimeout; the stuck call
is abandoned in a daemon thread. Abandoned calls are capped: once `max_abandoned`
are still alive the gate fails fast rather than piling up threads and NDIF jobs.
"""

from __future__ import annotations

import threading
from concurrent.futures import Future, TimeoutError as FutureTimeout
from typing import Callable, TypeVar

from app.errors import BackendTimeout, QueueFull, ScoringUnavailable

T = TypeVar("T")


class ScoringGate:
    def __init__(self, concurrency: int, acquire_timeout_s: float = 30.0,
                 call_timeout_s: float | None = None, max_abandoned: int | None = None):
        self._sem = threading.BoundedSemaphore(max(1, concurrency))
        self._timeout = acquire_timeout_s
        self._call_timeout = call_timeout_s
        self._max_abandoned = max_abandoned if max_abandoned is not None else 2 * max(1, concurrency)
        self._abandoned = 0
        self._lock = threading.Lock()

    @property
    def abandoned(self) -> int:
        with self._lock:
            return self._abandoned

    def run(self, fn: Callable[..., T], *args, **kwargs) -> T:
        if self._call_timeout is not None and self.abandoned >= self._max_abandoned:
            raise ScoringUnavailable("The model backend (NDIF) is not answering right now — "
                                     "please try again later.")
        acquired = self._sem.acquire(timeout=self._timeout)
        if not acquired:
            raise QueueFull("Scoring is busy right now — please try again in a moment.")
        if self._call_timeout is None:
            try:
                return fn(*args, **kwargs)
            finally:
                self._sem.release()

        fut: Future = Future()
        state = {"abandoned": False}

        def worker():
            try:
                fut.set_result(fn(*args, **kwargs))
            except BaseException as e:  # noqa: BLE001 — re-raised in the caller
                fut.set_exception(e)
            finally:
                with self._lock:
                    if state["abandoned"]:
                        self._abandoned -= 1

        threading.Thread(target=worker, daemon=True, name="ndif-call").start()
        try:
            return fut.result(timeout=self._call_timeout)
        except FutureTimeout:
            with self._lock:
                if not fut.done():
                    state["abandoned"] = True
                    self._abandoned += 1
            if fut.done():          # finished in the instant after the timeout
                return fut.result()
            raise BackendTimeout(f"The model backend (NDIF) did not answer within "
                                 f"{self._call_timeout:.0f}s — its queue may be stalled. "
                                 f"Please try again later.") from None
        finally:
            self._sem.release()
