"""The live feed behind /ws: solves and scored entries, pushed as they happen.

Submissions are judged in FastAPI's threadpool, but sockets live on the event loop.
Each subscriber records the loop it was created on. `publish()` may be called from any
thread: it hands the event to that loop with call_soon_threadsafe, and the loop puts it
in the client's bounded queue. A client that falls a full queue behind is dropped
rather than letting memory grow, and its socket is closed. Clients refetch what
changed, so a dropped client loses nothing it cannot get back by reconnecting.

In the server every socket shares one loop. Recording it per subscriber also keeps
the feed correct where that is not true (the test client runs one loop per socket).
"""

from __future__ import annotations

import asyncio
import threading


class Subscriber:
    def __init__(self, size: int, loop: asyncio.AbstractEventLoop):
        self.queue: asyncio.Queue = asyncio.Queue(maxsize=size)
        self.loop = loop

    async def next(self) -> dict | None:
        """The next event, or None once this client has been dropped."""
        return await self.queue.get()


class Broadcaster:
    def __init__(self, queue_size: int = 100, max_clients: int = 500):
        self.queue_size = queue_size
        self.max_clients = max_clients
        self._subs: set[Subscriber] = set()
        self._lock = threading.Lock()

    @property
    def clients(self) -> int:
        with self._lock:
            return len(self._subs)

    def subscribe(self) -> Subscriber | None:
        """Call on the event loop. Returns None when the server is at max_clients."""
        with self._lock:
            if len(self._subs) >= self.max_clients:
                return None
            sub = Subscriber(self.queue_size, asyncio.get_running_loop())
            self._subs.add(sub)
        return sub

    def unsubscribe(self, sub: Subscriber) -> None:
        with self._lock:
            self._subs.discard(sub)

    def publish(self, event: dict) -> None:
        """Thread-safe and never blocks."""
        with self._lock:
            subs = list(self._subs)
        for sub in subs:
            try:
                sub.loop.call_soon_threadsafe(self._deliver, sub, event)
            except RuntimeError:        # that loop has closed, so the socket is gone
                self.unsubscribe(sub)

    def _deliver(self, sub: Subscriber, event: dict) -> None:
        # Runs on the subscriber's own loop, so touching its queue is safe.
        try:
            sub.queue.put_nowait(event)
        except asyncio.QueueFull:
            self.unsubscribe(sub)
            while not sub.queue.empty():
                sub.queue.get_nowait()
            sub.queue.put_nowait(None)      # tells the socket handler to close
