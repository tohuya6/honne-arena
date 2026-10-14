import re

import pytest
from fastapi.testclient import TestClient

from app.challenges import steering
from app.config import Settings
from app.db import Database
from app.email_send import ConsoleSender
from app.main import create_app

ADMIN = {"X-Admin-Token": "admin-secret"}
ADMIN_EMAIL = "boss@northeastern.edu"
TOKEN_RE = re.compile(r"verify\.html\?token=([A-Za-z0-9_\-]+)")


@pytest.fixture()
def cfg(tmp_path):
    # Never a real NDIF key in tests. ndif_enabled is flipped per test when needed,
    # and steering tests always replace the scorer factory with a fake.
    # The per-IP sign-in limit is raised because every test client shares one IP.
    return Settings(_env_file=None, db_path=":memory:", admin_token="admin-secret",
                    ip_hash_salt="salt", ndif_enabled=False, ndif_api_key="",
                    cache_dir=str(tmp_path / "cache"), public_dir=str(tmp_path / "public"),
                    admin_emails=ADMIN_EMAIL, email_backend="console",
                    dev_expose_links=False, cookie_secure=False, login_per_ip_hour=1000,
                    public_base_url="http://arena.test")


@pytest.fixture()
def db():
    d = Database(":memory:")
    d.migrate()
    d.create_season("Season 1")
    return d


@pytest.fixture(autouse=True)
def _reset_scorers():
    steering.reset_scorers()
    yield
    steering.reset_scorers()


@pytest.fixture()
def outbox():
    return []


@pytest.fixture()
def app(cfg, db, outbox):
    return create_app(cfg, db=db, mailer=ConsoleSender(outbox=outbox))


@pytest.fixture()
def client(app):
    """Signed out."""
    return TestClient(app)


def last_token(outbox) -> str:
    return TOKEN_RE.search(outbox[-1]["body"]).group(1)


def sign_in(app, email: str, handle: str | None = None,
            client: TestClient | None = None) -> TestClient:
    """Sign in through the real flow: request a link, read it from the captured
    outbox, POST the token, then set a handle. Returns a client holding the cookie."""
    c = client or TestClient(app)
    r = c.post("/api/auth/request", json={"email": email})
    assert r.status_code == 200, r.text
    r = c.post("/api/auth/verify", json={"token": last_token(app.state.mailer.outbox)})
    assert r.status_code == 200, r.text
    if handle:
        r = c.post("/api/me/handle", json={"handle": handle})
        assert r.status_code == 200, r.text
    return c


@pytest.fixture()
def login(app):
    """login("alice") -> a client signed in as alice@northeastern.edu with handle alice."""
    def _login(handle: str = "alice", email: str | None = None) -> TestClient:
        email = email or re.sub(r"[^a-z0-9]", "", handle.lower()) + "@northeastern.edu"
        return sign_in(app, email, handle)
    return _login


@pytest.fixture()
def admin_client(login):
    return login("boss", ADMIN_EMAIL)


def submit(c: TestClient, slug: str, answer: str):
    return c.post("/api/submit", json={"challenge": slug, "answer": answer})


def make_flag(c: TestClient, slug="warmup", flag="nuais{hello}", **kw):
    body = {"slug": slug, "title": "Warmup", "kind": "flag", "flag": flag,
            "track": "ctf", "tier": "compete", "config": {"flag_format": "nuais{...}"},
            "points_initial": 500, "points_min": 100, "points_decay": 2, **kw}
    r = c.post("/api/admin/challenges", json=body, headers=ADMIN)
    assert r.status_code == 200, r.text
    return r.json()
