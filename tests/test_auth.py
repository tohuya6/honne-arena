"""NU email sign-in: domains, tokens, cookies, sessions, handles."""

from datetime import datetime, timedelta, timezone

import pytest
from fastapi.testclient import TestClient

from app import auth
from app.email_send import EmailError
from app.main import create_app
from tests.conftest import last_token, sign_in, submit


def _request(c, email):
    return c.post("/api/auth/request", json={"email": email})


# ── domains ──

@pytest.mark.parametrize("email", [
    "x@gmail.com",
    "x@northeastern.edu.evil.com",
    "x@evil.northeastern.edu",
    "x@NORTHEASTERN.EDU.evil.com",
    "northeastern.edu",
    "x@",
    "@northeastern.edu",
    "x y@northeastern.edu",
    "x@northeastern.edu@gmail.com",
    "",
])
def test_rejects_other_domains_and_bad_syntax(client, outbox, email):
    r = _request(client, email)
    assert r.status_code == 400
    assert "error" in r.json()
    assert outbox == []


def test_uppercase_email_is_lowercased(client, outbox, db):
    assert _request(client, "  Alice.Smith@NorthEastern.EDU ").json() == {"ok": True}
    assert outbox[-1]["to"] == "alice.smith@northeastern.edu"
    r = client.post("/api/auth/verify", json={"token": last_token(outbox)})
    assert r.json()["user"]["email"] == "alice.smith@northeastern.edu"
    assert db.get_user_by_email("alice.smith@northeastern.edu")


def test_same_answer_for_new_and_existing_accounts(app, client, outbox):
    sign_in(app, "old@northeastern.edu")
    assert _request(client, "old@northeastern.edu").json() == {"ok": True}
    assert _request(client, "new@northeastern.edu").json() == {"ok": True}


# ── tokens ──

def test_link_points_at_verify_page(client, outbox):
    _request(client, "a@northeastern.edu")
    assert "http://arena.test/verify.html?token=" in outbox[-1]["body"]


def test_only_the_hash_is_stored(client, outbox, db):
    _request(client, "a@northeastern.edu")
    token = last_token(outbox)
    rows = db._all("select * from login_tokens")
    assert token not in str(rows)
    assert rows[0]["token_sha256"] == auth.sha256(token)


def test_token_is_single_use(client, outbox):
    _request(client, "a@northeastern.edu")
    token = last_token(outbox)
    assert client.post("/api/auth/verify", json={"token": token}).status_code == 200
    r = TestClient(client.app).post("/api/auth/verify", json={"token": token})
    assert r.status_code == 400


def test_token_expires(cfg, db, outbox):
    sent_at = datetime.now(timezone.utc) - timedelta(minutes=16)
    from app.email_send import ConsoleSender
    auth.request_link(db, ConsoleSender(outbox), cfg, email="a@northeastern.edu",
                      ip_hash="h", now=sent_at)
    with pytest.raises(Exception) as ei:
        auth.verify_link(db, cfg, token=last_token(outbox))
    assert ei.value.http_status == 400


def test_token_works_just_before_expiry(cfg, db, outbox):
    from app.email_send import ConsoleSender
    sent_at = datetime.now(timezone.utc) - timedelta(minutes=14)
    auth.request_link(db, ConsoleSender(outbox), cfg, email="a@northeastern.edu",
                      ip_hash="h", now=sent_at)
    user, _session = auth.verify_link(db, cfg, token=last_token(outbox))
    assert user["email"] == "a@northeastern.edu"


def test_unknown_token(client):
    assert client.post("/api/auth/verify", json={"token": "nope"}).status_code == 400
    assert client.post("/api/auth/verify", json={"token": ""}).status_code == 400


def test_no_get_route_consumes_a_token(client, outbox):
    _request(client, "a@northeastern.edu")
    token = last_token(outbox)
    # What a mail scanner would do with the link.
    client.get(f"/verify.html?token={token}")
    client.get(f"/api/auth/verify?token={token}")
    assert client.post("/api/auth/verify", json={"token": token}).status_code == 200


# ── rate limits ──

def test_per_email_limit(client):
    for _ in range(3):
        assert _request(client, "a@northeastern.edu").status_code == 200
    assert _request(client, "A@northeastern.edu").status_code == 429
    assert _request(client, "b@northeastern.edu").status_code == 200


def test_per_ip_limit(app, cfg, client):
    cfg.login_per_ip_hour = 2
    assert _request(client, "a@northeastern.edu").status_code == 200
    assert _request(client, "b@northeastern.edu").status_code == 200
    assert _request(client, "c@northeastern.edu").status_code == 429


# ── email backends ──

def test_dev_link_only_when_enabled_with_console(cfg, db, outbox):
    from app.email_send import ConsoleSender
    c = TestClient(create_app(cfg, db=db, mailer=ConsoleSender(outbox)))
    assert "dev_link" not in _request(c, "a@northeastern.edu").json()
    cfg.dev_expose_links = True
    body = _request(c, "b@northeastern.edu").json()
    assert body["dev_link"].startswith("http://arena.test/verify.html?token=")
    cfg.email_backend = "smtp"
    assert "dev_link" not in _request(c, "c@northeastern.edu").json()


def test_send_failure_is_a_generic_503(cfg, db):
    class Broken:
        def send(self, to, subject, body):
            raise EmailError("relay said no")
    c = TestClient(create_app(cfg, db=db, mailer=Broken()))
    r = _request(c, "a@northeastern.edu")
    assert r.status_code == 503
    assert "relay" not in r.json()["error"]


def test_smtp_backend_needs_host(cfg):
    from app.email_send import make_sender
    cfg.email_backend = "smtp"
    with pytest.raises(ValueError):
        make_sender(cfg)
    cfg.smtp_host, cfg.email_from = "smtp.example.com", "arena@northeastern.edu"
    sender = make_sender(cfg)
    assert sender.port == 587 and sender.starttls and sender.timeout_s == 10


# ── cookies and sessions ──

def test_cookie_flags(client, outbox):
    _request(client, "a@northeastern.edu")
    r = client.post("/api/auth/verify", json={"token": last_token(outbox)})
    cookie = r.headers["set-cookie"]
    assert cookie.startswith("honne_session=")
    low = cookie.lower()
    assert "httponly" in low and "samesite=lax" in low and "path=/" in low
    assert "secure" not in low.replace("samesite", "")
    assert f"max-age={30 * 86400}" in low


def test_cookie_secure_when_configured(cfg, client, outbox):
    cfg.cookie_secure = True
    _request(client, "a@northeastern.edu")
    r = client.post("/api/auth/verify", json={"token": last_token(outbox)})
    assert "; secure" in r.headers["set-cookie"].lower()


def test_session_token_not_stored_in_plain(app, db):
    c = sign_in(app, "a@northeastern.edu")
    raw = c.cookies.get("honne_session")
    assert raw and raw not in str(db._all("select * from sessions"))


def test_me_signed_out_and_in(app, client):
    assert client.get("/api/me").json() == {"user": None, "team": None, "tiers": {}}
    c = sign_in(app, "a@northeastern.edu")
    me = c.get("/api/me").json()
    assert me["user"]["email"] == "a@northeastern.edu"
    assert me["user"]["handle"] is None and me["user"]["is_admin"] is False
    assert me["team"] is None
    assert me["tiers"]["ctf"] == {"tier": 0, "rank": None, "points": 0}


def test_admin_flag_from_email(app):
    c = sign_in(app, "Boss@northeastern.edu")
    assert c.get("/api/me").json()["user"]["is_admin"] is True


def test_logout(app, db):
    c = sign_in(app, "a@northeastern.edu")
    raw = c.cookies.get("honne_session")
    r = c.post("/api/auth/logout", json={})
    assert r.json() == {"ok": True}
    assert "honne_session=" in r.headers["set-cookie"]
    assert c.get("/api/me").json()["user"] is None
    # The old cookie value is dead even if someone kept a copy.
    other = TestClient(c.app, cookies={"honne_session": raw})
    assert other.get("/api/me").json()["user"] is None


def test_expired_session_is_rejected(app, db):
    c = sign_in(app, "a@northeastern.edu")
    past = (datetime.now(timezone.utc) - timedelta(seconds=1)).isoformat()
    db._exec("update sessions set expires_at = ?", (past,))
    assert c.get("/api/me").json()["user"] is None
    assert c.post("/api/me/handle", json={"handle": "a"}).status_code == 401


def test_session_lasts_30_days(app, db):
    sign_in(app, "a@northeastern.edu")
    row = db._one("select * from sessions")
    life = datetime.fromisoformat(row["expires_at"]) - datetime.fromisoformat(row["created_at"])
    assert life == timedelta(days=30)


def test_signing_in_again_replaces_the_old_session(app, db):
    c = sign_in(app, "a@northeastern.edu")
    sign_in(app, "a@northeastern.edu", client=c)
    assert db._count("select count(*) from sessions") == 1


# ── handles ──

def test_handle_unique_ignoring_case(app):
    a = sign_in(app, "a@northeastern.edu", "Alice")
    b = sign_in(app, "b@northeastern.edu")
    r = b.post("/api/me/handle", json={"handle": "alice"})
    assert r.status_code == 409
    assert b.post("/api/me/handle", json={"handle": "ALICE "}).status_code == 409
    assert b.post("/api/me/handle", json={"handle": "bob"}).json()["user"]["handle"] == "bob"
    # Changing the case of your own handle is fine.
    assert a.post("/api/me/handle", json={"handle": "alice"}).status_code == 200


@pytest.mark.parametrize("handle", ["", "x" * 33, "<script>", "a/b", "tab\there"])
def test_bad_handles(app, handle):
    c = sign_in(app, "a@northeastern.edu")
    assert c.post("/api/me/handle", json={"handle": handle}).status_code == 400


def test_handle_needs_sign_in(client):
    assert client.post("/api/me/handle", json={"handle": "a"}).status_code == 401


def test_submit_401_signed_out_and_400_without_handle(app, client, login):
    from tests.conftest import make_flag
    make_flag(client)
    r = submit(client, "warmup", "nuais{hello}")
    assert r.status_code == 401
    assert r.json() == {"error": "Sign in first."}
    c = sign_in(app, "a@northeastern.edu")
    r = submit(c, "warmup", "nuais{hello}")
    assert r.status_code == 400
    assert "handle" in r.json()["error"].lower()
