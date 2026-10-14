"""CAPTCHA gating. No network: covers the disabled and missing-token branches, plus the
/api/submit 400 when enabled without a token. (Ported from Steering Arena.)"""

from fastapi.testclient import TestClient

from app import captcha
from app.main import create_app


def test_disabled_when_no_secret():
    assert captcha.verify("", "") is True
    assert captcha.verify("anything", "") is True


def test_enabled_rejects_missing_token():
    # secret set but no token → fail closed, without any network call
    assert captcha.verify("", "a-secret") is False


def test_submit_blocked_without_captcha_when_enabled(cfg, db):
    cfg.turnstile_secret = "test-secret"
    client = TestClient(create_app(cfg, db=db))
    r = client.post("/api/submit", json={"handle": "alice", "challenge": "x", "answer": "y"})
    assert r.status_code == 400
    assert "captcha" in r.json()["error"].lower()
