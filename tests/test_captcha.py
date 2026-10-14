"""CAPTCHA gating. No network: covers the disabled and missing-token branches, plus the
/api/submit 400 when enabled without a token. (Ported from Steering Arena.)"""

from app import captcha


def test_disabled_when_no_secret():
    assert captcha.verify("", "") is True
    assert captcha.verify("anything", "") is True


def test_enabled_rejects_missing_token():
    # secret set but no token → fail closed, without any network call
    assert captcha.verify("", "a-secret") is False


def test_submit_blocked_without_captcha_when_enabled(cfg, app, login):
    cfg.turnstile_secret = "test-secret"
    alice = login("alice")
    r = alice.post("/api/submit", json={"challenge": "x", "answer": "y"})
    assert r.status_code == 400
    assert "captcha" in r.json()["error"].lower()
