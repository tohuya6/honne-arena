"""Sign-in by NU email, with a magic link.

1. POST /api/auth/request {email}: if the domain is allowed, store the sha256 of a
   fresh random token and email a link to /verify.html?token=...
2. The verify page shows a button that POSTs the token to /api/auth/verify. No GET
   route ever consumes a token, because mail scanners (Outlook Safe Links) fetch every
   link in a message before the person clicks it.
3. Verify consumes the token once, creates the user if needed, and starts a session.
   The session token goes in an HttpOnly cookie. Only its sha256 is stored.

Plain functions over an injected db and settings, so all of it is testable without HTTP.
"""

from __future__ import annotations

import hashlib
import re
import secrets
from datetime import datetime, timedelta, timezone
from urllib.parse import quote

from app.email_send import EmailError
from app.errors import RateLimited, ServiceUnavailable, Unauthorized, ValidationError

COOKIE_NAME = "honne_session"
# Deliberately narrow: the plain address forms NU hands out, nothing exotic.
EMAIL_RE = re.compile(r"^[a-z0-9][a-z0-9._%+\-]{0,63}@([a-z0-9\-]+(?:\.[a-z0-9\-]+)+)$")


def sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def normalize_email(email: str, settings) -> str:
    """Lowercase and check the address. The domain must equal an allowed domain
    exactly, so `x@evil.northeastern.edu` and `x@northeastern.edu.evil.com` fail."""
    email = (email or "").strip().lower()
    m = EMAIL_RE.match(email)
    if not m or len(email) > 254:
        raise ValidationError("Enter a valid email address.")
    allowed = settings.allowed_domain_set
    if m.group(1) not in allowed:
        shown = " or ".join(f"@{d}" for d in sorted(allowed))
        raise ValidationError(f"Sign in with your {shown} email.")
    return email


def request_link(db, sender, settings, *, email: str, ip_hash: str,
                 now: datetime | None = None) -> str:
    """Email a sign-in link. Returns the link (the API shows it only in dev mode).

    The answer is the same whether or not the address has an account."""
    now = now or datetime.now(timezone.utc)
    email = normalize_email(email, settings)

    window = (now - timedelta(minutes=15)).isoformat()
    if db.count_login_tokens_since(window, email=email) >= settings.login_per_email:
        raise RateLimited("Too many sign-in emails. Please wait 15 minutes.")
    hour_ago = (now - timedelta(hours=1)).isoformat()
    if db.count_login_tokens_since(hour_ago, ip_hash=ip_hash) >= settings.login_per_ip_hour:
        raise RateLimited("Too many sign-in emails from your connection. Try again later.")

    token = secrets.token_urlsafe(32)
    expires = now + timedelta(minutes=settings.login_token_minutes)
    db.create_login_token(token_sha256=sha256(token), email=email, ip_hash=ip_hash,
                          expires_at=expires.isoformat(), created_at=now.isoformat())
    link = f"{settings.public_base_url.rstrip('/')}/verify.html?token={quote(token)}"
    body = (f"Sign in to Honne Arena:\n\n{link}\n\n"
            f"The link works once and expires in {settings.login_token_minutes} minutes. "
            f"If you did not ask for it, ignore this email.\n")
    try:
        sender.send(email, "Your Honne Arena sign-in link", body)
    except EmailError:
        raise ServiceUnavailable("Could not send the email right now. "
                                 "Please try again later.") from None
    return link


def verify_link(db, settings, *, token: str,
                now: datetime | None = None) -> tuple[dict, str]:
    """Consume a sign-in token. Returns (user, session_token)."""
    now = now or datetime.now(timezone.utc)
    stamp = now.isoformat()
    email = db.consume_login_token(sha256((token or "").strip()), stamp) if token else None
    if not email:
        raise ValidationError("That sign-in link is invalid or has expired. "
                              "Request a new one.")
    db.purge_expired_auth(stamp)
    user = db.get_or_create_user(email)
    session = secrets.token_urlsafe(32)
    db.create_session(token_sha256=sha256(session), user_id=user["id"],
                      expires_at=(now + timedelta(days=settings.session_days)).isoformat(),
                      created_at=stamp)
    return user, session


def session_user(db, session_token: str | None, now: datetime | None = None) -> dict | None:
    """The signed-in user for a cookie value, or None if missing, unknown, or expired."""
    if not session_token:
        return None
    now = now or datetime.now(timezone.utc)
    return db.session_user(sha256(session_token), now.isoformat())


def require_user(user: dict | None) -> dict:
    if user is None:
        raise Unauthorized("Sign in first.")
    return user


def end_session(db, session_token: str | None) -> None:
    if session_token:
        db.delete_session(sha256(session_token))


def is_admin(user: dict | None, settings) -> bool:
    return bool(user) and user["email"] in settings.admin_email_set


def user_view(user: dict, settings) -> dict:
    """The User object. Holds the email, so only ever send it to its owner."""
    return {"id": user["id"], "handle": user["handle"], "email": user["email"],
            "is_admin": is_admin(user, settings)}


def set_cookie(response, session_token: str, settings) -> None:
    response.set_cookie(COOKIE_NAME, session_token,
                        max_age=settings.session_days * 86400, path="/",
                        httponly=True, samesite="lax", secure=settings.cookie_secure)


def clear_cookie(response, settings) -> None:
    response.delete_cookie(COOKIE_NAME, path="/", httponly=True, samesite="lax",
                           secure=settings.cookie_secure)
