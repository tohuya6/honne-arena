"""Sending the sign-in email, behind one small interface.

`EMAIL_BACKEND=console` (the default) logs the message instead of sending it, for local
dev. `EMAIL_BACKEND=smtp` sends it through an SMTP relay. Both have `send(to, subject,
body)` and raise EmailError on failure, which the API turns into a generic 503.

Tests pass a ConsoleSender with an `outbox` list and read the link from it.
"""

from __future__ import annotations

import logging
import smtplib
import ssl
from email.message import EmailMessage

_log = logging.getLogger("honne")


class EmailError(Exception):
    """The message could not be sent. Details go to the server log only."""


class ConsoleSender:
    def __init__(self, outbox: list | None = None):
        # Only tests pass an outbox. A real dev server just logs.
        self.outbox = outbox

    def send(self, to: str, subject: str, body: str) -> None:
        if self.outbox is not None:
            self.outbox.append({"to": to, "subject": subject, "body": body})
        # Warning level so it shows under uvicorn's default logging.
        _log.warning("console email (dev only, not sent) to=%s subject=%r\n%s",
                     to, subject, body)


class SmtpSender:
    def __init__(self, *, host: str, port: int, user: str, password: str,
                 starttls: bool, sender: str, timeout_s: float):
        self.host, self.port = host, port
        self.user, self.password = user, password
        self.starttls, self.sender, self.timeout_s = starttls, sender, timeout_s

    def send(self, to: str, subject: str, body: str) -> None:
        msg = EmailMessage()
        msg["From"], msg["To"], msg["Subject"] = self.sender, to, subject
        msg.set_content(body)
        try:
            with smtplib.SMTP(self.host, self.port, timeout=self.timeout_s) as smtp:
                if self.starttls:
                    smtp.starttls(context=ssl.create_default_context())
                if self.user:
                    smtp.login(self.user, self.password)
                smtp.send_message(msg)
        except (OSError, smtplib.SMTPException) as exc:
            _log.error("sign-in email to %s failed: %s", to, exc)
            raise EmailError("send failed") from exc


def make_sender(settings):
    backend = settings.email_backend.lower()
    if backend == "console":
        return ConsoleSender()
    if backend == "smtp":
        if not (settings.smtp_host and settings.email_from):
            raise ValueError("EMAIL_BACKEND=smtp needs SMTP_HOST and EMAIL_FROM")
        return SmtpSender(host=settings.smtp_host, port=settings.smtp_port,
                          user=settings.smtp_user, password=settings.smtp_password,
                          starttls=settings.smtp_starttls, sender=settings.email_from,
                          timeout_s=settings.smtp_timeout_s)
    raise ValueError(f"unknown EMAIL_BACKEND {settings.email_backend!r}")
