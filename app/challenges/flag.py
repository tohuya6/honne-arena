"""CTF-style flag: one right string, checked against its SHA-256.

Only the hash is stored, so a leaked database or config dump does not leak the flags.
The comparison is constant-time so response timing says nothing about how close a
guess was.
"""

from __future__ import annotations

import hashlib
import hmac
import re

from app.challenges.base import Challenge, Outcome

_HEX64 = re.compile(r"^[0-9a-f]{64}$")


def hash_flag(flag: str, case_sensitive: bool = False) -> str:
    text = flag.strip() if case_sensitive else flag.strip().lower()
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class FlagChallenge(Challenge):
    kind = "flag"
    mode = "solve"

    def validate_config(self) -> None:
        h = self.config.get("flag_sha256", "")
        if not _HEX64.match(h):
            raise ValueError("flag challenge needs config.flag_sha256 (64 lowercase hex chars)")

    @property
    def case_sensitive(self) -> bool:
        return bool(self.config.get("case_sensitive", False))

    def normalize(self, answer: str) -> str:
        return answer.strip() if self.case_sensitive else answer.strip().lower()

    def evaluate(self, answer: str) -> Outcome:
        got = hash_flag(answer, self.case_sensitive)
        ok = hmac.compare_digest(got, self.config["flag_sha256"])
        return Outcome(correct=ok, score=None, detail={})

    def public_config(self) -> dict:
        # e.g. "nuais{...}", so players know what shape of answer to look for.
        fmt = self.config.get("flag_format")
        return {"flag_format": fmt} if fmt else {}
