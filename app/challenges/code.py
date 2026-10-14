"""Competitive programming: submit a program, it runs against hidden tests.

The answer is Python source. It runs once per test case in the sandbox from
`app/runner.py`, reading the test's input on stdin, and its stdout is compared with the
expected output. It is correct only when every test passes.

Tests run in order, the public samples first and then the hidden tests, and judging
stops at the first failure. They run one after another inside a submission. Parallel
submissions are bounded by the local job gate (`local_concurrency`), so at most that
many containers run at once.

What a player learns:
- the verdict, how many tests passed, and the number of the first failing test
- for a failing sample, the expected and actual output (the sample is public anyway)
- nothing about hidden tests beyond the verdict: no input, no output, no stderr

Config:
  language        "python" (the only one so far)
  samples         [{input, output}], public, shown to players
  tests_file      path to a JSON list of {input, output}, under data/private/, or
  tests           the same list inline (unit tests and tiny problems)
  time_limit_s    per test, default 2
  memory_mb       per test, default 256
  max_code_chars  default 50000
"""

from __future__ import annotations

import json
from pathlib import Path, PurePath

from app import runner
from app.challenges.base import Challenge, Outcome
from app.errors import ScoringUnavailable, ValidationError

LANGUAGES = ("python",)
MAX_TIME_LIMIT_S = 10
MIN_MEMORY_MB, MAX_MEMORY_MB = 32, 1024
SHOWN_CHARS = 500          # how much of a failing sample's output to show

VERDICTS = {
    "timeout": "TLE",
    "memory": "MLE",
    "runtime_error": "RE",
    "output_limit": "OLE",
    "internal_error": "IE",
}


def normalize_output(text: str) -> str:
    """Strip trailing whitespace on each line and drop trailing blank lines."""
    lines = [ln.rstrip() for ln in text.replace("\r\n", "\n").replace("\r", "\n").split("\n")]
    while lines and not lines[-1]:
        lines.pop()
    return "\n".join(lines)


def outputs_match(got: str, expected: str) -> bool:
    return normalize_output(got) == normalize_output(expected)


def _truncate(text: str, n: int = SHOWN_CHARS) -> str:
    return text if len(text) <= n else text[:n] + "\n[truncated]"


def _check_cases(cases, where: str) -> None:
    if not isinstance(cases, list):
        raise ValueError(f"code challenge {where} must be a list of {{input, output}}")
    for i, case in enumerate(cases, 1):
        if (not isinstance(case, dict) or not isinstance(case.get("input"), str)
                or not isinstance(case.get("output"), str)):
            raise ValueError(f"code challenge {where}[{i}] needs string input and output")


def _under_private(path: str) -> bool:
    parts = PurePath(path).parts
    if ".." in parts:
        return False
    return any(parts[i:i + 2] == ("data", "private") for i in range(len(parts) - 1))


class CodeChallenge(Challenge):
    kind = "code"
    mode = "solve"
    compute = "local"

    def validate_config(self) -> None:
        c = self.config
        if c.get("language", "python") not in LANGUAGES:
            raise ValueError(f"code challenge language must be one of {LANGUAGES}")
        _check_cases(c.get("samples", []), "samples")
        if "tests" in c:
            _check_cases(c["tests"], "tests")
        tests_file = c.get("tests_file")
        if tests_file is not None:
            if not isinstance(tests_file, str) or not _under_private(tests_file):
                raise ValueError("code challenge tests_file must be a path under data/private/")
        if not (c.get("samples") or c.get("tests") or tests_file):
            raise ValueError("code challenge needs config.tests_file, config.tests, or samples")
        t = c.get("time_limit_s", 2)
        if isinstance(t, bool) or not isinstance(t, (int, float)) or not 0 < t <= MAX_TIME_LIMIT_S:
            raise ValueError(f"time_limit_s must be a number in (0, {MAX_TIME_LIMIT_S}]")
        m = c.get("memory_mb", 256)
        if isinstance(m, bool) or not isinstance(m, int) or not MIN_MEMORY_MB <= m <= MAX_MEMORY_MB:
            raise ValueError(f"memory_mb must be an integer in [{MIN_MEMORY_MB}, {MAX_MEMORY_MB}]")
        n = c.get("max_code_chars", 50000)
        if isinstance(n, bool) or not isinstance(n, int) or n <= 0:
            raise ValueError("max_code_chars must be a positive integer")

    @property
    def time_limit_s(self) -> float:
        return float(self.config.get("time_limit_s", 2))

    @property
    def memory_mb(self) -> int:
        return int(self.config.get("memory_mb", 256))

    @property
    def samples(self) -> list[dict]:
        return list(self.config.get("samples", []))

    def hidden_tests(self) -> list[dict]:
        if "tests" in self.config:
            return list(self.config["tests"])
        path = self.config.get("tests_file")
        if not path:
            return []
        try:
            cases = json.loads(Path(path).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            raise ScoringUnavailable("This problem's tests are not installed yet.") from None
        try:
            _check_cases(cases, "tests_file")
        except ValueError:
            raise ScoringUnavailable("This problem's tests are broken.") from None
        return cases

    def prepare(self, settings, on_remote_call) -> None:
        # A Challenge lives for one request, so keeping settings on it is safe.
        self._settings = settings

    def normalize(self, answer: str) -> str:
        # Leading whitespace matters in Python, so only line endings are touched.
        return answer.replace("\r\n", "\n")

    def evaluate(self, answer: str) -> Outcome:
        settings = getattr(self, "_settings", None)
        if settings is None:
            raise RuntimeError("prepare() must run before evaluate()")
        code = self.normalize(answer)
        limit = int(self.config.get("max_code_chars", 50000))
        if len(code) > limit:
            raise ValidationError(f"Code too long (max {limit} characters).")

        samples = self.samples
        cases = samples + self.hidden_tests()
        time_ms = 0
        for i, case in enumerate(cases):
            res = runner.run_python(code, case["input"], time_limit_s=self.time_limit_s,
                                    memory_mb=self.memory_mb, settings=settings)
            time_ms = max(time_ms, res.wall_ms)
            if res.status == "ok":
                if outputs_match(res.stdout, case["output"]):
                    continue
                verdict = "WA"
            else:
                verdict = VERDICTS.get(res.status, "IE")
            detail = {"verdict": verdict, "passed": i, "total": len(cases),
                      "failed_test": i + 1, "time_ms": time_ms}
            if i < len(samples):
                detail["sample"] = True
                if verdict == "WA":
                    detail["expected"] = _truncate(case["output"])
                    detail["got"] = _truncate(res.stdout)
                elif verdict == "RE":
                    detail["stderr"] = _truncate(res.stderr[-SHOWN_CHARS:])
            return Outcome(correct=False, score=None, detail=detail)
        return Outcome(correct=True, score=None,
                       detail={"verdict": "AC", "passed": len(cases), "total": len(cases),
                               "failed_test": None, "time_ms": time_ms})

    def public_config(self) -> dict:
        return {"language": self.config.get("language", "python"),
                "time_limit_s": self.time_limit_s,
                "memory_mb": self.memory_mb,
                "samples": self.samples}
