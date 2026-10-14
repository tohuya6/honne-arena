"""The `code` challenge kind. Most tests swap the runner for a fake, so they check the
compare and verdict logic alone. The Docker tests at the bottom run real containers."""

import json
from pathlib import Path

import pytest

from app import challenges, runner
from app.challenges.code import CodeChallenge, normalize_output, outputs_match
from app.config import Settings
from app.errors import ScoringUnavailable, ValidationError
from tests.test_runner import IMAGE, needs_docker

SEEDS = Path(__file__).resolve().parent.parent / "seeds" / "cp.json"

SAMPLES = [{"input": "1 2\n", "output": "3\n"}, {"input": "5 5\n", "output": "10\n"}]
HIDDEN = [{"input": "100 200\n", "output": "300\n"},
          {"input": "-7 7\n", "output": "0\n"},
          {"input": "123456789 987654321\n", "output": "1111111110\n"}]

SOLUTION = "a, b = map(int, input().split())\nprint(a + b)\n"


def make(config=None, **extra):
    cfg = {"language": "python", "samples": SAMPLES, "tests": HIDDEN, **(config or {})}
    row = {"id": 1, "slug": "add", "title": "Add", "kind": "code", "track": "cp",
           "tier": "learn", "config": cfg, **extra}
    return challenges.load(row)


def prepared(ch, **settings_kw):
    ch.prepare(Settings(_env_file=None, **settings_kw), lambda: None)
    return ch


class FakeRunner:
    """Answers each case from a table keyed by stdin, and records every call."""

    def __init__(self, table):
        self.table = table
        self.calls = []

    def __call__(self, code, stdin, *, time_limit_s, memory_mb, settings):
        self.calls.append(stdin)
        status, out, err = self.table.get(stdin, ("ok", "", ""))
        return runner.RunResult(status, out, err, 0 if status == "ok" else 1, 40)


@pytest.fixture()
def fake(monkeypatch):
    def install(table):
        f = FakeRunner(table)
        monkeypatch.setattr(runner, "run_python", f)
        return f
    return install


def all_correct(**override):
    table = {c["input"]: ("ok", c["output"], "") for c in SAMPLES + HIDDEN}
    table.update(override)
    return table


# ── Comparison ──

def test_trailing_whitespace_and_blank_lines_do_not_matter():
    assert outputs_match("3   \n4\t\n\n\n", "3\n4")
    assert outputs_match("3\r\n4\r\n", "3\n4\n")
    assert outputs_match("", "\n\n")


def test_other_whitespace_does_matter():
    assert not outputs_match(" 3\n", "3\n")
    assert not outputs_match("3\n\n4\n", "3\n4\n")
    assert not outputs_match("3 4\n", "3  4\n")
    assert normalize_output("a \nb  \n\n") == "a\nb"


# ── Verdicts with a fake runner ──

def test_all_pass_is_accepted(fake):
    f = fake(all_correct())
    out = prepared(make()).evaluate(SOLUTION)
    assert out.correct is True and out.score is None
    assert out.detail == {"verdict": "AC", "passed": 5, "total": 5, "failed_test": None,
                          "time_ms": 40}
    assert f.calls == [c["input"] for c in SAMPLES + HIDDEN]   # samples first, in order


@pytest.mark.parametrize("status,verdict", [("timeout", "TLE"), ("memory", "MLE"),
                                            ("runtime_error", "RE"), ("output_limit", "OLE"),
                                            ("internal_error", "IE")])
def test_runner_status_maps_to_verdict(fake, status, verdict):
    fake(all_correct(**{HIDDEN[1]["input"]: (status, "", "")}))
    out = prepared(make()).evaluate(SOLUTION)
    assert out.correct is False
    assert out.detail["verdict"] == verdict
    assert (out.detail["passed"], out.detail["total"], out.detail["failed_test"]) == (3, 5, 4)


def test_stops_at_first_failure(fake):
    f = fake(all_correct(**{SAMPLES[1]["input"]: ("ok", "11\n", "")}))
    prepared(make()).evaluate(SOLUTION)
    assert f.calls == [SAMPLES[0]["input"], SAMPLES[1]["input"]]


def test_failing_sample_shows_expected_and_got(fake):
    fake(all_correct(**{SAMPLES[1]["input"]: ("ok", "11\n" * 1000, "")}))
    d = prepared(make()).evaluate(SOLUTION).detail
    assert d["verdict"] == "WA" and d["failed_test"] == 2 and d["sample"] is True
    assert d["expected"] == "10\n"
    assert d["got"].startswith("11\n11\n") and d["got"].endswith("[truncated]")
    assert len(d["got"]) < 600


def test_sample_runtime_error_shows_stderr(fake):
    fake(all_correct(**{SAMPLES[0]["input"]: ("runtime_error", "", "ZeroDivisionError")}))
    d = prepared(make()).evaluate(SOLUTION).detail
    assert d["verdict"] == "RE" and d["stderr"] == "ZeroDivisionError"


@pytest.mark.parametrize("status", ["ok", "runtime_error", "timeout"])
def test_failing_hidden_test_reveals_nothing(fake, status):
    secret = HIDDEN[2]["input"]
    fake(all_correct(**{secret: (status, "leak " + secret, "leak " + secret)}))
    d = prepared(make()).evaluate(SOLUTION).detail
    assert d["failed_test"] == 5
    assert set(d) == {"verdict", "passed", "total", "failed_test", "time_ms"}
    assert "123456789" not in json.dumps(d) and "1111111110" not in json.dumps(d)


def test_code_length_limit(fake):
    fake(all_correct())
    ch = prepared(make({"max_code_chars": 10}))
    with pytest.raises(ValidationError):
        ch.evaluate("print(1)  # far too long")


def test_evaluate_needs_prepare(fake):
    fake(all_correct())
    with pytest.raises(RuntimeError):
        make().evaluate(SOLUTION)


def test_leading_whitespace_in_code_is_kept():
    assert make().normalize("  x = 1\r\n") == "  x = 1\n"


# ── Config ──

def test_public_config_hides_tests():
    ch = make({"time_limit_s": 1.5, "memory_mb": 128,
               "tests_file": "data/private/cp/add.json"})
    pub = ch.public_config()
    assert pub == {"language": "python", "time_limit_s": 1.5, "memory_mb": 128,
                   "samples": SAMPLES}
    assert ch.public()["kind"] == "code" and ch.compute_mode() == "local"
    assert ch.mode == "solve"


def test_defaults():
    ch = make()
    assert (ch.time_limit_s, ch.memory_mb) == (2.0, 256)


@pytest.mark.parametrize("bad", [
    {"language": "c++"},
    {"samples": "1 2"},
    {"samples": [{"input": "1"}]},
    {"tests": [{"input": 1, "output": "1"}]},
    {"time_limit_s": 0}, {"time_limit_s": 60}, {"time_limit_s": "2"}, {"time_limit_s": True},
    {"memory_mb": 8}, {"memory_mb": 4096}, {"memory_mb": 64.5},
    {"max_code_chars": 0},
    {"tests_file": "/etc/passwd"},
    {"tests_file": "data/private/../../etc/passwd"},
    {"tests_file": "seeds/cp.json"},
])
def test_bad_config_is_rejected(bad):
    with pytest.raises(ValueError):
        make(bad)


def test_needs_some_tests():
    with pytest.raises(ValueError):
        challenges.load({"id": 1, "slug": "x", "title": "x", "kind": "code",
                         "config": {"language": "python"}})


def test_tests_file_is_read(fake, tmp_path):
    path = tmp_path / "data" / "private" / "cp" / "add.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(HIDDEN))
    cfg = {"language": "python", "samples": SAMPLES, "tests_file": str(path)}
    ch = challenges.load({"id": 1, "slug": "add", "title": "Add", "kind": "code",
                          "config": cfg})
    f = fake(all_correct())
    assert prepared(ch).evaluate(SOLUTION).detail["total"] == 5
    assert len(f.calls) == 5


def test_missing_tests_file_is_scoring_unavailable(fake, tmp_path):
    cfg = {"samples": SAMPLES, "tests_file": str(tmp_path / "data/private/cp/none.json")}
    ch = challenges.load({"id": 1, "slug": "add", "title": "Add", "kind": "code",
                          "config": cfg})
    fake(all_correct())
    with pytest.raises(ScoringUnavailable):
        prepared(ch).evaluate(SOLUTION)


def test_seed_file_loads():
    rows = json.loads(SEEDS.read_text())
    assert {r["tier"] for r in rows} == {"learn", "compete"}
    for i, r in enumerate(rows, 1):
        ch = challenges.load({**r, "id": i})
        assert isinstance(ch, CodeChallenge) and r["track"] == "cp"
        assert ch.config["tests_file"].startswith("data/private/cp/")
        assert "tests" not in ch.config      # hidden tests never live in the repo
        assert ch.samples


# ── Through the submit pipeline, with the fake runner ──

def test_submit_pipeline_runs_code_behind_the_local_gate(fake):
    from app.db import Database
    from app.submission import process_submission

    db = Database(":memory:")
    db.migrate()
    season = db.create_season("Season 1")
    db.create_challenge(season["id"], "add", "Add", "code", track="cp", tier="learn",
                        config={"language": "python", "samples": SAMPLES, "tests": HIDDEN})
    gate_calls = []

    def run_local(fn):
        gate_calls.append(fn)
        return fn()

    s = Settings(_env_file=None, db_path=":memory:", ip_hash_salt="salt")
    fake(all_correct(**{HIDDEN[0]["input"]: ("ok", "wrong\n", "")}))
    res = process_submission(handle="ada", slug="add", answer=SOLUTION, ip_hash="h", db=db,
                             settings=s, run_local=run_local)
    assert res["correct"] is False and len(gate_calls) == 1
    fake(all_correct())
    res = process_submission(handle="ada", slug="add", answer=SOLUTION, ip_hash="h", db=db,
                             settings=s, run_local=run_local)
    assert res["correct"] is True and len(gate_calls) == 2


# ── Docker ──

class TestDockerJudging:
    pytestmark = needs_docker

    def settings_kw(self):
        return {"runner_backend": "docker", "runner_image": IMAGE}

    def test_correct_solution_is_accepted(self):
        out = prepared(make(), **self.settings_kw()).evaluate(SOLUTION)
        assert out.correct is True and out.detail["verdict"] == "AC"
        assert out.detail["passed"] == out.detail["total"] == 5

    def test_wrong_solution_is_wrong_answer(self):
        out = prepared(make(), **self.settings_kw()).evaluate(
            "a, b = map(int, input().split())\nprint(a - b if a > 100 else a + b)\n")
        assert out.detail["verdict"] == "WA" and out.detail["failed_test"] == 5
        assert "expected" not in out.detail

    def test_slow_solution_is_time_limit_exceeded(self):
        ch = prepared(make({"time_limit_s": 1}), **self.settings_kw())
        out = ch.evaluate("import time\na, b = map(int, input().split())\n"
                          "if a > 50:\n    time.sleep(5)\nprint(a + b)\n")
        assert out.detail["verdict"] == "TLE" and out.detail["failed_test"] == 3
