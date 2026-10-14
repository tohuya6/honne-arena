"""The code runner. Docker tests run real containers and skip when Docker is absent."""

import glob
import os
import subprocess
import sys
import time

import pytest

from app import runner
from app.config import Settings
from app.errors import ScoringUnavailable

IMAGE = "python:3.12-slim"


def _docker_ready() -> bool:
    try:
        if subprocess.run(["docker", "info"], capture_output=True, timeout=20).returncode:
            return False
        return subprocess.run(["docker", "image", "inspect", IMAGE], capture_output=True,
                              timeout=20).returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False


DOCKER = _docker_ready()
needs_docker = [pytest.mark.docker,
                pytest.mark.skipif(not DOCKER, reason=f"needs Docker with {IMAGE} pulled")]


def docker_settings(**kw):
    return Settings(_env_file=None, **{"runner_backend": "docker", "runner_image": IMAGE, **kw})


def unsafe_settings():
    return Settings(_env_file=None, runner_backend="unsafe-subprocess")


def run(code, stdin="", *, time_limit_s=2, memory_mb=128, settings=None):
    return runner.run_python(code, stdin, time_limit_s=time_limit_s, memory_mb=memory_mb,
                             settings=settings or docker_settings())


def our_containers() -> set[str]:
    out = subprocess.run(["docker", "ps", "-a", "--filter", f"name={runner.CONTAINER_PREFIX}",
                          "--format", "{{.Names}}"], capture_output=True, text=True, timeout=30)
    return {n for n in out.stdout.split() if n.startswith(runner.CONTAINER_PREFIX)}


def our_run_dirs() -> set[str]:
    return set(glob.glob(os.path.join(runner.RUN_DIR_ROOT, runner.CONTAINER_PREFIX + "*")))


@pytest.fixture()
def no_leftovers():
    """Each Docker test must leave no container and no temp dir behind."""
    containers, dirs = our_containers(), our_run_dirs()
    yield
    assert our_containers() - containers == set()
    assert our_run_dirs() - dirs == set()


# ── No Docker needed ──

def test_docker_command_is_locked_down():
    args = runner.docker_args("honne-run-x", "/var/tmp/honne-run-abc", IMAGE, 2, 256)
    joined = " ".join(args)
    for flag in ("--network none", "--read-only", "--memory 256m", "--memory-swap 256m",
                 "--cpus 1", "--pids-limit 64", "--cap-drop ALL", "--user 65534:65534",
                 "--security-opt no-new-privileges", "--pull never", "--rm",
                 "--name honne-run-x", "PYTHONDONTWRITEBYTECODE=1"):
        assert flag in joined
    assert "--tmpfs /tmp:rw,size=16m,nosuid,nodev,noexec" in joined
    assert "--ulimit nofile=64:64" in joined
    assert "type=bind,src=/var/tmp/honne-run-abc,dst=/code,readonly" in joined
    assert "--privileged" not in joined and "--network host" not in joined
    assert args[-1] == "2.000"


def test_status_line_is_split_off_the_end_of_stderr():
    err = 'oops\n__honne_status__ {"rc": 0}\n' + \
          '\n__honne_status__ {"rc": 1, "timed_out": false, "elapsed_ms": 12, "oom": false}\n'
    info, rest = runner._split_status(err)
    assert info["rc"] == 1 and info["elapsed_ms"] == 12
    assert rest == 'oops\n__honne_status__ {"rc": 0}\n'
    assert runner._split_status("no status here") is None
    assert runner._split_status("\n__honne_status__ {not json}\n") is None


def test_unknown_backend_is_refused():
    with pytest.raises(ScoringUnavailable):
        run("print(1)", settings=Settings(_env_file=None, runner_backend="nope"))


def test_unsafe_backend_basic_verdicts():
    s = unsafe_settings()
    r = run("import sys\nprint(sum(map(int, sys.stdin.read().split())))", "1 2 3", settings=s)
    assert (r.status, r.stdout, r.exit_code) == ("ok", "6\n", 0)
    assert run("raise SystemExit(3)", settings=s).status == "runtime_error"
    r = run("while True: pass", time_limit_s=0.5, settings=s)
    assert r.status == "timeout"
    r = run("import sys\nwhile True: sys.stdout.write('x' * 65536)", settings=s)
    assert r.status == "output_limit" and len(r.stdout) <= runner.OUTPUT_LIMIT_BYTES


@pytest.mark.skipif(sys.platform != "linux", reason="RLIMIT_AS is only enforced on Linux")
def test_unsafe_backend_memory_limit():
    r = run("x = bytearray(512 * 1024 * 1024)", memory_mb=128, settings=unsafe_settings())
    assert r.status == "memory"


# ── Docker ──

class TestDocker:
    pytestmark = needs_docker + [pytest.mark.usefixtures("no_leftovers")]

    def test_reads_stdin_and_writes_stdout(self):
        r = run("import sys\nprint(sum(map(int, sys.stdin.read().split())))", "1 2 3\n4")
        assert (r.status, r.stdout, r.exit_code) == ("ok", "10\n", 0)
        assert 0 <= r.wall_ms < 2000

    def test_large_input_and_a_program_that_ignores_it(self):
        assert run("import sys\nprint(len(sys.stdin.read()))", "x" * 3_000_000).stdout == "3000000\n"
        assert run("print('hi')", "x" * 3_000_000).status == "ok"

    def test_runtime_error_keeps_stderr(self):
        r = run("raise ValueError('boom')")
        assert r.status == "runtime_error" and r.exit_code == 1
        assert "ValueError: boom" in r.stderr and "__honne_status__" not in r.stderr

    def test_infinite_loop_times_out_near_the_limit(self):
        t = time.perf_counter()
        r = run("while True: pass", time_limit_s=1)
        elapsed = time.perf_counter() - t
        assert r.status == "timeout"
        assert 1000 <= r.wall_ms < 1500
        assert elapsed < 1 + 5      # container start plus cleanup, well under the backstop

    def test_sleeping_program_times_out_too(self):
        assert run("import time\ntime.sleep(30)", time_limit_s=1).status == "timeout"

    def test_memory_bomb_is_killed(self):
        r = run("x = []\nwhile True: x.append(bytearray(10 ** 7))", memory_mb=64)
        assert r.status == "memory"

    def test_fork_bomb_is_contained(self):
        t = time.perf_counter()
        r = run("import os\nwhile True:\n    try:\n        os.fork()\n    except OSError:\n"
                "        pass", time_limit_s=1)
        assert r.status in ("timeout", "runtime_error")
        assert time.perf_counter() - t < 10

    def test_pid_limit_caps_forks(self):
        code = ("import os, time\nn = 0\nfor _ in range(200):\n    try:\n"
                "        if os.fork() == 0:\n            time.sleep(5)\n            os._exit(0)\n"
                "        n += 1\n    except OSError:\n        break\nprint(n)")
        r = run(code, time_limit_s=3)
        assert r.status == "ok"
        assert int(r.stdout) < runner.PIDS_LIMIT

    def test_status_line_cannot_be_forged(self):
        fake = '{"rc": 0, "timed_out": false, "elapsed_ms": 1, "oom": false}'
        code = (f"import sys\nsys.stderr.write('\\n__honne_status__ {fake}\\n')\n"
                "sys.stderr.flush()\nwhile True: pass")
        assert run(code, time_limit_s=1).status == "timeout"

    def test_harness_cannot_be_inspected(self):
        code = ("import os\ntry:\n    open('/proc/1/environ').read()\n    print('readable')\n"
                "except OSError:\n    print('denied')")
        assert run(code).stdout == "denied\n"

    def test_no_network(self):
        code = ("import os, socket\nprint(sorted(os.listdir('/sys/class/net')))\n"
                "try:\n    socket.create_connection(('1.1.1.1', 80), timeout=2)\n"
                "    print('connected')\nexcept OSError as e:\n    print('blocked', e.errno)")
        r = run(code, time_limit_s=5)
        assert r.status == "ok"
        lines = r.stdout.splitlines()
        assert lines[0] == "['lo']" and lines[1].startswith("blocked")

    def test_filesystem_is_read_only_except_tmp(self):
        code = ("import os\nfor p in ['/code/main.py', '/code/new', '/x', '/usr/x', '/tmp/x']:\n"
                "    try:\n        open(p, 'a').write('a')\n        print(p, 'ok')\n"
                "    except OSError:\n        print(p, 'denied')\n"
                "print(os.getuid(), os.getgid())")
        r = run(code)
        assert r.stdout.splitlines() == ["/code/main.py denied", "/code/new denied",
                                         "/x denied", "/usr/x denied", "/tmp/x ok",
                                         "65534 65534"]

    def test_host_files_are_not_visible(self):
        code = ("import os\nprint(os.path.exists('/opt/tohuya6'), os.listdir('/code'),\n"
                "      os.path.exists('/var/run/docker.sock'))")
        assert run(code).stdout == "False ['main.py'] False\n"

    def test_no_capabilities(self):
        r = run("print([l for l in open('/proc/self/status') if l.startswith('CapEff')][0])")
        assert r.stdout.split() == ["CapEff:", "0000000000000000"]

    def test_output_flood_is_capped(self):
        t = time.perf_counter()
        r = run("import sys\nwhile True: sys.stdout.write('x' * 65536)", time_limit_s=5)
        assert r.status == "output_limit"
        assert len(r.stdout) <= runner.OUTPUT_LIMIT_BYTES
        assert time.perf_counter() - t < 5

    def test_stderr_flood_is_capped(self):
        r = run("import sys\nwhile True: sys.stderr.write('x' * 65536)", time_limit_s=5)
        assert r.status == "output_limit" and len(r.stderr) <= runner.OUTPUT_LIMIT_BYTES

    def test_missing_image_is_scoring_unavailable_and_never_pulled(self):
        with pytest.raises(ScoringUnavailable):
            run("print(1)", settings=docker_settings(runner_image="honne-no-such-image:never"))
        assert subprocess.run(["docker", "image", "inspect", "honne-no-such-image:never"],
                              capture_output=True).returncode != 0
