"""Run untrusted Python against one input and report what happened.

`run_python()` is the only entry point. It picks a backend from `settings.runner_backend`:

- "docker": the real sandbox. Each call starts one fresh container from
  `settings.runner_image` with no network, a read-only root, a small tmpfs at /tmp,
  hard memory and pid limits, every capability dropped, and the nobody user. The code is
  bind-mounted read-only and the input is piped to stdin. The container is named
  `honne-run-<uuid>` and is force-removed if anything goes wrong.
- "unsafe-subprocess": a plain child process on this machine with rlimits and a
  timeout. It is NOT a sandbox: the code can read every file this server can, open
  network connections, and talk to other processes. It exists so tests and dev boxes
  without Docker can exercise the judging logic, and it only runs when the setting
  says exactly "unsafe-subprocess".

Inside the container a small trusted harness is PID 1. It starts the player's program,
enforces the time limit, kills every other process when the program ends, and then
writes one status line to stderr. Because nothing else is alive at that point, the
last status line is always the harness's own, so the program cannot forge it. The host
also keeps a wall-clock backstop that removes the container if the harness never
finishes (a stuck daemon, or a container that never started).

The image is never pulled at run time. If it is missing the run raises
ScoringUnavailable and an admin has to `docker pull` it by hand.
"""

from __future__ import annotations

import json
import logging
import os
import resource
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from typing import Callable, NamedTuple

from app.errors import ScoringUnavailable

_log = logging.getLogger("honne")

STATUSES = ("ok", "timeout", "memory", "runtime_error", "output_limit", "internal_error")

OUTPUT_LIMIT_BYTES = 1 << 20   # per stream; more than this ends the run as output_limit
TMPFS_MB = 16                  # size of the writable /tmp inside the container
PIDS_LIMIT = 64
NOFILE_LIMIT = 64
CONTAINER_PREFIX = "honne-run-"
# Slack on top of the time limit before the host gives up on a container. It covers
# container start (about 0.5 s here) and a busy daemon. The time limit itself is
# enforced inside the container, so this only matters when something is broken.
HOST_SLACK_S = 10.0
DOCKER_CLI_TIMEOUT_S = 30.0

# Where code is written before it is mounted. The Docker daemon has to see this path.
# On this server the daemon is a snap whose /tmp is private, so /tmp would not work,
# while /var/tmp is shared with it (and with Docker Desktop on macOS).
RUN_DIR_ROOT = "/var/tmp" if os.path.isdir("/var/tmp") else tempfile.gettempdir()

_MARK = "\n__honne_status__ "


class RunResult(NamedTuple):
    status: str        # one of STATUSES
    stdout: str
    stderr: str
    exit_code: int | None
    wall_ms: int       # the program's own run time, container start not included


# ── The harness. Runs as PID 1 in the container and is passed with `python -c`. ──
# Kept free of anything the player can influence: it reads only its argv and the
# container's own cgroup counters.
_HARNESS = r"""
import json, os, select, signal, subprocess, sys, time

def oom_kills():
    try:
        with open("/sys/fs/cgroup/memory.events") as f:
            for line in f:
                k, v = line.split()
                if k == "oom_kill":
                    return int(v)
    except OSError:
        pass
    return 0

# Not dumpable, so the program (same uid) cannot ptrace us or read /proc/1/mem,
# even on a host where Yama allows ptrace between unrelated processes.
try:
    import ctypes
    ctypes.CDLL(None).prctl(4, 0, 0, 0, 0)   # PR_SET_DUMPABLE, 0
except Exception:
    pass

limit = float(sys.argv[1])
before = oom_kills()
t0 = time.perf_counter()
p = subprocess.Popen([sys.executable, "-I", "-B", "/code/main.py"])
try:
    # A pidfd wakes us the moment the program exits. Popen.wait(timeout) polls in
    # steps of up to 50 ms, which would blur the reported time.
    timed_out = not select.select([os.pidfd_open(p.pid)], [], [], limit)[0]
except (AttributeError, OSError):
    try:
        p.wait(timeout=limit)
        timed_out = False
    except subprocess.TimeoutExpired:
        timed_out = True
elapsed = time.perf_counter() - t0
if timed_out:
    p.kill()
rc = p.wait()
try:
    os.kill(-1, signal.SIGKILL)   # everything except PID 1, so nothing writes after us
except OSError:
    pass
deadline = time.monotonic() + 1.0
while time.monotonic() < deadline:
    try:
        if os.waitpid(-1, os.WNOHANG) == (0, 0):
            time.sleep(0.01)
    except ChildProcessError:
        break
code = rc if rc >= 0 else 128 - rc
sys.stderr.flush()
os.write(2, ("\n__honne_status__ " + json.dumps({
    "rc": code, "timed_out": timed_out, "elapsed_ms": int(elapsed * 1000),
    "oom": oom_kills() > before,
}) + "\n").encode())
os._exit(0)
"""


def run_python(code: str, stdin: str, *, time_limit_s: float, memory_mb: int,
               settings) -> RunResult:
    """Run `code` as a Python 3 program with `stdin` as its input."""
    backend = settings.runner_backend
    if backend == "docker":
        return _run_docker(code, stdin, time_limit_s, memory_mb, settings.runner_image)
    if backend == "unsafe-subprocess":
        return _run_unsafe_subprocess(code, stdin, time_limit_s, memory_mb)
    raise ScoringUnavailable(f"Unknown code runner backend {backend!r}.")


# ── Shared plumbing ──

class _Streams(NamedTuple):
    stdout: bytes
    stderr: bytes
    overflow: bool
    timed_out: bool


def _communicate(proc: subprocess.Popen, data: bytes, timeout_s: float, cap: int,
                 kill: Callable[[], None]) -> _Streams:
    """Feed stdin and read both pipes with a byte cap, killing on overflow or timeout.

    Every pipe has its own thread, so a program that never reads its input, or one that
    floods one stream, cannot block the other side."""
    bufs = {"out": bytearray(), "err": bytearray()}
    overflow = threading.Event()
    killed = threading.Lock()

    def kill_once():
        # In its own thread, so a reader that hits the cap keeps draining its pipe
        # while the kill runs. Docker will not stop a container whose output is stuck.
        if killed.acquire(blocking=False):
            threading.Thread(target=kill, daemon=True).start()

    def reader(name, pipe):
        buf = bufs[name]
        while True:
            chunk = pipe.read1(65536) if hasattr(pipe, "read1") else pipe.read(65536)
            if not chunk:
                break
            if overflow.is_set():
                continue                # keep draining so the writer is not blocked
            room = cap - len(buf)
            buf.extend(chunk[:room])
            if len(chunk) > room:
                overflow.set()
                kill_once()

    def writer():
        try:
            proc.stdin.write(data)
        except (BrokenPipeError, OSError, ValueError):
            pass                        # the program exited without reading it all
        finally:
            try:
                proc.stdin.close()
            except (BrokenPipeError, OSError, ValueError):
                pass

    threads = [threading.Thread(target=reader, args=("out", proc.stdout), daemon=True),
               threading.Thread(target=reader, args=("err", proc.stderr), daemon=True),
               threading.Thread(target=writer, daemon=True)]
    for t in threads:
        t.start()
    timed_out = False
    try:
        proc.wait(timeout=timeout_s)
    except subprocess.TimeoutExpired:
        timed_out = True
        kill_once()
        try:
            proc.wait(timeout=DOCKER_CLI_TIMEOUT_S)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
    for t in threads:
        t.join(timeout=5)
    return _Streams(bytes(bufs["out"]), bytes(bufs["err"]), overflow.is_set(), timed_out)


def _decode(b: bytes) -> str:
    return b.decode("utf-8", errors="replace")


def _looks_like_memory_error(stderr: str) -> bool:
    lines = [ln for ln in stderr.strip().splitlines() if ln.strip()]
    return bool(lines) and lines[-1].startswith("MemoryError")


def _make_run_dir(code: str) -> str:
    """A temp dir holding main.py, readable by the container's nobody user."""
    d = tempfile.mkdtemp(prefix=CONTAINER_PREFIX, dir=RUN_DIR_ROOT)
    os.chmod(d, 0o755)
    path = os.path.join(d, "main.py")
    with open(path, "w", encoding="utf-8") as f:
        f.write(code)
    os.chmod(path, 0o644)
    return d


# ── Docker backend ──

_image_ok: set[str] = set()
_nnp_ok: dict[str, bool] = {}
_probe_lock = threading.Lock()


def _docker(*args: str, timeout: float = DOCKER_CLI_TIMEOUT_S) -> subprocess.CompletedProcess:
    try:
        return subprocess.run(["docker", *args], capture_output=True, timeout=timeout,
                              stdin=subprocess.DEVNULL)
    except FileNotFoundError:
        raise ScoringUnavailable("The code runner is not available (no Docker).") from None
    except subprocess.TimeoutExpired:
        raise ScoringUnavailable("The code runner is not answering right now.") from None


def remove_container(name: str, wait_s: float = 10.0) -> None:
    """Force-remove one container by its exact name and wait until it is gone.

    `docker rm -f` can return early when a removal is already under way (the --rm
    auto-removal racing ours), so we poll until the name no longer resolves."""
    deadline = time.monotonic() + wait_s
    try:
        subprocess.run(["docker", "rm", "-f", name], capture_output=True,
                       timeout=DOCKER_CLI_TIMEOUT_S, stdin=subprocess.DEVNULL)
        while time.monotonic() < deadline:
            r = subprocess.run(["docker", "container", "inspect", "--format", "{{.Id}}", name],
                               capture_output=True, timeout=DOCKER_CLI_TIMEOUT_S,
                               stdin=subprocess.DEVNULL)
            if r.returncode != 0:
                return
            time.sleep(0.1)
        _log.warning("container %s is still there after removal", name)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        _log.warning("could not remove container %s", name)


def _check_image(image: str) -> None:
    if image in _image_ok:
        return
    r = _docker("image", "inspect", "--format", "{{.Id}}", image)
    if r.returncode != 0:
        _log.error("runner image %s is missing; pull it by hand", image)
        raise ScoringUnavailable("The code runner is not set up on this server yet.")
    _image_ok.add(image)


def _supports_no_new_privileges(image: str) -> bool:
    """Whether this daemon can run with --security-opt no-new-privileges.

    Docker installed as a snap fails every exec under that flag ("operation not
    permitted"), so on such a host the flag has to go. The run is still non-root with
    an empty capability bounding set, so a setuid binary gains a root uid but no
    capabilities on a read-only root filesystem. Checked once per process."""
    with _probe_lock:
        if image not in _nnp_ok:
            r = _docker("run", "--rm", "--pull", "never", "--network", "none",
                        "--log-driver", "none", "--name", CONTAINER_PREFIX + uuid.uuid4().hex,
                        "--security-opt", "no-new-privileges", "--user", "65534:65534",
                        image, "python3", "-c", "pass")
            ok = r.returncode == 0
            if not ok and b"operation not permitted" not in r.stderr:
                ok = True   # failed for some other reason, keep the flag and let the run report it
            if not ok:
                _log.warning("this Docker daemon cannot exec under no-new-privileges; "
                             "code runs without that flag")
            _nnp_ok[image] = ok
        return _nnp_ok[image]


def docker_args(name: str, run_dir: str, image: str, time_limit_s: float, memory_mb: int,
                no_new_privileges: bool = True) -> list[str]:
    """The full `docker run` command for one test case."""
    args = [
        "docker", "run", "-i", "--rm", "--name", name,
        "--pull", "never",
        "--network", "none",
        "--read-only",
        "--tmpfs", f"/tmp:rw,size={TMPFS_MB}m,nosuid,nodev,noexec",
        "--memory", f"{memory_mb}m", "--memory-swap", f"{memory_mb}m",
        "--cpus", "1",
        "--pids-limit", str(PIDS_LIMIT),
        "--cap-drop", "ALL",
        "--user", "65534:65534",
        "--ulimit", f"nofile={NOFILE_LIMIT}:{NOFILE_LIMIT}",
        "--ulimit", "core=0",
        "--log-driver", "none",       # output reaches us over the attach stream only
        "--hostname", "sandbox",
        "--workdir", "/tmp",
        "-e", "PYTHONDONTWRITEBYTECODE=1",
        "-e", "HOME=/tmp",
        "--mount", f"type=bind,src={run_dir},dst=/code,readonly",
    ]
    if no_new_privileges:
        args += ["--security-opt", "no-new-privileges"]
    args += [image, "python3", "-I", "-B", "-c", _HARNESS, f"{time_limit_s:.3f}"]
    return args


def _run_docker(code: str, stdin: str, time_limit_s: float, memory_mb: int,
                image: str) -> RunResult:
    _check_image(image)
    nnp = _supports_no_new_privileges(image)
    name = CONTAINER_PREFIX + uuid.uuid4().hex
    run_dir = _make_run_dir(code)
    start = time.perf_counter()
    abnormal = True
    try:
        try:
            proc = subprocess.Popen(
                docker_args(name, run_dir, image, time_limit_s, memory_mb, nnp),
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        except FileNotFoundError:
            raise ScoringUnavailable("The code runner is not available (no Docker).") from None

        def kill():
            proc.kill()
            remove_container(name)

        s = _communicate(proc, stdin.encode("utf-8"), time_limit_s + HOST_SLACK_S,
                         OUTPUT_LIMIT_BYTES, kill)
        host_ms = int((time.perf_counter() - start) * 1000)
        out, err = _decode(s.stdout), _decode(s.stderr)
        if s.overflow:
            return RunResult("output_limit", out, err, None, host_ms)
        if s.timed_out:
            return RunResult("timeout", out, err, None, host_ms)

        status = _split_status(err)
        if status is None:
            # No status line: the harness never ran or was killed. 125 is Docker's own
            # "could not start" code. 137 with no status line means the cgroup OOM
            # killer took the harness too.
            rc = proc.returncode
            if rc == 125 and b"No such image" in s.stderr:
                _image_ok.discard(image)
                raise ScoringUnavailable("The code runner is not set up on this server yet.")
            if rc == 137:
                return RunResult("memory", out, "", rc, host_ms)
            _log.error("runner container %s ended without a status (rc=%s): %s",
                       name, rc, err[-500:])
            return RunResult("internal_error", out, "", rc, host_ms)

        info, err = status
        abnormal = False
        rc, ms = info["rc"], info["elapsed_ms"]
        if info["timed_out"]:
            return RunResult("timeout", out, err, None, ms)
        if info["oom"] or (rc != 0 and _looks_like_memory_error(err)):
            return RunResult("memory", out, err, rc, ms)
        if rc != 0:
            return RunResult("runtime_error", out, err, rc, ms)
        return RunResult("ok", out, err, 0, ms)
    finally:
        if abnormal:
            remove_container(name)
        shutil.rmtree(run_dir, ignore_errors=True)


def _split_status(stderr: str) -> tuple[dict, str] | None:
    """Pull the harness's status line off the end of stderr."""
    i = stderr.rfind(_MARK)
    if i < 0:
        return None
    line = stderr[i + len(_MARK):].split("\n", 1)[0]
    try:
        info = json.loads(line)
    except ValueError:
        return None
    if not isinstance(info, dict) or not {"rc", "timed_out", "elapsed_ms", "oom"} <= info.keys():
        return None
    return info, stderr[:i]


# ── Unsafe subprocess backend (no isolation, see the module docstring) ──

def _run_unsafe_subprocess(code: str, stdin: str, time_limit_s: float,
                           memory_mb: int) -> RunResult:
    run_dir = _make_run_dir(code)
    limit_bytes = memory_mb * 1024 * 1024

    def limits():
        for res, value in ((resource.RLIMIT_AS, limit_bytes),
                           (resource.RLIMIT_CPU, int(time_limit_s) + 1),
                           (resource.RLIMIT_NOFILE, NOFILE_LIMIT),
                           (resource.RLIMIT_FSIZE, OUTPUT_LIMIT_BYTES),
                           (resource.RLIMIT_CORE, 0)):
            try:
                resource.setrlimit(res, (value, value))
            except (ValueError, OSError):
                pass   # macOS refuses some of these

    try:
        start = time.perf_counter()
        proc = subprocess.Popen(
            [sys.executable, "-I", "-B", os.path.join(run_dir, "main.py")],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            cwd=run_dir, env={"PYTHONDONTWRITEBYTECODE": "1", "PATH": "/usr/bin:/bin"},
            preexec_fn=limits, start_new_session=True)

        def kill():
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except OSError:
                pass

        s = _communicate(proc, stdin.encode("utf-8"), time_limit_s, OUTPUT_LIMIT_BYTES, kill)
        ms = int((time.perf_counter() - start) * 1000)
        kill()   # take down anything it forked
        out, err = _decode(s.stdout), _decode(s.stderr)
        rc = proc.returncode
        if s.overflow:
            return RunResult("output_limit", out, err, None, ms)
        if s.timed_out or rc == -signal.SIGXCPU:
            return RunResult("timeout", out, err, None, ms)
        if rc != 0 and _looks_like_memory_error(err):
            return RunResult("memory", out, err, rc, ms)
        if rc != 0:
            return RunResult("runtime_error", out, err, rc, ms)
        return RunResult("ok", out, err, 0, ms)
    finally:
        shutil.rmtree(run_dir, ignore_errors=True)
