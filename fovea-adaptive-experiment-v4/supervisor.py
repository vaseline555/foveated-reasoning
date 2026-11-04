"""Wait for the existing smoke, then launch exactly one frozen Adaptive worker.

This supervisor makes no API calls and never retries a failed smoke or worker.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
SHARED = Path(
    "/workspace/fovea-reproduction-8b-recovery/runs/vstar191-8b-only-20261103"
)
TERMINAL = {"complete", "stopped", "paused", "attention_required"}
METHOD = "ADAPTIVE_V4"


def read(path: Path):
    return json.loads(path.read_text()) if path.exists() else None


def save(path: Path, value: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def process(pid: int):
    """Return (state, start ticks, argv); absence/zombie means ended."""
    try:
        root = Path("/proc") / str(pid)
        fields = (root / "stat").read_text().rsplit(") ", 1)[1].split()
        if fields[0] in {"Z", "X"}:
            return None
        argv = (root / "cmdline").read_bytes().split(b"\0")
        return fields[0], fields[19], [x.decode() for x in argv if x]
    except (FileNotFoundError, ProcessLookupError):
        return None


def stopped(output: Path):
    return (output / "STOP").exists() or (SHARED / "STOP").exists()


def attention(output: Path, reason: str, **details):
    state = {
        "kind": "adaptive_vstar191",
        "status": "attention_required",
        "method": METHOD,
        "output": str(output),
        "supervisor_pid": os.getpid(),
        "finished_at": time.time(),
        "reason": reason,
        **details,
    }
    save(output / "supervisor_error.json", state)
    save(output / "supervisor_state.json", state)
    save(SHARED / "adaptive_job.json", state)
    return 1


def supervise(output: Path, smoke_pid: int | None = None, smoke_case="vstar_0"):
    output = output.resolve()
    lock = (output / "supervisor.lock").open("a+")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    smoke_child = None
    if smoke_pid is None:
        if stopped(output):
            return attention(output, "STOP present; smoke and full run not launched")
        command = [
            sys.executable,
            "-u",
            str(HERE / "run.py"),
            "smoke",
            str(output),
            "--case",
            smoke_case,
        ]
        with (output / "smoke.log").open("a") as log:
            smoke_child = subprocess.Popen(
                command,
                cwd=HERE,
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
            )
        smoke_pid = smoke_child.pid
    initial = process(smoke_pid) if smoke_child is None else None
    if initial:
        argv = initial[2]
        valid_command = any(
            argv[index : index + 2] == ["smoke", str(output)]
            for index in range(len(argv) - 1)
        )
        if not any(Path(arg).name == "run.py" for arg in argv) or not valid_command:
            return attention(
                output,
                "Smoke PID does not match the expected command",
                smoke_pid=smoke_pid,
            )

    state = {
        "kind": "adaptive_vstar191",
        "status": "waiting_for_smoke",
        "method": METHOD,
        "output": str(output),
        "pid": smoke_pid,
        "smoke_pid": smoke_pid,
        "smoke_owned": smoke_child is not None,
        "smoke_case": smoke_case,
        "supervisor_pid": os.getpid(),
        "started_at": time.time(),
    }
    save(output / "supervisor_state.json", state)
    save(SHARED / "adaptive_job.json", state)
    if smoke_child is not None:
        while smoke_child.poll() is None:
            time.sleep(2)
        if smoke_child.returncode != 0:
            return attention(
                output,
                "Owned smoke failed; full run not launched",
                smoke_pid=smoke_pid,
                returncode=smoke_child.returncode,
            )
    while initial:
        current = process(smoke_pid)
        if current is None or current[1] != initial[1]:
            break
        # STOP is handled by the smoke's own client. Do not launch a replacement
        # while it can still have an in-flight physical request.
        time.sleep(2)

    if stopped(output):
        return attention(
            output,
            "STOP present after smoke; full run not launched",
            smoke_pid=smoke_pid,
        )
    row = read(output / "rows" / (smoke_case + ".json"))
    if not row or row.get("status") != "complete":
        return attention(
            output,
            "Smoke ended without a complete result; full run not launched",
            smoke_pid=smoke_pid,
        )
    health = read(output / "health.json")
    if not health or not health.get("ready"):
        return attention(
            output,
            "Fixed smoke cases and actual crop-observation health check are not complete; full run not launched",
            health=health,
        )

    # Another manually launched worker must not be replaced. The child's own
    # identical lock remains the final atomic duplicate-worker guard.
    worker_lock = (output / "worker.lock").open("a+")
    try:
        fcntl.flock(worker_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        return attention(
            output, "An Adaptive worker already owns worker.lock; no worker launched"
        )
    fcntl.flock(worker_lock, fcntl.LOCK_UN)
    worker_lock.close()
    state.update(status="launching", pid=os.getpid(), smoke_completed_at=time.time())
    save(output / "supervisor_state.json", state)
    save(SHARED / "adaptive_job.json", state)
    command = [sys.executable, "-u", str(HERE / "run.py"), "run", str(output)]
    with (output / "execution.log").open("a") as log:
        child = subprocess.Popen(
            command, cwd=HERE, stdout=log, stderr=subprocess.STDOUT
        )
        state.update(
            status="running",
            pid=child.pid,
            worker_pid=child.pid,
            launched_at=time.time(),
        )
        save(output / "supervisor_state.json", state)
        # run.py owns shared adaptive_job.json from this point onward.
        while child.poll() is None:
            time.sleep(2)
    result = read(output / "state.json")
    if (
        child.returncode != 0
        or not result
        or result.get("pid") != child.pid
        or result.get("status") not in TERMINAL
    ):
        return attention(
            output,
            "Adaptive worker exited without a valid terminal state",
            worker_pid=child.pid,
            returncode=child.returncode,
            worker_state=result,
        )
    state.update(
        status=result["status"], finished_at=time.time(), returncode=child.returncode
    )
    save(output / "supervisor_state.json", state)
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--smoke-case",
        default="vstar_0",
        help="Fixed manifest case to check; prior successful health cases are reused.",
    )
    parser.add_argument(
        "--smoke-pid",
        type=int,
        help="Attach to this smoke; omit to launch and own the smoke process.",
    )
    args = parser.parse_args()

    def request_stop(*_):
        (args.output / "STOP").write_text(
            "Supervisor signal requested a graceful stop.\n"
        )

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    raise SystemExit(supervise(args.output, args.smoke_pid, args.smoke_case))
