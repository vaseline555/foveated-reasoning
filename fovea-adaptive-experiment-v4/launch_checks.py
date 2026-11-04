"""Attach the first live smoke, finish fixed health checks, then run once.

Operational orchestration only. Frozen run.py owns all API, ledger and budget
operations. This file never changes frozen files or the shared job metadata.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import signal
import sqlite3
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
METHOD = "ADAPTIVE_V4"
MODEL = "qwen/qwen3-vl-8b-instruct"


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def process(pid):
    try:
        root = Path("/proc") / str(pid)
        fields = (root / "stat").read_text().rsplit(") ", 1)[1].split()
        if fields[0] in {"Z", "X"}:
            return None
        return {
            "start_ticks": fields[19],
            "argv": [
                x.decode() for x in (root / "cmdline").read_bytes().split(b"\0") if x
            ],
        }
    except (FileNotFoundError, ProcessLookupError):
        return None


def is_first_smoke(info, output):
    argv = info["argv"]
    scripts = [
        Path(arg)
        for arg in argv
        if Path(arg).name == "run.py" and Path(arg).is_absolute()
    ]
    command_ok = any(
        argv[i : i + 2] == ["smoke", str(output)] for i in range(len(argv) - 1)
    )
    selected = argv[argv.index("--case") + 1] if "--case" in argv else "vstar_0"
    return HERE / "run.py" in scripts and command_ok and selected == "vstar_0"


def completed(output, case):
    path = output / "rows" / f"{case}.json"
    if not path.exists():
        return False
    row = read(path)
    return (
        row.get("status") == "complete"
        and row.get("method") == METHOD
        and row.get("models") == [MODEL]
    )


def health_ready(output):
    path = output / "health.json"
    if not path.exists():
        return False
    health = read(path)
    return (
        health.get("ready") is True
        and completed(output, "vstar_0")
        and completed(output, "vstar_115")
        and completed(output, "vstar_1")
        and bool(health.get("crop_observation_cases"))
    )


def guard(output, shared):
    if (output / "STOP").exists() or (shared / "STOP").exists():
        return "STOP marker present"
    with sqlite3.connect(f"file:{shared / 'ledger.sqlite'}?mode=ro", uri=True) as db:
        count = db.execute(
            "SELECT COUNT(*) FROM attempts WHERE status='unknown' "
            "OR (method=? AND cost IS NULL)",
            (METHOD,),
        ).fetchone()[0]
        if count:
            return "Unresolved Adaptive reservation or unknown charge; inspect ledger"
        if db.execute("SELECT paused FROM api_health WHERE singleton=1").fetchone()[0]:
            return "Shared API health is paused"
    return None


def main(output, first_pid):
    output = output.resolve()
    with (output / "launch_checks.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        config = read(output / "config.json")
        shared = Path(config["shared_root"])
        state = {
            "status": "waiting_first_smoke",
            "method": METHOD,
            "monitor_pid": os.getpid(),
            "first_smoke_pid": first_pid,
            "output": str(output),
            "started_at": time.time(),
            "orchestrator_sha256": hashlib.sha256(
                Path(__file__).read_bytes()
            ).hexdigest(),
        }

        def update(**details):
            state.update(details, updated_at=time.time())
            save(output / "validation_state.json", state)

        def fail(reason, status="attention_required", **details):
            update(status=status, reason=reason, finished_at=time.time(), **details)
            save(output / f"validation_error.{time.time_ns()}.json", state)
            save(output / "validation_error.json", state)
            return 1

        def wait_child(command, stage, log_name):
            problem = guard(output, shared)
            if problem:
                return None, problem
            update(status="launching", stage=stage, child_pid=None, command=command)
            with (output / log_name).open("a") as log:
                child = subprocess.Popen(
                    command,
                    cwd=HERE,
                    stdin=subprocess.DEVNULL,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                )
            update(status="running", stage=stage, child_pid=child.pid)
            while child.poll() is None:
                time.sleep(2)
            update(returncode=child.returncode, child_finished_at=time.time())
            return child, None

        if config.get("method") != METHOD or config.get("shared_global_cap_usd") != 15:
            return fail("Expected frozen Adaptive V3 with the existing $15 campaign")
        initial = process(first_pid)
        if initial:
            if not is_first_smoke(initial, output):
                return fail("First smoke PID command does not match this frozen run")
            update(first_smoke_start_ticks=initial["start_ticks"])
            while True:
                current = process(first_pid)
                if current is None:
                    break
                if current["start_ticks"] != initial["start_ticks"]:
                    return fail("First smoke PID was reused")
                time.sleep(2)
        else:
            update(first_smoke_already_ended=True)
        worker_state = read(output / "state.json")
        if (
            not completed(output, "vstar_0")
            or worker_state.get("pid") != first_pid
            or worker_state.get("status") != "smoke_complete"
        ):
            return fail(
                "First smoke ended without its completed result",
                worker_state=worker_state,
            )

        for case in ("vstar_115", "vstar_1", "vstar_2"):
            if case == "vstar_2" and health_ready(output):
                break
            command = [
                sys.executable,
                "-u",
                str(HERE / "run.py"),
                "smoke",
                str(output),
                "--case",
                case,
            ]
            child, problem = wait_child(command, "smoke_" + case, "smoke.log")
            if problem:
                return fail(problem)
            result = read(output / "state.json")
            if child.returncode != 0 or not completed(output, case):
                return fail(
                    "Smoke failed; no further calls started", worker_state=result
                )
        if not health_ready(output):
            return fail(
                "Fixed checks did not establish actual crop-observation health",
                health=read(output / "health.json"),
            )
        update(status="healthy", health=read(output / "health.json"))
        command = [sys.executable, "-u", str(HERE / "run.py"), "run", str(output)]
        child, problem = wait_child(command, "full191", "execution.log")
        if problem:
            return fail(problem)
        result = read(output / "state.json")
        if (
            child.returncode != 0
            or result.get("pid") != child.pid
            or result.get("status")
            not in {"complete", "stopped", "paused", "attention_required"}
        ):
            return fail("Full worker ended; no automatic retry", worker_state=result)
        update(status=result["status"], finished_at=time.time(), worker_state=result)
        return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--first-smoke-pid", required=True, type=int)
    args = parser.parse_args()

    def stop(*_):
        with (args.output / "STOP").open("a") as target:
            target.write("Graceful stop requested by validation orchestrator signal.\n")

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    try:
        raise SystemExit(main(args.output, args.first_smoke_pid))
    except (OSError, ValueError, KeyError, sqlite3.Error) as exc:
        record = {
            "status": "attention_required",
            "monitor_pid": os.getpid(),
            "reason": type(exc).__name__ + ": " + str(exc),
            "at": time.time(),
        }
        save(args.output / f"validation_error.{time.time_ns()}.json", record)
        save(args.output / "validation_error.json", record)
        save(args.output / "validation_state.json", record)
        raise SystemExit(1)
