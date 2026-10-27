"""Detached workers, cooperative stop, and cheap file-based progress monitoring."""

import os
import subprocess
import sys
import time
from pathlib import Path

from ..common import REPO, read, save


def progress(root, **changes):
    path = root / "progress.json"
    value = read(path) if path.exists() else {}
    value.update(changes, updated_unix=time.time())
    save(path, value)
    return value


def status(root):
    path = root / "progress.json"
    value = read(path) if path.exists() else dict(status="PREPARED")
    marker = root / "worker.json"
    pid = value.get("pid") or (read(marker).get("pid") if marker.exists() else None)
    alive = False
    if pid:
        try:
            os.kill(pid, 0)
            alive = True
        except ProcessLookupError:
            pass
        except PermissionError:
            # EPERM still establishes that the PID exists across a sandbox boundary.
            alive = True
    value["worker_alive"] = alive
    value["seconds_since_update"] = round(time.time() - value.get("updated_unix", time.time()), 1)
    if value.get("status") in {"RUNNING", "STARTING"} and not alive:
        value["status"] = "WORKER_EXITED_CHECK_WORKER_LOG"
    return value


def start(root, approve=False, key_file=None, dry_run=False, resume=False):
    from .accounting import approval

    if not dry_run:
        approval(root, approve)
    if status(root)["worker_alive"]:
        raise RuntimeError("A worker is already running for this output")
    command = [sys.executable, str(REPO / "run_experiment.py"), "run", str(root)]
    if dry_run:
        command.append("--dry-run")
    if resume:
        command.append("--resume")
    if key_file:
        command += ["--api-key-file", str(Path(key_file).resolve())]
    with (root / "worker.log").open("a") as log:
        worker = subprocess.Popen(
            command,
            cwd=REPO,
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    # Worker owns progress.json; parent writes a separate marker to avoid a race.
    save(root / "worker.json", dict(pid=worker.pid, started_unix=time.time()))
    return dict(pid=worker.pid, output=str(root), log=str(root / "worker.log"))


def stop(root):
    save(root / "stop_requested.json", dict(requested_unix=time.time()))
    return "Stop requested; worker exits at the next request boundary."
