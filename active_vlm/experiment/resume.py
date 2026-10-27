"""Resume a quiescent worker without discarding receipts or reissuing paid calls."""

import shutil
import time

from ..common import digest, read, save


def prepare_resume(root):
    ended = root / "execution_end.json"
    if not ended.exists() or not read(ended).get("terminal"):
        raise RuntimeError("Resume requires a terminal, stopped worker")
    if read(ended)["status"] in {"COMPLETE", "COMPLETE_WITH_FAILURES"}:
        raise RuntimeError("Experiment is already complete")
    requests = list((root / "requests").glob("*.json"))
    if any(read(path).get("status") == "pending" for path in requests):
        raise RuntimeError("Reconcile pending HTTP requests before resuming")
    ledger = read(root / "physical_ledger.json")
    if ledger["stopped"] or any(
        h["status"] in {"unknown", "dispatched"} for h in ledger["held"].values()
    ):
        raise RuntimeError("Reconcile unresolved dispatches before resuming")
    history_file = root / "worker_history.json"
    history = (
        read(history_file)
        if history_file.exists()
        else dict(original_started_unix=read(root / "progress.json")["started_unix"], segments=[])
    )
    segment = root / "operations" / f"resume_{len(history['segments']) + 1:03d}"
    segment.mkdir(parents=True, exist_ok=False)
    for name in [
        "physical_ledger.json",
        "progress.json",
        "execution_end.json",
        "stop_requested.json",
    ]:
        if (root / name).exists():
            shutil.copy2(root / name, segment / name)
    history["segments"].append(
        dict(
            resumed_unix=time.time(),
            archive=str(segment),
            old_physical_paid_usd=ledger["physical_paid"],
            request_hashes={str(p): digest(p.read_bytes()) for p in requests},
        )
    )
    save(history_file, history)
    ended.unlink()
    (root / "stop_requested.json").unlink(missing_ok=True)
