"""Portable receipt accounting, exact result reuse, and frozen run configuration."""

from copy import deepcopy
from decimal import Decimal
from pathlib import Path

from ..answers import answer_mode, direct_mode
from ..common import REPO, digest, read, save
from ..protocol import crop_protocol
from ..readers import recovery_policy


def snapshot(exclude=None, account_root=None):
    """Include historical holds once, plus all new physical attempts in results/."""
    account = Path(account_root).resolve() if account_root else REPO
    baseline = account / "results/accounting_baseline.json"
    initial = read(baseline) if baseline.exists() else {}
    known = Decimal(str(initial.get("known_usd", 0)))
    held = Decimal(str(initial.get("reserved_usd", 0)))
    unresolved = initial.get("unresolved", 0)
    receipts = {}
    for path in (account / "results").glob("*/requests/*.json"):
        if exclude and path.parent.parent.resolve() == Path(exclude).resolve():
            continue
        record = read(path)
        if record.get("backend") == "mock" or record.get("imported_from") or record.get("cached"):
            continue
        amount = record.get("cost_usd")
        if amount is None:
            held += Decimal(str(record["reserve_usd"]))
            unresolved += 1
        else:
            receipt = record.get("receipt_id") or str(path)
            charge = Decimal(str(amount))
            if receipt in receipts and receipts[receipt] != charge:
                raise ValueError("RECEIPT_COST_CHANGED")
            receipts[receipt] = charge
    known += sum(receipts.values(), Decimal(0))
    return dict(
        known_usd=float(known),
        reserved_usd=float(held),
        known_or_reserved_usd=float(known + held),
        unresolved=unresolved,
    )


def returned(result):
    return {"direct", "aggregator"} <= {event.get("stage") for event in result.get("events", [])}


def finished(result):
    """A recorded local failure is terminal too; never silently pay to rerun it."""
    if "path_terminal" in result:
        return result["path_terminal"] is True
    return returned(result) or result.get("status") == "planning_failed"


def relocate(value, mappings):
    if isinstance(value, str):
        for old, new in sorted(mappings.items(), key=lambda item: -len(item[0])):
            if value == old or value.startswith(old + "/"):
                return new + value[len(old) :]
    if isinstance(value, list):
        return [relocate(item, mappings) for item in value]
    if isinstance(value, dict):
        return {key: relocate(item, mappings) for key, item in value.items()}
    return value


def verify(root, mappings=None):
    frozen = read(root / "freeze.json")
    if digest(frozen["hashes"]) != frozen["fingerprint"]:
        raise RuntimeError("INVALID_SOURCE_FINGERPRINT")
    for name, checksum in frozen["hashes"].items():
        path = Path(relocate(name, mappings or {}))
        if not path.is_file() or digest(path.read_bytes()) != checksum:
            raise RuntimeError(f"Frozen input/source changed: {path}")
    return frozen


def freeze(root):
    files = list((REPO / "active_vlm").rglob("*.py"))
    files += list((REPO / "active_vlm/baselines/templates").glob("*"))
    files += [REPO / "run_experiment.py"]
    files += [REPO / name for name in ["pyproject.toml", "requirements.txt", ".python-version"]]
    files += [
        root / name
        for name in [
            "config.json",
            "plan.json",
            "inference_manifest.json",
            "dataset_sources.json",
            "budget_quote.json",
        ]
    ]
    hashes = {str(path.resolve()): digest(path.read_bytes()) for path in files if path.is_file()}
    save(root / "freeze.json", dict(hashes=hashes, fingerprint=digest(hashes)))


def approval(root, approve=False):
    frozen = verify(root)
    quote = read(root / "budget_quote.json")
    if not quote["ready_for_run"]:
        raise RuntimeError("Preparation blockers: " + str(quote["blockers"]))
    bound = dict(
        fingerprint=frozen["fingerprint"],
        scope=quote["scope"],
        cumulative_cap_usd=quote["cumulative_hard_cap_usd"],
    )
    if approve:
        save(root / "approval.json", dict(**bound, human_authorized=True))
    path = root / "approval.json"
    if not path.exists():
        raise RuntimeError("Review budget_quote.json, then use --approve")
    record = read(path)
    if not record.get("human_authorized") or any(record.get(k) != v for k, v in bound.items()):
        raise RuntimeError("Approval does not match this preparation")
    return record


def reuse(cases, arms, roots, output, config, mappings=None):
    completed, cached = {}, {}
    bykey = {case["key"]: case for case in cases}
    for root in roots:
        verify(root, mappings)
        previous = {case["key"]: case for case in read(root / "inference_manifest.json")}
        cfg = read(root / "config.json")
        if answer_mode(cfg) != answer_mode(config):
            raise ValueError("Reuse protocol mismatch: final_answer_mode")
        if direct_mode(cfg) != direct_mode(config):
            raise ValueError("Reuse protocol mismatch: direct_answer_mode")
        if recovery_policy(cfg) != recovery_policy(config):
            raise ValueError("Reuse protocol mismatch: native_reader_recovery")
        for field in ["model", "K", "temperature", "max_tokens"]:
            if cfg.get(field) != config[field]:
                raise ValueError(f"Reuse protocol mismatch: {field}")
        for arm in arms:
            if arm in {"B", "F"}:
                continue
            if crop_protocol(cfg, arm) != crop_protocol(config, arm):
                raise ValueError(f"Reuse protocol mismatch: crop_probe_mode for {arm}")
        if cfg.get("seed") != config["seed"]:
            raise ValueError("Reuse seed mismatch")
        if cfg.get("provider", config["provider"]) != config["provider"]:
            raise ValueError("Reuse provider mismatch")
        for key, case in bykey.items():
            if key not in previous:
                continue
            for field in ["question", "image_sha256", "pixel_sha256", "allowed_letters"]:
                if previous[key][field] != case[field]:
                    raise ValueError(f"Reuse input mismatch: {key}/{field}")
            for arm in arms:
                if arm in {"B", "F"}:
                    continue  # Graph and flat controls always run afresh.
                path = root / "trajectories" / key / arm / "result.json"
                if path.exists() and returned(read(path)):
                    completed[key + "/" + arm] = dict(
                        path=str(path), sha256=digest(path.read_bytes())
                    )
        for path in (root / "requests").glob("*.json"):
            record = read(path)
            if record.get("case") not in bykey or record.get("imported_from"):
                continue
            if record.get("arm") == "B" and record.get("stage") != "direct":
                continue
            if record.get("status") == "ok" and record.get("backend") == "api":
                cached[record["request_hash"]] = dict(
                    path=str(path), sha256=digest(path.read_bytes())
                )
    for key, proof in cached.items():
        record = deepcopy(read(proof["path"]))
        record["request"] = relocate(record["request"], mappings or {})
        record.update(imported_from=proof["path"], imported_sha256=proof["sha256"])
        save(output / "requests" / (key + ".json"), record)
    pending = [
        case["key"] + "/" + arm
        for case in cases
        for arm in case["arm_order"]
        if arm in arms and case["key"] + "/" + arm not in completed
    ]
    return dict(completed=completed, pending=pending, cache_imports=cached)
