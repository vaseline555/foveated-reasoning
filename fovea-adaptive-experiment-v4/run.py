"""Run the isolated Adaptive draft alongside the frozen 8B baselines."""

from __future__ import annotations

import argparse
import asyncio
import fcntl
import hashlib
import json
import os
import platform
import random
import signal
import time
import traceback
from dataclasses import asdict, is_dataclass
from importlib.metadata import version
from pathlib import Path

from backend import Backend, adaptive
from transport import (
    AdaptiveClient,
    BudgetStop,
    RateLimitPause,
    RequestFailure,
    UnknownCharge,
)

HERE = Path(__file__).resolve().parent
SHARED = Path(
    "/workspace/fovea-reproduction-8b-recovery/runs/vstar191-8b-only-20261103"
)
BASE_CODE = Path("/workspace/fovea-reproduction-8b-recovery-v2")
PREP = Path("/workspace/vstar-prepared")
POLICY = Path("/workspace/fovea-official/src/cv_agent/policies/adaptive.py")
METHOD = "ADAPTIVE_V4"


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def plain(value):
    if is_dataclass(value):
        return plain(asdict(value))
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, set):
        return [plain(v) for v in sorted(value)]
    if isinstance(value, (tuple, list)):
        return [plain(v) for v in value]
    return value


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(plain(value), ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def parse_answer(text):
    import re

    match = re.search(r"<answer>(.*?)</answer>", text or "", re.IGNORECASE | re.DOTALL)
    cleaned = (match.group(1) if match else text or "").strip().upper()
    if "ERROR" in cleaned or "FAILED" in cleaned:
        return None
    match = re.search(r"(?:^|[^A-Z])\(?([A-D])\)?\.?$", cleaned)
    return match.group(1) if match else None


def direct_result(case):
    for folder in ("result_overlays", "rows"):
        path = SHARED / folder / "Direct" / (case["key"] + ".json")
        if path.exists():
            row = read(path)
            if row.get("status") != "complete" or row.get("models") != [
                "qwen/qwen3-vl-8b-instruct"
            ]:
                raise ValueError("A completed pure8B Direct report is required")
            return row, path
    raise ValueError("Direct result missing: " + case["key"])


def client_for(output, config):
    cfg = read(SHARED / "config-v2.json")
    return AdaptiveClient(
        SHARED,
        cfg["models"],
        cap=cfg["cap_usd"],
        historical_usd=cfg["historical"]["committed_usd"],
        min_request_interval=cfg["dispatch_interval_seconds"],
        rate_retry_limit=cfg["rate_retry_limit"],
        stop_callback=lambda: (output / "STOP").exists(),
    )


def verify_sources(config):
    for path, expected in config["source_hashes"].items():
        if sha(path) != expected:
            raise ValueError("Frozen source/input changed: " + path)
    if (
        config.get("runtime_versions")
        and config["runtime_versions"] != runtime_versions()
    ):
        raise ValueError("Frozen Python/library versions changed")


def runtime_versions():
    return {
        "python": platform.python_version(),
        "Pillow": version("Pillow"),
        "jsonschema": version("jsonschema"),
    }


def prepare(output):
    output.mkdir(parents=True, exist_ok=True)
    if (output / "config.json").exists():
        return read(output / "config.json")
    shared = read(SHARED / "config-v2.json")
    if shared["cap_usd"] != 15 or any(
        m["id"] != "qwen/qwen3-vl-8b-instruct" for m in shared["models"].values()
    ):
        raise ValueError("Shared campaign must remain 8B-only and capped at $15")
    manifest = read(PREP / "manifest.json")
    if len(manifest["cases"]) != 191:
        raise ValueError("Expected complete V*191 manifest")
    inputs = {
        str(p): sha(p)
        for p in [
            POLICY,
            HERE / "backend.py",
            HERE / "transport.py",
            HERE / "run.py",
            HERE / "supervisor.py",
            BASE_CODE / "client.py",
            BASE_CODE / "official_adapter.py",
            PREP / "manifest.json",
            SHARED / "config-v2.json",
        ]
    }
    for case in manifest["cases"]:
        if sha(case["image_path"]) != case["image_sha256"]:
            raise ValueError("Source image changed")
        direct_result(case)
    cfg = {
        "created_at": time.time(),
        "method": METHOD,
        "shared_root": str(SHARED),
        "shared_global_cap_usd": 15,
        "historical_in_shared_cap": shared["historical"]["committed_usd"],
        "models": shared["models"],
        "seed": shared["seed"],
        "max_workers": 1,
        "max_consecutive_structured_failures": 3,
        "require_crop_observation_health": True,
        "health_cases": ["vstar_0", "vstar_115", "vstar_1"],
        "max_tokens": 2048,
        "temperature": 1.0,
        "dispatch_interval_seconds": shared["dispatch_interval_seconds"],
        "limits": plain(
            adaptive.Limits(
                max_steps=10,
                max_model_calls=24,
                max_candidates=32,
                branch_steps=1,
                progress_bonus=1,
                no_progress_steps=1,
                max_visits=2,
            )
        ),
        "source_hashes": inputs,
        "runtime_versions": runtime_versions(),
        "scope": "All191 pure8B with same model/provider, images, sampling and final Direct report as baselines.",
        "differences": [
            "Adaptive uses explicit role-plan/proposal/observation calls instead of Greedy 3x5 probes.",
            "Hard maximum10 crops and24 logical backend calls including final; bounded format retries are billed.",
            "No geometric merging, area rejection, center-distance ranking, or 30B fallback.",
            "Dollar cost and pacing share the existing campaign ledger with concurrently running baselines.",
        ],
    }
    save(output / "config.json", cfg)
    return cfg


def smoke_records(output):
    return [read(p) for p in (output / "smoke_attempts").glob("*.json")]


def refresh_health(output):
    rows = [read(p) for p in (output / "rows").glob("*.json")]
    complete = {r["case"]: r for r in rows if r["status"] == "complete"}
    unresolved = sorted(
        {r["case"] for r in smoke_records(output) if r["status"] == "execution_failure"}
        - complete.keys()
    )
    observed = sorted(
        k
        for k, r in complete.items()
        if r.get("crops", 0) > 0 and r.get("observed_crops", 0) > 0
    )
    config_path = output / "config.json"
    required = (
        read(config_path).get("health_cases", ["vstar_0", "vstar_115", "vstar_1"])
        if config_path.exists()
        else ["vstar_0", "vstar_115", "vstar_1"]
    )
    missing = sorted(set(required) - complete.keys())
    health = {
        "updated_at": time.time(),
        "method": METHOD,
        "ready": bool(complete and observed and not unresolved and not missing),
        "required_cases": required,
        "missing_required_cases": missing,
        "completed_final_cases": sorted(complete),
        "crop_observation_cases": observed,
        "unresolved_smoke_failures": unresolved,
        "requirement": "All three fixed smoke cases completed, at least one actual crop+observation, and no unresolved smoke failure. Accuracy is not a health criterion.",
    }
    save(output / "health.json", health)
    return health


def publish_state(output, state):
    save(output / "state.json", state)
    save(SHARED / "adaptive_job.json", state)


def error_category(event):
    error = event.get("error") or ""
    if error.startswith("HTTP 429"):
        return "api_rate_limit"
    if error.startswith("HTTP "):
        return "api_error"
    response = Path(event["response_path"]) if event.get("response_path") else None
    if response and response.exists():
        choices = read(response).get("choices") or []
        if choices and choices[0].get("finish_reason") == "length":
            return "output_length"
    if event["status"] == "unknown":
        return "unresolved_charge"
    return "output_validation" if event["status"] == "invalid" else "api_rejection"


def summarize(output, client):
    rows = [read(p) for p in (output / "rows").glob("*.json")]
    complete = [r for r in rows if r["status"] == "complete"]
    labels = read(PREP / "labels.evaluation-only.json")
    correct = sum(r["prediction"] == labels[r["case"]] for r in complete)
    with client.connect() as db:
        # Canceled/pending final-reader reservations are not physical API attempts.
        physical = dict(
            db.execute(
                """SELECT COUNT(*) attempts,COALESCE(SUM(cost),0) paid_usd,
            COALESCE(SUM(CASE WHEN cost IS NULL THEN reserved ELSE 0 END),0) held_usd
            FROM attempts WHERE method=? AND status NOT IN ('reserved_final','cancelled_reservation')""",
                (METHOD,),
            ).fetchone()
        )
        errors = [
            dict(r)
            for r in db.execute(
                """SELECT case_id,stage,status,error,model,response_path,cost
            FROM attempts WHERE method=? AND status IN ('invalid','rejected','unknown') ORDER BY started""",
                (METHOD,),
            )
        ]
        requested_cases = {
            r[0]
            for r in db.execute(
                "SELECT DISTINCT case_id FROM attempts WHERE method=?", (METHOD,)
            )
            if r[0]
        }
    for event in errors:
        event["category"] = error_category(event)
    complete_cases = {r["case"] for r in complete}
    failed_rows = {r["case"] for r in rows if r["status"] != "complete"}
    failed_smoke = {
        r["case"] for r in smoke_records(output) if r["status"] == "execution_failure"
    } - complete_cases
    failed_cases = failed_rows | failed_smoke
    attempted = (
        {r["case"] for r in rows}
        | {r["case"] for r in smoke_records(output)}
        | requested_cases
    )
    incomplete = {r["case"]: r for r in rows if r["status"] != "complete"}
    for record in sorted(smoke_records(output), key=lambda r: r.get("completed_at", 0)):
        if record["case"] not in complete_cases:
            incomplete[record["case"]] = record
    counted_work = complete + list(incomplete.values())
    categories = {}
    for event in errors:
        categories[event["category"]] = categories.get(event["category"], 0) + 1
    report = {
        "updated_at": time.time(),
        "method": METHOD,
        "planned_n": 191,
        "completed": len(complete),
        "correct": correct,
        "accuracy_completed": correct / len(complete) if complete else None,
        "accuracy_full191": correct / 191 if len(complete) == 191 else None,
        "failed": len(failed_cases),
        "failed_result_rows": len(failed_rows),
        "failed_smoke_cases": len(failed_smoke),
        "pending": 191 - len(complete_cases | failed_cases),
        "not_attempted": 191 - len(attempted),
        "partial_attempted_cases": len(attempted - complete_cases - failed_cases),
        "no_answer": sum(r["prediction"] is None for r in complete),
        "crops": sum(r.get("crops", 0) for r in counted_work),
        "crop_attempts": sum(
            r.get("crop_attempts", r.get("crops", 0)) for r in counted_work
        ),
        "physical": physical,
        "physical_scope": "ADAPTIVE_V4 requests only; all earlier Adaptive costs remain included in shared_campaign and are not added again.",
        "request_error_events": len(errors),
        "request_error_categories": categories,
        "shared_campaign": client.stats(),
        "reservation_accounting": client.final_reservation_stats(),
    }
    save(output / "progress.json", report)
    save(
        output / "harness_errors.json",
        {
            "updated_at": time.time(),
            "request_events": errors,
            "failed_results": [r for r in rows if r["status"] != "complete"],
            "smoke_attempts": smoke_records(output),
            "note": "Model NO_ANSWER results are scored zero and excluded from harness failures.",
        },
    )
    return report


async def execute_case(case, output, client, config):
    path = output / "rows" / (case["key"] + ".json")
    if path.exists():
        return read(path)
    if (output / "STOP").exists() or (SHARED / "STOP").exists():
        raise BudgetStop("STOP requested")
    direct, direct_path = direct_result(case)
    folder = output / "cases" / case["key"]
    folder.mkdir(parents=True, exist_ok=True)
    client.reserve_final(case["key"])
    try:
        backend = Backend(case, SHARED, folder, client, direct)
        policy = adaptive.AdaptivePolicy(adaptive.Limits(**config["limits"]))
        result = await policy.run(
            case["question"], backend.original_image_key, backend, backend.executor
        )
        row = {
            "case": case["key"],
            "method": METHOD,
            "status": "complete",
            "answer": result.answer,
            "prediction": parse_answer(result.answer),
            "grade": "NO_ANSWER" if parse_answer(result.answer) is None else "MCQ",
            "models": ["qwen/qwen3-vl-8b-instruct"],
            "crops": len(backend.image_registry) - 1,
            "crop_attempts": result.state.steps,
            "logical_model_calls": result.state.model_calls,
            "observed_crops": sum(
                1
                for o in getattr(result.state, "observations", ())
                if o.candidate is not None and not o.error
            ),
            "stop_reason": result.state.stop_reason,
            "direct_source": str(direct_path),
            "direct_sha256": sha(direct_path),
            "completed_at": time.time(),
        }
        save(folder / "state.json", result.state)
        provenance = (
            backend.provenance() if callable(backend.provenance) else backend.provenance
        )
        save(folder / "provenance.json", provenance)
        save(path, row)
        return row
    except Exception as exc:
        if "backend" in locals():
            states = [
                event["state"]
                for event in backend.events
                if isinstance(event.get("state"), dict)
            ]
            latest = states[-1] if states else {}
            save(
                folder / "partial.json",
                {
                    "case": case["key"],
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "crops": len(backend.image_registry) - 1,
                    "crop_attempts": latest.get(
                        "steps", len(backend.image_registry) - 1
                    ),
                    "observed_crops": sum(
                        event.get("event") == "call_complete"
                        and str(event.get("stage", "")).startswith("observe_")
                        for event in backend.events
                    ),
                    "provenance": backend.provenance,
                },
            )
        raise
    finally:
        client.release_final(case["key"])


def failure_record(case, output, exc, phase):
    partial_path = output / "cases" / case["key"] / "partial.json"
    partial = read(partial_path) if partial_path.exists() else {}
    return {
        "case": case["key"],
        "method": METHOD,
        "phase": phase,
        "status": "execution_failure",
        "error_type": type(exc).__name__,
        "error": str(exc),
        "completed_at": time.time(),
        "crops": partial.get("crops", 0),
        "crop_attempts": partial.get("crop_attempts", 0),
        "observed_crops": partial.get("observed_crops", 0),
    }


def smoke(output, case_key=None):
    with (output / "worker.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return _smoke_locked(output, case_key)


def _smoke_locked(output, case_key):
    config = read(output / "config.json")
    verify_sources(config)
    cases = read(PREP / "manifest.json")["cases"]
    case = (
        next((c for c in cases if c["key"] == case_key), None) if case_key else cases[0]
    )
    if case is None:
        raise ValueError("Unknown manifest case: " + str(case_key))
    client = client_for(output, config)
    state = {
        "status": "smoke_running",
        "phase": "smoke",
        "case": case["key"],
        "pid": os.getpid(),
        "method": METHOD,
        "output": str(output),
        "started_at": time.time(),
    }
    publish_state(output, state)
    exit_code = 0
    try:
        row = asyncio.run(execute_case(case, output, client, config))
        if row["status"] != "complete":
            raise RequestFailure("Existing terminal case is not complete")
        record = dict(row, phase="smoke")
        state.update(status="smoke_complete")
    except (RateLimitPause, BudgetStop, UnknownCharge) as exc:
        status = "paused" if isinstance(exc, RateLimitPause) else "stopped"
        record = dict(failure_record(case, output, exc, "smoke"), status=status)
        state.update(status=status, reason=str(exc))
        exit_code = 1
    except Exception as exc:
        record = failure_record(case, output, exc, "smoke")
        state.update(status="attention_required", reason=record["error"])
        save(
            output / "smoke_failure.json",
            dict(record, traceback=traceback.format_exc()),
        )
        exit_code = 1
    attempt = (
        output / "smoke_attempts" / f"{case['key']}.{time.time_ns()}.{os.getpid()}.json"
    )
    save(attempt, record)
    health = refresh_health(output)
    report = summarize(output, client)
    state.update(finished_at=time.time(), health=health, summary=report)
    publish_state(output, state)
    print(
        json.dumps({"result": record, "health": health, "summary": report}, indent=2),
        flush=True,
    )
    return exit_code


def run(output):
    with (output / "worker.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return _run_locked(output)


def _run_locked(output):
    config = read(output / "config.json")
    verify_sources(config)
    health = refresh_health(output)
    if config.get("require_crop_observation_health", True) and not health["ready"]:
        state = {
            "status": "attention_required",
            "pid": os.getpid(),
            "method": METHOD,
            "output": str(output),
            "reason": "Health check requires a completed final reader and actual crop observation before full191.",
            "finished_at": time.time(),
            "health": health,
        }
        publish_state(output, state)
        return
    client = client_for(output, config)
    cases = read(PREP / "manifest.json")["cases"]
    random.Random(config["seed"]).shuffle(cases)
    started = time.time()
    state = {
        "status": "running",
        "pid": os.getpid(),
        "started_at": started,
        "method": METHOD,
        "output": str(output),
    }
    save(output / "state.json", state)
    save(SHARED / "adaptive_job.json", state)
    status = "complete"
    reason = None
    consecutive_failures = 0
    for case in cases:
        if (output / "STOP").exists() or (SHARED / "STOP").exists():
            status = "stopped"
            reason = "STOP requested"
            break
        try:
            row = asyncio.run(execute_case(case, output, client, config))
            consecutive_failures = (
                0 if row["status"] == "complete" else consecutive_failures + 1
            )
        except RateLimitPause as exc:
            status = "paused"
            reason = str(exc)
            break
        except (BudgetStop, UnknownCharge) as exc:
            status = "stopped"
            reason = str(exc)
            break
        except (RequestFailure, ValueError, KeyError, TypeError) as exc:
            save(
                output / "rows" / (case["key"] + ".json"),
                failure_record(case, output, exc, "full_run"),
            )
            consecutive_failures += 1
        except Exception as exc:
            status = "stopped"
            reason = type(exc).__name__ + ": " + str(exc)
            save(
                output / "fatal_error.json",
                {
                    "case": case["key"],
                    "reason": reason,
                    "traceback": traceback.format_exc(),
                },
            )
            break
        report = summarize(output, client)
        print("ADAPTIVE_PROGRESS " + json.dumps(report), flush=True)
        if consecutive_failures >= config.get("max_consecutive_structured_failures", 3):
            status = "attention_required"
            reason = f"Stopped after {consecutive_failures} consecutive structured/execution failures; remaining cases were not attempted."
            break
    report = summarize(output, client)
    if status == "complete" and report["completed"] != 191:
        status = "stopped"
    state.update(status=status, finished_at=time.time(), reason=reason, summary=report)
    save(output / "state.json", state)
    save(SHARED / "adaptive_job.json", state)
    save(output / "final.json", report)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["prepare", "run", "status", "smoke"])
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--case", help="Manifest case ID for smoke only; never selected using labels."
    )
    args = parser.parse_args()
    if args.command in {"run", "smoke"}:
        for sig in (signal.SIGINT, signal.SIGTERM):
            signal.signal(
                sig,
                lambda *_: (args.output / "STOP").write_text("Signal stop requested\n"),
            )
    if args.command == "prepare":
        print(json.dumps(prepare(args.output), indent=2))
    elif args.command == "status":
        cfg = read(args.output / "config.json")
        print(
            json.dumps(summarize(args.output, client_for(args.output, cfg)), indent=2)
        )
    elif args.command == "smoke":
        raise SystemExit(smoke(args.output, args.case))
    else:
        run(args.output)
