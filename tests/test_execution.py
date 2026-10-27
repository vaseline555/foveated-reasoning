"""Offline checks of actual selectors, shared readers, retries, and process control."""

import io
import json
import subprocess
import sys
import time
import urllib.error
from copy import deepcopy

import pytest
from PIL import Image

from active_vlm.baselines.runner import FoveaRun
from active_vlm.budget import BudgetStop, BudgetStore, UnknownCharge
from active_vlm.common import MODEL, PROVIDER, REPO, canonical, read, save
from active_vlm.experiment import accounting, process
from active_vlm.experiment.cli import main
from active_vlm.experiment.data import materialize
from active_vlm.transport import Transport, semantic_request

from .replay import no_dispatch


@pytest.fixture(autouse=True)
def block_network():
    with no_dispatch():
        yield


def config():
    return dict(
        model=MODEL,
        provider=PROVIDER,
        seed=0,
        K=3,
        temperature=0.7,
        max_tokens=2048,
        request_upper_usd=0.067072,
        per_case_usd_cap=3,
        cumulative_cap_usd=30,
        input_price_per_million=0.25,
        output_price_per_million=0.75,
        retry_cooldown_seconds=0,
        min_dispatch_interval_seconds=0,
    )


def make_case(tmp_path):
    path = tmp_path / "input.png"
    Image.new("RGB", (300, 200), "blue").save(path)
    return materialize(
        [
            dict(
                dataset="vstar",
                id="fixture",
                image_path=str(path),
                question="What color is the image?\n(A) Blue\n(B) Red",
            )
        ],
        tmp_path,
    )[0]


def response(meta, message=None):
    return dict(
        id="mock-" + meta["request_hash"],
        model=MODEL,
        provider="Parasail",
        usage=dict(cost=0.001),
        choices=[
            dict(
                finish_reason="stop",
                message=message or dict(role="assistant", content="<answer>A</answer>"),
            )
        ],
    )


class SelectorResponder:
    def __init__(self):
        self.inputs = []

    def __call__(self, request, meta):
        self.inputs.append(deepcopy(meta))
        stage = meta["stage"]
        message = dict(role="assistant", content="<answer>A</answer>")
        if stage in {"direct", "aggregator"} and request.get("response_format"):
            message["content"] = '{"answer":"A"}'  # Labelled schema-conforming mock response.
        if stage == "planner_1" or stage.startswith("expansion_"):
            inner = stage.startswith("expansion_")
            args = dict(coordinates=[200, 200, 600, 600])
            args["image" if inner else "image_url"] = "image_1" if inner else "original"
            message.update(
                content="Inspect the target.",
                tool_calls=[
                    dict(
                        id="crop",
                        type="function",
                        function=dict(
                            name="crop" if inner else "cropping", arguments=canonical(args)
                        ),
                    )
                ],
            )
        elif stage.startswith("probe_"):
            message["content"] = "<answer>yes</answer>"
        if request.get("tools", [{}])[0].get("function", {}).get("name") == "plan_regions":
            properties = request["tools"][0]["function"]["parameters"]["properties"]
            message = dict(
                role="assistant",
                content="Initial regions.",
                tool_calls=[
                    dict(
                        id="regions",
                        type="function",
                        function=dict(
                            name="plan_regions",
                            arguments=canonical({k: [200, 200, 600, 600] for k in properties}),
                        ),
                    )
                ],
            )
        return response(meta, message)


def test_all_fovea_selectors_run_with_one_shared_direct(tmp_path):
    case = make_case(tmp_path)
    arms = ["A", "M", "L"]
    budget = BudgetStore(
        tmp_path / "physical_ledger.json",
        config(),
        0,
        30,
        [case["key"] + "/" + arm for arm in arms],
    )
    responder = SelectorResponder()
    transport = Transport(tmp_path, budget, config(), responder=responder)
    results = [FoveaRun(case, arm, transport, tmp_path / arm).execute() for arm in arms]
    assert all(r["prediction"] == "A" and r["crop_rounds"] == 1 for r in results)
    assert all(not r["errors"] for r in results)
    assert sum(m["stage"] == "direct" for m in responder.inputs) == 1
    assert all(r["probe_calls"] % 3 == 0 and r["probe_calls"] >= 3 for r in results)
    assert any(m["stage"].startswith("expansion_") for m in responder.inputs)
    state = read(budget.path)
    assert not state["held"] and not state["final_pool"]


def test_transient_retries_preserve_payload_and_unknown_money(tmp_path):
    cfg = config()
    budget = BudgetStore(tmp_path / "ledger.json", cfg, 0, 30, ["case/A"])
    attempts = []

    def responder(request, meta):
        attempts.append(deepcopy(request))
        if len(attempts) <= 2:
            raise urllib.error.HTTPError("offline", 429, "rate limit", {}, io.BytesIO(b"{}"))
        return response(meta)

    transport = Transport(tmp_path, budget, cfg, responder=responder)
    token = budget.reserve("case/A", ["planner_1"])[0]
    message, _ = transport.call("case", "A", "planner_1", [], token)
    assert message["content"] == "<answer>A</answer>"
    assert len(attempts) == 3 and all(r == attempts[0] for r in attempts)
    state = read(budget.path)
    assert state["paths"]["case/A"]["calls"] == 1
    assert len(state["held"]) == 2 and not state["stopped"]
    assert all(h["status"] == "authorized_unknown" for h in state["held"].values())
    assert len(list((tmp_path / "requests").glob("*.failed_attempt_*.json"))) == 2
    assert process.status(tmp_path)["failed_attempts"] == 2


def test_retry_cannot_exceed_cumulative_cap(tmp_path):
    cfg = config()
    # Two final readers + one exploratory attempt fit; a second attempt cannot.
    budget = BudgetStore(tmp_path / "ledger.json", cfg, 0, cfg["request_upper_usd"] * 3, ["case/A"])
    token = budget.reserve("case/A", ["planner_1"])[0]
    budget.dispatched(token)
    with pytest.raises(UnknownCharge):
        budget.settle(token, None)
    with pytest.raises(BudgetStop, match="CUMULATIVE"):
        budget.retry_token(token, "hash")


def test_auth_error_is_not_retried(tmp_path):
    calls = []

    def denied(request, meta):
        calls.append(meta)
        raise urllib.error.HTTPError("offline", 401, "denied", {}, io.BytesIO(b"{}"))

    budget = BudgetStore(tmp_path / "ledger.json", config(), 0, 30, ["case/A"])
    transport = Transport(tmp_path, budget, config(), responder=denied)
    token = budget.reserve("case/A", ["planner_1"])[0]
    with pytest.raises(urllib.error.HTTPError):
        transport.call("case", "A", "planner_1", [], token)
    assert len(calls) == 1 and read(budget.path)["stopped"]


def test_background_worker_dry_run_and_stop(tmp_path):
    root = tmp_path / "account/results/dry_run"
    case = make_case(tmp_path)
    manifest = tmp_path / "manifest.json"
    save(manifest, [case])
    try:
        main(
            [
                "prepare",
                "--manifest",
                str(manifest),
                "--benchmarks",
                "vstar",
                "--samples",
                "1",
                "--methods",
                "graph",
                "--account-root",
                str(tmp_path / "account"),
                "--output",
                str(root),
            ]
        )
        with pytest.raises(RuntimeError, match="--approve"):
            accounting.approval(root)
        worker = process.start(root, dry_run=True)
        for _ in range(100):
            if read(root / "progress.json")["status"] == "DRY_RUN_COMPLETE":
                break
            time.sleep(0.03)
        assert process.status(root)["status"] == "DRY_RUN_COMPLETE"
        assert worker["pid"] > 0 and not (root / "requests").exists()
        assert not (root / "approval.json").exists()
        process.stop(root)
        assert (root / "stop_requested.json").exists()
        output = subprocess.check_output(
            [sys.executable, str(REPO / "run_experiment.py"), "status", str(root)], text=True
        )
        assert json.loads(output)["status"] == "DRY_RUN_COMPLETE"
    finally:
        import shutil

        shutil.rmtree(root)


def test_request_units_seed_and_english_contract(tmp_path):
    budget = BudgetStore(tmp_path / "ledger.json", config(), 0, 30, ["case/A"])
    transport = Transport(tmp_path, budget, config(), responder=lambda *args: None)
    request = transport.request("case", "A", "direct", [])
    assert request["seed"] == transport.request("case", "B", "direct", [])["seed"]
    assert request["provider"]["max_price"]["prompt"] == 0.25
    with pytest.raises(ValueError, match="ENGLISH"):
        transport.request("case", "A", "planner_1", [dict(role="user", content="한글")])
    assert semantic_request(request) == request
