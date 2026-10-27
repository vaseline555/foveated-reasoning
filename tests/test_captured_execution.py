"""Optional real-receipt parity checks. Replays never measure new accuracy."""

from copy import deepcopy
from pathlib import Path

import pytest

from active_vlm.answers import parse_answer
from active_vlm.baselines.runner import FoveaRun
from active_vlm.budget import BudgetStore
from active_vlm.common import read
from active_vlm.graph.runner import GraphRun
from active_vlm.transport import Transport, semantic_request

from .replay import no_dispatch
from .test_execution import response

RESULTS = Path(__file__).resolve().parents[1] / "results"


@pytest.mark.parametrize(
    "arm,mode",
    [(a, "native") for a in ["A", "M", "L", "B", "G", "F"]]
    + [(a, "graph-single") for a in ["A", "M", "L"]],
)
def test_entire_captured_trajectory_retains_every_wire_crop_and_answer(tmp_path, arm, mode):
    name = "stratified60" if arm in {"A", "M", "L"} else "subgoal_coverage_backpack_pilot"
    key = "vstar_ec1742ad00f7" if arm in {"A", "M", "L"} else "vstar_70beb59fd7ab"
    if arm == "M":
        key = (
            "mme_b0d1d94816a0"  # Six rounds and 96 captured calls exercise repeated MCMC proposals.
        )
    if arm == "L":
        key = "vstar_8334c554c727"  # All three expansions and 27 leaf probes.
    root = RESULTS / name
    if not (root / "inference_manifest.json").exists():
        pytest.skip("Optional local paid receipts are not distributed with the code")
    case = next(c for c in read(root / "inference_manifest.json") if c["key"] == key)
    recorded_arm = "G" if arm == "B" else arm
    original = read(root / "trajectories" / key / recorded_arm / "result.json")
    receipts = {r["stage"]: r for r in original["receipts"]}
    cfg = read(root / "config.json")
    cfg = dict(cfg, min_dispatch_interval_seconds=0, pacing=dict(cfg["pacing"], mode="fixed"))
    if mode == "graph-single":
        cfg.update(crop_probe_mode=mode, task_policy=dict(cfg.get("task_policy", {}), quorum=1))
    cfg["seed_arm_aliases"] = dict(cfg.get("seed_arm_aliases", {}), B="A")
    seen = []

    def replay(request, meta):
        source = read(root / "requests" / (receipts[meta["stage"]]["request_hash"] + ".json"))
        # Graph planning packets intentionally changed; native probes/readers
        # and every baseline request must still retain their captured wire.
        compare_wire = arm in {"A", "M", "L"} or not meta["stage"].startswith("planner_")
        if compare_wire and semantic_request(request) != semantic_request(source["request"]):
            import difflib
            import json

            before = json.dumps(
                semantic_request(source["request"]), ensure_ascii=False, indent=2
            ).splitlines()
            after = json.dumps(semantic_request(request), ensure_ascii=False, indent=2).splitlines()
            print("\n".join(difflib.unified_diff(before, after)))
            raise AssertionError("Captured request changed: " + meta["stage"])
        seen.append(meta["stage"])
        captured = response(meta, deepcopy(source["message"]))
        captured["id"] = source["receipt_id"]  # Later planner packets cite this exact ID.
        return captured

    with no_dispatch():
        store = BudgetStore(tmp_path / "ledger.json", cfg, 0, 30, [key + "/" + arm])
        transport = Transport(tmp_path, store, cfg, responder=replay)
        cls = GraphRun if arm in {"B", "F", "G"} else FoveaRun
        result = cls(case, arm, transport, tmp_path / arm).execute()
    final = next(
        e["message"]["content"] for e in original["events"] if e.get("stage") == "aggregator"
    )
    assert result["prediction"] == parse_answer(final, case["allowed_letters"])
    assert result["calls"] == original["calls"]
    assert result["crop_rounds"] == original["crop_rounds"]
    assert result["stop_reason"] == original["stop_reason"]
    assert [r["selected_id"] for r in result["rounds"]] == [
        r["selected_id"] for r in original["rounds"]
    ]
    assert len(seen) == original["calls"]
    assert all(read(p)["backend"] == "mock" for p in (tmp_path / "requests").glob("*.json"))
