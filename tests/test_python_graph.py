"""Local graph control preserves native probes and never fabricates evidence."""

from copy import deepcopy

import pytest

from active_vlm.baselines.runner import FoveaRun
from active_vlm.budget import BudgetStore
from active_vlm.common import canonical, read
from active_vlm.graph.runner import GraphRun
from active_vlm.graph.search import CropSearch
from active_vlm.graph.state import TaskGraph
from active_vlm.transport import Transport, semantic_request

from .replay import no_dispatch
from .test_execution import config, make_case, response


@pytest.fixture(autouse=True)
def offline():
    with no_dispatch():
        yield


def make_run(tmp_path, probe_text=None, regions=None, bad_regions=False):
    case = make_case(tmp_path)
    case["question"] = "What color is the sign?\n(A) Blue\n(B) Red"
    cfg = dict(
        config(),
        crop_probe_mode="graph-single",
        graph_controller="python",
        seed_arm_aliases={"B": "A", "F": "A"},
    )
    seen = []

    def respond(request, meta):
        seen.append(deepcopy(meta))
        if meta["stage"].startswith("planner_"):
            tool = request["tools"][0]["function"]
            assert tool["name"] == "plan_regions"
            payload = regions or {k: [200, 200, 600, 600] for k in tool["parameters"]["properties"]}
            if bad_regions:
                payload = {"invented_id": [0, 0, 1000, 1000]}
            message = dict(
                role="assistant",
                content="Initial regions.",
                tool_calls=[
                    dict(
                        id="call-" + meta["stage"],
                        type="function",
                        function=dict(name="plan_regions", arguments=canonical(payload)),
                    )
                ],
            )
        elif meta["stage"].startswith("probe_"):
            message = dict(role="assistant", content=probe_text or "No")
        else:
            message = dict(role="assistant", content="<answer>A</answer>")
        return response(meta, message)

    budget = BudgetStore(tmp_path / "ledger.json", cfg, 0, 30, [case["key"] + "/B"])
    transport = Transport(tmp_path, budget, cfg, responder=respond)
    return GraphRun(case, "B", transport, tmp_path / "B"), seen


def test_one_plan_one_probe_then_python_completion_and_common_readers(tmp_path):
    run, seen = make_run(
        tmp_path,
        "The cropped region shows the sign. The sign in the cropped region is blue. <answer>Yes</answer>",
    )
    result = run.execute()
    assert result["stop_reason"] == "ALL_GOALS_SUPPORTED"
    assert result["prediction"] == "A" and not result["errors"]
    assert result["crop_rounds"] == result["probe_calls"] == 1
    assert [x["stage"] for x in seen] == ["planner_1", "probe_1_0_0_0", "direct", "aggregator"]
    assert result["task_guidance"]["python_search"]["subsequent_planner_requests"] == 0
    assert run.tasks.focus() is None
    assert not read(run.transport.budget.path)["held"]


def test_failed_plan_is_repaired_without_crop_or_reader_bypass(tmp_path):
    run, seen = make_run(tmp_path, bad_regions=True)
    result = run.execute()
    assert len(seen) == 3
    assert all(x["stage"].startswith("planner_") for x in seen)
    assert result["status"] == "planning_failed" and result["prediction"] is None
    assert result["stop_reason"] == "REGION_PLAN_VALIDATION_FAILED"
    assert not run.rounds


@pytest.mark.parametrize(
    "regions",
    [
        None,
        {},
        {"g1": [0, 0, 0, 100]},
        {"g1": [0, 0, True, 100]},
        {"g1": [0, 0, float("nan"), 100]},
        {"g1": [0, 0, 1001, 100]},
        {"g1": [0, 0, 100, 100], "foreign": [0, 0, 100, 100]},
    ],
)
def test_initial_region_binding_is_atomic_and_strict(regions):
    tasks = TaskGraph("What color is the sign?\n(A) Blue\n(B) Red")
    search = CropSearch()
    with pytest.raises((ValueError, TypeError)):
        search.accept(tasks, regions)
    assert search.regions is None
    assert not any(n["status"] == "supported_by_reports" for n in tasks.nodes)


def test_region_plan_cannot_be_replaced_or_used_before_valid_graph():
    search = CropSearch()
    pending = TaskGraph("Explain the purpose of the situation.")
    with pytest.raises(ValueError, match="VALID_PLAN_REQUIRED"):
        search.accept(pending, {})
    tasks = TaskGraph("What color is the sign?\n(A) Blue\n(B) Red")
    search.accept(tasks, {"g1": [100, 100, 500, 500]})
    with pytest.raises(ValueError, match="ALREADY_FIXED"):
        search.accept(tasks, {"g1": [0, 0, 1000, 1000]})


def test_truncated_region_plan_is_repaired_before_any_probe(tmp_path):
    run, seen = make_run(
        tmp_path,
        "The cropped region shows the sign. The sign in the cropped region is blue. <answer>Yes</answer>",
    )
    respond = run.transport.responder

    def truncated_once(request, meta):
        body = respond(request, meta)
        if meta["stage"] == "planner_1":
            body["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"] = '{"g1": [200,'
        return body

    run.transport.responder = truncated_once
    result = run.execute()
    assert result["stop_reason"] == "ALL_GOALS_SUPPORTED"
    assert [x["stage"] for x in seen] == [
        "planner_1",
        "planner_2",
        "probe_2_0_0_0",
        "direct",
        "aggregator",
    ]
    assert len(result["errors"]) == 1 and result["errors"][0]["stage"] == "region_plan"
    assert result["task_guidance"]["python_search"]["initial_region_plan_requests"] == 2
    assert result["task_guidance"]["python_search"]["subsequent_planner_requests"] == 0
    for record in seen[:2]:
        messages = record["request"]["messages"]
        assert len(messages) == 2
        assert sum(b["type"] == "image_url" for b in messages[1]["content"]) == 1


def test_direct_finish_cannot_bypass_initial_region_validation(tmp_path):
    run, _ = make_run(tmp_path)
    with pytest.raises(ValueError, match="INITIAL_REGION_PLAN_REQUIRED"):
        run.finish()
    assert not run.transport.records


def test_no_observation_keeps_goals_unresolved_and_uses_no_later_planner(tmp_path):
    run, seen = make_run(tmp_path)
    result = run.execute()
    assert result["prediction"] == "A"  # Only an offline final-reader fixture.
    assert not any(rd["task_progress"]["newly_supported"] for rd in result["rounds"])
    assert not result["task_guidance"]["all_goals_supported"]
    assert sum(x["stage"].startswith("planner_") for x in seen) == 1
    assert 1 < result["probe_calls"] <= 9
    assert all(rd["paid_probe_count"] == 1 for rd in result["rounds"])
    boxes = [tuple(rd["candidates"][0]["original_xyxy"]) for rd in result["rounds"]]
    assert len(boxes) == len(set(boxes))
    assert not result["errors"]


def test_coarse_no_does_not_exclude_other_quadrants(tmp_path):
    run, seen = make_run(tmp_path, regions={"g1": [0, 0, 1000, 1000]})
    result = run.execute()
    boxes = [rd["candidates"][0]["normalized"] for rd in result["rounds"]]
    # A coarse No permits a different detail partition, but is not a reason to
    # exhaust all four partitions when consecutive probes add no evidence.
    assert any([x, y, x + 500, y + 500] in boxes for x in (0, 500) for y in (0, 500))
    assert len(boxes) == 2 and result["stop_reason"] == "NO_PRODUCTIVE_SEARCH"
    assert not any(rd["task_progress"]["newly_supported"] for rd in result["rounds"])
    assert sum(x["stage"].startswith("planner_") for x in seen) == 1


def test_exhausted_first_root_does_not_block_other_ready_root(tmp_path):
    run, _ = make_run(tmp_path)
    run.tasks.nodes = [
        dict(
            id=f"g{i}",
            kind="locate",
            task="Locate target",
            targets=[f"target {i}"],
            depends_on=[],
            status="unresolved",
            citations=[],
        )
        for i in (1, 2)
    ]
    search = CropSearch()
    search.accept(run.tasks, {"g1": [100, 100, 300, 300], "g2": [700, 700, 900, 900]})
    search.tried.update(
        ("g1", tuple(round(v) for v in b)) for _, b in search.boxes(run.tasks, run.tasks.nodes[0])
    )
    decision = search.next(run.tasks, run.images)
    assert decision["focus_id"] == "g2" and run.tasks.focus()["id"] == "g2"
    assert not run.tasks.complete


def test_completed_or_blocked_frontier_never_dispatches_crop(tmp_path):
    run, _ = make_run(tmp_path)
    search = CropSearch()
    for node in run.tasks.nodes:
        node["status"] = "supported_by_reports"
    assert search.next(run.tasks, run.images)["stop"] == "ALL_GOALS_SUPPORTED"
    run.tasks.nodes[0]["status"] = "unavailable"
    run.tasks.nodes[-1]["status"] = "unresolved"
    assert search.next(run.tasks, run.images)["stop"] == "NO_READY_VISUAL_GOAL"
    assert not run.transport.records


def test_same_crop_native_probe_payload_is_unchanged(tmp_path):
    run, seen = make_run(tmp_path)
    run.turn = 1
    proposal = dict(anchor=dict(parent="original", box=[100, 100, 500, 500]), branches=[])
    candidates = run.crop_candidates(proposal)
    run.probes(candidates, 0)
    graph_request = seen[-1]["request"]
    cfg = dict(run.transport.config)
    path = tmp_path / "native"
    budget = BudgetStore(path / "ledger.json", cfg, 0, 30, [run.case["key"] + "/B"])
    inputs = []

    def responder(request, meta):
        inputs.append(request)
        return response(meta, dict(role="assistant", content="No"))

    native = FoveaRun(run.case, "B", Transport(path, budget, cfg, responder=responder), path / "B")
    native.turn = 1
    native.probes(native.crop_candidates(proposal), 0)
    assert semantic_request(graph_request) == semantic_request(inputs[0])
