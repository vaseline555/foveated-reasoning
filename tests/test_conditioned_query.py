"""Captured failures + adversarial controls; synthetic corrections are not new VLM results."""

import json
from copy import deepcopy
from pathlib import Path

import pytest

from active_vlm.common import canonical, read, save
from active_vlm.decision.partition import PartitionDAG
from active_vlm.decision.partition_prompts import dependency_tool
from active_vlm.decision.question_query import (
    dependency_errors,
    query_region_patch,
    question_plan,
    repair_dependencies,
)
from active_vlm.experiment import cli

from .replay import no_dispatch
from .test_execution import make_case, response
from .test_partition_graph import PartitionResponder, node, report

ROOT = Path(__file__).resolve().parents[1] / "results/question_query_graph30_round04_20261026"


@pytest.fixture(autouse=True)
def offline():
    with no_dispatch():
        yield


@pytest.mark.skipif(not ROOT.exists(), reason="Optional historical responses are not distributed")
def test_all_actual_dependency_failures_are_localized_without_automatic_edge_changes():
    count = 0
    for path in (ROOT / "trajectories").glob("*/B/result.json"):
        result = read(path)
        bad = {
            r["stage"]
            for r in result["task_guidance"]["plan_rejections"]
            if r["error"] == "DEPENDENCIES_REQUIRE_EARLIER_INTEGERS"
        }
        for event in result["events"]:
            if event.get("stage") not in bad:
                continue
            payload = json.loads(event["message"]["tool_calls"][0]["function"]["arguments"])
            steps = payload["steps"]
            before = deepcopy(steps)
            errors = dependency_errors(steps)
            assert errors and all(e["index"] in e["supplied"] for e in errors.values())
            assert steps == before
            # HUMAN structural control: retain earlier valid edges and omit self.
            # This is not an observed model correction or a forecast of accuracy.
            patches = {k: e["preserve"] for k, e in errors.items()}
            fixed = repair_dependencies(steps, patches)
            question_plan(fixed, "Exact human question control", ["A", "B"])
            for i in range(len(steps)):
                for field in ["task", "target", "box"]:
                    assert steps[i][field] == fixed[i][field]
            assert steps == before
            count += 1
    assert count == 13


@pytest.mark.parametrize(
    "patch", [{"g0": [0]}, {"g0": [True]}, {"g0": [], "g1": []}, {}, {"g0": "[]"}]
)
def test_invalid_patch_cannot_pass_or_mutate_draft(patch):
    steps = [{k: v for k, v in node("locate", requires=[0]).items() if k != "kind"}]
    before = deepcopy(steps)
    with pytest.raises(ValueError):
        repair_dependencies(steps, patch)
    assert before == steps


def test_valid_edges_and_unrelated_nodes_are_immutable():
    steps = [
        {k: v for k, v in node("locate", requires=d).items() if k != "kind"}
        for d in [[], [], [0, 2]]
    ]
    with pytest.raises(ValueError, match="PRESERVE"):
        repair_dependencies(steps, {"g2": [1]})
    fixed = repair_dependencies(steps, {"g2": [0, 1]})
    assert fixed[:2] == steps[:2] and fixed[2]["requires"] == [0, 1]
    schema = dependency_tool(dependency_errors(steps))
    assert schema["function"]["parameters"]["properties"]["replacements"]["properties"]["g2"][
        "items"
    ]["enum"] == [0, 1]


@pytest.mark.skipif(not ROOT.exists(), reason="Optional historical responses are not distributed")
def test_actual_failed_truck_prerequisite_repair_reopens_query_without_gain():
    result = read(ROOT / "trajectories/vstar_33c8cbbbc2d5/B/result.json")
    events = result["events"]
    initial = next(e for e in events if e.get("stage") == "planner_1")
    plan = json.loads(initial["message"]["tool_calls"][0]["function"]["arguments"])
    d = PartitionDAG(["A", "B"])
    d.accept(
        question_plan(plan["steps"], "Is the blue truck left or right?", d.letters), lambda _: None
    )
    obs = result["partition_graph"]["observations"]
    added = d.observe_batch(
        [o["reported"] for o in obs], [0, 1, 2], obs[0]["crop_id"], obs[0]["receipt_id"]
    )
    assert not any(o["observed"] for o in added)
    assert [n["status"] for n in d.nodes] == ["unobserved", "pending", "pending"]
    patches = [
        {k: x[k] for k in ["node_index", "box", "reason"]}
        for x in result["partition_graph"]["region_revisions"]
    ]
    d.revise(patches, lambda _: None)
    d.revise(query_region_patch(d), lambda _: None)
    assert d.select()[0]["focus"] == 0
    assert d.nodes[-1]["box"] == [540, 467, 650, 500]
    assert all(not o["observed"] and o["realized_gain"] == 0 for o in d.observations)
    assert d.possible == {"A", "B"}
    # Old failed reports are not reused, even after the nodes become pending.
    with pytest.raises(ValueError, match="REPEATED"):
        d.observe_batch([report(0)], [0], obs[0]["crop_id"], "new")


def test_observer_gets_one_actual_crop_and_dependency_repair_has_no_image(tmp_path):
    case = make_case(tmp_path)
    save(tmp_path / "manifest.json", [case])
    seen = []

    class BrokenPlan(PartitionResponder):
        def __call__(self, req, meta):
            if meta["stage"] == "planner_2_dependencies":
                seen.append((meta["stage"], deepcopy(req)))
                return response(
                    meta,
                    dict(
                        role="assistant",
                        content="",
                        tool_calls=[
                            dict(
                                function=dict(
                                    name="repair_plan_dependencies",
                                    arguments=canonical(dict(replacements={"g0": []})),
                                )
                            )
                        ],
                    ),
                )
            ans = super().__call__(req, meta)
            if meta["stage"] == "planner_1":
                fun = ans["choices"][0]["message"]["tool_calls"][0]["function"]
                data = json.loads(fun["arguments"])
                data["steps"][0]["requires"] = [0]
                fun["arguments"] = canonical(data)
            if meta["stage"].startswith("observe") or meta["stage"].startswith("probe"):
                seen.append((meta["stage"], deepcopy(req)))
            return ans

    root = cli.main(
        [
            "prepare",
            "--manifest",
            str(tmp_path / "manifest.json"),
            "--full",
            "--methods",
            "graph",
            "--graph-controller",
            "partition",
            "--account-root",
            str(tmp_path / "account"),
            "--output",
            "results/conditional",
        ]
    )
    assert cli.run(root, approve=True, responder=BrokenPlan()) == "COMPLETE"
    result = read(root / "trajectories" / case["key"] / "B/result.json")
    assert result["calls"] == 6 and result["planner_turns"] == 2 and result["probe_calls"] == 1
    assert result["task_guidance"]["complete"] and not result["errors"]
    for stage, req in seen:
        blocks = [
            x for m in req["messages"] if isinstance(m.get("content"), list) for x in m["content"]
        ]
        images = [x for x in blocks if x["type"] == "image_url"]
        if stage.endswith("_dependencies"):
            assert len(images) == 0
        if stage.startswith("observe"):
            assert len(images) == 1
    summary = read(root / "summary.json")
    assert summary["partition_graph_diagnostics"]["admitted_facts"] == 2
    assert summary["graph_diagnostics"]["positive_gain_rounds"] == 1
    assert summary["graph_diagnostics"]["supported_nodes"] == 2


def test_conditional_failure_is_not_success_even_with_explicit_answer():
    d = PartitionDAG(["A", "B"])
    d.accept([node("locate"), node(requires=[0])], lambda _: None)
    out = d.observe_batch([report(0, target_visible=False), report(1, 0)], [0, 1], "c", "r")
    assert d.nodes[1]["status"] == "pending" and not out[1]["observed"]
    assert d.select()[0] is None and d.possible == {"A", "B"}


def test_end_to_end_failed_context_actually_executes_repaired_view(tmp_path):
    case = make_case(tmp_path)
    save(tmp_path / "manifest.json", [case])

    class RepairContext(PartitionResponder):
        def __call__(self, req, meta):
            name = req.get("tools", [{}])[0].get("function", {}).get("name")
            if name == "repair_regions":
                return response(
                    meta,
                    dict(
                        role="assistant",
                        content="",
                        tool_calls=[
                            dict(
                                function=dict(
                                    name=name,
                                    arguments=canonical(
                                        dict(
                                            patches=[
                                                dict(
                                                    node_index=0,
                                                    box=[100, 100, 400, 400],
                                                    reason="CONTROL magnify same target",
                                                )
                                            ]
                                        )
                                    ),
                                )
                            )
                        ],
                    ),
                )
            ans = super().__call__(req, meta)
            if meta["stage"].startswith("observe_1_"):
                fun = ans["choices"][0]["message"]["tool_calls"][0]["function"]
                data = json.loads(fun["arguments"])
                for row in data["observations"].values():
                    row.update(
                        target_visible=False,
                        resolved=False,
                        observation="CONTROL not identified in context view",
                    )
                data["answer"] = None
                fun["arguments"] = canonical(data)
            return ans

    root = cli.main(
        [
            "prepare",
            "--manifest",
            str(tmp_path / "manifest.json"),
            "--full",
            "--methods",
            "graph",
            "--graph-controller",
            "partition",
            "--account-root",
            str(tmp_path / "account"),
            "--output",
            "results/repaired",
        ]
    )
    assert cli.run(root, approve=True, responder=RepairContext()) == "COMPLETE"
    result = read(root / "trajectories" / case["key"] / "B/result.json")
    assert (
        result["probe_calls"] == 2 and result["calls"] == 8 and result["task_guidance"]["complete"]
    )
    observations = result["partition_graph"]["observations"]
    assert len({o["crop_id"] for o in observations}) == 2
    assert all(not o["observed"] and o["realized_gain"] == 0 for o in observations[:2])
    assert observations[-1]["realized_gain"] > 0 and all(o["observed"] for o in observations[2:])
    assert result["task_guidance"]["nodes"][-1]["box"] == [100, 100, 400, 400]
