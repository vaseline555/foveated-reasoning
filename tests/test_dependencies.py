"""Real omitted-edge regressions and bounded structural controls; no inference."""

import json
from copy import deepcopy
from pathlib import Path

import pytest

from active_vlm.common import read
from active_vlm.graph.control import plan_messages
from active_vlm.graph.dependencies import DependencyRepair
from active_vlm.graph.planning import normalize_root_fields, plan_limits
from active_vlm.graph.state import TaskGraph

from .replay import no_dispatch
from .test_execution import config, make_case, response
from .test_plan_gate import call, graph_run

ROOT = Path(__file__).resolve().parents[1] / "results/bounded_graph_pilot30_20261023"
CASES = [
    "treebench_037c6767c876",
    "treebench_abcae921ffa5",
    "mme_54a0faa18b69",
    "mme_aba49185501f",
    "treebench_5b0b141fbb83",
]
# Explicit HUMAN-WRITTEN structural controls, never captured/new model plans.
CONTROL_PATCHES = {
    "mme_aba49185501f": {"6": [4, 5]},
    "treebench_5b0b141fbb83": {
        "5": [0, 1],
        "6": [0, 2],
        "7": [0, 3],
        "8": [0, 4],
        "9": [5, 6, 7, 8],
    },
}


@pytest.fixture(autouse=True)
def offline():
    with no_dispatch():
        yield


def pending(tmp_path, responder=None):
    case = make_case(tmp_path)
    case["question"] = "Explain the visible relationship.\n(A) Related\n(B) Separate"
    cfg = dict(
        config(),
        graph_controller="python",
        crop_probe_mode="graph-single",
        seed_arm_aliases={"B": "A"},
    )
    return graph_run(case, tmp_path / "run", responder or (lambda req, meta: response(meta)), cfg)


def steps():
    return [
        dict(kind="locate", task="Locate the target", targets=["target"]),
        dict(kind="attribute", task="Read target attribute", targets=["target"]),
    ]


@pytest.mark.parametrize("key", CASES)
def test_actual_five_failed_plans_preserve_targets_and_gate_evidence(tmp_path, key):
    if not (ROOT / "inference_manifest.json").exists():
        pytest.skip("Optional private captured logs are not distributed")
    case = next(c for c in read(ROOT / "inference_manifest.json") if c["key"] == key)
    old = read(ROOT / "trajectories" / key / "B/result.json")
    assert old["status"] == "planning_failed"
    cfg = dict(read(ROOT / "config.json"), min_dispatch_interval_seconds=0)
    run = graph_run(case, tmp_path, lambda req, meta: response(meta), cfg)
    calls = [c for e in old["events"] for c in e.get("message", {}).get("tool_calls", [])]
    for c in calls:
        if c["function"]["name"] == "read_question":
            run.control_call(c)
    original = next(c for c in reversed(calls) if c["function"]["name"] == "define_goals")
    raw_steps = json.loads(original["function"]["arguments"])["steps"]
    before = deepcopy(raw_steps)
    run.control_call(original)
    if key in CONTROL_PATCHES:
        assert run.tasks.phase == "define_goals" and run.dependency_repair is not None
        assert run.planner_tool_choice()["function"]["name"] == "repair_dependencies"
        assert "VALID_PLAN_REQUIRED" in run.control_call(call("cropping", {}))["content"]
        with pytest.raises(ValueError, match="VALID_PLAN_REQUIRED"):
            run.finish()
        run.control_call(call("repair_dependencies", dict(requires=CONTROL_PATCHES[key])))
    assert run.tasks.phase == "search" and run.dependency_repair is None
    assert len(run.tasks.nodes) == len(raw_steps)
    assert [(n["kind"], n["task"], n["targets"]) for n in run.tasks.nodes] == [
        (s["kind"], s["task"], s["targets"]) for s in raw_steps
    ]
    assert before == raw_steps  # Input capture stays immutable.
    assert not run.tasks.complete and not run.tasks.attempts and not run.rounds
    assert all(n["status"] == "unresolved" and not n["citations"] for n in run.tasks.nodes)
    assert not run.transport.records


@pytest.mark.parametrize(
    "bad",
    [
        None,
        {},
        {"1": []},
        {"1": [1]},
        {"1": [2]},
        {"1": [-1]},
        {"1": ["0"]},
        {"1": [True]},
        {"1": [0, 0]},
        {"1": [0], "0": []},
        {"1": [[0]]},
    ],
)
def test_bad_dependency_patch_is_atomic_and_cannot_bypass_gate(tmp_path, bad):
    run = pending(tmp_path)
    run.control_call(call("define_goals", dict(steps=steps())))
    before = deepcopy(run.dependency_repair.steps)
    msg = run.control_call(call("repair_dependencies", dict(requires=bad)))
    assert "INVALID_DEPENDENCY_PATCH" in msg["content"]
    assert run.dependency_repair.steps == before
    assert run.tasks.phase == "define_goals" and not run.tasks.complete
    assert not run.rounds and not run.transport.records


def test_repair_schema_and_prompt_expose_exact_zero_based_fields(tmp_path):
    run = pending(tmp_path)
    run.control_call(call("define_goals", dict(steps=steps())))
    tool = run.planner_tools()[0]
    deps = tool["function"]["parameters"]["properties"]["requires"]
    assert deps["required"] == ["1"] and deps["additionalProperties"] is False
    assert deps["properties"]["1"]["items"] == dict(type="integer", minimum=0, maximum=0)
    assert deps["properties"]["1"]["minItems"] == 1
    packet = json.loads(plan_messages(run, "repair_dependencies")[-1]["content"][-1]["text"])
    assert packet["required_tool"] == "repair_dependencies"
    assert packet["correction"]["step_index_map"][1]["index"] == 1
    run.control_call(call("repair_dependencies", dict(requires={"1": [0]})))
    assert run.tasks.nodes[-1]["depends_on"] == ["g1"]
    assert run.dependency_repairs == [dict(turn=run.turn, requires={"1": [0]})]


@pytest.mark.parametrize(
    "question,allowed",
    [
        ("Which object is located to the right of the tower?", True),
        ("What item is positioned above the fence?", True),
        ("What color is the object?", False),
        ("How many objects are there?", False),
    ],
)
def test_dependent_localization_only_answers_object_retrieval(question, allowed):
    tasks = TaskGraph("Explain the scene.", open_questions=True)
    tasks.question = question
    plan = [
        dict(kind="locate", task="Locate the tower", targets=["tower"], requires=[]),
        dict(
            kind="locate",
            task="Locate object relative to tower",
            targets=["object right of tower"],
            requires=[0],
        ),
    ]
    if allowed:
        tasks.define_goals(plan)
        assert not tasks.complete and tasks.ready() == [tasks.nodes[0]]
    else:
        with pytest.raises(ValueError, match="QUESTION_ACTION_REQUIRED"):
            tasks.define_goals(plan)


def test_repair_never_converts_a_nonlocalization_first_step_or_unknown_fields():
    plan, _ = normalize_root_fields(steps())
    plan[0]["kind"] = "attribute"
    assert DependencyRepair.for_plan(plan, plan_limits("Explain the scene.")) is None
    plan, _ = normalize_root_fields(steps())
    plan[1]["answer"] = "A"
    assert DependencyRepair.for_plan(plan, plan_limits("Explain the scene.")) is None


def test_dependency_patch_does_not_hide_a_remaining_semantic_plan_error(tmp_path):
    run = pending(tmp_path)
    plan = steps()
    plan[1]["kind"] = "locate"
    plan[1]["requires"] = [1]  # Explicit self-edge, never normalized to [].
    run.control_call(call("define_goals", dict(steps=plan)))
    assert run.dependency_repair.indices == [1]
    reply = run.control_call(call("repair_dependencies", dict(requires={"1": [0]})))
    assert "QUESTION_ACTION_REQUIRED" in reply["content"]
    assert run.tasks.phase == "define_goals" and run.dependency_repair is None
    assert run.planner_tool_choice()["function"]["name"] == "define_goals"
    assert run.latest_plan_error["rejected_arguments"]["steps"][1]["requires"] == [0]
    assert not run.tasks.complete and not run.transport.records


def test_repeated_invalid_dependency_repairs_stop_the_path_before_any_probe(tmp_path):
    def respond(req, meta):
        name = req["tools"][0]["function"]["name"]
        payload = dict(steps=steps()) if name == "define_goals" else dict(requires={"1": [1]})
        return response(
            meta, dict(role="assistant", content=None, tool_calls=[call(name, payload)])
        )

    run = pending(tmp_path, respond)
    result = run.execute()
    assert result["status"] == "planning_failed" and result["calls"] == 4
    assert result["task_guidance"]["plan_stop_trigger"] == "PLAN_REPAIR_STALLED"
    assert not result["rounds"] and result["prediction"] is None
    assert all(e.get("stage", "").startswith("planner_") for e in result["events"])


def test_native_model_controller_keeps_the_original_required_plan_contract(tmp_path):
    run = pending(tmp_path)
    run.controller = "model"
    schema = run.planner_tools()[0]["function"]["parameters"]["properties"]["steps"]["items"]
    assert "requires" in schema["required"]
    run.control_call(call("define_goals", dict(steps=steps())))
    assert run.dependency_repair is None and not run.plan_normalizations
    assert run.tasks.phase == "define_goals" and not run.tasks.complete


def test_dependency_repair_executes_before_regions_and_shared_native_probe(tmp_path):
    seen = []

    def respond(req, meta):
        if meta["stage"].startswith("planner_"):
            tool = req["tools"][0]["function"]["name"]
            assert tool == req["tool_choice"]["function"]["name"]
            seen.append(tool)
            payload = {
                "define_goals": dict(steps=steps()),
                "repair_dependencies": dict(requires={"1": [0]}),
                "plan_regions": dict(g1=[100, 100, 500, 500]),
            }[tool]
            return response(
                meta, dict(role="assistant", content=None, tool_calls=[call(tool, payload)])
            )
        seen.append(meta["stage"])
        return response(
            meta,
            dict(
                role="assistant",
                content="No" if meta["stage"].startswith("probe_") else "<answer>A</answer>",
            ),
        )

    run = pending(tmp_path, respond)
    result = run.execute()
    assert result["status"] == "ok"
    assert seen[:3] == ["define_goals", "repair_dependencies", "plan_regions"]
    assert result["probe_calls"] == result["crop_rounds"] == 2
    assert len(result["task_guidance"]["dependency_repairs"]) == 1
    assert result["task_guidance"]["python_search"]["subsequent_planner_requests"] == 0
