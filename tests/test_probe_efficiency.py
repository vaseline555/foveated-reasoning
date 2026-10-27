"""Real failed responses and offline search/reader contract regressions."""

import json
from copy import deepcopy
from pathlib import Path

import pytest

from active_vlm.answers import choice_schema, parse_final_answer
from active_vlm.baselines.runner import FoveaRun
from active_vlm.budget import BudgetStore
from active_vlm.common import canonical, read
from active_vlm.experiment import accounting
from active_vlm.graph.control import plan_messages
from active_vlm.graph.planning import normalize_root_fields
from active_vlm.graph.search import CropSearch
from active_vlm.graph.state import TaskGraph
from active_vlm.transport import Transport

from .replay import no_dispatch
from .test_execution import config, make_case, response
from .test_plan_gate import call, graph_run
from .test_python_graph import make_run

ROOT = Path(__file__).resolve().parents[1] / "results/python_graph_pilot30_20261023"
PLAN_CASES = ["treebench_bc9912c53b60", "mme_3ac2b04de6ab", "mme_e27494e9b523"]


@pytest.fixture(autouse=True)
def offline():
    with no_dispatch():
        yield


def captured(key):
    if not (ROOT / "inference_manifest.json").exists():
        pytest.skip("Optional private captured logs are not distributed")
    case = next(c for c in read(ROOT / "inference_manifest.json") if c["key"] == key)
    return case, read(ROOT / "trajectories" / key / "B/result.json")


@pytest.mark.parametrize("key", PLAN_CASES)
def test_actual_failed_questions_compile_to_complete_instructions_not_facts(key):
    case, old = captured(key)
    assert old["status"] == "planning_failed"
    tasks = TaskGraph(case["question"], open_questions=True)
    assert tasks.phase == "search" and not tasks.complete and not tasks.attempts
    assert any(n["kind"] == "locate" for n in tasks.nodes)
    assert any(n["kind"] in {"attribute", "relation"} for n in tasks.nodes)
    assert all(n["status"] == "unresolved" and not n["citations"] for n in tasks.nodes)
    assert case["question"].splitlines()[0] in tasks.nodes[-1]["task"]
    if key == "mme_e27494e9b523":
        assert "front of the ego car" in tasks.nodes[0]["targets"][0]
    assert tasks.ready() and tasks.nodes[-1] not in tasks.ready()


def test_actual_helmet_plan_defaults_all_independent_locates_without_adding_edges():
    case, old = captured(PLAN_CASES[0])
    message = next(e for e in old["events"] if e.get("stage") == "planner_2")
    original = json.loads(message["message"]["tool_calls"][0]["function"]["arguments"])["steps"]
    steps, changes = normalize_root_fields(original)
    assert len(changes) == 2 and "requires" not in original[0]
    assert steps[0] == dict(original[0], requires=[])
    assert steps[1] == dict(original[1], requires=[])
    assert steps[2] == original[2]
    tasks = TaskGraph(case["question"])
    tasks.define_goals(steps)
    assert tasks.phase == "search" and not tasks.complete
    assert tasks.nodes[-1]["depends_on"] == ["g1", "g2"]


@pytest.mark.parametrize("bad", [[0], ["0"], [True], [-1], [8]])
def test_normalization_never_repairs_invalid_model_edges(bad):
    steps = [dict(kind="locate", task="Locate target", targets=["target"], requires=bad)]
    normalized, changes = normalize_root_fields(steps)
    assert normalized == steps and not changes
    with pytest.raises(ValueError, match="DEPENDENCIES"):
        TaskGraph("Explain the purpose.").define_goals(normalized)


def test_repair_request_contains_rejected_payload_and_field_instruction(tmp_path):
    case = make_case(tmp_path)
    case["question"] = "Explain the visible bucket purpose.\n(A) Water\n(B) Waste"
    cfg = dict(config(), graph_controller="python", seed_arm_aliases={"B": "A"})
    run = graph_run(case, tmp_path / "repair", lambda req, meta: response(meta), cfg)
    payload = dict(
        steps=[dict(kind="locate", task="Locate bucket", targets=["bucket"], requires=[0])]
    )
    run.control_call(call("define_goals", payload))
    run.search = CropSearch()
    request = plan_messages(run, "define_goals")
    correction = json.loads(request[-1]["content"][-1]["text"])["correction"]
    assert correction["rejected_arguments"] == payload
    assert correction["error"]["path"] == "steps[0].requires"
    assert correction["step_index_map"][0]["index"] == 0
    assert correction["dependency_example"][0]["requires"] == []
    assert not run.tasks.complete and not run.rounds


def test_zero_gain_changes_region_then_stops_unresolved_without_more_planner(tmp_path):
    run, seen = make_run(tmp_path)
    result = run.execute()
    decisions = result["task_guidance"]["python_search"]["decisions"]
    assert [d["reason"] for d in decisions] == ["initial_region", "new_region"]
    assert result["crop_rounds"] == result["probe_calls"] == 2
    assert result["stop_reason"] == "NO_PRODUCTIVE_SEARCH"
    assert not run.tasks.complete and all(n["status"] == "unresolved" for n in run.tasks.nodes)
    assert sum(x["stage"].startswith("planner_") for x in seen) == 1


def test_dependency_view_is_not_paid_for_again_under_a_different_focus(tmp_path):
    run, _ = make_run(tmp_path, "The cropped region clearly shows the sign. <answer>Yes</answer>")
    result = run.execute()
    ids = [r["candidates"][0]["id"] for r in result["rounds"]]
    assert len(ids) == len(set(ids))
    assert not run.tasks.complete and any(
        n["status"] == "supported_by_reports" for n in run.tasks.nodes
    )


@pytest.mark.parametrize("values", [(True, 2), (0, 2), (6, -1), (6, "2")])
def test_invalid_search_budgets_fail_before_dispatch(values):
    with pytest.raises(ValueError, match="POSITIVE_INTEGERS"):
        CropSearch(*values)


def test_global_probe_budget_preserves_unresolved_roots_and_never_fakes_completion(tmp_path):
    run, _ = make_run(tmp_path)
    run.tasks.nodes = [
        dict(
            id=f"g{i}",
            kind="locate",
            task="Locate target",
            targets=[f"target{i}"],
            depends_on=[],
            status="unresolved",
            citations=[],
        )
        for i in range(1, 8)
    ]
    search = CropSearch()
    search.accept(run.tasks, {f"g{i}": [i * 100, 100, i * 100 + 60, 200] for i in range(1, 8)})
    for _ in range(6):
        decision = search.next(run.tasks, run.images)
        assert "stop" not in decision
        crop = run.images.candidate("original", decision["proposal"]["anchor"]["box"])
        run.tasks.observe(crop, [dict(receipt_id=crop["id"], message=dict(content="No"))])
    assert search.next(run.tasks, run.images)["stop"] == "PROBE_BUDGET_EXHAUSTED"
    assert not run.tasks.complete and all(n["status"] == "unresolved" for n in run.tasks.nodes)
    assert not run.transport.records


def test_grammar_handles_unseen_properties_without_an_attribute_dictionary():
    graph = TaskGraph(
        "What is the fluorescence pattern of the sample marker?\n(A) Striped\n(B) Uniform",
        open_questions=True,
    )
    assert graph.phase == "search" and graph.nodes[-1]["attribute"] == "fluorescence pattern"
    assert graph.nodes[0]["targets"] == ["the sample marker"]
    assert not graph.complete


def test_changed_direct_contract_cannot_reuse_historical_reader_results(tmp_path):
    case, _ = captured(PLAN_CASES[0])
    cfg = dict(read(ROOT / "config.json"), direct_answer_mode="choice")
    mappings = read(
        ROOT.parents[2] / "experiment_archive/before_probe_efficiency_20261023/relocations.json"
    )
    with pytest.raises(ValueError, match="direct_answer_mode"):
        accounting.reuse([case], ["A"], [ROOT], tmp_path / "reuse", cfg, mappings)


def test_visible_unconfirmed_object_keeps_detail_without_borrowing_another_objects_position(
    tmp_path,
):
    case, old = captured("treebench_037c6767c876")
    cfg = dict(
        config(),
        crop_probe_mode="graph-single",
        graph_controller="python",
        seed_arm_aliases={"B": "A"},
    )
    run = graph_run(case, tmp_path, lambda req, meta: response(meta), cfg)
    run.tasks.nodes = deepcopy(old["task_guidance"]["goals"])
    for node in run.tasks.nodes:
        node.update(status="unresolved", citations=[])
    search = CropSearch()
    search.accept(run.tasks, old["task_guidance"]["python_search"]["initial_regions"])
    for original in old["rounds"][:3]:
        decision = search.next(run.tasks, run.images)
        crop = run.images.candidate("original", decision["proposal"]["anchor"]["box"])
        assert crop["id"] == original["candidates"][0]["id"]
        update = run.tasks.observe(crop, original["raw_reports"][0])
        if original is old["rounds"][2]:
            assert decision["reason"] == "detail" and update["newly_supported"] == []
            # The crop explicitly contains a railing, but "far to the right"
            # describes the power poles OUTSIDE it. That qualifier cannot
            # identify the question's far-right railing in this crop.
            assert (
                "while the power poles are located far to the right"
                in original["raw_reports"][0][0]["message"]["content"]
            )
    assert not run.tasks.complete and not run.transport.records


def test_context_rescue_preserves_actual_gain_after_two_zero_gain_crops(tmp_path):
    case, old = captured("mme_51f3cce18c1d")
    cfg = dict(
        config(),
        crop_probe_mode="graph-single",
        graph_controller="python",
        seed_arm_aliases={"B": "A"},
    )
    run = graph_run(case, tmp_path, lambda req, meta: response(meta), cfg)
    run.tasks.nodes = deepcopy(old["task_guidance"]["goals"])
    for node in run.tasks.nodes:
        node.update(status="unresolved", citations=[])
    search = CropSearch()
    search.accept(run.tasks, old["task_guidance"]["python_search"]["initial_regions"])
    for original in old["rounds"][:3]:
        decision = search.next(run.tasks, run.images)
        crop = run.images.candidate("original", decision["proposal"]["anchor"]["box"])
        assert crop["id"] == original["candidates"][0]["id"]
        update = run.tasks.observe(crop, original["raw_reports"][0])
    assert decision["context_recovery"] and decision["reason"] == "context"
    assert update["newly_supported"] == ["g1"]
    assert not run.tasks.complete and not run.transport.records


@pytest.mark.parametrize("bad_capture", [False, True])
def test_actual_double_truncation_uses_two_bounded_readers_no_extra_calls(tmp_path, bad_capture):
    case, old = captured("treebench_bbabf4b8e7b2")
    old_messages = {
        e["stage"]: e for e in old["events"] if e.get("stage") in {"direct", "aggregator"}
    }
    assert all(e["finish"] == "length" for e in old_messages.values())
    assert all(
        parse_final_answer(e["message"]["content"], case["allowed_letters"], "choice") is None
        for e in old_messages.values()
    )
    cfg = dict(
        config(),
        crop_probe_mode="graph-single",
        final_answer_mode="choice",
        direct_answer_mode="choice",
    )
    seen = []

    def respond(request, meta):
        seen.append(meta["stage"])
        assert request["max_tokens"] == 64
        assert request["response_format"]["json_schema"]["schema"] == choice_schema(
            case["allowed_letters"]
        )
        if meta["stage"] == "aggregator":
            assert "Wait — perhaps" not in canonical(request["messages"][-2])
        # This is a labelled output-contract control, never a new model answer.
        msg = (
            deepcopy(old_messages[meta["stage"]]["message"])
            if bad_capture
            else dict(role="assistant", content='{"answer":"A"}')
        )
        return response(meta, msg)

    store = BudgetStore(tmp_path / "ledger.json", cfg, 0, 30, [case["key"] + "/B"])
    run = FoveaRun(case, "B", Transport(tmp_path, store, cfg, responder=respond), tmp_path / "B")
    run.final_history = deepcopy(old["final_history"])
    assert run.finish() == (None if bad_capture else "A")
    assert seen == ["direct", "aggregator"]
    assert bool(run.errors) == bad_capture
    assert old["prediction"] is None  # Never overwrite or infer an old answer.
