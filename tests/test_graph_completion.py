"""Real chart-prefix replay and labelled completion-routing controls; no network."""

import json
from copy import deepcopy
from pathlib import Path

import pytest

from active_vlm.baselines.runner import FoveaRun
from active_vlm.budget import BudgetStore
from active_vlm.common import canonical, read
from active_vlm.graph.runner import GraphRun
from active_vlm.graph.state import TaskGraph
from active_vlm.transport import Transport, semantic_request

from .replay import no_dispatch
from .test_execution import config, make_case, response

ROOT = Path(__file__).resolve().parents[1] / "results/question_evidence_pilot8_20261022"
KEY = "mme_7f8799322e5f"


@pytest.fixture(autouse=True)
def block_network():
    with no_dispatch():
        yield


def replay_chart(output, arm="B"):
    if not (ROOT / "inference_manifest.json").exists():
        pytest.skip("Optional local captured chart logs are not distributed")
    case = next(c for c in read(ROOT / "inference_manifest.json") if c["key"] == KEY)
    old = read(ROOT / "trajectories" / KEY / "B/result.json")
    cfg = dict(
        read(ROOT / "config.json"), min_dispatch_interval_seconds=0, pacing=dict(mode="fixed")
    )
    cfg["seed_arm_aliases"] = dict(cfg["seed_arm_aliases"], **{arm: "A"})
    receipts = {r["stage"]: r for r in old["receipts"] if r["status"] == "ok"}
    seen, identical = [], []
    reader_source = []

    def replay(request, meta):
        stage = meta["stage"]
        assert stage in {
            "planner_1",
            "planner_2",
            "planner_3",
            "direct",
            "aggregator",
        } or stage.startswith("probe_3_0_")
        saved = read(ROOT / "requests" / (receipts[stage]["request_hash"] + ".json"))
        if stage.startswith("planner_"):
            # Plan limits, repair feedback and support metadata intentionally
            # changed. Reuse real replies as fixtures, not a new plan measurement.
            assert request["messages"][0] == saved["request"]["messages"][0]

            def original_input(messages):
                return next(m for m in messages if isinstance(m.get("content"), list))

            assert semantic_request({"messages": [original_input(request["messages"])]}) == (
                semantic_request({"messages": [original_input(saved["request"]["messages"])]})
            )
        elif stage != "aggregator":
            assert semantic_request(request) == semantic_request(saved["request"])
            identical.append(stage)
        else:
            # Common reader implementation, but a shorter exploration history.
            # The captured output is only a routing/parse fixture, not a newly
            # measured response to this different aggregator input.
            assert request["messages"][0] == saved["request"]["messages"][0]
            assert semantic_request(request) != semantic_request(saved["request"])
            assert canonical(request).count("TOOL OBSERVATION:") == 1
            assert "Python maintains visual subgoals" not in canonical(request)
            reader_source.append("captured_reader_output_as_offline_routing_fixture_only")
        seen.append(stage)
        captured = response(meta, deepcopy(saved["message"]))
        captured["id"] = saved["receipt_id"]
        return captured

    budget = BudgetStore(output / "ledger.json", cfg, 0, 30, [KEY + "/" + arm])
    transport = Transport(output, budget, cfg, responder=replay)
    run = GraphRun(case, arm, transport, output / arm)
    result = run.execute()
    assert result["stop_reason"] == "ALL_GOALS_SUPPORTED"
    assert result["task_guidance"]["all_goals_supported"]
    assert result["crop_rounds"] == 1 and result["calls"] == 14
    assert result["planner_turns"] == 3 and result["probe_calls"] == 9
    assert seen[-2:] == ["direct", "aggregator"] and len(seen) == 14
    assert len(identical) == 10 and reader_source
    assert result["rounds"][0]["selected_id"] == old["rounds"][0]["selected_id"]
    assert result["rounds"][0]["task_progress"]["newly_supported"] == ["g1", "g2"]
    assert result["rounds"][0]["scores"] == old["rounds"][0]["scores"]
    assert run.tasks.complete and run.tasks.focus(run.graph_enabled) is None
    assert len(result["errors"]) == 1 and result["errors"][0]["stage"] == "planning"
    assert not read(budget.path)["held"] and not read(budget.path)["final_pool"]

    # Replay every actual post-completion command directly too. Guards must run
    # before constructing a crop, reserving slots, or inspecting probe inputs.
    ledger_before = read(budget.path)
    for event in old["events"]:
        if event.get("stage", "").startswith("planner_") and int(
            event["stage"].split("_")[1]
        ) in range(4, 10):
            call = event["message"]["tool_calls"][0]
            args = json.loads(call["function"]["arguments"])
            proposal = dict(
                anchor=dict(parent=args["image_url"], box=args["coordinates"], target_nodes=[]),
                branches=[],
            )
            with pytest.raises(ValueError, match="ALL_GOALS_SUPPORTED"):
                run.crop_candidates(proposal)
    with pytest.raises(ValueError, match="ALL_GOALS_SUPPORTED"):
        run.probes([], 0)
    assert run.planner_tools() == []
    assert read(budget.path) == ledger_before and len(seen) == 14
    for rd in old["rounds"][1:]:
        decision = run.tasks.repeat_decision(rd["candidates"], run.graph_enabled)
        assert decision["reject"] and decision["reason"] == "no_unresolved_focus"
    assert all(read(p)["backend"] == "mock" for p in (output / "requests").glob("*.json"))
    kept = set(seen)
    removed = [r for r in old["receipts"] if r["status"] == "ok" and r["stage"] not in kept]
    return dict(
        arm=arm,
        key=KEY,
        original_successful_calls=old["calls"],
        replay_calls=result["calls"],
        original_crop_rounds=len(old["rounds"]),
        replay_crop_rounds=1,
        replay_stop=result["stop_reason"],
        executed_stages=seen,
        unchanged_prefix_and_direct_requests=len(identical),
        unchanged_native_probes=9,
        blocked_saved_post_completion_crops=6,
        extra_direct_probe_dispatches=0,
        captured_tail_not_dispatched=len(removed),
        captured_tail_known_cost_usd=sum(r["logical_usd"] for r in removed),
        shared_final_readers_reached=True,
        final_reader_fixture=reader_source,
        observed_new_model_accuracy=False,
        observed_new_model_cost=False,
        network_calls=0,
        paid_calls=0,
    )


@pytest.mark.parametrize("arm", ["B", "G"])
def test_actual_chart_stops_at_supported_prefix_then_uses_common_readers(tmp_path, arm):
    audit = replay_chart(tmp_path, arm)
    assert audit["captured_tail_not_dispatched"] == 61


def controller(tmp_path, responder=None, arm="B"):
    """Synthetic control image/reports; not measured visual reasoning."""
    case = make_case(tmp_path)
    case["question"] = "What is the color of the sign?\n(A) Blue\n(B) Red"
    cfg = dict(config(), seed_arm_aliases={"B": "A", "F": "A"})
    budget = BudgetStore(tmp_path / "ledger.json", cfg, 0, 30, [case["key"] + "/" + arm])
    transport = Transport(
        tmp_path, budget, cfg, responder=responder or (lambda req, meta: response(meta))
    )
    return GraphRun(case, arm, transport, tmp_path / arm)


@pytest.mark.parametrize("phase", ["pending", "embedded", "empty", "blocked", "partial"])
def test_missing_focus_or_plan_is_not_claimed_as_goal_completion(tmp_path, phase):
    run = controller(tmp_path)
    if phase == "pending":
        run.tasks = TaskGraph("Explain the situation and its purpose.")
    elif phase == "embedded":
        run.tasks = TaskGraph("Read the printed task.", embedded=True)
    elif phase == "empty":
        run.tasks.nodes = []
    elif phase == "blocked":
        # Deliberately inconsistent state, not a model-generated valid DAG.
        run.tasks.nodes[0]["status"] = "unavailable"
    else:
        run.tasks.nodes[0]["status"] = "supported_by_reports"
    assert not run.tasks.complete
    if phase == "blocked":
        assert run.tasks.focus() is None
        assert run.exploration_stop_reason() == "NO_READY_VISUAL_GOAL"
        assert run.stopping_reason() == "UNRESOLVED_GOALS_STOP"
        assert run.tasks.repeat_decision([])["reject"]
        with pytest.raises(ValueError, match="NO_READY_VISUAL_GOAL"):
            run.probes([], 0)
        with pytest.raises(ValueError, match="NO_READY_VISUAL_GOAL"):
            run.crop_candidates({})
    else:
        assert run.exploration_stop_reason() is None
        if phase == "empty":
            assert run.tasks.phase == "define_goals"
            with pytest.raises(ValueError, match="VALID_PLAN_REQUIRED"):
                run.probes([], 0)
    assert not run.transport.records


def test_specified_frame_and_supported_visual_nodes_count_as_complete():
    graph = TaskGraph("Which person from the right is wearing a hat?")
    assert any(n["status"] == "specified_by_question" for n in graph.nodes)
    for n in graph.nodes:
        if n["status"] == "unresolved":
            n["status"] = "supported_by_reports"  # Labelled state control, not new evidence.
    assert graph.complete
    with pytest.raises(ValueError, match="ALL_GOALS_SUPPORTED"):
        graph.require_search()


@pytest.mark.parametrize("arm", ["B", "F"])
def test_completion_in_first_tool_skips_remaining_tools_in_same_planner_response(tmp_path, arm):
    seen = []

    def respond(request, meta):
        seen.append(meta["stage"])
        message = dict(role="assistant", content="<answer>A</answer>")
        if meta["stage"] == "planner_1":
            message["tool_calls"] = [
                dict(
                    id=f"CONTROL-crop-{i}",
                    type="function",
                    function=dict(
                        name="cropping",
                        arguments=canonical(
                            dict(image_url="original", coordinates=[200, 200, 600, 600])
                        ),
                    ),
                )
                for i in range(2)
            ]
        elif meta["stage"].startswith("probe_"):
            message["content"] = "The cropped region shows a blue sign. <answer>Yes</answer>"
        return response(meta, message)

    run = controller(tmp_path, respond, arm)
    result = run.execute()
    assert result["stop_reason"] == "ALL_GOALS_SUPPORTED" and result["calls"] == 12
    assert result["crop_rounds"] == 1 and result["probe_calls"] == 9
    assert result["planner_turns"] == 1 and not result["errors"]
    assert seen[-2:] == ["direct", "aggregator"]
    assert not any(s.startswith("planner_2") or s.startswith("probe_1_1_") for s in seen)
    skipped = next(m for m in run.history if m.get("tool_call_id") == "CONTROL-crop-1")
    assert json.loads(skipped["content"]) == dict(
        status="not_executed", message="ALL_GOALS_SUPPORTED"
    )


def test_native_fovea_has_no_automatic_graph_stop(tmp_path):
    # The hook defaults to None; real A/M/L full-wire replays cover actual loops.
    case = make_case(tmp_path)
    cfg = config()
    budget = BudgetStore(tmp_path / "ledger.json", cfg, 0, 30, [case["key"] + "/A"])
    transport = Transport(tmp_path, budget, cfg, responder=lambda req, meta: response(meta))
    assert FoveaRun(case, "A", transport, tmp_path / "A").exploration_stop_reason() is None
