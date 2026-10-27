"""Captured plan/truncation failures and labelled output-contract controls; no API."""

import json
from copy import deepcopy
from pathlib import Path

import pytest

from active_vlm.answers import choice_schema, parse_answer, parse_final_answer
from active_vlm.budget import BudgetStore
from active_vlm.common import read
from active_vlm.experiment import accounting
from active_vlm.graph.planning import PlanError
from active_vlm.graph.runner import GraphRun
from active_vlm.graph.state import TaskGraph
from active_vlm.transport import Transport, semantic_request

from .replay import no_dispatch
from .test_execution import response
from .test_plan_gate import graph_run

ROOT = Path(__file__).resolve().parents[1] / "results/native_fovea_graph_single30_20261023"
PLAN_CASE = "treebench_abcae921ffa5"
FINAL_CASE = "mme_81daa3a8ece0"


@pytest.fixture(autouse=True)
def block_network():
    with no_dispatch():
        yield


def captured(key):
    if not (ROOT / "inference_manifest.json").exists():
        pytest.skip("Optional local captured failures are not distributed")
    case = next(c for c in read(ROOT / "inference_manifest.json") if c["key"] == key)
    return case, read(ROOT / "trajectories" / key / "B/result.json")


def replay_actual_plan(output):
    case, old = captured(PLAN_CASE)
    plans = [e["message"] for e in old["events"] if e.get("stage", "").startswith("planner_")]
    run = graph_run(case, output, lambda req, meta: response(meta), read(ROOT / "config.json"))
    rejections = []
    for message in plans[:3]:
        tool = run.control_call(deepcopy(message["tool_calls"][0]))
        rejections.append(json.loads(tool["content"])["status"])
    assert rejections == ["plan_required", "plan_required", "accepted_task_instructions"]
    assert [e["error"]["code"] for e in run.plan_rejections] == [
        "QUESTION_ACTION_REQUIRED",
        "INVALID_SUBGOAL_TEXT",
    ]
    assert run.plan_rejections[1]["error"]["path"] == "steps[6].targets"
    original_steps = json.loads(plans[2]["tool_calls"][0]["function"]["arguments"])["steps"]
    assert [n["targets"] for n in run.tasks.nodes] == [s["targets"] for s in original_steps]
    assert len(run.tasks.nodes[-1]["targets"][0]) == 114
    assert len(run.tasks.nodes) == 7 and run.tasks.phase == "search"
    assert all(n["status"] == "unresolved" and not n["citations"] for n in run.tasks.nodes)
    assert run.latest_plan_error is None and not run.plan_stalled
    assert not run.transport.records and not run.tasks.attempts and not run.tasks.complete
    return dict(
        case=case["key"],
        original_status=old["status"],
        captured_plan_messages_replayed=3,
        corrected_plan_source="unmodified_actual_planner_3_response",
        accepted_goals=7,
        preserved_target_characters=114,
        remaining_structural_guards=[
            "question action required",
            "at most two targets per task",
            "typed backward DAG",
        ],
        visual_facts_added=0,
        measured_new_model_response=False,
        paid_calls=0,
    )


def test_actual_114_character_target_validates_without_truncation_or_invented_evidence(tmp_path):
    replay_actual_plan(tmp_path)


def test_total_plan_budget_replaces_arbitrary_per_target_cap_atomically():
    graph = TaskGraph("Explain the bucket's purpose.")
    original = deepcopy(graph.nodes)
    excessive = "x" * graph.plan_limits["max_text_chars"]
    with pytest.raises(PlanError, match="PLAN_TEXT_BUDGET_EXCEEDED"):
        graph.define_goals(
            [
                dict(kind="locate", task="Locate the target", targets=[excessive], requires=[]),
                dict(kind="attribute", task="Read its purpose", targets=["bucket"], requires=[0]),
            ]
        )
    assert graph.nodes == original
    with pytest.raises(PlanError, match="INVALID_SUBGOAL_TEXT"):
        graph.define_goals(
            [dict(kind="locate", task="Locate the target", targets=["  "], requires=[])]
        )


def test_plan_schema_and_python_use_the_same_total_resource_budget(tmp_path):
    case, _ = captured(PLAN_CASE)
    run = graph_run(case, tmp_path, lambda req, meta: response(meta), read(ROOT / "config.json"))
    schema = run.planner_tools()[0]["function"]["parameters"]["properties"]["steps"]
    budget = run.tasks.plan_limits["max_text_chars"]
    assert schema["items"]["properties"]["targets"]["items"]["maxLength"] == budget > 114
    assert schema["items"]["properties"]["task"]["maxLength"] == budget
    assert schema["items"]["properties"]["targets"]["maxItems"] == 2


@pytest.mark.parametrize(
    "text,expected",
    [
        ('{"answer":"A"}', "A"),
        (' {"answer": "E"} ', "E"),
        ('{"answer":"AB"}', None),
        ('{"answer":"K"}', None),
        ('{"answer":"a"}', None),
        ('{"answer":true}', None),
        ('{"answer":["A"]}', None),
        ('{"answer":"A","answer":"B"}', None),
        ('{"answer":"A","reason":"extra prose"}', None),
        ('["A"]', None),
        ('{"answer":"A"', None),
        ("<answer>A</answer>", None),
        ('Reasoning first. {"answer":"A"}', None),
        ('```json\n{"answer":"A"}\n```', None),
    ],
)
def test_choice_only_parser_requires_one_complete_allowed_option(text, expected):
    assert parse_final_answer(text, "ABCDE", "choice") == expected


def test_actual_truncated_reply_has_no_recoverable_answer_and_is_not_relabelled():
    _, old = captured(FINAL_CASE)
    final = next(e for e in old["events"] if e.get("stage") == "aggregator")
    text = final["message"]["content"]
    assert final["finish"] == "length" and len(text) > 9000
    assert text.count("**Final Answer:**") >= 3
    assert "<answer>" not in text and parse_answer(text, "ABCDE") is None
    assert parse_final_answer(text, "ABCDE", "choice") is None
    assert old["prediction"] is None  # No guessed answer or old-score replacement.
    endpoint = next(
        e
        for e in read(ROOT / "endpoint_metadata.json")["data"]["endpoints"]
        if e["provider_name"] == "Parasail" and e["quantization"] == "bf16"
    )
    assert {"response_format", "structured_outputs"} <= set(endpoint["supported_parameters"])


def replay_final_contract(output, use_bad_capture=False):
    case, old = captured(FINAL_CASE)
    cfg = dict(
        read(ROOT / "config.json"),
        final_answer_mode="choice",
        min_dispatch_interval_seconds=0,
        pacing=dict(mode="fixed"),
    )
    by_stage = {
        r["stage"]: read(ROOT / "requests" / (r["request_hash"] + ".json"))
        for r in old["receipts"]
        if r["stage"] in {"direct", "aggregator"}
    }
    seen = []

    def respond(request, meta):
        stage = meta["stage"]
        seen.append(stage)
        saved = by_stage[stage]
        if stage == "direct":
            assert semantic_request(request) == semantic_request(saved["request"])
            return response(meta, deepcopy(saved["message"]))
        assert stage == "aggregator" and request["max_tokens"] == 64
        schema = request["response_format"]["json_schema"]
        assert schema["strict"] and schema["schema"] == choice_schema(case["allowed_letters"])
        assert "step-by-step" not in request["messages"][0]["content"].lower()
        # Same question, original/crop JPEGs, full report and Direct content.
        assert (
            semantic_request(request)["messages"][1:]
            == semantic_request(saved["request"])["messages"][1:]
        )
        if use_bad_capture:
            message = deepcopy(saved["message"])  # Actual malformed text remains a failure.
        else:
            # HUMAN-WRITTEN OUTPUT-CONTRACT CONTROL ONLY. This is not inferred
            # from the bad reply and does not measure a new model's answer.
            message = dict(role="assistant", content='{"answer":"E"}')
        body = response(meta, message)
        if use_bad_capture:
            body["choices"][0]["finish_reason"] = saved["finish_reason"]
        return body

    store = BudgetStore(output / "ledger.json", cfg, 0, 30, [case["key"] + "/B"])
    transport = Transport(output, store, cfg, responder=respond)
    run = GraphRun(case, "B", transport, output / "B")
    run.final_history = deepcopy(old["final_history"])
    run.stop = old["stop_reason"]
    prediction = run.finish()
    assert prediction == (None if use_bad_capture else "E")
    assert seen == ["direct", "aggregator"] and len(transport.records) == 2
    assert not read(store.path)["held"] and not run.rounds
    return dict(
        case=case["key"],
        native_direct_wire_identical=True,
        original_image_crop_reports_and_reader_evidence_unchanged=True,
        final_output_contract="one allowed choice, strict JSON, no analysis,64token cap",
        final_response_source="actual_bad_capture"
        if use_bad_capture
        else "human_written_schema_control",
        fixture_prediction=prediction,
        original_failure_not_overwritten=True,
        final_api_slots=2,
        extra_repair_or_analysis_slots=0,
        measured_new_model_response=False,
        measured_new_accuracy=False,
        paid_calls=0,
    )


@pytest.mark.parametrize("bad", [False, True])
def test_actual_images_and_reports_reach_bounded_final_contract_without_extra_calls(tmp_path, bad):
    replay_final_contract(tmp_path, bad)


def test_reader_mode_mismatch_cannot_reuse_or_rescore_old_native_results(tmp_path):
    case, _ = captured(FINAL_CASE)
    cfg = dict(read(ROOT / "config.json"), final_answer_mode="choice")
    archived = (
        ROOT.parents[2] / "experiment_archive/before_python_graph_control_20261023/relocations.json"
    )
    mappings = read(archived) if archived.exists() else None
    with pytest.raises(ValueError, match="final_answer_mode"):
        accounting.reuse([case], ["A"], [ROOT], tmp_path / "output", cfg, mappings)
