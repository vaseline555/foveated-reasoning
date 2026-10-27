"""Captured misbinding/plan/reader failures and labelled structural controls."""

import json
from copy import deepcopy
from pathlib import Path

import pytest

from active_vlm.common import read
from active_vlm.graph.dependencies import DependencyRepair
from active_vlm.graph.evidence import evidence_units
from active_vlm.graph.planning import normalize_root_fields
from active_vlm.graph.runner import GraphRun
from active_vlm.graph.state import TaskGraph
from active_vlm.readers import finalization_messages, recovery_policy, repetition_diagnostic
from active_vlm.transport import semantic_request

from .replay import no_dispatch
from .test_execution import config, make_case, response
from .test_native_readers import make_reader

ROOT = Path(__file__).resolve().parents[1] / "results/native_reader_graph_pilot30_20261023"


@pytest.fixture(autouse=True)
def offline():
    with no_dispatch():
        yield


def captured(key):
    if not (ROOT / "inference_manifest.json").exists():
        pytest.skip("Optional private captured logs are not distributed")
    case = next(c for c in read(ROOT / "inference_manifest.json") if c["key"] == key)
    return case, read(ROOT / "trajectories" / key / "B/result.json")


def test_real_green_shorts_keeps_tower_hint_but_rejects_inferred_boat_identity():
    case, old = captured("vstar_9a2fdb11a847")
    graph = TaskGraph(
        case["question"], quorum=1, image_size=case["original_size"], open_questions=True
    )
    updates = []
    for rd in old["rounds"]:
        candidate = rd["candidates"][rd["selected_index"]]
        reports = rd["raw_reports"][rd["selected_index"]]
        updates.append(graph.observe(candidate, reports))
        for node in graph.nodes:
            for citation in node["citations"]:
                text = next(
                    p["message"]["content"]
                    for p in reports
                    if p["receipt_id"] == citation["receipt_id"]
                )
                assert citation["quote"] in text
    assert [u["newly_supported"] for u in updates] == [[], ["g3"]]
    assert not graph.complete and graph.nodes[0]["status"] == "unresolved"
    hint = updates[0]["probe_feedback"]["unconfirmed_pool_hints"][0]
    assert "one of whom is wearing green shorts" in hint["quote"]
    assert hint["crop_id"] == old["rounds"][0]["selected_id"]
    assert hint["evidence_status"] == "unconfirmed_search_hint_only_no_gain"
    assert all(n["status"] == "unresolved" for n in graph.nodes if n["id"] != "g3")


@pytest.mark.parametrize("kind", ["locate", "condition", "attribute"])
@pytest.mark.parametrize(
    "claim",
    [
        "The cropped region contains two people, allowing us to identify the woman with a red bag on the left.",
        "The cropped region provides sufficient detail to identify the woman with a red bag on the left.",
        "The cropped region shows two people, which enables us to determine the woman with a red bag on the left.",
        "The original image shows the woman with a red bag, while the cropped region shows a boat.",
        "The cropped region is located on the left of the original image, while the woman with a red bag is elsewhere.",
    ],
)
def test_purpose_and_original_scene_mentions_are_never_identity_observations(kind, claim):
    node = dict(
        kind=kind,
        targets=["woman with red bag"],
        task="Locate the woman or read her position",
        attribute="position",
    )
    assert not list(evidence_units(node, claim))


def test_explicit_clothing_observation_is_not_blocked_by_whole_question_no():
    node = dict(kind="condition", targets=["woman with red bag"], task="Confirm her bag")
    text = "The cropped region shows a woman carrying a red bag. The boat is outside this crop. <answer>No</answer>"
    assert list(evidence_units(node, text))


def test_concrete_target_clause_survives_an_appended_purpose_clause():
    node = dict(kind="locate", targets=["pet collar"], task="Locate the pet collar")
    text = (
        "The question asks for the color of the pet collar. "
        "The candidate cropped region focuses on the collar, which is the exact region "
        "needed to determine the color."
    )
    units = list(evidence_units(node, text))
    assert units and units[0]["quote"] == "The candidate cropped region focuses on the collar"


def test_real_three_target_comparison_is_accepted_without_dropping_text_or_edges(tmp_path):
    case, old = captured("treebench_0e6791fa2da6")
    graph = TaskGraph(case["question"], open_questions=True)
    wire = old["events"][0]["message"]["tool_calls"][0]
    steps = json.loads(wire["function"]["arguments"])["steps"]
    assert len(steps[3]["targets"]) == 3
    accepted = graph.define_goals(steps)
    assert accepted["visual_facts_added"] == 0 and len(graph.nodes) == len(steps)
    for node, step in zip(graph.nodes, steps, strict=True):
        assert node["task"] == step["task"] and node["targets"] == step["targets"]
        assert node["depends_on"] == [f"g{i + 1}" for i in step["requires"]]
        assert node["status"] == "unresolved" and not node["citations"]
    cfg = dict(read(ROOT / "config.json"), min_dispatch_interval_seconds=0)
    run = make_reader(case, tmp_path, cfg, lambda req, meta: response(meta))
    run = GraphRun(case, "B", run.transport, tmp_path / "graph")
    schema = run.planner_tools()[0]["function"]["parameters"]["properties"]["steps"]
    assert (
        schema["items"]["properties"]["targets"]["maxItems"] == run.tasks.plan_limits["max_targets"]
    )


@pytest.mark.parametrize("key", ["mme_aba49185501f", "mme_4d5c96ed1ec9"])
def test_actual_dependency_patch_still_validates_under_expanded_target_limits(key):
    case, old = captured(key)
    graph = TaskGraph(case["question"], open_questions=True)
    functions = [
        call["function"]
        for event in old["events"]
        for call in event.get("message", {}).get("tool_calls", [])
    ]
    plan = next(f for f in functions if f["name"] == "define_goals")
    steps, _ = normalize_root_fields(json.loads(plan["arguments"])["steps"])
    with pytest.raises(ValueError):
        graph.define_goals(steps)
    repair = DependencyRepair.for_plan(steps, graph.plan_limits)
    patch = next(f for f in functions if f["name"] == "repair_dependencies")
    repaired = repair.apply(json.loads(patch["arguments"])["requires"])
    assert graph.define_goals(repaired)["visual_facts_added"] == 0
    for before, after in zip(steps, repaired, strict=True):
        assert before["task"] == after["task"] and before["targets"] == after["targets"]


@pytest.mark.parametrize("violation", ["count", "text", "type", "edge"])
def test_expanded_target_budget_still_validates_atomically(violation):
    graph = TaskGraph(
        "Compare the listed objects and determine their material.\n(A) wood\n(B) metal",
        open_questions=True,
    )
    steps = [
        dict(kind="locate", task="Locate all listed objects", targets=["object"], requires=[]),
        dict(
            kind="attribute", task="Read their visible material", targets=["object"], requires=[0]
        ),
    ]
    if violation == "count":
        steps[1]["targets"] = ["object"] * (graph.plan_limits["max_targets"] + 1)
    elif violation == "text":
        steps[1]["targets"] = ["x" * (graph.plan_limits["max_text_chars"] + 1)]
    elif violation == "type":
        steps[1]["targets"] = [True]
    else:
        steps[1]["requires"] = [1]
    before = deepcopy(graph.nodes)
    with pytest.raises(ValueError):
        graph.define_goals(steps)
    assert graph.nodes == before and not graph.complete


def finalizing_config():
    return dict(
        config(),
        final_answer_mode="native",
        direct_answer_mode="native",
        native_reader_recovery=dict(
            max_length_retries=1, max_tokens=512, strategy="finalize", analysis_max_chars=4096
        ),
    )


@pytest.mark.parametrize("bad_recovery", [False, True])
def test_real_direct_loop_gets_one_bounded_native_finalization(tmp_path, bad_recovery):
    case, old = captured("treebench_bbabf4b8e7b2")
    records = {
        r["stage"]: read(ROOT / "requests" / (r["request_hash"] + ".json"))
        for r in old["receipts"]
        if r["stage"] in {"direct", "direct_length_retry", "aggregator"}
    }
    assert repetition_diagnostic(records["direct_length_retry"]["message"]["content"])[
        "loop_detected"
    ]
    seen, requests = [], {}

    def replay(req, meta):
        stage = meta["stage"]
        seen.append(stage)
        requests[stage] = deepcopy(req)
        assert "response_format" not in req
        if stage == "direct":
            assert semantic_request(req) == semantic_request(records[stage]["request"])
            return deepcopy(records[stage]["response"])
        if stage == "direct_length_retry":
            assert req["max_tokens"] == 512
            assert req["messages"][:-2] == requests["direct"]["messages"]
            assert len(req["messages"][-2]["content"]) <= 4096 + len(
                "Incomplete, fallible draft:\n"
            )
            assert req["seed"] == requests["direct"]["seed"]
            if bad_recovery:
                # Rejected actual bad continuation (not a new model response).
                return deepcopy(records[stage]["response"])
            # HUMAN-WRITTEN format control, NOT a measured new model answer.
            return response(
                meta, dict(role="assistant", content="Brief format control. <answer>B</answer>")
            )
        if bad_recovery:
            assert "Direct reader returned no complete valid allowed choice" in str(req["messages"])
            assert records["direct_length_retry"]["message"]["content"] not in str(req["messages"])
        return deepcopy(records["aggregator"]["response"])

    cfg = dict(
        read(ROOT / "config.json"),
        native_reader_recovery=finalizing_config()["native_reader_recovery"],
        request_upper_usd=0.067072,
        min_dispatch_interval_seconds=0,
    )
    run = make_reader(case, tmp_path, cfg, replay)
    run.final_history = deepcopy(old["final_history"])
    run.stop = old["stop_reason"]
    assert run.finish() == "B"  # Captured judge output, not a new correct result.
    assert seen == ["direct", "direct_length_retry", "aggregator"]
    diagnostic = run.reader_diagnostics[0]
    assert diagnostic["recovery_strategy"] == "finalize"
    assert diagnostic["complete_valid_answer"] != bad_recovery
    assert diagnostic["recovery_attempted"]
    assert not read(run.transport.budget.path)["held"]


def test_deduplicated_draft_is_bounded_and_duplicate_choices_stay_distinct():
    paragraph = "Repeated fallible interpretation of the traffic scene. " * 12
    draft = "\n\n".join([paragraph] * 8)
    messages = [dict(role="user", content="Choose. (A) red (B) white (C) pink (D) red")]
    recovered = finalization_messages(messages, draft, "ABCD", 4096)
    assert recovered[0] == messages[0] and recovered[0] is not messages[0]
    assert recovered[-2]["content"].count(paragraph) == 1
    assert "A, B, C, D" in recovered[-1]["content"]
    assert repetition_diagnostic(draft)["loop_detected"]
    assert not repetition_diagnostic("The target is a car. The answer is A.")["loop_detected"]


@pytest.mark.parametrize(
    "bad",
    [
        dict(max_tokens=True),
        dict(max_tokens=4096),
        dict(analysis_max_chars=99999),
        dict(strategy="unknown"),
    ],
)
def test_finalization_policy_is_bounded_before_any_dispatch(bad):
    cfg = finalizing_config()
    cfg["native_reader_recovery"].update(bad)
    with pytest.raises(ValueError):
        recovery_policy(cfg)


def test_valid_native_stop_never_retries_even_if_analysis_repeats(tmp_path):
    case, seen = make_case(tmp_path), []

    def replay(req, meta):
        seen.append(meta["stage"])
        return response(
            meta,
            dict(
                role="assistant",
                content=("A long repeated sentence about the same target. " * 100)
                + "<answer>A</answer>",
            ),
        )

    run = make_reader(case, tmp_path, finalizing_config(), replay)
    assert run.finish() == "A" and seen == ["direct", "aggregator"]
    assert all(d["initial_repetition"]["loop_detected"] for d in run.reader_diagnostics)
    assert not any(d["recovery_attempted"] for d in run.reader_diagnostics)
