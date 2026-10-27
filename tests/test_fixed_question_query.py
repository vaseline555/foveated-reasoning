"""Real failure replay and adversarial controls, never new measured VLM accuracy."""

import json
from pathlib import Path
from copy import deepcopy
import pytest
from active_vlm.common import read
from active_vlm.decision.question_query import question_plan
from active_vlm.decision.partition import PartitionDAG
from active_vlm.decision.partition_prompts import bind_observations, plan_tool
from .test_partition_graph import node, report
from .test_partition_view_scope import wire
from .replay import no_dispatch

ROOT = Path(__file__).resolve().parents[1] / "results/question_led_graph30_round03_20261026"


@pytest.fixture(autouse=True)
def offline():
    with no_dispatch():
        yield


@pytest.mark.skipif(not ROOT.exists(), reason="Optional historical responses are not distributed")
def test_three_actual_river_plans_have_exact_question_query_without_visual_gain():
    r = read(ROOT / "trajectories/mme_b531ae97b13b/B/result.json")
    c = next(c for c in read(ROOT / "inference_manifest.json") if c["key"] == r["case_key"])
    count = 0
    for e in r["events"]:
        if not e.get("stage", "").startswith("planner"):
            continue
        p = json.loads(e["message"]["tool_calls"][0]["function"]["arguments"])
        # Explicit old/new wire translation removes the model-owned kind field;
        # actual task/target/box/dependency values are preserved verbatim.
        steps = [{k: v for k, v in n.items() if k != "kind"} for n in p["steps"]]
        d = PartitionDAG(c["allowed_letters"])
        d.accept(question_plan(steps, c["question"], c["allowed_letters"]), lambda _: None)
        assert d.nodes[-1]["task"] == c["question"] and d.nodes[-1]["requires"] == [0]
        assert d.nodes[0]["groups"] == [] and d.nodes[-1]["groups"] == [
            [a] for a in c["allowed_letters"]
        ]
        assert not d.observations and all(n["status"] == "pending" for n in d.nodes)
        assert d.possible == set(c["allowed_letters"])
        count += 1
    assert count == 3


@pytest.mark.skipif(not ROOT.exists(), reason="Optional historical responses are not distributed")
def test_actual_unresolved_missing_group_is_unknown_and_never_answer_gain():
    r = read(ROOT / "trajectories/vstar_33c8cbbbc2d5/B/result.json")
    nodes = r["task_guidance"]["nodes"]
    count = 0
    for e in r["events"]:
        if not e.get("stage", "").startswith("observe"):
            continue
        p = json.loads(e["message"]["tool_calls"][0]["function"]["arguments"])
        d = PartitionDAG(["A", "B"])
        d.accept(
            [{k: n[k] for k in ["task", "target", "kind", "requires", "box"]} for n in nodes],
            lambda _: None,
        )
        before = deepcopy(p)
        rows = bind_observations(p["observations"], d.nodes)
        assert p == before
        out = d.observe_batch(rows, list(range(len(nodes))), "actual", "receipt")
        assert not any(o["observed"] for o in out) and d.possible == {"A", "B"}
        assert all(o["realized_gain"] == 0 for o in out)
        count += 1
    assert count == 2


@pytest.mark.parametrize("resolved", ["false", None, 0, 1])
def test_nonboolean_resolution_stays_invalid(resolved):
    d = PartitionDAG(["A", "B"])
    d.accept([node()], lambda _: None)
    row = {k: v for k, v in report(0, 0).items() if k not in ["node_index", "group"]}
    row["resolved"] = resolved
    rows = bind_observations({"g0": row}, d.nodes)
    with pytest.raises(ValueError):
        d.observe_batch(rows, [0], "c", "r")
    assert not d.observations


def test_unmapped_fact_has_zero_answer_gain_and_is_not_goal_completion():
    d = PartitionDAG(["A", "B"])
    d.accept([node()], lambda _: None)
    rows = bind_observations(wire([report(0)], d.nodes), d.nodes)
    out = d.observe_batch(rows, [0], "c", "r")
    assert out[0]["observed"] and out[0]["realized_gain"] == 0
    assert out[0]["compatible_answers"] is None and d.possible == {"A", "B"}
    assert d.nodes[0]["status"] == "observed_unmapped" and d.select()[0] is None


def test_only_explicit_allowed_answer_maps_query_and_identity_still_gives_no_gain():
    d = PartitionDAG(["A", "B"])
    d.accept([node("locate"), node(requires=[0])], lambda _: None)
    rows = bind_observations(wire([report(0), report(1)], d.nodes), d.nodes, "B", ["A", "B"])
    out = d.observe_batch(rows, [0, 1], "c", "r")
    assert out[0]["realized_gain"] == 0 and out[1]["realized_gain"] == 0.5
    for invalid in [0, True, "option B", "C", ["B"]]:
        rows = bind_observations(
            wire([report(0), report(1)], d.nodes), d.nodes, invalid, ["A", "B"]
        )
        assert all(r["group"] is None for r in rows)


def test_query_requires_every_terminal_fact_preserves_semantics_and_atomic_guards():
    facts = [node("locate"), node("locate"), node("locate", requires=[0])]
    steps = [{k: v for k, v in n.items() if k != "kind"} for n in facts]
    before = deepcopy(steps)
    plan = question_plan(steps, "Which of these objects is larger?", ["A", "B"])
    assert steps == before
    assert plan[-1]["requires"] == [1, 2]
    assert [n["target"] for n in plan[:-1]] == [n["target"] for n in steps]
    assert all(n["kind"] == "locate" for n in plan[:-1])
    bad = deepcopy(steps)
    bad[0]["requires"] = [2]
    with pytest.raises(ValueError):
        question_plan(bad, "Question", ["A", "B"])
    bad = deepcopy(steps)
    bad[1]["box"] = [0, 0, float("nan"), 5]
    with pytest.raises(ValueError):
        question_plan(bad, "Question", ["A", "B"])
    with pytest.raises(ValueError):
        question_plan([], "Question", ["A", "B"])
    assert (
        "kind"
        not in plan_tool(False)["function"]["parameters"]["properties"]["steps"]["items"][
            "properties"
        ]
    )


def test_query_cannot_gain_from_location_alone_or_unverified_dependency():
    steps = [{k: v for k, v in node("locate").items() if k != "kind"}]
    d = PartitionDAG(["A", "B"])
    d.accept(question_plan(steps, "What color is the object?", ["A", "B"]), lambda _: None)
    out = d.observe_batch([report(0)], [0], "c0", "r0")
    assert out[0]["realized_gain"] == 0 and d.possible == {"A", "B"}
    d2 = PartitionDAG(["A", "B"])
    d2.accept(question_plan(steps, "What color is the object?", ["A", "B"]), lambda _: None)
    out = d2.observe_batch([report(1, 0)], [1], "c1", "r1")
    assert not out[0]["observed"] and out[0]["realized_gain"] == 0


@pytest.mark.parametrize("case", ["mme_9767ae1525f1", "mme_a001a8b697b3"])
@pytest.mark.skipif(not ROOT.exists(), reason="Optional historical responses are not distributed")
def test_actual_resolved_but_unmapped_facts_survive_without_choice_inference(case):
    result = read(ROOT / "trajectories" / case / "B/result.json")
    nodes = result["task_guidance"]["nodes"]
    count = 0
    manifest = next(c for c in read(ROOT / "inference_manifest.json") if c["key"] == case)
    for e in result["events"]:
        if not e.get("stage", "").startswith("observe"):
            continue
        try:
            data = json.loads(e["message"]["tool_calls"][0]["function"]["arguments"])
        except json.JSONDecodeError:
            continue  # Never salvage the truncated prefix.
        d = PartitionDAG(manifest["allowed_letters"])
        d.accept(
            [{k: n[k] for k in ["task", "target", "kind", "requires", "box"]} for n in nodes],
            lambda _: None,
        )
        rows = bind_observations(data["observations"], d.nodes)
        out = d.observe_batch(rows, list(range(len(nodes))), "actual", "captured")
        assert all(o["realized_gain"] == 0 and o["compatible_answers"] is None for o in out)
        assert d.possible == set(manifest["allowed_letters"])
        assert d.nodes[-1]["status"] == "observed_unmapped"
        assert (
            out[-1]["reported"]["observation"]
            == data["observations"][f"g{len(nodes) - 1}"]["observation"]
        )
        assert out[-1]["semantic_truth_verified"] is False
        count += 1
    assert count == (1 if case == "mme_9767ae1525f1" else 2)


@pytest.mark.skipif(not ROOT.exists(), reason="Optional historical responses are not distributed")
def test_actual_native_truncation_never_parses_even_if_tool_finish_hides_length():
    from active_vlm.decision.runner import parse_tool

    result = read(ROOT / "trajectories/mme_9767ae1525f1/B/result.json")
    request_hash = next(
        r["request_hash"] for r in result["receipts"] if r["stage"] == "observe_1_0"
    )
    record = next(
        r
        for p in (ROOT / "requests").glob("*.json")
        if (r := read(p))["request_hash"] == request_hash
    )
    assert record["finish_reason"] == "tool_calls"
    assert record["response"]["choices"][0]["native_finish_reason"] == "length"
    with pytest.raises(ValueError, match="TRUNCATED_TOOL_OUTPUT"):
        parse_tool(record["message"], record, "observe_partition")
    complete = deepcopy(record["message"])
    complete["tool_calls"][0]["function"]["arguments"] = '{"observations":{},"answer":"A"}'
    with pytest.raises(ValueError, match="TRUNCATED_TOOL_OUTPUT"):
        parse_tool(complete, record, "observe_partition")


@pytest.mark.skipif(not ROOT.exists(), reason="Optional historical responses are not distributed")
def test_all_actual_kind_and_mapping_failures_replay_without_invented_answer():
    cases = {c["key"]: c for c in read(ROOT / "inference_manifest.json")}
    plan_count = observation_count = 0
    for path in (ROOT / "trajectories").glob("*/B/result.json"):
        result = read(path)
        case = cases[result["case_key"]]
        errors = {e["stage"]: e["error"] for e in result["task_guidance"]["plan_rejections"]}
        for event in result["events"]:
            error = errors.get(event.get("stage"))
            if error not in ["ANSWER_MEASUREMENT_REQUIRED", "OBSERVATION_FIELDS_FOR_BOUND_KIND"]:
                continue
            data = json.loads(event["message"]["tool_calls"][0]["function"]["arguments"])
            if error == "ANSWER_MEASUREMENT_REQUIRED":
                steps = [{k: v for k, v in n.items() if k != "kind"} for n in data["steps"]]
                d = PartitionDAG(case["allowed_letters"])
                d.accept(
                    question_plan(steps, case["question"], case["allowed_letters"]), lambda _: None
                )
                assert not d.observations and d.select()[0] is not None
                assert [n["task"] for n in d.nodes[:-1]] == [n["task"] for n in data["steps"]]
                plan_count += 1
            else:
                nodes = result["task_guidance"]["nodes"]
                d = PartitionDAG(case["allowed_letters"])
                d.accept(
                    [
                        {k: n[k] for k in ["task", "target", "kind", "requires", "box"]}
                        for n in nodes
                    ],
                    lambda _: None,
                )
                rows = bind_observations(data["observations"], d.nodes)
                out = d.observe_batch(rows, [n["index"] for n in nodes], "actual", "captured")
                assert all(o["realized_gain"] == 0 and o["compatible_answers"] is None for o in out)
                observation_count += 1
    assert (plan_count, observation_count) == (21, 10)


def test_end_to_end_unmapped_value_reaches_native_reader_without_completion_claim(tmp_path):
    from active_vlm.common import save, canonical
    from active_vlm.experiment import cli
    from .test_execution import make_case
    from .test_partition_graph import PartitionResponder

    case = make_case(tmp_path)
    save(tmp_path / "manifest.json", [case])

    class NoMapping(PartitionResponder):
        def __call__(self, request, meta):
            response = super().__call__(request, meta)
            if meta["stage"].startswith("observe"):
                function = response["choices"][0]["message"]["tool_calls"][0]["function"]
                data = json.loads(function["arguments"])
                data.pop("answer")
                data["observations"]["g1"]["observation"] = "CONTROL_VISIBLE_VALUE_17"
                function["arguments"] = canonical(data)
            return response

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
            "results/unmapped",
        ]
    )
    assert cli.run(root, approve=True, responder=NoMapping()) == "COMPLETE"
    r = read(root / "trajectories" / case["key"] / "B/result.json")
    assert r["calls"] == 5 and r["probe_calls"] == 1 and not r["errors"]
    assert r["stop_reason"] == "UNMAPPED_QUERY_FACT_TO_NATIVE_READER"
    assert not r["task_guidance"]["complete"]
    assert "CONTROL_VISIBLE_VALUE_17" in canonical(r["final_history"])
    assert all(o["realized_gain"] == 0 for o in r["partition_graph"]["observations"])
