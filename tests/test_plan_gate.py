"""Required planning/repair and explicit answer-label parsing, without network."""

from copy import deepcopy
from pathlib import Path

import pytest

from active_vlm.answers import parse_answer
from active_vlm.budget import BudgetStore
from active_vlm.common import canonical, read
from active_vlm.graph.keywords import extract_keywords
from active_vlm.graph.runner import GraphRun
from active_vlm.graph.state import TaskGraph
from active_vlm.transport import Transport, semantic_request

from .replay import no_dispatch
from .test_execution import config, make_case, response

ROOT = Path(__file__).resolve().parents[1] / "results/release_smoke15_20261022"
FAILED_EIGHT = {
    "mme_734aa94f0f5d": ("building", "attribute", "Read the number on that same building"),
    "mme_eb1b19cad952": (
        "black bucket",
        "attribute",
        "Determine the purpose of that same black bucket from its visible context",
    ),
    "mme_f87eb313710a": (
        "truck",
        "relation",
        "Determine the orientation of the truck in the image",
    ),
    "treebench_ba37237a9294": (
        "person wearing a red jacket",
        "attribute",
        "Read the pose of the person wearing a red jacket",
    ),
    "mme_7f8799322e5f": (
        "EBITDA chart",
        "attribute",
        "Compare the labeled bars and read the year of the highest EBITDA",
    ),
    "treebench_224cf03028e6": (
        "black van with the yellow license plate",
        "relation",
        "Locate the black van within the overall street layout",
    ),
    "treebench_3325bfff5a80": (
        "beach chair",
        "attribute",
        "Read the material of that same beach chair",
    ),
    "treebench_ef13a5b34e2a": (
        "woman wearing a lime-green dress",
        "relation",
        "Compare contact between that woman and the dark wooden bench",
    ),
}


@pytest.fixture(autouse=True)
def network_blocked():
    with no_dispatch():
        yield


def call(name, args, ident="CONTROL"):
    return dict(id=ident, type="function", function=dict(name=name, arguments=canonical(args)))


def corrected_steps(key):
    """Human-written structural corrections, NOT newly measured model plans."""
    target, kind, task = FAILED_EIGHT[key]
    steps = [dict(kind="locate", task=f"Locate the {target}", targets=[target], requires=[])]
    targets, deps = [target], [0]
    if key == "treebench_ef13a5b34e2a":
        steps.append(
            dict(
                kind="locate",
                task="Locate the dark wooden bench",
                targets=["dark wooden bench"],
                requires=[],
            )
        )
        targets.append("dark wooden bench")
        deps.append(1)
    steps.append(dict(kind=kind, task=task, targets=targets, requires=deps))
    return steps


def graph_run(case, output, responder, cfg=None):
    cfg = cfg or dict(config(), seed_arm_aliases={"B": "A"})
    budget = BudgetStore(output / "ledger.json", cfg, 0, 30, [case["key"] + "/B"])
    transport = Transport(output, budget, cfg, responder=responder)
    return GraphRun(case, "B", transport, output / "trajectory")


def test_keywords_are_local_source_terms_and_preserve_option_contrasts():
    q = "Where is the black bucket?\n(A) Inside the building\n(B) Not inside the building"
    kw = extract_keywords(q)
    assert kw["question_terms"] == ["where", "black", "bucket"]
    assert kw["choice_contrasts"] == {"A": [], "B": ["not"]}
    assert kw["api_calls"] == 0 and kw["alternatives_are_not_visual_facts"]
    graph = TaskGraph(q)
    assert graph.phase == "define_goals" and graph.keywords == kw


@pytest.mark.parametrize("mode", ["crop", "answer", "broken_json"])
def test_no_plan_cannot_escape_via_crop_stop_or_final_reader(tmp_path, mode):
    case = make_case(tmp_path)
    case["question"] = "Explain the black bucket's visible purpose.\n(A) Waste\n(B) Water"
    seen = []

    def bypass(request, meta):
        seen.append(meta["stage"])
        assert [t["function"]["name"] for t in request["tools"]] == ["define_goals"]
        assert request["tool_choice"]["function"]["name"] == "define_goals"
        message = dict(role="assistant", content="<answer>A</answer>")
        if mode == "crop":
            message["tool_calls"] = [
                call("cropping", dict(image_url="original", coordinates=[0, 0, 500, 500]))
            ]
        elif mode == "broken_json":
            message["tool_calls"] = [call("define_goals", {})]
            message["tool_calls"][0]["function"]["arguments"] = '{"steps":['
        return response(meta, message)

    run = graph_run(case, tmp_path / "blocked", bypass)
    result = run.execute()
    assert (
        result["status"] == "planning_failed" and result["stop_reason"] == "PLAN_VALIDATION_FAILED"
    )
    assert not result["rounds"] and not result["prediction"]
    assert len(seen) == 3 and all(s.startswith("planner_") for s in seen)
    assert len(run.plan_rejections) == 3
    assert result["task_guidance"]["plan_stop_trigger"] == "PLAN_REPAIR_STALLED"
    assert not result["task_guidance"]["validated_plan"]
    with pytest.raises(ValueError, match="VALID_PLAN_REQUIRED"):
        run.finish()
    with pytest.raises(ValueError, match="VALID_PLAN_REQUIRED"):
        run.crop_candidates(dict(anchor=dict(parent="original", box=[0, 0, 500, 500])))
    with pytest.raises(ValueError, match="VALID_PLAN_REQUIRED"):
        run.probes([], 0)


def test_image_question_is_read_then_keywords_are_rebuilt_before_plan(tmp_path):
    case = make_case(tmp_path)
    case.update(question="Read the question and choices in the image.", embedded_question=True)
    run = graph_run(case, tmp_path / "read", lambda req, meta: response(meta))
    assert run.tasks.phase == "read_question"
    assert run.planner_tool_choice()["function"]["name"] == "read_question"
    rejected = run.control_call(call("cropping", {}))
    assert "VALID_PLAN_REQUIRED" in rejected["content"]
    run.control_call(
        call(
            "read_question",
            dict(text="What is the bucket used for?", options=["A. Waste", "B. Water"]),
        )
    )
    assert run.tasks.phase == "define_goals" and "bucket" in run.tasks.keywords["question_terms"]
    assert run.tasks.keywords["choice_contrasts"] == {"A": ["waste"], "B": ["water"]}
    assert run.planner_tool_choice()["function"]["name"] == "define_goals"


@pytest.mark.parametrize(
    "payload,expected",
    [
        ("A", "A"),
        ("(A) green", "A"),
        ("A. Green", "A"),
        ("Option A", "A"),
        ("(C) 4.", "C"),
        ("**(B) red**", "B"),
        ("A or B", None),
        ("(A) or (B)", None),
        ("(A) / B", None),
        ("A green car is visible", None),
        ("Neither", None),
        ("(K) other", None),
        ("(A) instead of (B)", None),
        ("A. red\nB. blue", None),
        ("A. red; B. blue", None),
    ],
)
def test_option_prefixes_are_parsed_but_ambiguous_or_out_of_range_answers_are_not(
    payload, expected
):
    assert parse_answer(f"<answer>{payload}</answer>", "ABCD") == expected
    assert parse_answer(payload, "ABCD") is None


def replay_failed_case(root, key, output):
    """Captured rejections/probes + labelled control repair; not a new VLM run."""
    original = read(root / "trajectories" / key / "B/result.json")
    case = next(c for c in read(root / "inference_manifest.json") if c["key"] == key)
    planners = [
        e["message"] for e in original["events"] if e.get("stage", "").startswith("planner_")
    ]
    crop_message = next(
        m
        for m in planners
        if any(c["function"]["name"] == "cropping" for c in m.get("tool_calls", []))
    )
    stop_message = next(m for m in planners if not m.get("tool_calls"))
    cfg = dict(
        read(root / "config.json"), min_dispatch_interval_seconds=0, pacing=dict(mode="fixed")
    )
    old_receipts = {r["stage"]: r for r in original["receipts"]}
    sources, probe_parity = [], []
    counter = 0
    run = None

    def responder(request, meta):
        nonlocal counter
        stage = meta["stage"]
        if stage.startswith("planner_"):
            counter += 1
            if counter == 1:
                message, source = planners[0], "captured_original_first_action"
            elif counter == 2:
                assert not run.tasks.attempts and not run.rounds and not run.tasks.phase == "search"
                message = dict(
                    role="assistant",
                    content="HUMAN-WRITTEN OFFLINE PLAN REPAIR",
                    tool_calls=[call("define_goals", dict(steps=corrected_steps(key)))],
                )
                source = "human_written_structural_repair_not_model_output"
                detail = run.plan_rejections[-1]["error"]
                if original["errors"]:
                    assert detail["path"].startswith("steps[")
            elif counter == 3:
                assert run.tasks.phase == "search"
                message, source = crop_message, "captured_original_crop_action"
            else:
                message, source = stop_message, "captured_original_stop_action"
            sources.append(dict(stage=stage, source=source))
        elif stage.startswith("probe_"):
            ci, k = map(int, stage.split("_")[-2:])
            rd = original["rounds"][0]
            old_stage = f"probe_{rd['turn']}_{rd['call_index']}_{ci}_{k}"
            record = read(root / "requests" / (old_receipts[old_stage]["request_hash"] + ".json"))
            old, new = semantic_request(record["request"]), semantic_request(request)
            # The repair adds planner turns, so the turn-derived seed changes.
            # Every other native probe field, JPEG hash/size and body is exact.
            old.pop("seed")
            new.pop("seed")
            assert old == new
            probe_parity.append(stage)
            captured = deepcopy(record["response"])
            return captured
        else:
            message = next(e["message"] for e in original["events"] if e.get("stage") == stage)
            sources.append(
                dict(stage=stage, source="captured_text_under_changed_history_not_new_inference")
            )
        return response(meta, deepcopy(message))

    run = graph_run(case, output, responder, cfg)
    result = run.execute()
    assert result["task_guidance"]["validated_plan"] and result["crop_rounds"] == 1
    assert len(probe_parity) == 9 and len(run.plan_rejections) == 1
    assert result["task_guidance"]["plan_rejections"][0]["required_tool"] == "define_goals"
    assert all(read(p)["backend"] == "mock" for p in (output / "requests").glob("*.json"))
    return dict(
        case=key,
        original_first_action=planners[0].get("tool_calls", [])[0]["function"]["name"],
        captured_first_action_rejected=True,
        rejection=run.plan_rejections[0]["error"],
        probes_before_valid_plan=0,
        corrected_plan_source="human_written_control_not_model_response",
        validated_plan=True,
        native_probe_bodies_and_images_matching=9,
        seed_changed_due_to_added_planning_turns=True,
        final_reader_complete={"direct", "aggregator"}
        <= {e.get("stage") for e in result["events"]},
        unresolved_nodes=sum(n["status"] == "unresolved" for n in run.tasks.nodes),
        response_sources=sources,
        model_accuracy_measured=False,
        network_calls=0,
    )


@pytest.mark.parametrize("key", FAILED_EIGHT)
def test_actual_eight_logs_reject_bypass_or_bad_plan_then_allow_corrected_plan(tmp_path, key):
    if not (ROOT / "inference_manifest.json").exists():
        pytest.skip("Optional actual failure logs are not distributed")
    replay_failed_case(ROOT, key, tmp_path)


def test_actual_nine_invalid_final_outputs_parse_only_their_explicit_label():
    if not (ROOT / "execution_audit.json").exists():
        pytest.skip("Optional captured invalid answers are not distributed")
    bad = read(ROOT / "execution_audit.json")["invalid_final"]
    assert len(bad) == 9
    for item in bad:
        value = parse_answer(item["output"])
        assert value in {"A", "C"}
        # The model's answer is not changed to gold: the three C answers remain C.
        if "(C) 4." in item["output"]:
            assert value == "C"


def test_actual_partial_repair_still_rejects_string_dependencies():
    if not (ROOT / "inference_manifest.json").exists():
        pytest.skip("Optional actual failure logs are not distributed")
    import json

    original = read(ROOT / "trajectories/treebench_ef13a5b34e2a/B/result.json")
    message = next(e["message"] for e in original["events"] if e.get("stage") == "planner_1")
    steps = json.loads(message["tool_calls"][0]["function"]["arguments"])["steps"]
    steps[2]["kind"] = "relation"  # Human-written partial correction only.
    graph = TaskGraph("Explain the contact of the woman and bench.")
    before = deepcopy(graph.nodes)
    with pytest.raises(ValueError, match=r"steps\[2\].requires"):
        graph.define_goals(steps)
    assert graph.nodes == before and graph.phase == "define_goals"


def test_reused_answers_use_the_same_parser_without_overwriting_saved_result(tmp_path):
    from active_vlm.common import save
    from active_vlm.experiment.report import report

    case = make_case(tmp_path)
    # Offline report fixture, no approval or model call. Labels are opened only
    # after the explicitly terminal marker by the normal reporter.
    save(tmp_path / "execution_end.json", dict(terminal=True, status="COMPLETE"))
    save(tmp_path / "plan.json", dict(arms=["A"], names=dict(A="Greedy"), reuse=dict(completed={})))
    save(tmp_path / "config.json", dict(engine="dependency_graph", account_root=str(tmp_path)))
    save(tmp_path / "inference_manifest.json", [case])
    label_file = tmp_path / "labels.json"
    save(label_file, {case["key"]: "A"})
    from active_vlm.common import digest

    save(
        tmp_path / "dataset_sources.json",
        dict(
            sources={}, labels_path=str(label_file), labels_sha256=digest(label_file.read_bytes())
        ),
    )
    saved = dict(
        case_key=case["key"],
        arm="A",
        prediction=None,
        stop_reason="MODEL_READY",
        calls=2,
        logical_usd=0.01,
        rounds=[],
        errors=[],
        events=[
            dict(stage="direct", message=dict(content="<answer>A</answer>")),
            dict(stage="aggregator", message=dict(content="<answer>(A) Blue</answer>")),
        ],
        receipts=[dict(stage="direct", status="ok"), dict(stage="aggregator", status="ok")],
    )
    path = tmp_path / "trajectories" / case["key"] / "A/result.json"
    save(path, saved)
    summary = report(tmp_path)
    assert summary["totals"]["A"]["correct"] == 1
    assert summary["normalized_saved_answers"][case["key"] + "/A"] == dict(stored=None, parsed="A")
    assert read(path)["prediction"] is None
