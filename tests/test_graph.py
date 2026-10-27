"""Recorded-image/request checks and separately labelled structural controls."""

import json
from copy import deepcopy
from pathlib import Path

import pytest

from active_vlm.budget import BudgetStore
from active_vlm.common import canonical, read
from active_vlm.experiment import cli
from active_vlm.graph.runner import GraphRun
from active_vlm.graph.state import TaskGraph, compile_question
from active_vlm.transport import Transport, semantic_request

from .replay import no_dispatch
from .test_execution import SelectorResponder, config, make_case, response

RESULTS = Path(__file__).resolve().parents[1] / "results"


@pytest.fixture(autouse=True)
def blocked_network(request):
    if (
        request.node.name.startswith(("test_actual", "test_recorded", "test_every_vstar"))
        and not (RESULTS / "stratified60/inference_manifest.json").exists()
    ):
        pytest.skip("Optional local captured experiment is not bundled")
    with no_dispatch():
        yield


def captured_case(key):
    path = RESULTS / "stratified60/inference_manifest.json"
    if not path.exists():
        if key == "vstar_70beb59fd7ab":
            return dict(
                key=key,
                question="Is the woman with a backpack on the left or right side of the person with a beanie?\n(A) right\n(B) left",
                original_size=[2250, 1500],
            )
        pytest.skip("Optional captured dataset is not bundled")
    return next(c for c in read(path) if c["key"] == key)


def task_config():
    return dict(
        config(), engine="fovea_task_graph", seed=1876186543, seed_arm_aliases={"F": "A", "G": "A"}
    )


def control_reports(text):
    """Human-written structural response; never a measured model observation."""
    return [dict(receipt_id=f"CONTROL-{i}", message=dict(content=text)) for i in range(3)]


def test_every_vstar_question_has_atomic_tasks_and_no_answer_leak():
    cases = [
        c for c in read(RESULTS / "stratified60/inference_manifest.json") if c["dataset"] == "vstar"
    ]
    assert len(cases) == 20
    for case in cases:
        plan = compile_question(case["question"])
        assert len(plan) >= 2 and not any(n["kind"] == "inspect" for n in plan)
        assert plan == compile_question(
            case["question"].split("\n")[0] + "\n(A) private alternative\n(B) another alternative"
        )
        assert not any({"answer", "label", "prediction", "box"} & n.keys() for n in plan)


def test_dependency_frontier_is_not_provenance_only():
    graph = TaskGraph("What is the pose of the woman with yellow backpack?")
    assert [n["kind"] for n in graph.ready()] == ["locate"]
    assert [n["kind"] for n in graph.ready(False)] == ["locate", "condition", "attribute"]
    assert graph.nodes[2]["depends_on"] == ["g2"]
    assert "dependencies" in graph.packet(True, full=True)
    assert "dependencies" not in graph.packet(False, full=True)
    assert "depends_on" not in graph.packet(False)["focus"]


def test_actual_negative_whole_question_probe_advances_local_goals_not_relation():
    key = "treebench_d9013b3ac73b"
    old = read(RESULTS / "minimal_graph_multicrop_pilot/trajectories" / key / "F/result.json")
    graph = TaskGraph(captured_case(key)["question"])
    states = []
    for rd in old["rounds"]:
        index = rd["selected_index"]
        states.append(graph.observe(rd["candidates"][index], rd["raw_reports"][index]))
    assert states[0]["newly_supported"] == ["g1", "g2"]
    assert old["rounds"][0]["scores"] == [0, 0, 0]
    assert states[1]["newly_supported"] == ["g3", "g4"]
    assert graph.focus()["kind"] == "frame"
    assert graph.nodes[-1]["id"] not in graph.packet()["ready_ids"]
    assert graph.nodes[-1]["status"] == "unresolved"
    assert all(c["receipt_id"].startswith("gen-") for n in graph.nodes for c in n["citations"])


def test_actual_visible_hydrant_accumulates_cited_native_evidence():
    key = "vstar_4bdfa55c6a2d"
    old = read(RESULTS / "stratified60/trajectories" / key / "A/result.json")
    graph = TaskGraph(captured_case(key)["question"])
    rd = old["rounds"][0]
    i = rd["selected_index"]
    report = graph.observe(rd["candidates"][i], rd["raw_reports"][i])
    assert report["newly_supported"] == ["g1", "g2"] and not graph.ready()
    assert all(len(n["citations"]) >= 2 for n in graph.nodes)
    assert report["semantic_truth_verified"] is False


@pytest.mark.parametrize(
    "text",
    [
        "<answer>Yes</answer>",
        "match",
        '{"identity_status":"match","findings":{"value":"standing"}}',
        "The original image shows a woman with yellow backpack standing.",
        "The cropped region does not show the woman with yellow backpack.",
        "The cropped region shows a woman next to a yellow backpack, standing.",
        "The cropped region might show a woman with yellow backpack standing.",
    ],
)
def test_controls_cannot_turn_match_or_nearby_bag_into_a_resolved_property(text):
    graph = TaskGraph("What is the pose of the woman with yellow backpack?")
    crop = dict(id="control", original_xyxy=[0, 0, 100, 100])
    graph.observe(crop, control_reports(text))
    assert graph.nodes[-1]["status"] == "unresolved"


def test_condition_requires_ownership_and_property_from_the_same_region():
    graph = TaskGraph("What is the pose of the woman with yellow backpack?")
    first = dict(id="woman_region", original_xyxy=[0, 0, 200, 200])
    graph.observe(
        first, control_reports("The cropped region shows a woman with a yellow backpack.")
    )
    assert graph.nodes[-1]["status"] == "unresolved"
    graph.observe(
        dict(id="other_region", original_xyxy=[300, 300, 400, 400]),
        control_reports("The cropped region shows a woman with a yellow backpack standing."),
    )
    assert graph.nodes[-1]["status"] == "unresolved"
    graph.observe(
        dict(id="nested_zoom", original_xyxy=[40, 40, 60, 60]),
        control_reports("The cropped region shows a woman with a yellow backpack standing."),
    )
    assert graph.nodes[-1]["status"] == "supported_by_reports"


def test_zero_progress_repeat_is_rejected_but_another_region_is_available():
    graph = TaskGraph("What is the color of the fire hydrant?")
    crop = dict(id="failed", original_xyxy=[10, 10, 110, 110])
    graph.observe(crop, control_reports("<answer>No</answer>"))
    assert graph.reject_repeat([crop, dict(crop, id="same_pixels")])
    assert not graph.reject_repeat([crop, dict(id="different", original_xyxy=[200, 200, 300, 300])])


def test_conflicting_property_values_and_duplicated_receipts_are_not_quorum():
    graph = TaskGraph("What is the color of the fire hydrant?")
    crop = dict(id="CONTROL", original_xyxy=[0, 0, 100, 100])
    reports = [
        dict(
            receipt_id=f"CONTROL-{i}",
            message=dict(content=f"The cropped region shows a {color} fire hydrant."),
        )
        for i, color in enumerate(["red", "blue", "green"])
    ]
    graph.observe(crop, reports)
    assert graph.nodes[0]["status"] == "supported_by_reports"
    assert graph.nodes[-1]["status"] == "unresolved"
    duplicate = control_reports("The cropped region shows a yellow fire hydrant.")
    for item in duplicate:
        item["receipt_id"] = "ONE-REPEATED-RECEIPT"
    graph.observe(crop, duplicate)
    assert graph.nodes[-1]["status"] == "unresolved"


def test_ordinal_uses_requested_order_without_inventing_a_facing_direction():
    graph = TaskGraph("Which person from the right is wearing a hat?")
    frame = next(n for n in graph.nodes if n["kind"] == "frame")
    assert frame["status"] == "specified_by_question"
    assert frame not in graph.ready()
    assert graph.focus()["kind"] == "locate"
    assert graph.nodes[-1]["targets"] == ["person wearing a hat"]
    crop = dict(id="CONTROL", original_xyxy=[0, 0, 100, 100])
    graph.observe(crop, control_reports("The cropped region shows three people wearing hats."))
    assert graph.nodes[-1]["status"] == "unresolved"


def test_controls_change_next_candidate_when_frame_dependency_is_unresolved():
    """A wiring counterfactual, not a claim about a new model response."""
    graph = TaskGraph(
        "From the viewpoint of the child dressed in pink, what is the relative directional position of the other child, who is wearing a blue-and-white-patterned hat?"
    )
    # Simulate already addressed locators/conditions, but not facing direction.
    for node in graph.nodes[:4]:
        node["status"] = "supported_by_reports"
    graph.attempts = [
        dict(focus_ids=["g5"], newly_supported=[], crop_id="old", original_xyxy=[0, 0, 100, 100])
    ]
    assert graph.focus(True)["kind"] == "frame" and graph.focus(False)["kind"] == "relation"
    candidates = [dict(id=f"CONTROL-{i}", original_xyxy=[0, 0, 100, 100]) for i in range(2)]
    reports = [
        control_reports("The cropped region shows the child dressed in pink facing the camera."),
        control_reports(
            "The cropped region shows the child dressed in pink and the child wearing a blue-and-white hat; from the pink child viewpoint the other child is to the right."
        ),
    ]
    assert graph.choose(candidates, [0.5, 1.0], reports, True)[0] == 0
    assert graph.choose(candidates, [0.5, 1.0], reports, False)[0] == 1


@pytest.mark.parametrize(
    "key",
    [
        "vstar_4bdfa55c6a2d",
        "vstar_94d16fa9d29e",
        "vstar_a76c84eb73a7",
        "vstar_70beb59fd7ab",
        "treebench_d9013b3ac73b",
    ],
)
def test_actual_first_pool_uses_identical_fovea_requests_images_and_votes(tmp_path, key):
    case = captured_case(key)
    # Original A has native FOVEA probes; never translate old graph JSON into Yes.
    original = read(RESULTS / "stratified60/trajectories" / key / "A/result.json")
    planner = next(e["message"] for e in original["events"] if e.get("stage") == "planner_1")
    args = json.loads(planner["tool_calls"][0]["function"]["arguments"])
    bystage = {r["stage"]: r for r in original["receipts"]}
    seen = []

    def replay(request, meta):
        old = read(
            RESULTS / "stratified60/requests" / (bystage[meta["stage"]]["request_hash"] + ".json")
        )
        assert semantic_request(request) == semantic_request(old["request"])
        assert "response_format" not in request and "tools" not in request
        seen.append(meta["stage"])
        return response(meta, deepcopy(old["message"]))

    cfg = task_config()
    store = BudgetStore(tmp_path / "ledger.json", cfg, 0, 30, [key + "/F", key + "/G"])
    transport = Transport(tmp_path, store, cfg, responder=replay)
    runs = []
    for arm in ["F", "G"]:
        run = GraphRun(case, arm, transport, tmp_path / arm)
        run.turn = 1
        pool = run.crop_candidates(
            dict(
                anchor=dict(parent=args["image_url"], box=args["coordinates"], target_nodes=[]),
                branches=[],
            )
        )
        run.probes(pool, 0)
        if key == "vstar_70beb59fd7ab":
            with pytest.raises(ValueError, match="REPEATED_UNPRODUCTIVE_CROP"):
                run.crop_candidates(
                    dict(
                        anchor=dict(
                            parent=args["image_url"], box=args["coordinates"], target_nodes=[]
                        ),
                        branches=[],
                    )
                )
            assert len(run.receipts) == 9
        runs.append(run)
    assert len(seen) == 9  # Nine more logical calls share exactly the same inputs.
    assert (
        runs[0].rounds[0]["scores"]
        == runs[1].rounds[0]["scores"]
        == original["rounds"][0]["scores"]
    )
    assert [c["id"] for c in runs[0].rounds[0]["candidates"]] == [
        c["id"] for c in original["rounds"][0]["candidates"]
    ]
    assert all(read(p).get("backend") == "mock" for p in (tmp_path / "requests").glob("*.json"))


def test_recorded_legacy_rejections_do_not_become_native_observations():
    for key in ["vstar_70beb59fd7ab", "vstar_a76c84eb73a7", "vstar_94d16fa9d29e"]:
        old = read(RESULTS / "graph_crop60/trajectories" / key / "B/result.json")
        graph = TaskGraph(captured_case(key)["question"])
        for rd in old["rounds"]:
            raw = rd["raw_reports"][0]
            graph.observe(rd["candidates"][0], raw)
        assert not any(n["status"] == "supported_by_reports" for n in graph.nodes)


def test_read_embedded_task_before_compiling_target_relations():
    graph = TaskGraph(
        "Recognize the question and options in the image and answer it.", embedded=True
    )
    assert graph.focus()["kind"] == "read_task"
    # Human transcription of captured image; explicitly not new model OCR.
    outcome = graph.read_question(
        "What is the relative position of the man wearing a hat to the anime poster?",
        ["A. right", "B. left", "C. above", "D. below"],
    )
    assert outcome["independently_verified"] is False
    assert graph.nodes[-1]["kind"] == "relation" and len(graph.nodes[-1]["depends_on"]) == 2


@pytest.mark.parametrize("dependency", [[0], [-1], [True], [1.0]])
def test_invalid_dependencies_never_partially_replace_the_plan(dependency):
    graph = TaskGraph("Explain the visible situation.")
    before = deepcopy(graph.nodes)
    with pytest.raises(ValueError):
        graph.define_goals(
            [dict(kind="locate", task="Locate the person", targets=["person"], requires=dependency)]
        )
    assert graph.nodes == before


def test_task_plan_is_sent_once_and_stop_is_bounded(tmp_path):
    case = make_case(tmp_path)
    cfg = task_config()
    store = BudgetStore(tmp_path / "ledger.json", cfg, 0, 30, [case["key"] + "/G"])
    transport = Transport(tmp_path, store, cfg, responder=lambda request, meta: response(meta))
    run = GraphRun(case, "G", transport, tmp_path / "G")
    result = run.execute()
    assert result["stop_reason"] == "UNRESOLVED_GOALS_STOP"
    assert len([r for r in result["receipts"] if r["stage"].startswith("planner_")]) == 2
    assert result["calls"] == 4 and not result["errors"]
    assert (
        len(
            [
                m
                for m in run.history
                if "Python maintains visual subgoals" in str(m.get("content", ""))
            ]
        )
        == 1
    )
    assert all("dependencies" not in str(p["packet"]) for p in run.packets)
    assert "task_guidance" not in canonical(run.final_history)


def test_standalone_preparation_does_not_approve_or_dispatch(tmp_path):
    case = make_case(tmp_path)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps([case]))
    account = tmp_path / "account"
    root = cli.main(
        [
            "prepare",
            "--manifest",
            str(manifest),
            "--full",
            "--benchmarks",
            "vstar",
            "--methods",
            "flat",
            "graph",
            "--account-root",
            str(account),
            "--output",
            "results/check",
        ],
    )
    assert read(root / "config.json")["engine"] == "dependency_graph"
    assert not (root / "approval.json").exists() and not (root / "requests").exists()
    # A mock CLI execution also verifies engine routing, terminal reporting,
    # and bounded unresolved stops; never a real approval or model measurement.
    assert cli.run(root, approve=True, responder=SelectorResponder()) == "COMPLETE"
    summary = read(root / "summary.json")
    assert summary["simulation"] and "task_guidance" in summary
    for arm in ["F", "B"]:
        result = read(root / "trajectories" / case["key"] / arm / "result.json")
        assert result["task_guidance"]["intervention"] == "task_dependency_frontier"
