"""Real woman/bench receipts plus separately labelled reference controls."""

from copy import deepcopy
from pathlib import Path

import pytest

from active_vlm.baselines.runner import FoveaRun
from active_vlm.budget import BudgetStore
from active_vlm.common import canonical, read
from active_vlm.graph.evidence import evidence_units
from active_vlm.graph.runner import GraphRun
from active_vlm.graph.state import TaskGraph
from active_vlm.transport import Transport, semantic_request

from .replay import no_dispatch
from .test_execution import response
from .test_question_evidence import real_plan

ROOT = Path(__file__).resolve().parents[1] / "results/graph_completion_pilot8_20261022"
KEY = "treebench_ef13a5b34e2a"


@pytest.fixture(autouse=True)
def block_network():
    with no_dispatch():
        yield


def saved():
    if not (ROOT / "inference_manifest.json").exists():
        pytest.skip("Optional local captured woman/bench logs are not distributed")
    case = next(c for c in read(ROOT / "inference_manifest.json") if c["key"] == KEY)
    return case, read(ROOT / "trajectories" / KEY / "B/result.json")


def new_graph(case, old):
    graph = TaskGraph(case["question"], image_size=case["original_size"])
    real_plan(graph, old)
    return graph


@pytest.mark.parametrize(
    "index,expected", [(0, ["g1", "g2", "g3"]), (1, ["g1", "g2"]), (2, []), (3, ["g1", "g2", "g3"])]
)
def test_actual_crop_references_recover_presence_without_inventing_full_body_coverage(
    index, expected
):
    case, old = saved()
    graph = new_graph(case, old)
    rd = old["rounds"][index]
    i = rd["selected_index"]
    reports = rd["raw_reports"][i]
    result = graph.observe(rd["candidates"][i], reports)
    assert result["newly_supported"] == expected
    source = {p["receipt_id"]: p["message"]["content"] for p in reports}
    for node in graph.nodes:
        if node["citations"]:
            assert len({c["receipt_id"] for c in node["citations"]}) >= 2
        for citation in node["citations"]:
            quotes = citation["reference_quotes"] + [
                citation.get(k)
                for k in ["quote", "context_quote", "source_quote", "coverage_quote"]
            ]
            assert all(q in source[citation["receipt_id"]] for q in quotes if q)
            if node["kind"] == "relation":
                assert (
                    citation["reported_value"][1]
                    == "there is a clear gap; she is not touching the bench at all."
                )
                assert citation["coverage_quote"]
    if index in {1, 2}:
        # All three partial-body probes said Yes, but not one supplied a valid
        # whole-body value. The blurry 27x70 crop supplies no presence either.
        inventory = [t for n in graph.nodes for t in n["targets"]]
        assert all(
            not list(evidence_units(graph.nodes[-1], p["message"]["content"], inventory))
            for p in reports
        )


def locate_units(text, inventory=None):
    """Human-written language controls, not new model/visual measurements."""
    target = "woman in lime-green dress"
    node = dict(kind="locate", targets=[target], task="Locate the question's woman")
    return list(evidence_units(node, text, inventory or [target]))


DEFINITION = "The question asks about the woman in the lime-green dress. "


@pytest.mark.parametrize(
    "reference", ["this specific woman", "this exact woman", "the woman in question"]
)
def test_question_definition_only_identifies_a_candidate_visual_reference(reference):
    units = locate_units(DEFINITION + f"The candidate cropped region shows {reference}.")
    assert units and units[0]["source_quote"] == DEFINITION.strip()
    assert not locate_units(DEFINITION + "<answer>Yes</answer>")
    assert not locate_units("The candidate cropped region shows this woman. " + DEFINITION)


@pytest.mark.parametrize(
    "candidate",
    [
        "The original image shows this woman.",
        "The parent image shows this woman.",
        "The agent's cropped image shows this woman.",
        "The candidate cropped region might show this woman.",
        "The candidate cropped region shows another woman.",
        "The candidate cropped region shows a different woman.",
        "The candidate cropped region does not show the woman.",
        "The candidate cropped region contains no visible woman.",
        "The candidate cropped region shows the bench. The woman is not visible.",
        "The candidate cropped region shows this woman wearing a red dress.",
        "The candidate cropped region is too blurry and lacks any discernible features of the woman.",
    ],
)
def test_references_never_override_absence_ambiguity_identity_or_uncertainty(candidate):
    assert not locate_units(DEFINITION + candidate)


def test_two_question_women_cannot_share_one_short_reference():
    inventory = ["woman in lime-green dress", "woman in red dress"]
    assert not locate_units(DEFINITION + "The cropped region shows this woman.", inventory)


def test_alignment_negation_does_not_negate_the_subject_or_contact_observation():
    text = DEFINITION + (
        "The cropped region shows this woman. "
        "The cropped region is not on a different part of the image; it is centered on the woman in the lime-green dress."
    )
    units = locate_units(text)
    assert len(units) == 2
    assert any("it is centered" in u["quote"] for u in units)


def test_reference_is_not_proof_of_wearing_or_ownership():
    target = "woman with yellow backpack"
    text = "The question asks about the woman with a yellow backpack. The cropped region shows this woman standing near a yellow backpack."
    for kind in ["condition", "attribute"]:
        node = dict(kind=kind, targets=[target], task="Determine her pose", attribute="pose")
        assert not list(evidence_units(node, text))


def replay_native(output):
    """Replay all 36 saved native probe wires, even diagnostically later crops."""
    case, old = saved()
    cfg = dict(
        read(ROOT / "config.json"), min_dispatch_interval_seconds=0, pacing=dict(mode="fixed")
    )
    receipts = {r["stage"]: r for r in old["receipts"] if r["status"] == "ok"}
    seen = []

    def replay(request, meta):
        source = read(ROOT / "requests" / (receipts[meta["stage"]]["request_hash"] + ".json"))
        assert semantic_request(request) == semantic_request(source["request"])
        seen.append(meta["stage"])
        captured = response(meta, deepcopy(source["message"]))
        captured["id"] = source["receipt_id"]
        return captured

    budget = BudgetStore(output / "ledger.json", cfg, 0, 30, [KEY + "/B"])
    run = GraphRun(case, "B", Transport(output, budget, cfg, responder=replay), output / "B")
    real_plan(run.tasks, old)
    for rd in old["rounds"]:
        run.turn = rd["turn"]
        base = rd["candidates"][1]
        pool = FoveaRun.crop_candidates(
            run,
            dict(
                anchor=dict(parent=base["parent"], box=base["normalized"], target_nodes=[]),
                branches=[],
            ),
        )
        assert [c["id"] for c in pool] == [c["id"] for c in rd["candidates"]]
        FoveaRun.probes(run, pool, rd["call_index"])
        assert run.rounds[-1]["scores"] == rd["scores"]
        run.images.commit(pool[rd["selected_index"]])
    assert len(seen) == 36
    assert all(read(p)["backend"] == "mock" for p in (output / "requests").glob("*.json"))
    return dict(unchanged_native_probes=36, unchanged_candidates=12, network_calls=0)


def test_actual_all_crop_candidates_and_probe_wires_are_unchanged(tmp_path):
    replay_native(tmp_path)


def replay_completed_prefix(output):
    """Exercise stop + common readers using saved replies, not new accuracy."""
    case, old = saved()
    cfg = dict(
        read(ROOT / "config.json"), min_dispatch_interval_seconds=0, pacing=dict(mode="fixed")
    )
    receipts = {r["stage"]: r for r in old["receipts"] if r["status"] == "ok"}
    seen = []

    def replay(request, meta):
        stage = meta["stage"]
        assert stage in {"planner_1", "planner_2", "direct", "aggregator"} or stage.startswith(
            "probe_2_0_"
        )
        source = read(ROOT / "requests" / (receipts[stage]["request_hash"] + ".json"))
        if stage.startswith("probe_"):
            assert semantic_request(request) == semantic_request(source["request"])
        if stage == "aggregator":
            assert canonical(request).count("TOOL OBSERVATION:") == 1
            assert request["messages"][0] == source["request"]["messages"][0]
        seen.append(stage)
        captured = response(meta, deepcopy(source["message"]))
        captured["id"] = source["receipt_id"]
        return captured

    budget = BudgetStore(output / "ledger.json", cfg, 0, 30, [KEY + "/B"])
    run = GraphRun(case, "B", Transport(output, budget, cfg, responder=replay), output / "B")
    result = run.execute()
    assert result["stop_reason"] == "ALL_GOALS_SUPPORTED"
    assert result["crop_rounds"] == 1 and result["calls"] == 13
    assert result["task_guidance"]["all_goals_supported"] and seen[-2:] == ["direct", "aggregator"]
    assert result["rounds"][0]["selected_id"] == old["rounds"][0]["selected_id"]
    assert not result["errors"]
    assert all(read(p)["backend"] == "mock" for p in (output / "requests").glob("*.json"))
    return dict(
        replay_calls=len(seen),
        original_calls=old["calls"],
        replay_crops=1,
        original_crops=4,
        stop=result["stop_reason"],
        final_reader_output_is_fixture=True,
        network_calls=0,
    )


def test_actual_supported_first_crop_stops_before_later_exploration(tmp_path):
    replay_completed_prefix(tmp_path)
