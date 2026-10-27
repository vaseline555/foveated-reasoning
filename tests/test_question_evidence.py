"""Question-defined values: captured eight-case replay and labelled controls."""

from copy import deepcopy
from pathlib import Path

import pytest

from active_vlm.baselines.runner import FoveaRun
from active_vlm.budget import BudgetStore
from active_vlm.common import read
from active_vlm.graph.runner import GraphRun
from active_vlm.graph.state import TaskGraph
from active_vlm.transport import Transport, semantic_request

from .replay import no_dispatch
from .test_execution import response
from .test_graph import control_reports

ROOT = Path(__file__).resolve().parents[1] / "results/required_graph_plan_pilot8_20261022"
RECOVERED = {
    "mme_734aa94f0f5d": "54",
    "mme_7f8799322e5f": "fy 2024",
    "mme_f87eb313710a": "the right of the image",
    "treebench_ba37237a9294": "actively riding and moving forward on the bicycle",
    "treebench_3325bfff5a80": "wicker/rattan",
    "treebench_ef13a5b34e2a": "there is a clear gap; she is not touching the bench at all.",
}
# Five attributes and one contact relation; no value is read from the gold label.
KEYS = [*RECOVERED, "treebench_224cf03028e6", "mme_eb1b19cad952"]


@pytest.fixture(autouse=True)
def block_network():
    with no_dispatch():
        yield


def saved_case(key):
    if not (ROOT / "inference_manifest.json").exists():
        pytest.skip("Optional local paid logs are not distributed with the package")
    case = next(c for c in read(ROOT / "inference_manifest.json") if c["key"] == key)
    return case, read(ROOT / "trajectories" / key / "B/result.json")


def real_plan(graph, old):
    graph.define_goals(
        [
            dict(
                kind=n["kind"],
                task=n["task"],
                targets=n["targets"],
                requires=[int(d[1:]) - 1 for d in n["depends_on"]],
            )
            for n in old["task_guidance"]["goals"]
        ]
    )


def replay_evidence(key):
    case, old = saved_case(key)
    graph = TaskGraph(case["question"], image_size=case["original_size"])
    real_plan(graph, old)
    assert not any(n["citations"] for n in graph.nodes)
    for rd in old["rounds"]:
        i = rd["selected_index"]
        graph.observe(
            rd["candidates"][i],
            rd["raw_reports"][i],
            candidate_pool=rd["candidates"],
            candidate_reports=rd["raw_reports"],
        )
    reports = {
        p["receipt_id"]: p["message"]["content"]
        for rd in old["rounds"]
        for p in rd["raw_reports"][rd["selected_index"]]
    }
    for node in graph.nodes:
        citations = node["citations"]
        if citations:
            assert len({c["receipt_id"] for c in citations}) >= 2
        for citation in citations:
            for field in ["quote", "context_quote", "source_quote", "coverage_quote"]:
                if citation.get(field):
                    assert citation[field] in reports[citation["receipt_id"]]
    return case, old, graph


@pytest.mark.parametrize("key", KEYS)
def test_actual_eight_retains_provenance_and_recovers_only_observed_values(key):
    _, old, graph = replay_evidence(key)
    goal = graph.nodes[-1]
    if key in RECOVERED:
        assert goal["status"] == "supported_by_reports"
        assert all(
            c["reported_value"] == ("question_choice", RECOVERED[key]) for c in goal["citations"]
        )
        assert old["task_guidance"]["goals"][-1]["status"] == "unresolved"
    else:
        assert goal["status"] == "unresolved"
    if key == "treebench_ef13a5b34e2a":
        # Prior question definitions resolve explicit candidate "this woman" /
        # "the bench" observations. Full-body coverage and two agreeing actual
        # contact reports additionally support the relation, not the definition.
        assert all(n["status"] == "supported_by_reports" for n in graph.nodes)
        assert all(c["coverage_quote"] for c in goal["citations"])


@pytest.mark.parametrize("key", KEYS)
def test_actual_eight_native_crop_probe_requests_and_images_are_unchanged(tmp_path, key):
    case, old = saved_case(key)
    cfg = dict(
        read(ROOT / "config.json"), min_dispatch_interval_seconds=0, pacing=dict(mode="fixed")
    )
    receipts = {r["stage"]: r for r in old["receipts"]}
    seen = []

    def responder(request, meta):
        source = read(ROOT / "requests" / (receipts[meta["stage"]]["request_hash"] + ".json"))
        assert semantic_request(request) == semantic_request(source["request"])
        assert "tools" not in request and "response_format" not in request
        seen.append(meta["stage"])
        captured = response(meta, deepcopy(source["message"]))
        captured["id"] = source["receipt_id"]
        return captured

    budget = BudgetStore(tmp_path / "ledger.json", cfg, 0, 30, [key + "/B"])
    transport = Transport(tmp_path, budget, cfg, responder=responder)
    run = GraphRun(case, "B", transport, tmp_path / key)
    real_plan(run.tasks, old)
    for rd in old["rounds"]:
        run.turn = rd["turn"]
        base = rd["candidates"][1]  # Native 1.0x proposal; retain captured parent/turn.
        proposal = dict(
            anchor=dict(parent=base["parent"], box=base["normalized"], target_nodes=[]), branches=[]
        )
        pool = FoveaRun.crop_candidates(run, proposal)
        assert [c["id"] for c in pool] == [c["id"] for c in rd["candidates"]]
        # This checks every unchanged native request. Selection may change under
        # the new checker; keep the recorded trajectory to isolate the checker.
        FoveaRun.probes(run, pool, rd["call_index"])
        assert run.rounds[-1]["scores"] == rd["scores"]
        i = rd["selected_index"]
        run.images.commit(pool[i])
        run.tasks.observe(
            pool[i], rd["raw_reports"][i], candidate_pool=pool, candidate_reports=rd["raw_reports"]
        )
    assert len(seen) == len(old["rounds"]) * 9
    assert all(read(p)["backend"] == "mock" for p in (tmp_path / "requests").glob("*.json"))


def question_graph(question, target="object", kind="attribute"):
    """Human-written structural plan; never a new model observation."""
    graph = TaskGraph(question)
    # Use the validated-plan path even for new property names, not field lists.
    graph.nodes = [dict(kind="inspect")]
    graph.define_goals(
        [
            dict(kind="locate", task="Locate the " + target, targets=[target], requires=[]),
            dict(kind=kind, task=question.splitlines()[0], targets=[target], requires=[0]),
        ]
    )
    return graph


def observe(graph, text, reports=None):
    graph.observe(
        dict(id="CONTROL", original_xyxy=[0, 0, 100, 100]), reports or control_reports(text)
    )
    return graph.nodes[-1]["status"]


@pytest.mark.parametrize(
    "question,claim",
    [
        (
            "What number is on the object?\n(A) 917\n(B) 928",
            "The cropped region shows the object with number 917.",
        ),
        (
            "What number is on the object?\n(A) 3700\n(B) 928",
            "The cropped region shows the object with number 3,700.0.",
        ),
        (
            "What word is on the object?\n(A) One\n(B) Two",
            "The cropped region shows the object with the word one.",
        ),
        (
            "What is the object used for?\n(A) Seed sorting\n(B) Water storage",
            "The cropped region shows the object being used for seed sorting.",
        ),
        (
            "Which direction is the object?\n(A) Northeast\n(B) Southwest",
            "The cropped region shows the object located to the northeast.",
        ),
        (
            "What is the object doing?\n(A) Somersaulting\n(B) Hovering",
            "The cropped region shows the object somersaulting.",
        ),
        (
            "Which year has the highest value?\n(A) FY 2091\n(B) FY 2093",
            "The cropped region shows the complete object chart. The highest bar is FY 2093.",
        ),
        (
            "Which year has the highest value?\n(A) FY2091\n(B) FY2093",
            "The cropped region shows the complete object chart. The highest bar is FY2093.",
        ),
        (
            "What is the object's material?\n(A) Aerogel\n(B) Cork",
            "The cropped region shows the object made of aerogel.",
        ),
    ],
)
def test_unseen_values_from_question_choices_need_no_value_dictionary(question, claim):
    assert observe(question_graph(question), claim) == "supported_by_reports"


@pytest.mark.parametrize(
    "claim",
    [
        "<answer>A</answer>",
        "match",
        "The original image shows the object made of aerogel.",
        "The agent's cropped image shows the object made of aerogel.",
        "The cropped region might show the object made of aerogel.",
        "The cropped region does not show the object made of aerogel.",
        "The cropped region shows the object. The original image shows a second object. It is made of aerogel.",
        "The cropped region shows the object. Another object is made of aerogel.",
        "The cropped region shows the object next to a box made of aerogel.",
        "The question asks if the object is aerogel or cork. <answer>Yes</answer>",
        "The cropped region shows the object. This corresponds to option A, aerogel.",
        "The cropped region shows the object. It may be aerogel. It is aerogel.",
    ],
)
def test_no_question_echo_uncertain_claim_or_other_object_can_supply_a_value(claim):
    assert (
        observe(question_graph("What is the object's material?\n(A) Aerogel\n(B) Cork"), claim)
        == "unresolved"
    )


def test_conflicts_and_repeated_receipts_are_not_agreement():
    q = "What is the number on the object?\n(A) 917\n(B) 928\n(C) 939"
    for values, ids in [([917, 928, 939], ["C1", "C2", "C3"]), ([917] * 3, ["DUP"] * 3)]:
        reports = [
            dict(
                receipt_id=i,
                message=dict(content=f"The cropped region shows the object with number {v}."),
            )
            for v, i in zip(values, ids, strict=True)
        ]
        assert observe(question_graph(q), "", reports) == "unresolved"


@pytest.mark.parametrize(
    "claim,expected",
    [
        ("The cropped region shows the object touching the bench.", "Touching the bench"),
        ("The cropped region shows the object not touching the bench.", "Not touching the bench"),
    ],
)
def test_negative_relation_has_distinct_polarity_not_target_absence(claim, expected):
    graph = question_graph(
        "What is the contact state?\n(A) Touching the bench\n(B) Not touching the bench",
        kind="relation",
    )
    assert observe(graph, claim) == "supported_by_reports"
    assert graph.nodes[-1]["citations"][0]["reported_value"] == (
        "question_choice",
        expected.lower(),
    )


@pytest.mark.parametrize(
    "claim",
    [
        "The cropped region shows the object chart with FY 2093.",
        "The cropped region shows a few bars from the object chart. The highest visible bar is FY 2093.",
        "The cropped region shows the object chart. The highest bar is FY 2093.",
    ],
)
def test_year_label_or_local_maximum_is_not_a_global_comparison(claim):
    assert (
        observe(
            question_graph("Which year has the highest value?\n(A) FY 2091\n(B) FY 2093"), claim
        )
        == "unresolved"
    )


def test_identity_words_and_camera_viewpoint_do_not_vote_for_an_answer():
    graph = question_graph(
        "Where is the yellow object?\n(A) Under a yellow awning\n(B) In the central lane",
        "yellow object",
        "relation",
    )
    assert (
        observe(
            graph,
            "The cropped region shows the yellow object, which is the central subject of the question.",
        )
        == "unresolved"
    )
    graph = question_graph("Where is the object in the image?\n(A) At the top\n(B) At the bottom")
    assert (
        observe(graph, "The cropped region shows the object from a top-down perspective.")
        == "unresolved"
    )
