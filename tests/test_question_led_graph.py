"""Question/option separation controls; no new model correctness measurements."""

import json
from pathlib import Path

import pytest

from active_vlm.common import read, save
from active_vlm.decision.question_context import question_context
from active_vlm.experiment import cli

from .replay import no_dispatch
from .test_execution import make_case
from .test_partition_graph import PartitionResponder


@pytest.mark.parametrize(
    "values",
    [
        ["black", "blue", "red", "yellow"],
        ["12 months", "12 years", "24 months", "24 years"],
        ["wood", "steel", "glass", "plastic"],
        ["left of the man", "right of the man", "behind him", "in front of him"],
        ["red", "blue", "yellow", "red"],
    ],
)
def test_verbatim_hypotheses_do_not_rewrite_referent_or_create_evidence(values):
    stem = "What is the requested property of the object held by the woman?"
    q = stem + "\nOptions:\n" + "\n".join(f"{a}. {v}" for a, v in zip("ABCD", values, strict=True))
    c = question_context(q, list("ABCD"))
    assert c["question_spine"] == stem
    assert c["option_hypotheses"] == [
        dict(letter=a, text=v) for a, v in zip("ABCD", values, strict=True)
    ]
    assert c["options_are_visual_evidence"] is False and c["additional_api_calls"] == 0


@pytest.mark.parametrize(
    "q",
    [
        "Read the question and options inside the image.",
        "Which is taller? Options: A. tree C. house",
        "What does the marker A. stand for?",
    ],
)
def test_ambiguous_options_keep_the_authoritative_question(q):
    c = question_context(q, list("AB"))
    assert c["question_spine"] == q and c["option_hypotheses"] == []


def test_actual_thirty_questions_are_parsed_without_labels_or_semantic_dictionary():
    root = Path(__file__).resolve().parents[1] / "results/decision_graph30_restart_20261026"
    if not root.exists():
        pytest.skip("Optional historical responses are not distributed")
    cases = read(root / "inference_manifest.json")
    parsed, image_reading = 0, 0
    for case in cases:
        packet = question_context(case["question"], case["allowed_letters"])
        if case.get("embedded_question"):
            assert not packet["option_hypotheses"]
            assert packet["question_spine"] == case["question"]
            image_reading += 1
            continue
        assert packet["complete_explicit_option_block"]
        parsed += 1
        assert [c["letter"] for c in packet["option_hypotheses"]] == list(case["allowed_letters"])
        assert packet["question_spine"]
        assert all(c["text"] in case["question"] for c in packet["option_hypotheses"])
    assert (len(cases), parsed, image_reading) == (30, 29, 1)


def test_prompt_separation_adds_no_inference_calls_and_preserves_single_probe(tmp_path):
    case = make_case(tmp_path)
    manifest = tmp_path / "manifest.json"
    save(manifest, [case])
    packets = []

    class Inspect(PartitionResponder):
        def __call__(self, request, meta):
            if meta["stage"].startswith(("planner_", "observe_")):
                packet = json.loads(request["messages"][1]["content"][0]["text"])
                assert "question_spine" in packet and "option_hypotheses" in packet
                assert not packet["options_are_visual_evidence"]
                packets.append(packet)
            return super().__call__(request, meta)

    with no_dispatch():
        root = cli.main(
            [
                "prepare",
                "--manifest",
                str(manifest),
                "--full",
                "--methods",
                "graph",
                "--graph-controller",
                "partition",
                "--account-root",
                str(tmp_path / "account"),
                "--output",
                "results/ql",
            ]
        )
        assert cli.run(root, approve=True, responder=Inspect()) == "COMPLETE"
    r = read(root / "trajectories" / case["key"] / "B/result.json")
    assert r["calls"] == 5 and r["probe_calls"] == 1 and len(packets) == 2
