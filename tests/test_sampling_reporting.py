"""Stratified cohort coverage and conservative postmortem classifications."""

from copy import deepcopy

from active_vlm.experiment.data import balanced_sample, select
from active_vlm.experiment.postmortem import postmortem


def test_balanced_sampling_covers_tasks_and_subtypes_reproducibly():
    rows = [
        dict(dataset="mme", id=f"{group}/{sub}/{i}", category=str(group), subcategory=str(sub))
        for group in range(9)
        for sub in range(3)
        for i in range(5)
    ]
    chosen = select(rows, ["mme"], 20, 42, stratified=True)
    assert chosen == select(list(reversed(rows)), ["mme"], 20, 42, stratified=True)
    assert len({r["id"] for r in chosen}) == 20
    for group in range(9):
        members = [r for r in chosen if r["category"] == str(group)]
        assert 2 <= len(members) <= 3
        assert len({r["subcategory"] for r in members}) == len(members)


def test_stratified_sampling_redistributes_exhausted_small_strata():
    rows = [dict(id="small", category="small")] + [
        dict(id=str(i), category="big") for i in range(8)
    ]
    chosen = balanced_sample(rows, 7, 17)
    assert len(chosen) == 7 and sum(r["category"] == "small" for r in chosen) == 1
    assert len(balanced_sample(rows, 0, 17)) == 0


def test_postmortem_uses_paired_answers_and_actual_evidence_gain(tmp_path):
    cases = [
        dict(key=str(i), dataset="vstar", question="Question", image_path="image.png")
        for i in range(3)
    ]
    graph = dict(
        prediction="B",
        calls=4,
        logical_usd=0.01,
        stop_reason="STRUCTURAL_ACTION_ITEMS_COMPLETE",
        graph=dict(potential=1, resolved=["t1"], reports={}),
        errors=[],
        rounds=[dict(realized_report_gain=1)],
        events=[dict(stage="direct", message=dict(content="<answer>A</answer>"))],
    )
    baseline = dict(prediction="A", calls=10, logical_usd=0.03)
    results = {f"{i}/B": deepcopy(graph) for i in range(3)}
    results.update({f"{i}/A": deepcopy(baseline) for i in range(3)})
    results["1/B"]["prediction"] = "A"
    results["1/M"] = dict(baseline, prediction="B")
    results["2/B"]["prediction"] = "A"
    results["2/B"]["rounds"] = []
    out = postmortem(
        tmp_path,
        cases,
        results,
        dict.fromkeys(["0", "1", "2"], "A"),
        dict(A="Greedy", M="MCMC", B="Graph"),
    )
    assert out["counts"]["baseline_correct_graph_wrong"] == 1
    assert out["counts"]["near_miss"] == 1
    assert out["counts"]["graph_correct_baseline_wrong"] == 1
    assert out["counts"]["efficient_graph_success"] == 1
    assert (tmp_path / "postmortem.md").exists()
