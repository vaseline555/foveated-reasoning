"""Economic identities, adversarial contracts and real shared harness execution.

All synthetic semantic reports below are human-written controls, not new VLM data.
"""

import json
import math
import random
import time
from copy import deepcopy

import pytest

from active_vlm.budget import BudgetStop, BudgetStore
from active_vlm.common import canonical, read, save
from active_vlm.decision.model import EvidenceDAG, posterior, voi
from active_vlm.decision.runner import DecisionGraphRun, parse_tool
from active_vlm.experiment import cli
from active_vlm.transport import Transport

from .replay import no_dispatch
from .test_execution import SelectorResponder, config, make_case, response


@pytest.fixture(autouse=True)
def offline():
    with no_dispatch():
        yield


def step(test="Is the target visibly blue?", likelihood=None, deps=None, box=None):
    return dict(
        test=test,
        requires=deps or [],
        box=box or [100, 100, 900, 900],
        positive_if_answer=likelihood or [0.95, 0.05],
        visibility=0.95,
    )


def observation(**fields):
    return dict(
        dict(
            outcome="positive",
            confidence=0.95,
            source="crop",
            observation="Blue surface is visibly present.",
            target_description="The requested object in the crop.",
            needs_repair=False,
            reason="",
        ),
        **fields,
    )


def test_exact_voi_matches_explicit_expectation_and_is_nonnegative():
    rng = random.Random(93)
    for n in range(2, 12):
        for _ in range(50):
            p = [rng.random() for _ in range(n)]
            p = [x / sum(p) for x in p]
            likelihood = [rng.random() for _ in p]
            z = sum(a * b for a, b in zip(p, likelihood, strict=True))
            explicit = (
                z * max(posterior(p, likelihood))
                + (1 - z) * max(posterior(p, [1 - x for x in likelihood]))
                - max(p)
            )
            assert voi(p, likelihood) == pytest.approx(explicit)
            assert voi(p, likelihood) >= 0
            assert voi(p, [0.8] * n) == pytest.approx(0)


def test_dependency_macro_path_can_pay_for_zero_immediate_information():
    dag = EvidenceDAG(2)
    dag.accept([step("Locate the specified person.", [0.5, 0.5]), step(deps=[0])])
    best, _ = dag.select()
    assert best["focus"] == 0 and best["path"] == ["r1g1", "r1g2"]
    r = dag.observe(0, observation(), "receipt1", "crop", [1, 2, 3, 4], 0.002)
    assert r["realized_gain"] == 0 and r["reward"] < 0
    assert dag.select()[0]["focus"] == 1
    # Same crop can measure a different test, without another native probe.
    r = dag.observe(1, observation(), "receipt2", "crop", [1, 2, 3, 4], 0.001)
    assert r["realized_gain"] > 0 and dag.select()[0] is None


@pytest.mark.parametrize("source", ["original", "question", "inference"])
def test_non_crop_claims_have_no_evidence_gain(source):
    dag = EvidenceDAG(2)
    dag.accept([step()])
    r = dag.observe(0, observation(source=source), "r", "c", [0] * 4, 0.001)
    assert not r["observed"] and r["realized_gain"] == 0 and r["reward"] < 0


def test_negative_measurement_and_confidence_are_not_positive_node_count_rewards():
    dag = EvidenceDAG(2)
    dag.accept([step()])
    r = dag.observe(0, observation(outcome="negative"), "r", "c", [0] * 4, 0.001)
    assert dag.belief[1] > dag.belief[0] and r["realized_gain"] > 0
    assert not r["semantic_truth_verified"]
    for q in [0, 0.5]:
        other = EvidenceDAG(2)
        other.accept([step()])
        assert not other.observe(0, observation(confidence=q), "r", "c", [0] * 4, 0.001)["observed"]


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), True, "1", -1, 2, []])
def test_bad_probabilities_fail_atomically(bad):
    dag = EvidenceDAG(2)
    dag.accept([step()])
    old = deepcopy(dag.nodes)
    with pytest.raises(ValueError):
        dag.accept([step(likelihood=[bad, 0.2])])
    assert dag.nodes == old and dag.revision == 1


@pytest.mark.parametrize("deps", [[0], [True], [1], [-1], ["0"], [[0]], [0, 0]])
def test_cycles_types_and_forward_dependencies_rejected(deps):
    dag = EvidenceDAG(2)
    with pytest.raises(ValueError):
        dag.accept([step(deps=deps)])


@pytest.mark.parametrize("bad", [[], {}, None, 3])
def test_observation_malformed_enums_do_not_crash_python(bad):
    dag = EvidenceDAG(2)
    dag.accept([step()])
    with pytest.raises(ValueError):
        dag.observe(0, observation(outcome=bad), "r", "c", [0] * 4, 0.001)


def test_revisions_cannot_recount_same_test_same_crop():
    dag = EvidenceDAG(2)
    dag.accept([step()])
    dag.observe(0, observation(), "r1", "c", [0] * 4, 0.001)
    belief = list(dag.belief)
    dag.accept([step()])
    with pytest.raises(ValueError, match="TWICE"):
        dag.observe(0, observation(), "r2", "c", [0] * 4, 0.001)
    assert dag.belief == belief


@pytest.mark.parametrize(
    "message",
    [
        {},
        dict(tool_calls=[]),
        dict(tool_calls=[None]),
        dict(tool_calls=[dict(function={})]),
        dict(tool_calls=[dict(function=dict(name="wanted", arguments="[]"))]),
        dict(tool_calls=[dict(function=dict(name="wanted", arguments="{"))]),
    ],
)
def test_tool_parse_failures_are_controlled(message):
    with pytest.raises(ValueError):
        parse_tool(message, dict(finish_reason="tool_calls"), "wanted")


def test_even_closed_tool_output_is_invalid_when_truncated():
    with pytest.raises(ValueError, match="TRUNCATED"):
        parse_tool({}, dict(finish_reason="length"), "wanted")


class DecisionResponder(SelectorResponder):
    def __call__(self, request, meta):
        if meta["arm"] != "B" or meta["stage"] in {"direct", "aggregator"}:
            return super().__call__(request, meta)
        self.inputs.append(deepcopy(meta))
        if meta["stage"].startswith("probe_"):
            return response(meta, dict(role="assistant", content="No"))
        name = request["tools"][0]["function"]["name"]
        payload = (
            dict(
                reading=None,
                steps=[step("Locate the specified person.", [0.5, 0.5]), step(deps=[0])],
            )
            if name == "design_experiments"
            else observation()
        )
        return response(
            meta,
            dict(
                role="assistant",
                content="",
                tool_calls=[
                    dict(
                        id="return",
                        type="function",
                        function=dict(name=name, arguments=canonical(payload)),
                    )
                ],
            ),
        )


def test_all_four_paths_share_native_primitives_and_final_reader(tmp_path):
    case = make_case(tmp_path)
    manifest = tmp_path / "manifest.json"
    save(manifest, [case])
    root = cli.main(
        [
            "prepare",
            "--manifest",
            str(manifest),
            "--benchmarks",
            "vstar",
            "--full",
            "--graph-controller",
            "decision",
            "--account-root",
            str(tmp_path / "account"),
            "--output",
            "results/decision",
        ]
    )
    responder = DecisionResponder()
    assert cli.run(root, approve=True, responder=responder) == "COMPLETE"
    graph = read(root / "trajectories" / case["key"] / "B/result.json")
    assert graph["prediction"] == "A" and not graph["errors"]
    assert graph["probe_calls"] == 1 and len(graph["decision_graph"]["observations"]) == 2
    assert all(x["observed"] for x in graph["decision_graph"]["observations"])
    assert sum(m["stage"] == "direct" for m in responder.inputs) == 1
    assert not read(root / "physical_ledger.json")["held"]
    assert read(root / "summary.json")["simulation"]
    # Native No relevance does not erase concrete semantic observation.
    assert graph["decision_graph"]["belief"][0] > 0.5
    assert "Blue surface" in canonical(graph["final_history"])


def test_unexpected_harness_error_aborts_study_without_label_report(tmp_path, monkeypatch):
    case = make_case(tmp_path)
    manifest = tmp_path / "manifest.json"
    save(manifest, [case])
    root = cli.main(
        [
            "prepare",
            "--manifest",
            str(manifest),
            "--benchmarks",
            "vstar",
            "--methods",
            "graph",
            "--full",
            "--graph-controller",
            "decision",
            "--account-root",
            str(tmp_path / "account"),
            "--output",
            "results/bug",
        ]
    )

    def bug(self):
        raise KeyError("injected implementation defect")

    monkeypatch.setattr(DecisionGraphRun, "explore", bug)
    assert cli.run(root, approve=True, responder=DecisionResponder()) == "STOPPED_HarnessError"
    assert not (root / "answers.csv").exists() and not (root / "summary.json").exists()


def test_expired_deadline_makes_no_physical_attempt_or_unknown_hold(tmp_path):
    cfg = dict(config(), deadline_unix=time.time() - 1)
    budget = BudgetStore(tmp_path / "ledger.json", cfg, 0, 30, ["c/B"])
    transport = Transport(tmp_path, budget, cfg, responder=lambda *_: pytest.fail("No dispatch"))
    token = budget.reserve("c/B", ["planner_1"])[0]
    with pytest.raises(BudgetStop, match="DEADLINE"):
        transport.call("c", "B", "planner_1", [], token)
    budget.release_unused([token])
    assert not read(budget.path)["held"]
    assert not list((tmp_path / "requests").glob("*.json"))


def test_reward_telescopes_with_all_observation_costs():
    dag = EvidenceDAG(2)
    dag.accept([step("One measurement.", [0.8, 0.2]), step("Another measurement.", [0.7, 0.3])])
    rows = [dag.observe(i, observation(), str(i), str(i), [0] * 4, 0.003) for i in range(2)]
    assert math.fsum(r["reward"] for r in rows) == pytest.approx(max(dag.belief) - 0.5 - 0.006)


def test_declared_target_or_task_failure_cannot_gain_even_with_match_outcome():
    dag = EvidenceDAG(2)
    dag.accept([step()])
    row = dag.observe(0, observation(needs_repair=True), "r", "c", [0] * 4, 0.001)
    assert not row["observed"] and row["realized_gain"] == 0


def test_native_pixel_validation_is_atomic_before_plan_acceptance():
    dag = EvidenceDAG(2)
    dag.accept([step()])
    prior = deepcopy(dag.nodes)

    def invalid_native_pixels(_):
        raise ValueError("EMPTY_NATIVE_CROP")

    with pytest.raises(ValueError, match="EMPTY"):
        dag.accept([step()], invalid_native_pixels)
    assert dag.nodes == prior and dag.revision == 1


def test_unobserved_native_relevance_gets_costed_replan_and_new_crop(tmp_path):
    cfg = dict(config(), crop_probe_mode="graph-single", strict_harness=True)
    case = make_case(tmp_path)
    budget = BudgetStore(tmp_path / "ledger.json", cfg, 0, 30, [case["key"] + "/B"])
    seen = []

    def responder(request, meta):
        seen.append(meta["stage"])
        if meta["stage"].startswith("probe_"):
            return response(meta, dict(role="assistant", content="No"))
        if meta["stage"] in {"direct", "aggregator"}:
            return response(meta)
        name = request["tools"][0]["function"]["name"]
        if name == "design_experiments":
            box = [0, 0, 400, 400] if meta["stage"] == "planner_1" else [500, 500, 1000, 1000]
            payload = dict(reading=None, steps=[step(box=box)])
        else:
            payload = (
                observation(
                    outcome="unobserved",
                    source="question",
                    needs_repair=True,
                    observation="The native report only repeats the question.",
                )
                if meta["stage"] == "observe_1_0"
                else observation()
            )
        return response(
            meta,
            dict(
                role="assistant",
                content="",
                tool_calls=[dict(function=dict(name=name, arguments=canonical(payload)))],
            ),
        )

    run = DecisionGraphRun(
        case, "B", Transport(tmp_path, budget, cfg, responder=responder), tmp_path / "B"
    )
    result = run.execute()
    assert result["prediction"] == "A" and not result["errors"]
    assert result["decision_graph"]["revisions"] == 1 and result["probe_calls"] == 2
    rows = result["decision_graph"]["observations"]
    assert rows[0]["realized_gain"] == 0 and rows[1]["realized_gain"] > 0
    assert rows[0]["crop_id"] != rows[1]["crop_id"]
    assert (
        sum(
            e["reward"]
            for e in result["decision_graph"]["cost_events"]
            if e["stage"].startswith("planner_")
        )
        < 0
    )
    assert seen[-2:] == ["direct", "aggregator"]


def test_redundant_ordinary_reading_is_ignored_and_schema_requires_null(tmp_path):
    case = make_case(tmp_path)
    cfg = dict(config(), crop_probe_mode="graph-single", strict_harness=True)
    store = BudgetStore(tmp_path / "ledger.json", cfg, 0, 30, [case["key"] + "/B"])

    class RedundantReading(DecisionResponder):
        def __call__(self, request, meta):
            result = super().__call__(request, meta)
            if meta["stage"].startswith("planner_"):
                assert request["tools"][0]["function"]["parameters"]["properties"]["reading"] == {
                    "type": "null"
                }
                function = result["choices"][0]["message"]["tool_calls"][0]["function"]
                payload = json.loads(function["arguments"])
                payload["reading"] = {
                    "question": "UNTRUSTED REWRITE",
                    "options": ["Wrong", "Wrong"],
                }
                function["arguments"] = canonical(payload)
            return result

    run = DecisionGraphRun(
        case, "B", Transport(tmp_path, store, cfg, responder=RedundantReading()), tmp_path / "B"
    )
    result = run.execute()
    assert not result["errors"] and not result["task_guidance"]["plan_rejections"]
    assert run.effective_question == case["question"] and run.plan_calls == 1
