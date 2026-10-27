"""Real failure replay and adversarial worker/protocol controls; never inference."""

import io
import json
import urllib.error
from copy import deepcopy
from pathlib import Path

import pytest

from active_vlm.answers import parse_answer
from active_vlm.baselines.runner import FoveaRun
from active_vlm.baselines.selectors import select_fovea
from active_vlm.budget import BudgetStop, BudgetStore, UnknownCharge
from active_vlm.common import canonical, read, save
from active_vlm.errors import IntegrityError
from active_vlm.experiment import accounting, cli
from active_vlm.experiment.report import report
from active_vlm.graph.runner import GraphRun
from active_vlm.graph.state import TaskGraph
from active_vlm.protocol import crop_protocol
from active_vlm.questions import normalize_reading
from active_vlm.transport import Transport

from .replay import no_dispatch
from .test_execution import config, make_case, response
from .test_plan_gate import call, graph_run

CAPTURED = Path(__file__).resolve().parents[1] / "results/sanitized_pilot30_20261022"
FAILED_KEY = "treebench_5b0b141fbb83"


@pytest.fixture(autouse=True)
def block_network():
    with no_dispatch():
        yield


def actual_failure():
    if not (CAPTURED / "inference_manifest.json").exists():
        pytest.skip("Optional local paid logs are not distributed")
    case = next(c for c in read(CAPTURED / "inference_manifest.json") if c["key"] == FAILED_KEY)
    result = read(CAPTURED / "trajectories" / FAILED_KEY / "B/result.json")
    messages = [e["message"] for e in result["events"] if e.get("stage", "").startswith("planner_")]
    return case, result, messages


def replay_actual_plan_failure(output):
    """Unmodified captured replies: raised limit exposes a second real defect."""
    case, old, messages = actual_failure()
    seen = []

    def captured(request, meta):
        assert meta["stage"].startswith("planner_")  # No crop or reader bypass.
        seen.append(meta["stage"])
        steps = request["tools"][0]["function"]["parameters"]["properties"]["steps"]
        assert steps["maxItems"] == 16
        return response(meta, deepcopy(messages[len(seen) - 1]))

    run = graph_run(case, output, captured)
    result = run.execute()
    assert result["path_terminal"] and result["status"] == "planning_failed"
    assert (
        len(seen) == 5 and result["calls"] == 5
    )  # 2 old target errors + 3 identical dependency errors.
    assert not result["rounds"] and result["prediction"] is None
    assert [r["error"]["path"] for r in run.plan_rejections] == (
        ["steps[1].targets"] * 2 + ["steps[5].requires"] * 3
    )
    assert run.plan_rejections[-1]["identical_invalid_attempts"] == 3
    assert run.plan_rejections[-1]["observed_step_count"] == 9
    assert len(run.plan_rejections[-1]["step_index_map"]) == 9
    assert run.tasks.phase == "define_goals" and not run.tasks.complete
    run.transport.budget.finish_path(run.path_key)
    assert not read(run.transport.budget.path)["final_pool"]
    return dict(
        case=case["key"],
        original_calls=old["calls"],
        replay_calls=result["calls"],
        actual_outputs_used=seen,
        plan_validated=False,
        original_defects=[
            "four targets in one step",
            "nine steps exceeded old max8",
            "string dependencies",
        ],
        new_stop=result["task_guidance"]["plan_stop_trigger"],
        native_crops_or_probes=0,
        measured_new_model_accuracy=False,
        paid_calls=0,
        network_calls=0,
    )


def test_real_ten_plan_outputs_stall_locally_without_fabricated_repair(tmp_path):
    replay_actual_plan_failure(tmp_path)


def test_real_nine_step_plan_needs_explicit_dependency_repair(tmp_path):
    case, _, messages = actual_failure()
    steps = json.loads(messages[2]["tool_calls"][0]["function"]["arguments"])["steps"]
    graph = TaskGraph(case["question"])
    before = deepcopy(graph.nodes)
    with pytest.raises(ValueError, match=r"steps\[5\].requires"):
        graph.define_goals(steps)
    assert graph.nodes == before
    # HUMAN-WRITTEN STRUCTURAL CONTROL ONLY: each comparison depends on wall and clothing.
    for i in range(5, 9):
        steps[i]["requires"] = [0, i - 4]
    accepted = graph.define_goals(steps)
    assert accepted["visual_facts_added"] == 0 and accepted["goals"] == 9
    assert graph.phase == "search" and not graph.complete and not graph.attempts
    assert all(n["status"] == "unresolved" and not n["citations"] for n in graph.nodes)


def test_actual_options_heading_is_not_part_of_ordinal_target():
    actual_failure()  # Optional corpus availability.
    r = read(CAPTURED / "trajectories/treebench_2209c12466a4/B/result.json")
    action = next(
        c
        for e in r["events"]
        for c in e.get("message", {}).get("tool_calls", [])
        if c["function"]["name"] == "read_question"
    )
    args = json.loads(action["function"]["arguments"])
    graph = TaskGraph("Read the question and choices in the image.", embedded=True)
    graph.read_question(**args)
    assert "Options:" not in graph.question and "(A) Green and Blue" in graph.question
    assert graph.phase == "define_goals"  # Model must explicitly plan population/filter/order/hat.
    assert all("Options" not in t for n in graph.nodes for t in n["targets"])
    assert "green" in graph.keywords["choice_contrasts"]["A"]


@pytest.mark.parametrize("deps", [[True], [1], [-1], ["0"], ["wall"], None])
def test_plan_dependency_types_cannot_mutate_state(deps):
    graph = TaskGraph("Explain the visible purpose of the bucket.")
    before = deepcopy(graph.nodes)
    with pytest.raises(ValueError, match="requires"):
        graph.define_goals(
            [
                dict(kind="locate", task="Locate the bucket", targets=["bucket"], requires=[]),
                dict(
                    kind="attribute",
                    task="Read its visible purpose",
                    targets=["bucket"],
                    requires=deps,
                ),
            ]
        )
    assert graph.nodes == before


def test_canonical_invalid_repetitions_and_success_clear_old_feedback(tmp_path):
    case = make_case(tmp_path)
    case.update(question="Read the printed task.", embedded_question=True)
    run = graph_run(case, tmp_path / "g", lambda req, meta: response(meta))
    bad = call("read_question", dict(text="What is it? Options:", options=["A. Red", "A. Blue"]))
    run.control_call(bad)
    differently_formatted = deepcopy(bad)
    differently_formatted["function"]["arguments"] = json.dumps(
        json.loads(bad["function"]["arguments"]), indent=2, sort_keys=True
    )
    run.control_call(differently_formatted)
    assert run.plan_rejections[-1]["identical_invalid_attempts"] == 2
    run.control_call(
        call(
            "read_question",
            dict(
                text="Explain the visible purpose of the bucket? Options:",
                options=["A. Waste", "B. Water"],
            ),
        )
    )
    assert run.tasks.phase == "define_goals" and run.latest_plan_error is None
    assert not run.invalid_plans and not run.plan_stalled
    assert "Required plan correction" not in run.planner_messages()[-1]["content"]
    run.control_call(
        call(
            "define_goals",
            dict(
                steps=[
                    dict(kind="locate", task="Locate the bucket", targets=["bucket"], requires=[]),
                    dict(
                        kind="attribute",
                        task="Read the visible purpose",
                        targets=["bucket"],
                        requires=[0],
                    ),
                ]
            ),
        )
    )
    assert run.tasks.phase == "search" and not run.plan_stalled


@pytest.mark.parametrize(
    "text,options",
    [
        ("What color? Options: (A) Red (B) Green", ["A. Red", "B. Blue"]),
        ("What color?", ["A. Red", "A. Blue"]),
        ("What color?", ["B. Red", "C. Blue"]),
        ("What color?", ["A. Red", "Blue"]),
        ("What color? Options: A. Red", ["A. Red", "B. Blue"]),
        ("What color?\n(A) Red", ["A. Red", "B. Blue"]),
    ],
)
def test_conflicting_or_partial_options_are_repaired_not_silently_dropped(text, options):
    with pytest.raises(ValueError):
        normalize_reading(text, options)


def test_multiline_question_and_duplicate_options_normalize_without_losing_filter():
    result = normalize_reading(
        "What color is the hat\nof the second man from the right?\nChoices:\n(A) Red\n(B) BLUE",
        ["A. red", "B. Blue"],
    )
    assert result == "What color is the hat of the second man from the right?\n(A) red\n(B) Blue"


@pytest.mark.parametrize("payload", ["", "AB", "BC", "A or B", "A. red; B. blue", "K"])
def test_final_label_cannot_use_substring_membership(payload):
    assert parse_answer(f"<answer>{payload}</answer>", "ABCD") is None


def single_config():
    return dict(
        config(),
        K=1,
        crop_probe_mode="single",
        seed_arm_aliases={"B": "A", "F": "A"},
        task_policy=dict(quorum=1, max_identical_invalid_plans=3),
    )


@pytest.mark.parametrize(
    "arm,mode",
    [(a, "single") for a in ["A", "M", "L", "B", "F"]] + [(a, "graph-single") for a in ["B", "F"]],
)
def test_each_turn_has_one_crop_and_one_probe_but_multiple_turns_allowed(tmp_path, arm, mode):
    case, cfg, seen = make_case(tmp_path), single_config(), []
    if mode == "graph-single":
        cfg.update(crop_probe_mode=mode, K=3)

    def respond(request, meta):
        stage = meta["stage"]
        seen.append(stage)
        message = dict(role="assistant", content="<answer>A</answer>")
        if stage in {"planner_1", "planner_2"}:
            box = [200, 200, 600, 600] if stage == "planner_1" else [0, 0, 300, 300]
            message["tool_calls"] = [
                call("cropping", dict(image_url="original", coordinates=box), ident=f"C-{i}")
                for i in range(2)
            ]
        elif stage.startswith("probe_"):
            message["content"] = "<answer>Yes</answer>"  # Relevance alone is NOT graph support.
        return response(meta, message)

    budget = BudgetStore(tmp_path / "ledger.json", cfg, 0, 30, [case["key"] + "/" + arm])
    transport = Transport(tmp_path, budget, cfg, responder=respond)
    cls = GraphRun if arm in {"B", "F"} else FoveaRun
    run = cls(case, arm, transport, tmp_path / arm)
    result = run.execute()
    assert result["prediction"] == "A" and not result["errors"]
    assert result["crop_rounds"] == result["probe_calls"] == 2
    assert [s for s in seen if s.startswith("probe_")] == ["probe_1_0_0_0", "probe_2_0_0_0"]
    assert all(len(r["candidates"]) == 1 for r in result["rounds"])
    assert not any(s.startswith("expansion_") for s in seen)
    assert not result["native_selector_enabled"]
    assert result["crop_probe_protocol"]["support_basis"] == "single_report"
    if arm in {"B", "F"}:
        assert not run.tasks.complete
        assert run.tasks.packet()["required_independent_reports"] == 1
        assert all(not a["newly_supported"] for a in run.tasks.attempts)
    assert sum("SINGLE_CROP_PER_TURN" in str(m.get("content")) for m in run.history) == 2
    run.turn = 2  # Attempt to re-enter the last actual crop/probe turn.
    before = read(budget.path)
    with pytest.raises(ValueError, match="SINGLE_PROBE_PER_TURN"):
        run.probes([{}], 1)
    with pytest.raises(ValueError, match="SINGLE_CROP_PER_TURN"):
        run.crop_candidates(dict(anchor=dict(parent="original", box=[0, 0, 300, 300])))
    with pytest.raises(ValueError, match="SINGLE"):
        select_fovea(run, {}, 0)
    assert read(budget.path) == before


def prepared_queue(tmp_path, methods=None):
    case = make_case(tmp_path)
    case["question"] = "Explain the bucket's visible purpose.\n(A) Waste\n(B) Water"
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
            "--account-root",
            str(tmp_path / "account"),
            "--output",
            "results/study",
            "--interval",
            "0",
            "--pacing",
            "fixed",
            "--final-answer-mode",
            "native",
            "--methods",
            *(methods or ["graph", "greedy", "mcmc", "lookahead"]),
        ]
    )
    # Force the failed graph to be first for this worker-routing control.
    plan = read(root / "plan.json")
    plan["reuse"]["pending"].sort(key=lambda p: p.rsplit("/", 1)[1] != "B")
    save(root / "plan.json", plan)
    accounting.freeze(root)
    return root, case


def test_one_bad_plan_and_invalid_answer_do_not_stop_other_paths_or_hide_cost(tmp_path):
    root, case = prepared_queue(tmp_path)
    seen = []

    def respond(request, meta):
        seen.append((meta["arm"], meta["stage"]))
        message = dict(role="assistant", content="<answer>A</answer>")
        if meta["arm"] == "B":
            message["tool_calls"] = [call("define_goals", dict(steps=[]))]
        elif meta["arm"] == "M" and meta["stage"] == "aggregator":
            message["content"] = "<answer>AB</answer>"
        return response(meta, message)

    assert cli.run(root, approve=True, responder=respond) == "COMPLETE_WITH_FAILURES"
    progress, summary = read(root / "progress.json"), read(root / "summary.json")
    assert progress["finished_paths"] == 4 and progress["final_reader_completed_paths"] == 3
    assert (
        progress["failed_paths"] == 2
        and summary["output_diagnostics"]["invalid_final_answers"]["M"] == 1
    )
    assert set(a for a, _ in seen) == {"A", "M", "L", "B"}
    assert summary["totals"]["B"]["logical_requests"] == 3
    assert summary["totals"]["B"]["logical_known_usd"] == pytest.approx(0.003)
    assert summary["totals"]["B"]["n"] == 1 and summary["totals"]["B"]["failed_paths"] == 1
    assert len(read(root / "path_outcomes.json")) == 4
    ledger = read(root / "physical_ledger.json")
    assert not ledger["held"] and not ledger["final_pool"] and not ledger["stopped"]
    assert all(p["finished"] for p in ledger["paths"].values())
    requests_before = {p.name: p.read_bytes() for p in (root / "requests").glob("*.json")}
    with pytest.raises(RuntimeError, match="already complete"):
        cli.run(root, responder=respond, resume=True)
    assert requests_before == {p.name: p.read_bytes() for p in (root / "requests").glob("*.json")}
    # Simulate an interrupted queue AFTER local graph failure, before later paths.
    save(root / "execution_end.json", dict(terminal=True, status="STOPPED_BY_USER"))
    cfg = read(root / "config.json")
    store = BudgetStore(
        root / "physical_ledger.json", cfg, 0, 30, read(root / "plan.json")["reuse"]["pending"]
    )
    transport = Transport(root, store, cfg, responder=respond, resume=True)
    assert not transport.replay_expected  # Recorded failures are terminal; never repaid.
    with pytest.raises(BudgetStop, match="PATH_ALREADY_TERMINAL"):
        store.reserve(case["key"] + "/B", ["planner_1"])


@pytest.mark.parametrize("fault", ["unknown_cost", "integrity", "provider", "auth"])
def test_financial_and_integrity_failures_still_stop_the_whole_queue(tmp_path, fault):
    root, _ = prepared_queue(tmp_path, methods=["graph", "greedy"])
    attempts = []

    def respond(request, meta):
        attempts.append(meta)
        if fault == "integrity":
            raise IntegrityError("OFFLINE_CORRUPT_RECEIPT_CONTROL")
        if fault == "auth":
            raise urllib.error.HTTPError("offline", 401, "denied", {}, io.BytesIO(b"{}"))
        body = response(meta)
        if fault == "unknown_cost":
            body["usage"] = {}
        else:
            body["provider"] = "different-provider"
        return body

    assert cli.run(root, approve=True, responder=respond) == "STOPPED_REQUEST_OR_BUDGET_FAILURE"
    assert len(attempts) == 1 and attempts[0]["arm"] == "B"
    ledger = read(root / "physical_ledger.json")
    assert ledger["stopped"] and not ledger["paths"][attempts[0]["case"] + "/B"].get("finished")
    assert (root / "summary.json").exists()


def test_failed_path_releases_only_unused_reader_protection_not_error_holds(tmp_path):
    cfg = config()
    store = BudgetStore(tmp_path / "ledger.json", cfg, 0, 30, ["case/B", "case/A"])
    token = store.reserve("case/B", ["planner_1"])[0]
    store.dispatched(token)
    with pytest.raises(UnknownCharge):
        store.settle(token, None)
    retry = store.retry_token(token, "fixture-hash")
    store.dispatched(retry)
    store.settle(retry, 0.001, "fixture-paid")
    store.finish_path("case/B")
    state = read(store.path)
    assert len(state["held"]) == 1 and state["held"][token]["status"] == "authorized_unknown"
    assert set(state["final_pool"]) == {"case/direct", "case/A/aggregator"}
    assert float(state["physical_paid"]) == 0.001
    store.finish_path("case/A")
    assert not read(store.path)["final_pool"] and len(read(store.path)["held"]) == 1


def test_fixed_denominator_includes_missing_failed_and_invalid_outputs(tmp_path):
    root, case = prepared_queue(tmp_path, methods=["greedy"])
    r = dict(
        case_key=case["key"],
        dataset="vstar",
        arm="A",
        status="planning_failed",
        path_terminal=True,
        prediction=None,
        stop_reason="CONTROL",
        receipts=[dict(status="ok")],
        events=[],
        logical_usd=0.001,
        errors=[],
    )
    save(root / "trajectories" / case["key"] / "A/result.json", r)
    label_file = tmp_path / "labels.json"
    save(label_file, {case["key"]: "A", "missing_case": "B"})
    # Terminal reporting fixture, not an inference input modification.
    save(
        root / "dataset_sources.json",
        dict(
            sources={},
            labels_path=str(label_file),
            labels_sha256=accounting.digest(label_file.read_bytes()),
        ),
    )
    cases = read(root / "inference_manifest.json")
    cases.append(dict(cases[0], key="missing_case"))
    save(root / "inference_manifest.json", cases)
    save(root / "execution_end.json", dict(terminal=True, status="COMPLETE_WITH_FAILURES"))
    summary = report(root)
    values = summary["totals"]["A"]
    assert values["correct"] == 0 and values["n"] == 2 and values["accuracy"] == 0
    assert values["failed_paths"] == 1 and values["missing_paths"] == 1
    assert values["logical_requests"] == 1 and values["logical_known_usd"] == 0.001
    assert (
        summary["paired_questions"] == 0
        and len((root / "answers.csv").read_text().splitlines()) == 3
    )


@pytest.mark.parametrize(
    "patch",
    [
        dict(K=3),
        dict(K=True),
        dict(task_policy=dict(quorum=2)),
        dict(task_policy=dict(max_identical_invalid_plans=True)),
    ],
)
def test_single_protocol_mismatch_fails_before_any_api_or_reservation(patch):
    with pytest.raises(ValueError):
        crop_protocol(dict(single_config(), **patch))


def test_last_planning_turn_can_correct_but_never_add_a_crop(tmp_path):
    case = make_case(tmp_path)
    case["question"] = "Explain the bucket's visible purpose.\n(A) Waste\n(B) Water"
    seen = []

    def respond(request, meta):
        stage = meta["stage"]
        seen.append(stage)
        message = dict(role="assistant", content="<answer>A</answer>")
        if stage.startswith("planner_"):
            turn = int(stage.split("_")[1])
            steps = [
                dict(
                    kind="locate",
                    task=f"Locate the bucket, attempt {turn}",
                    targets=["bucket"],
                    requires=[],
                )
            ]
            if turn == 10:
                assert "MUST stop calling tools" not in canonical(request["messages"])
                steps.append(
                    dict(
                        kind="attribute",
                        task="Read its visible purpose",
                        targets=["bucket"],
                        requires=[0],
                    )
                )
            message["tool_calls"] = [call("define_goals", dict(steps=steps))]
            if turn == 10:
                message["tool_calls"].append(
                    call("cropping", dict(image_url="original", coordinates=[0, 0, 500, 500]))
                )
        return response(meta, message)

    run = graph_run(case, tmp_path / "last", respond, single_config())
    result = run.execute()
    assert result["task_guidance"]["validated_plan"] and result["prediction"] == "A"
    assert result["calls"] == 12 and result["planner_turns"] == 10
    assert not result["rounds"] and not result["probe_calls"]
    assert not run.plan_stalled and run.latest_plan_error is None
    assert seen[-2:] == ["direct", "aggregator"]


def test_changed_frozen_crop_bytes_halt_before_any_new_dispatch(tmp_path):
    case, cfg, seen = make_case(tmp_path), single_config(), []
    budget = BudgetStore(tmp_path / "ledger.json", cfg, 0, 30, [case["key"] + "/A"])
    transport = Transport(tmp_path, budget, cfg, responder=lambda req, meta: seen.append(meta))
    run = FoveaRun(case, "A", transport, tmp_path / "A")
    view = run.history[1]["content"][0]["image_url"]["url"]
    Path(view["path"]).write_bytes(b"OFFLINE_TAMPER_CONTROL")
    result = run.execute()
    assert result["stop_reason"] == "INTEGRITY_GLOBAL_STOP" and not result["path_terminal"]
    assert result["calls"] == 0 and seen == [] and not read(budget.path)["held"]
    assert read(budget.path)["stopped"]


def test_protocol_reuse_rejects_native_results_in_single_mode(tmp_path):
    root, case = prepared_queue(tmp_path, methods=["greedy"])
    new_config = read(root / "config.json")
    new_config.update(crop_probe_mode="single", K=1, task_policy=dict(quorum=1))
    with pytest.raises(ValueError, match="Reuse protocol mismatch"):
        accounting.reuse([case], ["A"], [root], tmp_path / "other", new_config)


def test_single_specific_observation_can_support_and_stop_without_claiming_agreement(tmp_path):
    """Labelled structural routing control, NOT a newly measured sign recognition."""
    case = make_case(tmp_path)
    case["question"] = "What is the color of the sign?\n(A) Blue\n(B) Red"
    seen = []

    def respond(request, meta):
        stage = meta["stage"]
        seen.append(stage)
        message = dict(role="assistant", content="<answer>A</answer>")
        if stage == "planner_1":
            message["tool_calls"] = [
                call("cropping", dict(image_url="original", coordinates=[0, 0, 500, 500]))
            ]
        elif stage.startswith("probe_"):
            message["content"] = "The cropped region shows a blue sign. <answer>Yes</answer>"
        return response(meta, message)

    run = graph_run(case, tmp_path / "support", respond, single_config())
    result = run.execute()
    assert result["stop_reason"] == "ALL_GOALS_SUPPORTED"
    assert result["crop_rounds"] == result["probe_calls"] == 1 and result["calls"] == 4
    assert len(result["rounds"][0]["task_progress"]["newly_supported"]) == 2
    assert seen == ["planner_1", "probe_1_0_0_0", "direct", "aggregator"]
    assert result["task_guidance"]["evidence_support_basis"] == "single_report"
    assert all(len(n["citations"]) == 1 for n in run.tasks.nodes)


@pytest.mark.parametrize(
    "patch",
    [
        dict(max_tokens=2049),
        dict(max_tokens=True),
        dict(temperature=float("nan")),
        dict(temperature=-1),
    ],
)
def test_frozen_generation_settings_cannot_exceed_reserved_output_or_be_ignored(tmp_path, patch):
    cfg = dict(single_config(), **patch)
    store = BudgetStore(tmp_path / "ledger.json", cfg, 0, 30, ["case/A"])
    with pytest.raises(ValueError):
        Transport(tmp_path, store, cfg, responder=lambda *args: None)
    assert not read(store.path)["held"] and not (tmp_path / "requests").exists()


def test_bounded_generation_settings_match_the_actual_wire_request(tmp_path):
    cfg = dict(single_config(), max_tokens=1024, temperature=0.5)
    store = BudgetStore(tmp_path / "ledger.json", cfg, 0, 30, ["case/A"])
    transport = Transport(tmp_path, store, cfg, responder=lambda *args: None)
    request = transport.request("case", "A", "planner_1", [])
    assert request["max_tokens"] == 1024 and request["temperature"] == 0.5
