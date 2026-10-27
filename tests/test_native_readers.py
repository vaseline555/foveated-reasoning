"""Actual native reader wires, truncations, budgets and explicit repair controls."""

from copy import deepcopy
from pathlib import Path

import pytest

from active_vlm.baselines.runner import FoveaRun
from active_vlm.budget import BudgetStore
from active_vlm.common import read, save
from active_vlm.experiment import accounting, cli
from active_vlm.experiment.report import metrics
from active_vlm.readers import reader_content, recovery_policy
from active_vlm.transport import Transport, semantic_request

from .replay import no_dispatch
from .test_execution import SelectorResponder, config, make_case, response

ROOT = Path(__file__).resolve().parents[1] / "results/python_graph_pilot30_20261023"


@pytest.fixture(autouse=True)
def offline():
    with no_dispatch():
        yield


def native_config():
    return dict(
        config(),
        final_answer_mode="native",
        direct_answer_mode="native",
        native_reader_recovery=dict(max_length_retries=1, max_tokens=4096),
        request_upper_usd=0.068608,
    )


def make_reader(case, output, cfg, responder, cap=30):
    store = BudgetStore(output / "ledger.json", cfg, 0, cap, [case["key"] + "/B"])
    return FoveaRun(case, "B", Transport(output, store, cfg, responder=responder), output / "B")


def captured(key):
    if not (ROOT / "inference_manifest.json").exists():
        pytest.skip("Optional private captured logs are not distributed")
    case = next(c for c in read(ROOT / "inference_manifest.json") if c["key"] == key)
    old = read(ROOT / "trajectories" / key / "B/result.json")
    records = {
        r["stage"]: read(ROOT / "requests" / (r["request_hash"] + ".json"))
        for r in old["receipts"]
        if r["stage"] in {"direct", "aggregator"}
    }
    return case, old, records


@pytest.mark.parametrize(
    "key",
    ["treebench_2209c12466a4", "vstar_be42ae699d50", "vstar_f527aaa35b68", "mme_4d5c96ed1ec9"],
)
def test_actual_native_direct_and_final_wires_are_exactly_restored(tmp_path, key):
    case, old, records = captured(key)
    cfg = dict(
        read(ROOT / "config.json"),
        **{
            "native_reader_recovery": dict(max_length_retries=1, max_tokens=4096),
            "request_upper_usd": 0.068608,
            "min_dispatch_interval_seconds": 0,
        },
    )
    seen = []

    def respond(req, meta):
        seen.append(meta["stage"])
        source = records[meta["stage"]]
        assert semantic_request(req) == semantic_request(source["request"])
        assert req["max_tokens"] == 2048 and "response_format" not in req
        return deepcopy(source["response"])

    run = make_reader(case, tmp_path, cfg, respond)
    run.final_history, run.errors, run.stop = (
        deepcopy(old["final_history"]),
        deepcopy(old["errors"]),
        old["stop_reason"],
    )
    assert run.finish() == old["prediction"]  # Captured-response replay, not a new answer.
    assert seen == ["direct", "aggregator"]
    assert all(
        d["complete_valid_answer"] and not d["recovery_attempted"] for d in run.reader_diagnostics
    )
    assert not read(run.transport.budget.path)["held"]


@pytest.mark.parametrize("repeat_bad_capture", [False, True])
def test_actual_double_truncation_gets_one_same_input_retry_per_reader(
    tmp_path, repeat_bad_capture
):
    case, old, records = captured("treebench_bbabf4b8e7b2")
    cfg = dict(
        read(ROOT / "config.json"),
        native_reader_recovery=dict(max_length_retries=1, max_tokens=4096),
        request_upper_usd=0.068608,
        min_dispatch_interval_seconds=0,
    )
    assert all(r["finish_reason"] == "length" for r in records.values())
    seen, requests = [], {}

    def respond(req, meta):
        stage = meta["stage"]
        seen.append(stage)
        requests[stage] = deepcopy(req)
        base = stage.removesuffix("_length_retry")
        assert "response_format" not in req
        if stage == base:
            assert req["max_tokens"] == 2048
            return deepcopy(records[base]["response"])
        assert req["max_tokens"] == 4096
        original = deepcopy(requests[base])
        original["max_tokens"] = 4096
        assert req == original  # No prompt, image, seed or temperature change.
        if repeat_bad_capture:
            return deepcopy(records[base]["response"])
        # HUMAN-WRITTEN output-format control, NOT a new measured model reply.
        return response(
            meta, dict(role="assistant", content="Reasoning control. <answer>A</answer>")
        )

    run = make_reader(case, tmp_path, cfg, respond)
    run.final_history, run.errors, run.stop = (
        deepcopy(old["final_history"]),
        deepcopy(old["errors"]),
        old["stop_reason"],
    )
    prediction = run.finish()
    assert prediction == (None if repeat_bad_capture else "A")
    assert seen == ["direct", "direct_length_retry", "aggregator", "aggregator_length_retry"]
    assert all(d["recovery_attempted"] for d in run.reader_diagnostics)
    assert all(d["complete_valid_answer"] != repeat_bad_capture for d in run.reader_diagnostics)
    if repeat_bad_capture:
        # The 2048-token incomplete Direct text must not contaminate the judge.
        assert "Direct reader returned no complete valid allowed choice" in str(
            requests["aggregator"]["messages"]
        )
        assert records["direct"]["message"]["content"] not in str(
            requests["aggregator"]["messages"]
        )
    assert old["prediction"] is None and not read(run.transport.budget.path)["held"]


def test_optional_recovery_cannot_consume_the_protected_final_reader(tmp_path):
    cfg, seen = native_config(), []
    case = make_case(tmp_path)

    def respond(req, meta):
        seen.append(meta["stage"])
        body = response(meta)
        if meta["stage"] == "direct":
            body["choices"][0]["finish_reason"] = "length"
        return body

    run = make_reader(case, tmp_path, cfg, respond, cap=2 * cfg["request_upper_usd"])
    assert run.finish() == "A"
    assert seen == ["direct", "aggregator"]
    assert run.reader_diagnostics[0]["recovery_skipped"] == "CUMULATIVE_PHYSICAL_CAP"
    assert not run.reader_diagnostics[0]["complete_valid_answer"]
    assert not read(run.transport.budget.path)["held"]


def test_stop_without_answer_is_not_silently_retried(tmp_path):
    seen = []
    case = make_case(tmp_path)

    def respond(req, meta):
        seen.append(meta["stage"])
        return response(meta, dict(role="assistant", content="Uncertain, no tagged answer."))

    run = make_reader(case, tmp_path, native_config(), respond)
    assert run.finish() is None and seen == ["direct", "aggregator"]
    assert not any(d["recovery_attempted"] for d in run.reader_diagnostics)


def test_reporting_cannot_revive_a_length_truncated_answer_prefix(tmp_path):
    case = make_case(tmp_path)
    manifest = tmp_path / "manifest.json"
    save(manifest, [case])
    labels = tmp_path / "labels.json"
    save(labels, {case["key"]: "A"})
    root = cli.main(
        [
            "prepare",
            "--manifest",
            str(manifest),
            "--benchmarks",
            "vstar",
            "--full",
            "--methods",
            "greedy",
            "--labels",
            str(labels),
            "--account-root",
            str(tmp_path / "account"),
            "--output",
            "results/length",
        ]
    )
    selector = SelectorResponder()

    def respond(req, meta):
        body = selector(req, meta)
        if meta["stage"] in {"aggregator", "aggregator_length_retry"}:
            body["choices"][0].update(
                finish_reason="length",
                message=dict(
                    role="assistant", content="<answer>A</answer> However, reconsider ..."
                ),
            )
        return body

    assert cli.run(root, approve=True, responder=respond) == "COMPLETE_WITH_FAILURES"
    result = read(root / "trajectories" / case["key"] / "A/result.json")
    assert result["prediction"] is None and reader_content(result, "aggregator") is None
    summary = read(root / "summary.json")
    assert summary["totals"]["A"]["failed_paths"] == 1
    assert summary["normalized_saved_answers"] == {}
    assert (
        metrics([case], "A", {case["key"] + "/A": result}, {case["key"]: "A"}, True)["correct"] == 0
    )
    assert summary["output_diagnostics"]["native_reader_recovery_requests"] == 1
    assert summary["output_diagnostics"]["incomplete_native_reader_outputs"] == 1


@pytest.mark.parametrize(
    "bad",
    [
        True,
        {},
        {"max_length_retries": 2, "max_tokens": 4096},
        {"max_length_retries": True, "max_tokens": 4096},
        {"max_length_retries": 1, "max_tokens": 2048},
        {"max_length_retries": 1, "max_tokens": 8193},
    ],
)
def test_invalid_recovery_policy_fails_before_dispatch(bad):
    with pytest.raises(ValueError, match="NATIVE_READER_RECOVERY"):
        recovery_policy(dict(config(), native_reader_recovery=bad))


def test_larger_native_cap_requires_a_larger_money_reservation(tmp_path):
    cfg = dict(native_config(), request_upper_usd=0.067072)
    case = make_case(tmp_path)
    with pytest.raises(ValueError, match="NOT_COVERED_BY_RESERVATION"):
        make_reader(case, tmp_path / "bad", cfg, lambda req, meta: response(meta))


def test_reuse_checks_native_recovery_policy_before_loading_saved_results(tmp_path):
    case, _, _ = captured("vstar_be42ae699d50")
    cfg = dict(
        read(ROOT / "config.json"),
        native_reader_recovery=dict(max_length_retries=1, max_tokens=4096),
    )
    mappings = read(
        ROOT.parents[2] / "experiment_archive/before_probe_efficiency_20261023/relocations.json"
    )
    with pytest.raises(ValueError, match="native_reader_recovery"):
        accounting.reuse([case], ["A"], [ROOT], tmp_path / "reuse", cfg, mappings)
