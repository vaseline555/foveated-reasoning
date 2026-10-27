"""The documented command runs all four public methods without network access."""

from active_vlm.common import read, save
from active_vlm.experiment import cli

from .replay import no_dispatch
from .test_execution import SelectorResponder, make_case


def test_documented_default_four_method_run_and_budget_quote(tmp_path):
    case = make_case(tmp_path)
    manifest = tmp_path / "manifest.json"
    save(manifest, [case])
    with no_dispatch():
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
                "results/smoke",
            ]
        )
        plan = read(root / "plan.json")
        assert set(plan["arms"]) == {"A", "M", "L", "B"}
        assert len(plan["reuse"]["pending"]) == 4
        assert not (root / "approval.json").exists()
        responder = SelectorResponder()
        assert cli.run(root, approve=True, responder=responder) == "COMPLETE"
    result = read(root / "summary.json")
    assert result["simulation"] and result["paired_questions"] == 1
    assert set(result["totals"]) == {"A", "M", "L", "B"}
    assert result["task_guidance"]["by_arm"]["B"]["unresolved_nodes"] > 0
    assert result["new_failed_physical_attempts"] == 0
    assert result["crop_probe_protocol"] == "graph-single"
    assert all(result["crop_probe_by_arm"][a]["K"] == 3 for a in ["A", "M", "L"])
    assert result["crop_probe_by_arm"]["B"]["K"] == 1
    assert any(r["stage"].startswith("expansion_") and r["arm"] == "L" for r in responder.inputs)
    for arm in ["A", "M", "L", "B"]:
        path = root / "trajectories" / case["key"] / arm / "result.json"
        trajectory = read(path)
        if arm == "B":
            assert 1 <= trajectory["probe_calls"] <= 9
            assert trajectory["task_guidance"]["controller"] == "python"
            assert trajectory["task_guidance"]["python_search"]["subsequent_planner_requests"] == 0
        else:
            assert trajectory["probe_calls"] == {"A": 9, "M": 3, "L": 27}[arm]
    assert result["graph_diagnostics"]["support_basis"] == "single_report"
    assert result["final_answer_mode"] == "native"
    assert result["answer_parser"] == "explicit_tagged_option_label"
    assert read(root / "config.json")["native_reader_recovery"] == dict(
        max_length_retries=1, max_tokens=512, strategy="finalize", analysis_max_chars=4096
    )
