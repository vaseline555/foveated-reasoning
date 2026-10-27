"""Native FOVEA with only graph/flat restricted to one crop and one probe."""

import pytest

from active_vlm.common import read, save
from active_vlm.experiment import cli
from active_vlm.protocol import crop_protocol

from .replay import no_dispatch
from .test_execution import config, make_case


def test_method_specific_k_and_selector_flags_cannot_leak_between_methods():
    cfg = dict(config(), crop_probe_mode="graph-single", task_policy=dict(quorum=1))
    for arm in ["A", "M", "L"]:
        assert crop_protocol(cfg, arm)["K"] == 3
        assert crop_protocol(cfg, arm)["mode"] == "native"
        assert crop_protocol(cfg, arm)["multi_candidate_selectors_enabled"]
    for arm in ["B", "F", "G"]:
        assert crop_protocol(cfg, arm)["K"] == 1
        assert crop_protocol(cfg, arm)["mode"] == "single"
        assert not crop_protocol(cfg, arm)["multi_candidate_selectors_enabled"]
    with pytest.raises(ValueError, match="METHOD_REQUIRED"):
        crop_protocol(cfg)


def test_graph_first_queue_preserves_native_selectors_under_tight_initial_reserves(tmp_path):
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
                "results/tight",
                "--graph-first",
                "--budget",
                "0.5",
            ]
        )
        quote = read(root / "budget_quote.json")
        assert quote["ready_for_run"] and not quote["blockers"]
        assert quote["crop_probe_by_arm"]["B"]["K"] == 1
        assert quote["native_mcmc_lookahead_selectors_enabled"]
        assert read(root / "plan.json")["reuse"]["pending"][0].endswith("/B")
        assert (
            quote["initial_exploration_headroom_usd"]
            > 2 * read(root / "config.json")["request_upper_usd"]
        )
        assert (
            quote["largest_atomic_probe_reservation_by_arm_usd"]["L"]
            > quote["initial_exploration_headroom_usd"]
        )
        assert not (root / "approval.json").exists()
