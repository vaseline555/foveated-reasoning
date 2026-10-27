"""Public recheck configuration is validated and frozen before paid execution."""

import pytest

from active_vlm.common import read, save
from active_vlm.experiment import accounting, cli

from .replay import no_dispatch
from .test_execution import make_case


def test_retest_probability_reaches_frozen_config_and_quote(tmp_path):
    manifest = tmp_path / "manifest.json"
    save(manifest, [make_case(tmp_path)])
    with no_dispatch():
        root = cli.main([
            "prepare", "--manifest", str(manifest), "--benchmarks", "vstar", "--full",
            "--methods", "graph", "--graph-controller", "partition",
            "--retest-probability", "0.5", "--account-root", str(tmp_path / "account"),
            "--output", "results/retest",
        ])
        assert read(root / "config.json")["retest_probability"] == 0.5
        quote = read(root / "budget_quote.json")
        assert quote["maximum_additional_retest_requests"] == 1
        assert quote["scenarios"][0]["expected_additional_retest_requests"] == 0.5
        accounting.verify(root)
        assert not (root / "approval.json").exists()
        assert cli.run(root, dry_run=True) == "COMPLETE"


@pytest.mark.parametrize("value", ["-0.1", "1.1", "nan", "inf"])
def test_bad_probabilities_rejected_before_preparation(value):
    with no_dispatch(), pytest.raises(SystemExit):
        cli.main(["prepare", "--retest-probability", value])


def test_retest_rejects_controller_that_would_ignore_it():
    with no_dispatch(), pytest.raises(SystemExit):
        cli.main(["prepare", "--retest-probability", "0.5"])
