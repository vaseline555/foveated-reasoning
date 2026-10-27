"""Offline structural controls; synthetic replies are NOT measured VLM accuracy."""

from copy import deepcopy

import pytest

from active_vlm.common import canonical, read, save
from active_vlm.decision.partition import PartitionDAG, corners
from active_vlm.experiment import cli

from .replay import no_dispatch
from .test_execution import SelectorResponder, make_case, response


@pytest.fixture(autouse=True)
def offline():
    with no_dispatch():
        yield


def node(kind="measure", requires=None, box=None):
    return dict(
        task="Locate the specified object" if kind == "locate" else "Read its visible color",
        target="the specified object with the question's identifying properties",
        kind=kind,
        requires=requires or [],
        box=box or [100, 100, 900, 900],
    )


def report(i, group=None, **changes):
    return dict(
        dict(
            node_index=i,
            target_visible=True,
            identity_evidence="The specified marker is visible",
            observation="The surface is blue",
            source="crop",
            resolved=True,
            group=group,
            reason="",
        ),
        **changes,
    )


def dag():
    d = PartitionDAG(["A", "B"])
    d.accept([node("locate"), node(requires=[0])], lambda _: None)
    return d


def test_same_crop_batch_satisfies_dependencies_without_location_answer_gain():
    d = dag()
    best, _ = d.select()
    assert best["focus"] == 0 and best["expected_cost_usd"] == 0.003
    rows = d.observe_batch([report(1, 0), report(0)], [0, 1], "crop", "r")
    assert rows[0]["realized_gain"] == 0 and rows[1]["realized_gain"] == 0.5
    assert d.possible == {"A"} and d.select()[0] is None


def test_target_absence_and_wrong_source_cannot_become_negative_answer_evidence():
    for changed in [
        dict(target_visible=False),
        dict(source="original"),
        dict(identity_evidence=""),
    ]:
        d = dag()
        rows = d.observe_batch([report(0, **changed), report(1, 1)], [0, 1], "c", "r")
        assert not any(r["observed"] for r in rows) and d.possible == {"A", "B"}


def test_python_alone_binds_choice_groups():
    d = dag()
    assert d.nodes[0]["groups"] == []
    assert d.nodes[1]["groups"] == [["A"], ["B"]]
    from active_vlm.decision.partition_prompts import plan_tool

    properties = plan_tool(False)["function"]["parameters"]["properties"]["steps"]["items"][
        "properties"
    ]
    assert "groups" not in properties


def test_conflicting_observation_restores_uncertainty_and_keeps_both_facts():
    d = PartitionDAG(["A", "B"])
    d.accept([node(), node()], lambda _: None)
    a = d.observe_batch([report(0, 0)], [0], "first", "r1")
    b = d.observe_batch([report(1, 1)], [1], "second", "r2")
    assert a[0]["realized_gain"] == 0.5 and b[0]["realized_gain"] == -0.5
    assert d.possible == {"A", "B"} and b[0]["conflict"]
    assert len(d.observations) == 2


def test_reversed_corners_are_canonical_not_semantically_rewritten():
    assert corners([380, 817, 770, 677]) == [380, 677, 770, 817]
    for invalid in [[1, 2, 1, 3], [0, 0, float("nan"), 5], [False, 0, 10, 10], [-1, 0, 5, 10]]:
        with pytest.raises(ValueError):
            corners(invalid)


@pytest.mark.parametrize("bad", [None, {}, [[]], [["A"], ["A"]], [["A"], ["C"]], [["A", "B"]]])
def test_untrusted_option_groups_cannot_override_python_bindings(bad):
    n = node()
    n["groups"] = bad
    d = PartitionDAG(["A", "B"])
    with pytest.raises(ValueError):
        d.accept([node("locate"), n], lambda _: None)
    assert not d.nodes


def test_atomic_region_revision_cannot_overwrite_observed_identity():
    d = dag()
    d.observe_batch([report(0)], [0], "c", "r")
    before = deepcopy(d.nodes)
    with pytest.raises(ValueError, match="OBSERVED"):
        d.revise(
            [
                dict(node_index=1, box=[0, 0, 300, 300], reason="zoom"),
                dict(node_index=0, box=[0, 0, 400, 400], reason="replace"),
            ],
            lambda _: None,
        )
    assert before == d.nodes
    d.revise(
        [dict(node_index=1, box=[0, 0, 300, 300], reason="zoom same referent")], lambda _: None
    )
    assert d.nodes[0] == before[0] and d.nodes[1]["target"] == before[1]["target"]


def test_duplicate_facts_never_gain_twice_after_region_revision():
    d = dag()
    d.observe_batch([report(0, target_visible=False)], [0], "c", "r")
    d.revise([dict(node_index=0, box=[100, 100, 300, 300], reason="zoom")], lambda _: None)
    with pytest.raises(ValueError, match="REPEATED"):
        d.observe_batch([report(0)], [0], "c", "r2")


@pytest.mark.parametrize("bad", [True, [], {}, "0", -1, 2])
def test_malformed_group_is_controlled_and_batch_atomic(bad):
    d = dag()
    with pytest.raises(ValueError):
        d.observe_batch([report(0), report(1, bad)], [0, 1], "c", "r")
    assert not d.observations and all(n["status"] == "pending" for n in d.nodes)


class PartitionResponder(SelectorResponder):
    def __call__(self, request, meta):
        if meta["arm"] != "B" or meta["stage"] in {"direct", "aggregator"}:
            return super().__call__(request, meta)
        self.inputs.append(deepcopy(meta))
        if meta["stage"].startswith("probe_"):
            return response(meta, dict(role="assistant", content="No"))
        name = request["tools"][0]["function"]["name"]
        payload = (
            dict(reading=None, steps=[{k: v for k, v in node("locate").items() if k != "kind"}])
            if name == "design_partition"
            else dict(
                answer="A",
                observations={
                    "g0": {k: v for k, v in report(0).items() if k not in {"node_index", "group"}},
                    "g1": {
                        k: v for k, v in report(1, 0).items() if k not in {"node_index", "group"}
                    },
                },
            )
        )
        return response(
            meta,
            dict(
                role="assistant",
                content="",
                tool_calls=[
                    dict(
                        id="result",
                        type="function",
                        function=dict(name=name, arguments=canonical(payload)),
                    )
                ],
            ),
        )


def test_four_methods_native_wire_and_one_crop_one_observer_complete_stop(tmp_path):
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
            "partition",
            "--account-root",
            str(tmp_path / "account"),
            "--output",
            "results/partition",
        ]
    )
    responder = PartitionResponder()
    assert cli.run(root, approve=True, responder=responder) == "COMPLETE"
    r = read(root / "trajectories" / case["key"] / "B/result.json")
    assert r["prediction"] == "A" and not r["errors"]
    assert r["probe_calls"] == 1 and len(r["partition_graph"]["observations"]) == 2
    assert (
        r["planner_turns"] == 1
        and sum(x["stage"].startswith("observe_") for x in responder.inputs) == 1
    )
    assert r["stop_reason"] == "NO_POSITIVE_PARTITION_VALUE"
    assert sum(x["stage"] == "direct" for x in responder.inputs) == 1
    assert not read(root / "physical_ledger.json")["held"]


def test_unadmitted_absence_claim_is_preserved_in_audit_not_final_evidence(tmp_path):
    import json

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
            "--methods",
            "graph",
            "--graph-controller",
            "partition",
            "--account-root",
            str(tmp_path / "account"),
            "--output",
            "results/rejected",
        ]
    )

    class Absent(PartitionResponder):
        def __call__(self, request, meta):
            if "repair" in meta["stage"]:
                return response(
                    meta,
                    dict(
                        role="assistant",
                        content="",
                        tool_calls=[
                            dict(
                                function=dict(
                                    name="repair_regions", arguments=canonical(dict(patches=[]))
                                )
                            )
                        ],
                    ),
                )
            result = super().__call__(request, meta)
            if meta["stage"].startswith("observe_"):
                f = result["choices"][0]["message"]["tool_calls"][0]["function"]
                data = json.loads(f["arguments"])
                for row in data["observations"].values():
                    row.update(
                        target_visible=False,
                        resolved=False,
                        observation="UNSUPPORTED_CLAIM_NO_VEHICLE_ANYWHERE",
                    )
                f["arguments"] = canonical(data)
            return result

    assert cli.run(root, approve=True, responder=Absent()) == "COMPLETE"
    r = read(root / "trajectories" / case["key"] / "B/result.json")
    assert "UNSUPPORTED_CLAIM" in canonical(r["partition_graph"]["observations"])
    assert "UNSUPPORTED_CLAIM" not in canonical(r["final_history"])
    assert not any(o["observed"] for o in r["partition_graph"]["observations"])


def test_three_captured_single_group_plans_need_no_generated_option_mapping():
    import json
    from pathlib import Path

    source = Path(__file__).resolve().parents[1]
    path = (
        source / "results/partition_graph30_20261026/trajectories/vstar_d41be9ad11f4/B/result.json"
    )
    if not path.exists():
        pytest.skip("Optional historical responses are not distributed")
    captured = read(path)
    count = 0
    for event in captured["events"]:
        if not event.get("stage", "").startswith("planner"):
            continue
        raw = json.loads(event["message"]["tool_calls"][0]["function"]["arguments"])
        # Replay actual task, target, dependency and box values. The removed
        # grouping field is NOT a new model output or an accuracy measurement.
        steps = [{k: v for k, v in n.items() if k != "groups"} for n in raw["steps"]]
        d = PartitionDAG(["A", "B", "C", "D"])
        d.accept(steps, lambda _: None)
        assert all(
            n["groups"] == [["A"], ["B"], ["C"], ["D"]] for n in d.nodes if n["kind"] == "measure"
        )
        assert d.select()[0] is not None
        count += 1
    assert count == 3
