"""Actual failed-output replays plus explicit synthetic adversarial controls.

Wire conversion removes controller-owned IDs/locate groups; it never invents
observations, and is not measured new-model accuracy.
"""

from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import json
import pytest

from active_vlm.common import read, canonical, save
from active_vlm.baselines.geometry import rel_coords_to_abs
from active_vlm.decision.partition import PartitionDAG
from active_vlm.decision.partition_runner import PartitionGraphRun
from active_vlm.decision.partition_prompts import bind_observations, observe_tool
from active_vlm.experiment import cli
from .replay import no_dispatch
from .test_partition_graph import node, report, PartitionResponder
from .test_execution import make_case, response

ROOT = Path(__file__).resolve().parents[1] / "results/partition_graph30_round02_20261026"


@pytest.fixture(autouse=True)
def offline():
    with no_dispatch():
        yield


class Geometry:
    def __init__(self, size):
        self.size = size

    def candidate(self, parent, box):
        px = rel_coords_to_abs(box, self.size)
        return dict(id=str(px), original_xyxy=px, normalized=box)


def wire(rows, nodes):
    """Explicit old/new structural replay, no semantic content rewritten."""
    expected = {n["index"]: n for n in nodes}
    return {
        f"g{r['node_index']}": {k: v for k, v in r.items() if k not in {"node_index", "group"}}
        for r in rows
        if r["node_index"] in expected
    }


def captured(case):
    if not ROOT.exists():
        pytest.skip("Optional historical responses are not distributed")
    result = read(ROOT / "trajectories" / case / "B/result.json")
    c = next(c for c in read(ROOT / "inference_manifest.json") if c["key"] == case)
    d = PartitionDAG(c["allowed_letters"])
    for e in result["events"]:
        if e.get("stage", "").startswith("planner") and "repair" not in e["stage"]:
            p = json.loads(e["message"]["tool_calls"][0]["function"]["arguments"])
            try:
                d.accept(p["steps"], lambda _: None)
            except ValueError:
                continue
            break
    assert d.nodes
    run = object.__new__(PartitionGraphRun)
    run.dag = d
    run.images = Geometry(c["original_size"])
    return result, run


@pytest.mark.parametrize(
    "case,expected_next",
    [
        ("treebench_e3a4fdda0003", 1),
        ("treebench_b6ecf910aec2", 2),
        ("treebench_0e6791fa2da6", 1),
        ("treebench_b0686072e21b", 1),
    ],
)
def test_real_unseen_planned_region_remains_available(case, expected_next):
    result, run = captured(case)
    rd = result["rounds"][0]
    best = run.dag.select()[0]
    candidate = run.images.candidate("original", rd["candidates"][0]["normalized"])
    indices, attempted = run.bound_nodes(best["path"], candidate)
    assert 0 in indices and expected_next not in indices
    event = next(e for e in result["events"] if e.get("stage", "").startswith("observe"))
    raw = json.loads(event["message"]["tool_calls"][0]["function"]["arguments"])["observations"]
    nodes = [run.dag.nodes[i] for i in indices]
    rows = bind_observations(wire(raw, nodes), nodes)
    added = run.dag.observe_batch(rows, indices, candidate["id"], "captured", attempted)
    assert all(o["realized_gain"] == 0 for o in added)
    assert run.dag.select()[0]["focus"] == expected_next
    assert not run.dag.revisions


def test_real_truck_repair_uses_untested_blue_truck_view_before_another_paid_repair():
    result, run = captured("vstar_33c8cbbbc2d5")
    d = run.dag
    first = result["rounds"][0]
    rows = [o["reported"] for o in first["observations"]]
    d.observe_batch(rows, list(range(3)), "whole", "captured0")
    patches = [
        {k: v for k, v in p.items() if k != "previous_box"}
        for p in result["partition_graph"]["region_revisions"]
    ]
    d.revise(patches, lambda _: None)
    second = result["rounds"][1]
    candidate = run.images.candidate("original", second["candidates"][0]["normalized"])
    indices, attempted = run.bound_nodes(d.select()[0]["path"], candidate)
    assert indices == [0]
    rows = [o["reported"] for o in second["observations"] if o["node_index"] in indices]
    d.observe_batch(rows, indices, candidate["id"], "captured1", attempted)
    assert d.select()[0]["focus"] == 1
    assert d.nodes[1]["box"] == [620, 450, 650, 480]
    assert d.possible == set(["A", "B"])


def test_missing_bound_observation_cannot_erase_actual_prerequisite_or_invent_gain():
    d = PartitionDAG(["A", "B"])
    d.accept([node("locate"), node(requires=[0])], lambda _: None)
    rows = bind_observations(wire([report(0)], d.nodes), d.nodes)
    out = d.observe_batch(rows, [0, 1], "crop", "r", attempted=[0])
    assert out[0]["observed"] and not out[1]["observed"]
    assert d.nodes[1]["status"] == "pending" and d.select()[0]["focus"] == 1
    assert d.possible == {"A", "B"}


def test_locate_wire_has_no_answer_field_and_keys_cannot_rebind_targets():
    d = PartitionDAG(["A", "B"])
    d.accept([node("locate"), node(requires=[0])], lambda _: None)
    schema = observe_tool(d.nodes, ["A", "B"])["function"]["parameters"]["properties"][
        "observations"
    ]
    assert "group" not in schema["properties"]["g0"]["properties"]
    assert "node_index" not in schema["properties"]["g1"]["properties"]
    assert "group" not in schema["properties"]["g1"]["properties"]
    assert observe_tool(d.nodes, ["A", "B"])["function"]["parameters"]["properties"]["answer"][
        "anyOf"
    ][0]["enum"] == ["A", "B"]
    p = wire([report(0), report(1, 0)], d.nodes)
    for bad in [dict(p, g9=p["g0"]), dict(p, g0=dict(p["g0"], group=0))]:
        with pytest.raises(ValueError):
            bind_observations(bad, d.nodes)
    assert not d.observations


def test_failure_in_larger_view_is_not_failure_of_untried_zoom():
    d = PartitionDAG(["A", "B"])
    d.accept([node("locate"), node(requires=[0], box=[200, 200, 300, 300])], lambda _: None)
    d.observe_batch(
        [report(0), report(1, target_visible=False, resolved=False)],
        [0, 1],
        "large",
        "r",
        attempted=[0],
    )
    assert d.nodes[1]["status"] == "pending"
    assert d.select()[0]["focus"] == 1
    d.observe_batch(
        [report(1, target_visible=False, resolved=False)], [1], "zoom", "r2", attempted=[1]
    )
    assert d.select()[0] is None
    with pytest.raises(ValueError, match="REPEATED"):
        d.observe_batch([report(1, 0)], [1], "zoom", "r3")


def test_end_to_end_two_planned_crops_no_geometry_repair_and_no_extra_probe(tmp_path):
    case = make_case(tmp_path)
    save(tmp_path / "manifest.json", [case])

    class TwoViews(PartitionResponder):
        def __call__(self, request, meta):
            stage = meta["stage"]
            if stage.startswith(("planner", "observe")):
                assert "repair" not in stage
                name = request["tools"][0]["function"]["name"]
                if name == "design_partition":
                    data = dict(
                        reading=None,
                        steps=[
                            {k: v for k, v in n.items() if k != "kind"}
                            for n in [
                                node("locate", box=[0, 0, 400, 400]),
                                node(requires=[0], box=[500, 500, 900, 900]),
                            ]
                        ],
                    )
                else:
                    packet = json.loads(request["messages"][1]["content"][0]["text"])
                    ns = packet["bound_nodes"]
                    data = dict(
                        answer="A" if any(n["kind"] == "measure" for n in ns) else None,
                        observations=wire(
                            [report(n["index"], 0 if n["kind"] == "measure" else None) for n in ns],
                            ns,
                        ),
                    )
                self.inputs.append(deepcopy(meta))
                return response(
                    meta,
                    dict(
                        role="assistant",
                        content="",
                        tool_calls=[
                            dict(
                                type="function",
                                id="x",
                                function=dict(name=name, arguments=canonical(data)),
                            )
                        ],
                    ),
                )
            return super().__call__(request, meta)

    root = cli.main(
        [
            "prepare",
            "--manifest",
            str(tmp_path / "manifest.json"),
            "--full",
            "--methods",
            "graph",
            "--graph-controller",
            "partition",
            "--account-root",
            str(tmp_path / "account"),
            "--output",
            "results/two",
        ]
    )
    responder = TwoViews()
    assert cli.run(root, approve=True, responder=responder) == "COMPLETE"
    result = read(root / "trajectories" / case["key"] / "B/result.json")
    assert result["prediction"] == "A" and result["calls"] == 7 and result["probe_calls"] == 2
    assert result["planner_turns"] == 1 and not result["errors"]
    assert not result["partition_graph"]["region_revisions"]
