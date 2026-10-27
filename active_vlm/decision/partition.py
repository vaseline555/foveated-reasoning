"""Finite version-space planning without model-generated answer probabilities.

Location is a prerequisite, never answer evidence. A categorical observation
restricts a set of possible answers; conflicting sets restore uncertainty.
The uniform version-space value is an optimistic planning surrogate, not a
calibrated posterior or a guarantee about the independent native final reader.
"""

import math
from copy import deepcopy

from ..imaging import validate_seed


def corners(box):
    """The wire supplies opposite corners, whose order has no meaning."""
    if (
        not isinstance(box, list)
        or len(box) != 4
        or any(
            type(v) not in {int, float} or not math.isfinite(v) or not 0 <= v <= 1000 for v in box
        )
    ):
        raise ValueError("BOX_REQUIRES_FOUR_FINITE_ORIGINAL_COORDINATES")
    a, b, c, d = box
    return list(validate_seed([min(a, c), min(b, d), max(a, c), max(b, d)]))


class PartitionDAG:
    def __init__(self, letters, cost_weight=1.0, pair_cost=0.003):
        if (
            not isinstance(letters, list)
            or not 2 <= len(letters) <= 11
            or any(not isinstance(x, str) for x in letters)
            or len(set(letters)) != len(letters)
        ):
            raise ValueError("INVALID_CHOICES")
        if any(
            type(v) not in {int, float} or not math.isfinite(v) or v <= 0
            for v in [cost_weight, pair_cost]
        ):
            raise ValueError("INVALID_ECONOMICS")
        self.letters = list(letters)
        self.possible = set(letters)
        self.cost_weight, self.pair_cost = cost_weight, pair_cost
        self.nodes, self.observations, self.revisions = [], [], []

    def accept(self, steps, validate_pixels):
        if self.nodes:
            raise ValueError("INITIAL_PLAN_ALREADY_ACCEPTED")
        self.nodes = self.validated_nodes(steps, validate_pixels)

    def validated_nodes(self, steps, validate_pixels, require_measurement=True):
        """Pure structural validation, also used before attaching the fixed query."""
        if not isinstance(steps, list) or not 1 <= len(steps) <= 12:
            raise ValueError("INITIAL_PLAN_REQUIRES_1_TO_12_NODES")
        nodes = []
        for i, raw in enumerate(steps):
            if not isinstance(raw, dict) or set(raw) != {
                "task",
                "target",
                "kind",
                "requires",
                "box",
            }:
                raise ValueError("PLAN_NODE_FIELDS")
            if any(not isinstance(raw[k], str) or not raw[k].strip() for k in ["task", "target"]):
                raise ValueError("TASK_AND_REFERENT_REQUIRED")
            if raw["kind"] not in ("locate", "measure"):
                raise ValueError("INVALID_NODE_KIND")
            deps = raw["requires"]
            if not isinstance(deps, list) or any(
                type(d) is not int or not 0 <= d < i for d in deps
            ):
                raise ValueError("DEPENDENCIES_REQUIRE_EARLIER_INTEGERS")
            if len(set(deps)) != len(deps):
                raise ValueError("DUPLICATE_DEPENDENCY")
            # The answer alphabet belongs to the dataset, not the planner.
            # Locate nodes carry no answer evidence; only question-completing
            # measurements distinguish the fixed choices.
            groups = [] if raw["kind"] == "locate" else [[a] for a in self.letters]
            box = corners(raw["box"])
            validate_pixels(box)
            nodes.append(
                dict(
                    deepcopy(raw),
                    box=box,
                    groups=groups,
                    index=i,
                    id=f"g{i}",
                    status="pending",
                    citations=[],
                )
            )
        if sum(len(n["task"]) + len(n["target"]) for n in nodes) > 12000:
            raise ValueError("PLAN_TEXT_BUDGET")
        if require_measurement and not any(n["kind"] == "measure" for n in nodes):
            raise ValueError("ANSWER_MEASUREMENT_REQUIRED")
        return nodes

    def ancestors(self, index):
        seen = set()

        def visit(i):
            for p in self.nodes[i]["requires"]:
                if p not in seen:
                    visit(p)
            seen.add(i)

        visit(index)
        return sorted(seen)

    def select(self):
        choices = []
        for leaf in self.nodes:
            if leaf["kind"] != "measure" or leaf["status"] != "pending":
                continue
            path = [self.nodes[i] for i in self.ancestors(leaf["index"])]
            if any(n["status"] == "unobserved" for n in path):
                continue
            pending = [n for n in path if n["status"] == "pending"]
            nonempty = sum(bool(self.possible.intersection(g)) for g in leaf["groups"])
            # Perfect categorical measurement under a uniform version space:
            # E[max p(y|o)] - max p(y) = (#nonempty groups - 1) / |S|.
            gain = (nonempty - 1) / len(self.possible)
            cost = len({tuple(n["box"]) for n in pending}) * self.pair_cost
            choices.append(
                dict(
                    focus=pending[0]["index"],
                    leaf=leaf["index"],
                    path=[n["index"] for n in pending],
                    expected_gain=gain,
                    expected_cost_usd=cost,
                    utility=gain - self.cost_weight * cost,
                )
            )
        best = max(choices, key=lambda c: (c["utility"], -c["expected_cost_usd"]), default=None)
        return (best if best and best["utility"] > 0 else None), choices

    def revise(self, patches, validate_pixels):
        """Atomic region repair: semantic targets, tasks and prior facts are immutable."""
        if not isinstance(patches, list) or not patches:
            raise ValueError("NONEMPTY_REGION_PATCH_REQUIRED")
        prepared, seen = [], set()
        for p in patches:
            if not isinstance(p, dict) or set(p) != {"node_index", "box", "reason"}:
                raise ValueError("REGION_PATCH_FIELDS")
            i = p["node_index"]
            if type(i) is not int or not 0 <= i < len(self.nodes) or i in seen:
                raise ValueError("REGION_PATCH_INDEX")
            if self.nodes[i]["status"] == "observed":
                raise ValueError("CANNOT_REWRITE_OBSERVED_REFERENT")
            if not isinstance(p["reason"], str) or not p["reason"].strip():
                raise ValueError("REGION_PATCH_REASON_REQUIRED")
            box = corners(p["box"])
            validate_pixels(box)
            if box == self.nodes[i]["box"]:
                raise ValueError("REGION_PATCH_MUST_CHANGE_REGION")
            prepared.append(dict(p, box=box))
            seen.add(i)
        for p in prepared:
            n = self.nodes[p["node_index"]]
            self.revisions.append(dict(p, previous_box=n["box"]))
            n["box"], n["status"] = p["box"], "pending"

    def observe_batch(self, rows, allowed, crop_id, receipt_id, attempted=None):
        """Validate the whole batch first, then apply in DAG order, once per crop.

        Same-crop descendant observations may consume a prerequisite from this
        batch. Missing identity/dependency is recorded, never promoted to gain.
        """
        if not isinstance(rows, list) or not rows:
            raise ValueError("OBSERVATION_BATCH_REQUIRED")
        attempted = set(allowed if attempted is None else attempted)
        if not attempted.issubset(allowed):
            raise ValueError("ATTEMPT_OUTSIDE_BOUND_NODES")
        prepared, seen = [], set()
        for r in rows:
            if not isinstance(r, dict) or set(r) != {
                "node_index",
                "target_visible",
                "identity_evidence",
                "observation",
                "source",
                "resolved",
                "group",
                "reason",
            }:
                raise ValueError("OBSERVATION_FIELDS")
            i = r["node_index"]
            if type(i) is not int or i not in allowed or i in seen:
                raise ValueError("OBSERVATION_BOUND_INDEX")
            if any(type(r[k]) is not bool for k in ["target_visible", "resolved"]):
                raise ValueError("OBSERVATION_BOOLEAN_TYPES")
            if any(
                not isinstance(r[k], str) for k in ["identity_evidence", "observation", "reason"]
            ):
                raise ValueError("OBSERVATION_TEXT_TYPES")
            if r["source"] not in ("crop", "original", "inference", "question"):
                raise ValueError("OBSERVATION_SOURCE")
            if sum(len(r[k]) for k in ["identity_evidence", "observation", "reason"]) > 4000:
                raise ValueError("OBSERVATION_TEXT_BUDGET")
            node = self.nodes[i]
            group = r["group"]
            if group is not None and (
                type(group) is not int or not 0 <= group < len(node["groups"])
            ):
                raise ValueError("INVALID_OPTION_GROUP")
            if node["status"] != "pending" or any(
                o["node_index"] == i and o["crop_id"] == crop_id for o in self.observations
            ):
                raise ValueError("REPEATED_MEASUREMENT")
            seen.add(i)
            prepared.append(r)
        added = []
        for r in sorted(prepared, key=lambda x: x["node_index"]):
            i, before = r["node_index"], set(self.possible)
            node = self.nodes[i]
            dependencies = all(self.nodes[d]["status"] == "observed" for d in node["requires"])
            admitted = (
                r["source"] == "crop"
                and r["resolved"]
                and r["target_visible"]
                and bool(r["identity_evidence"].strip())
                and bool(r["observation"].strip())
                and dependencies
            )
            supported = (
                node["groups"][r["group"]]
                if admitted and node["kind"] == "measure" and r["group"] is not None
                else None
            )
            row = dict(
                node_index=i,
                node_id=node["id"],
                task=node["task"],
                target=node["target"],
                crop_id=crop_id,
                receipt_id=receipt_id,
                reported=deepcopy(r),
                observed=admitted,
                compatible_answers=supported,
                dependency_ready=dependencies,
                possible_before=sorted(before),
                semantic_truth_verified=False,
            )
            # Failure in a prerequisite's crop does not exhaust this node's
            # different planned view. Only an actual attempt of its own view
            # may block the node and require a new region proposal.
            if admitted:
                node["status"] = (
                    "observed_unmapped"
                    if node["kind"] == "measure" and supported is None
                    else "observed"
                )
            else:
                # A conditional task was not tested if its prerequisite failed.
                # A later repaired prerequisite may make its new view eligible;
                # no prior failed report is thereby admitted as evidence.
                node["status"] = "unobserved" if i in attempted and dependencies else "pending"
            row["answer_mapping_available"] = supported is not None
            row["planned_view_attempted"] = i in attempted
            self.observations.append(row)
            constraints = [
                set(o["compatible_answers"]) for o in self.observations if o["compatible_answers"]
            ]
            intersection = set.intersection(set(self.letters), *constraints)
            self.possible = intersection or set(self.letters)
            row.update(
                possible_after=sorted(self.possible),
                conflict=not bool(intersection),
                realized_gain=1 / len(self.possible) - 1 / len(before),
            )
            node["citations"].append(row)
            added.append(row)
        return added
