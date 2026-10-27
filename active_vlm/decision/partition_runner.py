"""Single-crop native FOVEA measurements with batched semantic interpretation."""

from copy import deepcopy

from ..baselines.geometry import rel_coords_to_abs
from ..baselines.runner import FoveaRun
from ..common import canonical, save
from . import partition_prompts as prompts
from .partition import PartitionDAG
from .question_context import question_context
from .question_query import (
    dependency_errors,
    query_region_patch,
    question_plan,
    repair_dependencies,
)
from .runner import parse_tool
from .retest import QueryRetest


class PartitionGraphRun(FoveaRun):
    def __init__(self, case, arm, transport, output):
        super().__init__(case, arm, transport, output)
        if arm != "B" or self.crop_protocol["mode"] != "single":
            raise ValueError("PARTITION_REQUIRES_GRAPH_SINGLE")
        self.dag = PartitionDAG(list(case["allowed_letters"]))
        self.valid_plan = False
        self.plan_calls = 0
        self.decisions, self.rejections, self.cost_events = [], [], []
        self.tested = {}
        self.question = case["question"]
        self.options = case.get("options")
        self.repairs = 0
        self.retest = QueryRetest(transport.config, case['key'])

    def validate_pixels(self, box):
        p = rel_coords_to_abs(box, self.case["original_size"])
        if p[0] >= p[2] or p[1] >= p[3]:
            raise ValueError("EMPTY_NATIVE_PIXELS")

    def tool(self, stage, system, packet, schema, crop=None, include_original=True):
        blocks = [dict(type="text", text=canonical(packet))]
        if include_original:
            blocks.append(self.images.block("original"))
        if crop is not None:
            blocks += [
                dict(type="text", text="CURRENT CROP (only evidence source):"),
                self.images.block(crop),
            ]
        token = self.group([stage])[0]
        message, record = self.ask(
            stage,
            [dict(role="system", content=system), dict(role="user", content=blocks)],
            token,
            tools=[schema],
            tool_choice=dict(type="function", function=dict(name=schema["function"]["name"])),
        )
        self.cost_events.append(
            dict(
                stage=stage,
                receipt_id=record["receipt_id"],
                actual_cost_usd=record["cost_usd"],
                reward=-record["cost_usd"],
            )
        )
        return parse_tool(message, record, schema["function"]["name"]), record

    def plan(self):
        error, draft = None, None
        for _ in range(3):
            self.plan_calls += 1
            stage = f"planner_{self.plan_calls}"
            p = None
            try:
                invalid = dependency_errors(draft["steps"]) if draft is not None else {}
                if invalid:
                    stage += "_dependencies"
                    patch, _ = self.tool(
                        stage,
                        prompts.DEPENDENCY_REPAIR,
                        dict(plan=draft["steps"], invalid_dependencies=invalid, format_error=error),
                        prompts.dependency_tool(invalid),
                        include_original=False,
                    )
                    if set(patch) != {"replacements"}:
                        raise ValueError("DEPENDENCY_REPAIR_ROOT_FIELDS")
                    p = deepcopy(draft)
                    p["steps"] = repair_dependencies(draft["steps"], patch["replacements"])
                else:
                    p, _ = self.tool(
                        stage,
                        prompts.PLAN,
                        dict(
                            allowed_letters=self.case["allowed_letters"],
                            format_error=error,
                            **question_context(self.question, self.case["allowed_letters"]),
                        ),
                        prompts.plan_tool(bool(self.case.get("embedded_question"))),
                    )
                if set(p) != {"reading", "steps"}:
                    raise ValueError("PLAN_ROOT_FIELDS")
                question, options, context = self.read_question(p)
                self.dag.accept(
                    question_plan(
                        p["steps"], context["question_spine"], self.case["allowed_letters"]
                    ),
                    self.validate_pixels,
                )
                self.question, self.options, self.valid_plan = question, options, True
                return True
            except ValueError as exc:
                error = str(exc)
                self.rejections.append(dict(stage=stage, error=error))
                if (
                    isinstance(p, dict)
                    and set(p) == {"reading", "steps"}
                    and dependency_errors(p["steps"])
                ):
                    draft = deepcopy(p)
                elif p is not None or not invalid:
                    draft = None
        self.stop = "BOUNDED_PLAN_FAILURE"
        return False

    def read_question(self, p):
        question, options = self.question, self.options
        if self.case.get("embedded_question"):
            r = p["reading"]
            if (
                not isinstance(r, dict)
                or set(r) != {"question", "options"}
                or not isinstance(r["question"], str)
                or not 5 <= len(r["question"]) <= 2400
            ):
                raise ValueError("EMBEDDED_QUESTION_REQUIRED")
            if (
                not isinstance(r["options"], list)
                or len(r["options"]) != len(self.case["allowed_letters"])
                or any(not isinstance(x, str) or not x.strip() for x in r["options"])
            ):
                raise ValueError("EMBEDDED_OPTIONS_REQUIRED")
            options = r["options"]
            question = (
                r["question"]
                + "\nOptions:\n"
                + "\n".join(
                    f"{a}. {b}" for a, b in zip(self.case["allowed_letters"], options, strict=True)
                )
            )
        context = question_context(question, self.case["allowed_letters"])
        return question, options, context

    def repair(self, crop_id):
        if self.repairs >= 2:
            return False
        indices = [n["index"] for n in self.dag.nodes if n["status"] != "observed"]
        if not indices:
            return False
        self.repairs += 1
        error = None
        for _ in range(2):
            self.plan_calls += 1
            try:
                p, _ = self.tool(
                    f"planner_{self.plan_calls}_repair",
                    prompts.REPAIR,
                    dict(
                        question=self.question,
                        unresolved=[self.node_packet(i) for i in indices],
                        previous_observations=self.facts(),
                        tested_boxes=[v[0]["normalized"] for v in self.tested.values()],
                        format_error=error,
                    ),
                    prompts.repair_tool(indices),
                    crop_id,
                )
                if set(p) != {"patches"}:
                    raise ValueError("REPAIR_ROOT_FIELDS")
                self.dag.revise(p["patches"], self.validate_pixels)
                query_index = len(self.dag.nodes) - 1
                patched = {x["node_index"] for x in p["patches"]}
                if query_index not in patched and patched.intersection(
                    self.dag.ancestors(query_index)[:-1]
                ):
                    derived = query_region_patch(self.dag)
                    if derived:
                        self.dag.revise(derived, self.validate_pixels)
                return True
            except ValueError as exc:
                error = str(exc)
                self.rejections.append(dict(stage=f"planner_{self.plan_calls}_repair", error=error))
        return False

    def node_packet(self, i):
        n = self.dag.nodes[i]
        return {
            k: deepcopy(n[k])
            for k in ["index", "task", "target", "kind", "requires", "groups", "box"]
        }

    def facts(self):
        return [
            dict(
                node_index=o["node_index"],
                task=o["task"],
                target=o["target"],
                crop_id=o["crop_id"],
                admitted=o["observed"],
                **{k: o["reported"][k] for k in ["observation", "identity_evidence", "reason"]},
            )
            for o in self.dag.observations
        ]

    def bound_nodes(self, path, candidate):
        """Batch only planned regions contained in the actual crop.

        Geometry limits the work requested; it never establishes visual truth.
        Dependencies must already have evidence or be requested in this batch.
        """
        x0, y0, x1, y1 = candidate["original_xyxy"]
        indices = []
        attempted = []
        for i in path:
            node = self.dag.nodes[i]
            view = self.images.candidate("original", node["box"])
            a, b, c, d = view["original_xyxy"]
            if not (x0 <= a < c <= x1 and y0 <= b < d <= y1):
                continue
            if any(
                o["node_index"] == i and o["crop_id"] == candidate["id"]
                for o in self.dag.observations
            ):
                continue
            if not all(
                self.dag.nodes[j]["status"] == "observed" or j in indices for j in node["requires"]
            ):
                continue
            indices.append(i)
            if view["id"] == candidate["id"]:
                attempted.append(i)
        return indices, attempted

    def finalization_allowed(self):
        return self.valid_plan

    def acquire_crop(self, candidate, focus):
        key = candidate["id"]
        start = len(self.receipts)
        cached = self.tested.get(key)
        if cached is None:
            candidates = self.crop_candidates(
                dict(anchor=dict(parent="original", box=focus["box"]), branches=[])
            )
            selected, native_evidence = self.probes(candidates, 0)
            native = deepcopy(self.rounds[-1]["raw_reports"])
            self.tested[key] = selected, native_evidence, native
            for r in self.receipts[start:]:
                self.cost_events.append(
                    dict(
                        stage=r["stage"],
                        receipt_id=r["receipt_id"],
                        actual_cost_usd=r["logical_usd"],
                        reward=-r["logical_usd"],
                    )
                )
        else:
            selected, native_evidence, native = cached
        return selected, native_evidence, native, cached is None

    def observe_crop(self, best, candidate, native):
        key = candidate["id"]
        indices, attempted = self.bound_nodes(best["path"], candidate)
        if not indices:
            raise ValueError("SELECTED_VIEW_HAS_NO_BOUND_TASK")
        packet = dict(
            allowed_letters=self.case["allowed_letters"],
            bound_nodes=[self.node_packet(i) for i in indices],
            crop_geometry=dict(
                original_normalized_box=candidate["normalized"],
                original_pixel_xyxy=candidate["original_xyxy"],
                original_size=self.case["original_size"],
            ),
            prior_observations=self.facts(),
            native_relevance_report=native,
            **question_context(self.question, self.case["allowed_letters"]),
        )
        error, added = None, None
        for attempt in range(2):
            try:
                p, record = self.tool(
                    f"observe_{self.turn}_{attempt}",
                    prompts.OBSERVE,
                    dict(packet, format_error=error),
                    prompts.observe_tool(
                        [self.node_packet(i) for i in indices], self.case["allowed_letters"]
                    ),
                    key,
                    include_original=False,
                )
                if "observations" not in p or set(p) - {"observations", "answer"}:
                    raise ValueError("OBSERVATION_ROOT_FIELDS")
                if p.get("answer") is not None and (
                    not isinstance(p["answer"], str)
                    or p["answer"] not in self.case["allowed_letters"]
                ):
                    self.rejections.append(
                        dict(
                            stage=f"observe_{self.turn}_{attempt}",
                            error="INVALID_OBSERVER_ANSWER_IGNORED",
                        )
                    )
                rows = prompts.bind_observations(
                    p["observations"],
                    [self.node_packet(i) for i in indices],
                    p.get("answer"),
                    self.case["allowed_letters"],
                )
                added = self.dag.observe_batch(
                    rows, indices, key, record["receipt_id"], attempted=attempted
                )
                self.cost_events.append(
                    dict(
                        stage="evidence_update",
                        actual_cost_usd=0,
                        reward=sum(r["realized_gain"] for r in added),
                    )
                )
                break
            except ValueError as exc:
                error = str(exc)
                self.rejections.append(dict(stage=f"observe_{self.turn}_{attempt}", error=error))
        if added is None:
            return None
        return added

    def record_observations(self, selected, native_evidence, added, fresh):
        audit = self.retest.consider(self, selected, added)
        evidence = dict(
            native_evidence,
            observations=[
                dict(
                    task=o["task"],
                    target=o["target"],
                    observed=o["observed"],
                    fact=o["reported"]["observation"] if o["observed"] else None,
                    identity=o["reported"]["identity_evidence"] if o["observed"] else None,
                    limitation=(
                        "Crop did not resolve this target/task; supports no answer."
                        if not o["observed"]
                        else o["reported"]["reason"]
                    ),
                )
                for o in added
            ],
            warning="Fallible crop observations. Unobserved target is not evidence of absence. No answer posterior is supplied.",
        )
        if audit is not None:
            evidence['independent_retest'] = audit
        self.record_selected(
            "Measure the fixed question referent in this crop.",
            selected,
            evidence,
            f"partition_{self.turn}",
        )
        if fresh:
            self.rounds[-1].update(
                observations=added, realized_report_gain=sum(r["realized_gain"] for r in added)
            )

    def explore(self):
        if not self.plan():
            return
        for turn in range(1, 7):
            best, proposals = self.dag.select()
            self.decisions.append(
                dict(possible=sorted(self.dag.possible), proposals=proposals, chosen=best)
            )
            if best is None:
                self.stop = (
                    "UNMAPPED_QUERY_FACT_TO_NATIVE_READER"
                    if any(n["status"] == "observed_unmapped" for n in self.dag.nodes)
                    else "NO_POSITIVE_PARTITION_VALUE"
                )
                break
            self.turn = turn
            focus = self.dag.nodes[best["focus"]]
            candidate = self.images.candidate("original", focus["box"])
            key = candidate["id"]
            if any(
                o["node_index"] == focus["index"] and o["crop_id"] == key
                for o in self.dag.observations
            ):
                focus["status"] = "unobserved"
                self.decisions.append(
                    dict(rejected="REPEATED_PHYSICAL_MEASUREMENT", node=focus["index"])
                )
                if not self.repair(key):
                    self.stop = "REGION_REPAIR_EXHAUSTED"
                    break
                continue
            selected, native_evidence, native, fresh = self.acquire_crop(candidate, focus)
            added = self.observe_crop(best, candidate, native)
            if added is None:
                self.stop = "BOUNDED_OBSERVATION_FORMAT_FAILURE"
                break
            self.record_observations(selected, native_evidence, added, fresh)
            if all(o["observed"] for o in added):
                continue
            if self.dag.select()[0] is not None:
                continue  # Other independent useful work before paying for repair.
            if not self.repair(key):
                self.stop = "REGION_REPAIR_EXHAUSTED"
                break
        else:
            self.stop = "MEASUREMENT_RESOURCE_LIMIT"

    def execute(self):
        result = super().execute()
        result["planner_turns"] = self.plan_calls
        result["task_guidance"] = dict(
            graph_enabled=True,
            controller="partition",
            nodes=self.dag.nodes,
            complete=bool(self.dag.nodes)
            and all(n["status"] == "observed" for n in self.dag.nodes),
            plan_rejections=self.rejections,
        )
        result["partition_graph"] = dict(
            possible_answers=sorted(self.dag.possible),
            observations=self.dag.observations,
            region_revisions=self.dag.revisions,
            decisions=self.decisions,
            cost_events=self.cost_events,
            surrogate_is_calibrated_probability=False,
        )
        result['query_retest'] = self.retest.opportunity
        save(self.output / "result.json", result)
        return result
