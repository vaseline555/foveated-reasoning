"""Dependency frontier and ranking from model reports, without visual truth certification."""

import re
from collections import defaultdict
from copy import deepcopy

from ..questions import normalize_reading
from .contracts import bind_values
from .evidence import evidence_units
from .feedback import pool_hints, repeat_decision, search_coverage, summarize_probe
from .keywords import extract_keywords
from .planning import (
    KINDS,
    PlanError,
    compile_question,
    plan_limits,
)


class TaskGraph:
    def __init__(self, question, embedded=False, quorum=2, image_size=None, open_questions=False):
        self.question = question
        self.keywords = extract_keywords(question)
        self.open_questions = open_questions
        self.nodes = compile_question(question, embedded, open_questions)
        bind_values(self.nodes, question)
        self.quorum = quorum
        self.attempts = []
        self.image_size = image_size
        self.focus_override = None
        self.partitioned_search = False

    @property
    def plan_limits(self):
        return plan_limits(self.question, multiple_targets=self.open_questions)

    @property
    def phase(self):
        if not self.nodes:
            return "define_goals"
        if any(n["kind"] == "read_task" for n in self.nodes):
            return "read_question"
        return "define_goals" if any(n["kind"] not in KINDS for n in self.nodes) else "search"

    def require_plan(self):
        if self.phase != "search":
            raise PlanError("action", "VALID_PLAN_REQUIRED", f"call {self.phase} first")

    @property
    def complete(self):
        # Empty plans or an empty dependency frontier are not evidence of
        # completion. Question-specified frames need no additional observation.
        return (
            self.phase == "search"
            and bool(self.nodes)
            and all(
                n["status"] in {"supported_by_reports", "specified_by_question"} for n in self.nodes
            )
        )

    def require_search(self, graph=True):
        self.require_plan()
        if self.complete:
            raise ValueError("ALL_GOALS_SUPPORTED: use the common final readers, not more crops")
        if self.focus(graph) is None:
            raise ValueError(
                "NO_READY_VISUAL_GOAL: retain unresolved goals; do not crop without a focus"
            )

    def ready(self, graph=True):
        done = {
            n["id"]
            for n in self.nodes
            if n["status"] in {"supported_by_reports", "specified_by_question"}
        }
        return [
            n
            for n in self.nodes
            if n["status"] == "unresolved" and (not graph or set(n["depends_on"]) <= done)
        ]

    def focus(self, graph=True):
        ready = self.ready(graph)
        if not ready:
            return None
        if self.focus_override is not None:
            return next((n for n in ready if n["id"] == self.focus_override), None)
        # Unproductive exploration rotates independent ready tasks; blocked tasks
        # never win merely because they have not been attempted yet.
        counts = {n["id"]: sum(n["id"] in a["focus_ids"] for a in self.attempts) for n in ready}
        return min(ready, key=lambda n: (counts[n["id"]], int(n["id"][1:])))

    def packet(self, graph=True, full=False):
        ready = self.ready(graph)
        focus = self.focus(graph)
        packet = dict(
            phase=self.phase,
            mode="dependency_graph" if graph else "flat_subgoals",
            focus=deepcopy(focus),
            unresolved_ids=[n["id"] for n in self.nodes if n["status"] == "unresolved"],
            ready_ids=[n["id"] for n in ready],
            recent_attempts=[
                {
                    k: deepcopy(v)
                    for k, v in a.items()
                    if k not in {"probe_feedback", "search_decision"}
                }
                for a in self.attempts[-3:]
            ],
            latest_probe_feedback=deepcopy(self.attempts[-1].get("probe_feedback"))
            if self.attempts
            else None,
            search_coverage=search_coverage(self.attempts, focus, self.image_size),
            reported_support_is_not_verified_visual_truth=True,
            support_basis="single_report" if self.quorum == 1 else "independent_report_agreement",
            required_independent_reports=self.quorum,
        )
        if packet["focus"] is not None and not graph:
            packet["focus"].pop("depends_on", None)
        if full:
            packet["question_keywords"] = deepcopy(self.keywords)
            packet["goals"] = [
                {k: deepcopy(n[k]) for k in ["id", "kind", "task", "targets", "status"]}
                for n in self.nodes
            ]
        if graph:
            packet["blocked_ids"] = [
                n["id"] for n in self.nodes if n["status"] == "unresolved" and n not in ready
            ]
            if full:
                packet["dependencies"] = [
                    dict(source=d, target=n["id"]) for n in self.nodes for d in n["depends_on"]
                ]
        return packet

    def evidence(self, node, candidate, reports):
        groups = defaultdict(list)
        seen_receipts = set()
        # A dependent property must stay spatially connected to its referent.
        for parent in self.nodes:
            if (
                parent["id"] not in node["depends_on"]
                or parent["kind"] not in {"locate", "condition"}
                or not parent["citations"]
            ):
                continue

            # Crops are regions, not object boxes: a valid nested zoom can have
            # tiny IoU. Require intersection of the smaller region instead.
            def connected(a, b):
                inter = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(
                    0, min(a[3], b[3]) - max(a[1], b[1])
                )
                small = min((a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1]))
                return small > 0 and inter / small >= 0.5

            if not any(
                connected(candidate["original_xyxy"], c["original_xyxy"])
                for c in parent["citations"]
            ):
                return []
        for i, report in enumerate(reports):
            if report["receipt_id"] in seen_receipts:
                continue
            seen_receipts.add(report["receipt_id"])
            text = report["message"].get("content") or ""
            if text.lstrip().startswith("{"):
                continue  # Legacy identity/match JSON is not a native FOVEA probe.
            inventory = [t for n in self.nodes for t in n["targets"]]
            units = list(evidence_units(node, text, inventory))
            signatures = {u["signature"] for u in units} - {None}
            if len(signatures) == 1:
                signature = next(iter(signatures))
                unit = next(u for u in units if u["signature"] == signature)
                groups[signature].append(
                    dict(
                        replicate=i,
                        receipt_id=report["receipt_id"],
                        quote=unit["quote"],
                        context_quote=unit["context_quote"],
                        source_quote=unit.get("source_quote"),
                        reference_quotes=unit.get("reference_quotes", []),
                        coverage_quote=unit.get("coverage_quote"),
                        crop_id=candidate["id"],
                        original_xyxy=list(candidate["original_xyxy"]),
                        reported_value=signature,
                    )
                )
        citations = max(groups.values(), key=len, default=[])
        return citations if len(citations) >= self.quorum else []

    def candidate_progress(self, candidate, reports, graph=True):
        # Reward downstream tasks that the SAME crop can legitimately unlock.
        # Otherwise graph mode would prefer a location-only crop over an equally
        # grounded crop that already exposes the requested attribute as well.
        preview = deepcopy(self)
        return preview.observe(candidate, reports, graph)["newly_supported"]

    def choose(self, candidates, scores, reports, graph=True):
        focus = self.focus(graph)
        audit = []
        for candidate, score, raw in zip(candidates, scores, reports, strict=True):
            supported = self.candidate_progress(candidate, raw, graph)
            rank = (int(focus is not None and focus["id"] in supported), len(supported), score)
            audit.append(
                dict(
                    crop_id=candidate["id"],
                    progress_nodes=supported,
                    native_relevance=score,
                    rank=list(rank),
                )
            )
        winner = max(range(len(audit)), key=lambda i: tuple(audit[i]["rank"]))
        return winner, audit

    def observe(self, candidate, reports, graph=True, candidate_pool=None, candidate_reports=None):
        focus = self.focus(graph)
        search = self.repeat_decision(candidate_pool or [candidate], graph)
        before = {n["id"] for n in self.nodes if n["status"] == "supported_by_reports"}
        # One crop can support several dependent tasks, but each has to supply its
        # own crop-grounded observations. Mere ID binding/Yes never suffices.
        for _ in range(len(self.nodes)):
            progressed = False
            for node in self.ready(graph):
                citations = self.evidence(node, candidate, reports)
                if citations:
                    node.update(status="supported_by_reports", citations=citations)
                    progressed = True
            if not progressed:
                break
        after = {n["id"] for n in self.nodes if n["status"] == "supported_by_reports"}
        record = dict(
            crop_id=candidate["id"],
            original_xyxy=list(candidate["original_xyxy"]),
            focus_ids=[focus["id"]] if focus else [],
            newly_supported=sorted(after - before),
            semantic_truth_verified=False,
            candidate_pool_ids=[c["id"] for c in (candidate_pool or [candidate])],
            candidate_pool_boxes=[
                list(c["original_xyxy"]) for c in (candidate_pool or [candidate])
            ],
            unsupported_goal_ids=[
                n["id"]
                for n in self.nodes
                if n["status"] == "unresolved" and not self.evidence(n, candidate, reports)
            ],
            search_refinement=search["reason"]
            in {"target_detail_refinement", "unconfirmed_detail_refinement", "partition_detail"},
            search_decision=search,
        )
        observed_pool = (
            list(zip(candidate_pool or [candidate], candidate_reports, strict=True))
            if candidate_reports is not None
            else [(candidate, reports)]
        )
        record["unsupported_pool_goal_ids"] = [
            n["id"]
            for n in self.nodes
            if n["status"] == "unresolved"
            and not any(self.evidence(n, c, raw) for c, raw in observed_pool)
        ]
        record["search_tested_boxes"] = [list(c["original_xyxy"]) for c, _ in observed_pool]
        record["probe_feedback"] = summarize_probe(
            candidate, reports, self.nodes, record["newly_supported"]
        )
        record["probe_feedback"]["unconfirmed_pool_hints"] = pool_hints(
            [c for c, _ in observed_pool], [raw for _, raw in observed_pool], self.nodes
        )
        self.attempts.append(record)
        return record

    def reject_repeat(self, candidates, graph=True):
        return self.repeat_decision(candidates, graph)["reject"]

    def repeat_decision(self, candidates, graph=True):
        return repeat_decision(
            self.attempts,
            self.focus(graph),
            self.nodes,
            candidates,
            partition_size=self.image_size if self.partitioned_search else None,
        )

    def read_question(self, text, options):
        if not any(n["kind"] == "read_task" for n in self.nodes):
            raise ValueError("READING_NOT_REQUESTED")
        question = normalize_reading(text, options)
        self.question = question
        self.keywords = extract_keywords(self.question)
        self.nodes = compile_question(self.question, open_questions=self.open_questions)
        bind_values(self.nodes, self.question)
        self.attempts = []
        return dict(status="reported_reading", independently_verified=False, goals=len(self.nodes))

    def define_goals(self, steps):
        if any(n["kind"] != "inspect" for n in self.nodes) or self.attempts:
            raise PlanError("steps", "EXISTING_PLAN_CANNOT_BE_REBOUND", "an unstarted pending plan")
        limit = self.plan_limits
        if not isinstance(steps, list) or not 1 <= len(steps) <= limit["max_steps"]:
            raise PlanError(
                "steps",
                "INVALID_SUBGOAL_COUNT",
                f"a list of 1..{limit['max_steps']} atomic visual tasks; do not drop question alternatives",
            )
        nodes = []
        text_chars = 0
        for i, step in enumerate(steps):
            path = f"steps[{i}]"
            if not isinstance(step, dict) or set(step) != {"kind", "task", "targets", "requires"}:
                raise PlanError(path, "INVALID_SUBGOAL_FIELDS", "kind, task, targets, requires")
            if step["kind"] not in KINDS:
                raise PlanError(path + ".kind", "INVALID_SUBGOAL_KIND", list(KINDS))
            if (
                not isinstance(step["task"], str)
                or len(step["task"].strip()) < 5
                or not isinstance(step["targets"], list)
                or not 1 <= len(step["targets"]) <= limit["max_targets"]
                or any(not isinstance(t, str) or not t.strip() for t in step["targets"])
            ):
                raise PlanError(
                    path
                    + (
                        ".targets"
                        if isinstance(step["targets"], list)
                        and len(step["targets"]) > limit["max_targets"]
                        else ".task_or_targets"
                    ),
                    "INVALID_SUBGOAL_TEXT",
                    f"task: nonempty text of at least5 chars; targets:1..{limit['max_targets']} nonempty strings. Preserve identifying qualifiers. Split independent objects into separate steps; retain all choices.",
                )
            text_chars += len(step["task"]) + sum(map(len, step["targets"]))
            if text_chars > limit["max_text_chars"]:
                raise PlanError(
                    "steps",
                    "PLAN_TEXT_BUDGET_EXCEEDED",
                    f"at most {limit['max_text_chars']} total task/target characters; remove duplicated prose, retain target qualifiers",
                )
            deps = step["requires"]
            if not isinstance(deps, list) or any(
                type(d) is not int or not 0 <= d < i for d in deps
            ):
                raise PlanError(
                    path + ".requires",
                    "DEPENDENCIES_MUST_POINT_TO_EARLIER_STEPS",
                    f"integer indices 0..{i - 1}; [] for a root, never names or strings",
                )
            nodes.append(
                dict(
                    id=f"g{i + 1}",
                    kind=step["kind"],
                    task=step["task"],
                    targets=step["targets"],
                    depends_on=[f"g{d + 1}" for d in deps],
                    status="unresolved",
                    citations=[],
                )
            )
        if not any(n["kind"] == "locate" for n in nodes):
            raise PlanError(
                "steps",
                "TARGET_LOCALIZATION_REQUIRED",
                "locate the question target before judging it",
            )
        for node in nodes:
            if node["kind"] != "locate" and not node["depends_on"]:
                raise PlanError(
                    node["id"] + ".requires",
                    "DEPENDENCY_REQUIRED",
                    "a prior target-localization step index",
                )
        retrieval = self.open_questions and re.search(
            r"\b(?:which|what)\s+(?:object|item)\s+is\s+(?:located|positioned)\b",
            self.question,
            re.I,
        )
        if not any(
            n["kind"] in {"condition", "attribute", "frame", "relation", "count"} for n in nodes
        ) and not (retrieval and nodes[-1]["kind"] == "locate" and nodes[-1]["depends_on"]):
            raise PlanError(
                "steps",
                "QUESTION_ACTION_REQUIRED",
                "an attribute, relation, condition or count after localization",
            )
        bind_values(nodes, self.question)
        self.nodes = nodes
        return dict(status="accepted_task_instructions", goals=len(nodes), visual_facts_added=0)
