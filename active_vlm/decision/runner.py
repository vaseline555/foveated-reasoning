"""Costed visual experiments with shared native FOVEA crop/probe and readers."""

import json
from copy import deepcopy

from ..baselines.runner import FoveaRun
from ..common import canonical, save
from .model import EvidenceDAG
from .prompts import OBSERVE_SYSTEM, PLAN_SYSTEM, observation_tool, plan_tool


def parse_tool(message, record, name):
    """Model output errors are data failures, never Python indexing failures."""
    choices = record.get("response", {}).get("choices", [])
    native_length = (
        isinstance(choices, list)
        and bool(choices)
        and isinstance(choices[0], dict)
        and choices[0].get("native_finish_reason") == "length"
    )
    if record.get("finish_reason") == "length" or native_length:
        raise ValueError("TRUNCATED_TOOL_OUTPUT")
    calls = message.get("tool_calls")
    if not isinstance(calls, list) or len(calls) != 1 or not isinstance(calls[0], dict):
        raise ValueError("EXACTLY_ONE_REQUIRED_TOOL_CALL")
    function = calls[0].get("function")
    if not isinstance(function, dict) or function.get("name") != name:
        raise ValueError("REQUIRED_TOOL_NAME_MISMATCH")
    arguments = function.get("arguments")
    if not isinstance(arguments, str):
        raise ValueError("TOOL_ARGUMENTS_MUST_BE_JSON_TEXT")
    try:
        payload = json.loads(arguments)
    except (ValueError, RecursionError) as error:
        raise ValueError("INVALID_TOOL_JSON") from error
    if not isinstance(payload, dict):
        raise ValueError("TOOL_ARGUMENTS_MUST_BE_OBJECT")
    return payload


class DecisionGraphRun(FoveaRun):
    def __init__(self, case, arm, transport, output):
        super().__init__(case, arm, transport, output)
        if arm != "B" or self.crop_protocol["mode"] != "single":
            raise ValueError("DECISION_GRAPH_REQUIRES_GRAPH_SINGLE_PROTOCOL")
        policy = transport.config.get("decision_graph", {})
        self.dag = EvidenceDAG(
            len(case["allowed_letters"]),
            cost_weight=policy.get("cost_weight", 1.0),
            expected_pair_usd=policy.get("expected_pair_usd", 0.003),
        )
        self.max_measurements = policy.get("max_measurements", 6)
        self.max_revisions = policy.get("max_revisions", 2)
        if (
            type(self.max_measurements) is not int
            or not 1 <= self.max_measurements <= 6
            or type(self.max_revisions) is not int
            or not 0 <= self.max_revisions <= 2
        ):
            raise ValueError("INVALID_DECISION_RESOURCE_BOUNDS")
        self.decisions, self.rejections, self.cost_events = [], [], []
        self.tested_crops = {}
        self.valid_plan = False
        self.plan_calls = 0
        self.effective_question = case["question"]

    def paid_tool(self, stage, system, packet, tool, crop=None):
        blocks = [dict(type="text", text=canonical(packet)), self.images.block("original")]
        if crop is not None:
            blocks.extend(
                [
                    dict(type="text", text="CURRENT CROP (only measurement source):"),
                    self.images.block(crop),
                ]
            )
        token = self.group([stage])[0]
        message, record = self.ask(
            stage,
            [dict(role="system", content=system), dict(role="user", content=blocks)],
            token,
            tools=[tool],
            tool_choice=dict(type="function", function=dict(name=tool["function"]["name"])),
        )
        cost = record["cost_usd"]
        self.cost_events.append(
            dict(
                stage=stage,
                receipt_id=record["receipt_id"],
                actual_cost_usd=cost,
                reward=-self.dag.cost_weight * cost,
            )
        )
        return parse_tool(message, record, tool["function"]["name"]), record

    def plan(self, reason):
        last_error = None
        # Ten total paid planning calls, including format corrections/revisions.
        for _ in range(3):
            if self.plan_calls >= 10:
                self.stop = "PLANNING_CALL_LIMIT"
                return False
            self.plan_calls += 1
            packet = dict(
                question=self.case["question"],
                allowed_letters=self.case["allowed_letters"],
                choice_order=self.case.get("options"),
                embedded_question=self.case.get("embedded_question", False),
                belief=self.dag.belief,
                revision_reason=reason,
                previous_error=last_error,
                previous_steps=[
                    dict(test=n["test"], status=n["status"], box=n["box"]) for n in self.dag.nodes
                ],
                prior_observations=self.dag.observations,
            )
            try:
                payload, _ = self.paid_tool(
                    f"planner_{self.plan_calls}",
                    PLAN_SYSTEM,
                    packet,
                    plan_tool(self.case.get("embedded_question", False)),
                )
                if set(payload) != {"reading", "steps"}:
                    raise ValueError("PLAN_ROOT_FIELDS_INVALID")
                reading = payload["reading"]
                question = self.case["question"]
                if self.case.get("embedded_question"):
                    if (
                        not isinstance(reading, dict)
                        or set(reading) != {"question", "options"}
                        or not isinstance(reading["question"], str)
                        or not 5 <= len(reading["question"]) <= 2400
                        or not isinstance(reading["options"], list)
                        or len(reading["options"]) != len(self.case["allowed_letters"])
                        or any(not isinstance(s, str) or not s.strip() for s in reading["options"])
                    ):
                        raise ValueError("EMBEDDED_QUESTION_AND_CHOICES_REQUIRED")
                    question = (
                        reading["question"]
                        + "\nOptions:\n"
                        + "\n".join(
                            f"{letter}. {text}"
                            for letter, text in zip(
                                self.case["allowed_letters"], reading["options"], strict=True
                            )
                        )
                    )
                # Ordinary question/choices are authoritative. Redundant model
                # reading is ignored; it can never replace the supplied text.
                # Tiny normalized boxes can round to empty native pixels.
                # Reject them in the planning correction loop, before commit.
                from ..baselines.geometry import rel_coords_to_abs

                def validate_pixels(box):
                    pixels = rel_coords_to_abs(box, self.case["original_size"])
                    if pixels[0] >= pixels[2] or pixels[1] >= pixels[3]:
                        raise ValueError("PLAN_BOX_ROUNDS_TO_EMPTY_NATIVE_CROP")

                self.dag.accept(payload["steps"], validate_pixels)
                self.effective_question = question
                self.valid_plan = True
                return True
            except ValueError as error:
                last_error = str(error)
                self.rejections.append(dict(stage=f"planner_{self.plan_calls}", error=last_error))
        self.stop = "BOUNDED_PLAN_FAILURE"
        return False

    def finalization_allowed(self):
        # No graph-free fallback after an invalid initial plan.
        return self.valid_plan

    def explore(self):
        if not self.plan("initial question-conditioned sequential experiment design"):
            return
        measurements = 0
        while measurements < self.max_measurements:
            best, proposals = self.dag.select()
            self.decisions.append(
                dict(belief=list(self.dag.belief), proposals=proposals, chosen=best)
            )
            if best is None:
                self.stop = "NO_POSITIVE_NET_INFORMATION_VALUE"
                break
            node = self.dag.nodes[best["focus"]]
            # Native single-crop guards use turn IDs; each measurement has a unique ID.
            self.turn = measurements + 1
            # Deduplicate physical pixels, not just normalized coordinate text.
            candidate = self.images.candidate("original", node["box"])
            key = candidate["id"]
            cached = self.tested_crops.get(key)
            cost_start = len(self.receipts)
            if any(
                r["crop_id"] == key and r["test"] == node["test"] for r in self.dag.observations
            ):
                node["status"] = "unobserved"
                self.decisions.append(
                    dict(rejected="ALREADY_MEASURED_TEST_AND_CROP", node_id=node["id"])
                )
                continue
            if cached is None:
                candidates = self.crop_candidates(
                    dict(anchor=dict(parent="original", box=node["box"]), branches=[])
                )
                selected, evidence = self.probes(candidates, 0)
                for r in self.receipts[cost_start:]:
                    self.cost_events.append(
                        dict(
                            stage=r["stage"],
                            receipt_id=r["receipt_id"],
                            actual_cost_usd=r["logical_usd"],
                            reward=-self.dag.cost_weight * r["logical_usd"],
                        )
                    )
                native = deepcopy(self.rounds[-1]["raw_reports"])
                self.tested_crops[key] = (selected, evidence, native)
            else:
                selected, evidence, native = cached
            measurements += 1
            packet = dict(
                question=self.effective_question,
                test=node["test"],
                prerequisite_observations=[r for r in self.dag.observations if r["observed"]],
                native_relevance_report=native,
            )
            last_error, accepted = None, None
            for correction in range(2):
                packet["format_error"] = last_error
                try:
                    payload, record = self.paid_tool(
                        f"observe_{measurements}_{correction}",
                        OBSERVE_SYSTEM,
                        packet,
                        observation_tool(),
                        selected["id"],
                    )
                    cost = sum(r["logical_usd"] for r in self.receipts[cost_start:])
                    accepted = self.dag.observe(
                        node["index"],
                        payload,
                        record["receipt_id"],
                        selected["id"],
                        selected["original_xyxy"],
                        cost,
                    )
                    # Each cost is recorded once: observer cost event is a debit;
                    # belief gain is a separate potential increment.
                    self.cost_events.append(
                        dict(
                            stage="belief_update",
                            reward=accepted["realized_gain"],
                            actual_cost_usd=0,
                        )
                    )
                    break
                except ValueError as error:
                    last_error = str(error)
                    self.rejections.append(
                        dict(stage=f"observe_{measurements}_{correction}", error=last_error)
                    )
            if accepted is None:
                self.stop = "BOUNDED_OBSERVATION_FORMAT_FAILURE"
                break
            evidence = dict(
                evidence,
                experiment=accepted,
                warning="Fallible visual model observation; no independently verified truth.",
            )
            self.record_selected(
                "Visual experiment: " + node["test"], selected, evidence, f"decision_{measurements}"
            )
            if cached is None:
                self.rounds[-1].update(
                    realized_report_gain=accepted["realized_gain"], observations=[accepted]
                )
            if not accepted["observed"] or payload["needs_repair"]:
                # Optimistic risk bound: a revision cannot improve more than the
                # current Bayes error. The revision itself always incurs a debit.
                if (
                    self.dag.revision >= 1 + self.max_revisions
                    or 1 - max(self.dag.belief) <= self.dag.cost_weight * self.dag.expected_pair_usd
                ):
                    self.stop = "REVISION_RESOURCE_LIMIT"
                    break
                if not self.plan(dict(report=payload, failed_test=node["test"])):
                    break
        else:
            self.stop = "MEASUREMENT_RESOURCE_LIMIT"

    def execute(self):
        result = super().execute()
        result["planner_turns"] = self.plan_calls
        nodes = [
            dict(
                n,
                task=n["test"],
                kind="binary_experiment",
                targets=[],
                depends_on=[self.dag.nodes[i]["id"] for i in n["requires"]],
            )
            for n in self.dag.nodes
        ]
        result["task_guidance"] = dict(
            graph_enabled=True,
            controller="decision",
            nodes=nodes,
            complete=bool(nodes) and all(n["status"] == "observed" for n in nodes),
            plan_rejections=self.rejections,
        )
        result["decision_graph"] = dict(
            belief=self.dag.belief,
            revisions=self.dag.revision - 1,
            retired_nodes=self.dag.retired,
            observations=self.dag.observations,
            decisions=self.decisions,
            cost_events=self.cost_events,
            likelihoods_are_uncalibrated_model_estimates=True,
        )
        save(self.output / "result.json", result)
        return result
