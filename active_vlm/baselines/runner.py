"""FOVEA Greedy, MCMC, and Lookahead exploration with common final readers."""

import json

from ..common import canonical, digest
from ..runner import BaseRun
from . import prompts
from .messages import probe_messages, yes_score


class FoveaRun(BaseRun):
    def planner_tools(self):
        return [prompts.CROP_TOOL]

    def planner_tool_choice(self):
        return None

    def control_call(self, call):
        return None

    def accept_stop(self, message):
        return True

    def stopping_reason(self):
        return "MODEL_READY"

    def exploration_stop_reason(self):
        """Optional search-controller stop; native FOVEA keeps its usual loop."""
        return None

    def crop_candidates(self, proposal):
        if self.crop_protocol["mode"] == "single":
            from ..imaging import validate_seed

            if self.turn in self.crop_turns:
                raise ValueError("SINGLE_CROP_PER_TURN")
            anchor = proposal["anchor"]
            if proposal.get("branches"):
                raise ValueError("SINGLE_CROP_PROTOCOL_FORBIDS_BRANCHES")
            if anchor["parent"] not in self.images.images:
                raise ValueError("UNKNOWN_PARENT")
            crop = self.images.candidate(anchor["parent"], validate_seed(anchor["box"]))
            crop["target_nodes"] = []
            self.crop_turns.add(self.turn)
            return [crop]
        return self.images.pool("A", proposal)

    def choose_candidate(self, candidates, scores, reports):
        return max(range(len(candidates)), key=lambda i: scores[i])

    def planner_messages(self):
        return self.history

    def turn_instruction(self, turn):
        return f"SYSTEM STATUS UPDATE: You are currently on Turn {turn} out of 10. If {turn} == 10, you MUST stop calling tools and provide your best final answer immediately in the required format."

    def probes(self, candidates, call_index):
        k_count = self.crop_protocol["K"]
        if self.crop_protocol["mode"] == "single" and len(candidates) != 1:
            raise ValueError("SINGLE_CROP_PROTOCOL_REQUIRES_ONE_CANDIDATE")
        if self.crop_protocol["mode"] == "single":
            if self.turn in self.probe_turns:
                raise ValueError("SINGLE_PROBE_PER_TURN")
            self.probe_turns.add(self.turn)
        stages = [
            f"probe_{self.turn}_{call_index}_{ci}_{k}"
            for ci in range(len(candidates))
            for k in range(k_count)
        ]
        snapshot = None
        frozen_hash = digest(snapshot)
        inputs = []
        for candidate in candidates:
            parent = candidate["parent"]
            messages = probe_messages(
                self.case["question"],
                self.images.images["original"],
                None if parent == "original" else self.images.images[parent],
                candidate["image"],
                lambda image: self.images.descriptor(image, probe=True),
                prompts.template("boed_resolvability.md"),
            )
            inputs.append(messages)
        tokens = self.group(stages)
        scores, raw = [], []
        used = 0
        try:
            for ci, candidate in enumerate(candidates):
                contents, records = [], []
                for k in range(k_count):
                    index = ci * k_count + k
                    message, record = self.ask(
                        stages[index], inputs[ci], tokens[index], schema=None
                    )
                    used += 1
                    contents.append(message.get("content"))
                    records.append(dict(receipt_id=record["receipt_id"], message=message))
                scores.append(yes_score(contents))
                raw.append(records)
            if digest(None) != frozen_hash:
                raise ValueError("ROUND_SNAPSHOT_CHANGED")
            winner = self.choose_candidate(candidates, scores, raw)
            gain = None
            selected = candidates[winner]
            self.images.commit(selected)
            audit = dict(
                turn=self.turn,
                call_index=call_index,
                frozen_state=snapshot,
                frozen_hash=frozen_hash,
                candidate_inputs_hashes=[digest(x) for x in inputs],
                candidates=[self.images.audit(c) for c in candidates],
                scores=scores,
                raw_reports=raw,
                observations=[],
                selected_index=winner,
                selected_id=selected["id"],
                mean_selected_score=scores[winner],
                realized_report_gain=gain,
                paid_probe_count=len(stages),
                ledger_after=None,
            )
            self.rounds.append(audit)
            self.selected.append(selected["id"])
            evidence = dict(
                status="success",
                output_image=selected["id"],
                selected_coordinates=selected["normalized"],
                original_xyxy=selected["original_xyxy"],
            )
            return (selected, evidence)
        finally:
            self.transport.budget.release_unused(tokens[used:])

    def explore(self):
        for turn in range(1, 11):
            stop_reason = self.exploration_stop_reason()
            if stop_reason is not None:
                self.stop = stop_reason
                break
            self.turn = turn
            if turn > 1:
                self.history.append(
                    dict(
                        role="user",
                        content=self.turn_instruction(turn),
                    )
                )
            stage = f"planner_{turn}"
            self.plan_schema = None
            token = self.group([stage])[0]
            message, _ = self.ask(
                stage,
                self.planner_messages(),
                token,
                tools=self.planner_tools(),
                schema=self.plan_schema,
                tool_choice=self.planner_tool_choice(),
            )
            self.history.append(message)
            calls = message.get("tool_calls", [])
            if not calls:
                if not self.accept_stop(message):
                    continue
                self.stop = self.stopping_reason()
                self.final_history.append(message)
                break
            if turn == 10 and self.planner_tool_choice() is None:
                self.stop = "TURN_LIMIT"
                break
            tool_messages, image_messages = ([], [])
            cropped_this_turn = False
            for ci, call in enumerate(calls):
                # An earlier crop in this same planner response may have
                # completed the graph. Do not execute the remaining tools.
                stop_reason = self.exploration_stop_reason()
                if stop_reason is not None:
                    self.stop = stop_reason
                    tool_messages.append(
                        dict(
                            role="tool",
                            tool_call_id=call["id"],
                            content=canonical(dict(status="not_executed", message=stop_reason)),
                        )
                    )
                    continue
                try:
                    control = self.control_call(call)
                    if control is not None:
                        tool_messages.append(control)
                        continue
                    if turn == 10:
                        self.stop = "TURN_LIMIT"
                        tool_messages.append(
                            dict(
                                role="tool",
                                tool_call_id=call["id"],
                                content=canonical(
                                    dict(status="not_executed", message="TURN_LIMIT")
                                ),
                            )
                        )
                        continue
                    if call["function"]["name"] != "cropping":
                        raise ValueError("UNKNOWN_TOOL")
                    if self.crop_protocol["mode"] == "single" and cropped_this_turn:
                        tool_messages.append(
                            dict(
                                role="tool",
                                tool_call_id=call["id"],
                                content=canonical(
                                    dict(status="not_executed", message="SINGLE_CROP_PER_TURN")
                                ),
                            )
                        )
                        continue
                    cropped_this_turn = True
                    args = json.loads(call["function"]["arguments"])
                    proposal = dict(
                        anchor=dict(
                            parent=args["image_url"], box=args["coordinates"], target_nodes=[]
                        ),
                        branches=[],
                    )
                    if (
                        self.arm in {"M", "L"}
                        and self.crop_protocol["multi_candidate_selectors_enabled"]
                    ):
                        from .selectors import select_fovea

                        selected, evidence = select_fovea(self, proposal, ci)
                    else:
                        candidates = self.crop_candidates(proposal)
                        selected, evidence = self.probes(candidates, ci)
                    tool, image = self.record_selected(
                        message.get("content"), selected, evidence, call["id"]
                    )
                    tool_messages.append(tool)
                    image_messages.append(image)
                except (KeyError, ValueError, TypeError) as error:
                    self.error("tool", error)
                    tool_messages.append(
                        dict(
                            role="tool",
                            tool_call_id=call["id"],
                            content=canonical(dict(status="error", message=str(error)[:300])),
                        )
                    )
            self.history.extend(tool_messages)
            self.history.extend(image_messages)
