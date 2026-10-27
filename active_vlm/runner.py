"""Common trajectory, shared final readers, and complete receipt accounting."""

import time
import urllib.error
from copy import deepcopy

from .answers import answer_mode, choice_schema, direct_mode, parse_final_answer, parser_name
from .baselines import prompts
from .baselines.messages import aggregator_messages
from .budget import BudgetStop, UnknownCharge
from .common import canonical, read, save
from .errors import HarnessError, IntegrityError
from .imaging import Images
from .protocol import crop_protocol
from .readers import read as read_final


def clean(message):
    return {
        k: deepcopy(message[k])
        for k in ["role", "content", "tool_calls", "tool_call_id"]
        if message.get(k) is not None
    }


class BaseRun:
    def __init__(self, case, arm, transport, output):
        offline = getattr(transport, "offline_only", False) and callable(transport.responder)
        if not offline and (not getattr(transport, "approval_bound", False)):
            raise RuntimeError("EXPLICIT_APPROVAL_REQUIRED")
        self.case, self.arm, self.transport, self.output = (case, arm, transport, output)
        self.crop_protocol = crop_protocol(transport.config, arm)
        self.final_answer_mode = answer_mode(transport.config)
        self.direct_answer_mode = direct_mode(transport.config)
        self.path_key = case["key"] + "/" + arm
        self.images = Images(case, output)
        self.events, self.rounds, self.errors, self.receipts = ([], [], [], [])
        self.reader_diagnostics = []
        self.turn = 0
        self.stop = "TURN_LIMIT"
        self.selected = []
        self.no_novel_rejections = 0
        self.global_stop_reason = None
        self.crop_turns = set()
        self.probe_turns = set()
        self.history = [
            dict(
                role="system",
                content=prompts.template("planner_react.j2"),
            ),
            dict(
                role="user",
                content=[
                    self.images.block("original"),
                    dict(type="text", text=case["question"]),
                    dict(type="text", text="\nOrignal image URL: original\n"),
                    dict(type="text", text="\nYou are on TURN 1 (max 10)"),
                ],
            ),
        ]
        self.final_history = deepcopy(self.history[:2])

    def group(self, stages, final=False):
        return self.transport.budget.reserve(self.path_key, stages, final)

    def ask(self, stage, messages, token, tools=None, schema=None, tool_choice=None):
        try:
            message, record = self.transport.call(
                self.case["key"], self.arm, stage, messages, token, tools, schema, tool_choice
            )
        except BaseException:
            # Only undispatched reservations are refundable. Failed HTTP holds
            # retain their ledger status and are deliberately left untouched.
            self.transport.budget.release_unused([token])
            raise
        self.receipts.append(
            dict(
                stage=stage,
                receipt_id=record["receipt_id"],
                logical_usd=record["cost_usd"],
                cached=record["cached"],
                request_hash=record["request_hash"],
                finish=record["finish_reason"],
            )
        )
        self.events.append(
            dict(
                event="response",
                stage=stage,
                message=message,
                receipt_id=record["receipt_id"],
                finish=record["finish_reason"],
            )
        )
        return (clean(message), record)

    def error(self, stage, error):
        self.errors.append(dict(stage=stage, type=type(error).__name__, message=str(error)[:300]))

    def record_selected(self, thought, selected, evidence, call_id):
        tool = dict(role="tool", tool_call_id=call_id, content=canonical(evidence))
        image = dict(
            role="user",
            content=[
                dict(
                    type="text",
                    text=f"The tool successfully generated a new image. You can now refer to it as url: {selected['id']}. It can works as tool input using the url.",
                ),
                self.images.block(selected["id"]),
            ],
        )
        report_message = dict(
            role="assistant",
            content=thought,
            tool_calls=[
                dict(id=call_id, type="function", function=dict(name="cropping", arguments="{}"))
            ],
        )
        self.final_history.extend([report_message, tool, image])
        return (tool, image)

    def finish(self):
        tokens = self.group(["direct", "aggregator"], final=True)
        try:
            prompt = prompts.template("direct_reasoning_system.j2").replace(
                "{{ question }}", self.case["question"]
            )
            direct_schema = None
            if self.direct_answer_mode == "choice":
                prompt = (
                    "Inspect the original image and choose the single best-supported allowed "
                    'option. Return ONLY the JSON object {"answer":"LETTER"}; no analysis, '
                    "markdown or additional fields. Do not repeat or rewrite the choices.\n"
                    + self.case["question"]
                )
                direct_schema = choice_schema(self.case["allowed_letters"])
            direct, direct_valid = read_final(
                self,
                "direct",
                [
                    dict(
                        role="user",
                        content=[dict(type="text", text=prompt), self.images.block("original")],
                    )
                ],
                tokens[0],
                self.direct_answer_mode,
                schema=direct_schema,
            )
            if (
                self.direct_answer_mode == "choice"
                and parse_final_answer(
                    direct.get("content"), self.case["allowed_letters"], "choice"
                )
                is None
            ):
                self.error("direct", ValueError("INVALID_BOUNDED_DIRECT_ANSWER"))
                direct = dict(
                    content="Direct reader returned no valid allowed choice; do not treat its output as evidence."
                )
            if direct_valid is False:
                direct = dict(
                    content="Direct reader returned no complete valid allowed choice; do not treat its incomplete analysis as evidence."
                )
            final_history = deepcopy(self.final_history)
            final_history.append(
                dict(
                    role="assistant",
                    content=canonical(
                        dict(exploration_stop_reason=self.stop, protocol_errors=self.errors)
                    ),
                )
            )
            system = prompts.template("aggregator_system.j2")
            schema = None
            if self.final_answer_mode == "choice":
                system = (
                    "You are the final judge. Compare the original image, the ReAct crop "
                    "evidence and the Direct report. Both reports can be wrong. Select the "
                    "single best-supported option from the supplied choices. Return ONLY "
                    'the JSON object {"answer":"LETTER"} matching the response schema. '
                    "Do not output analysis, repeated conclusions, markdown or any extra fields. "
                    "Do not invent an option or turn uncertainty into an unsupported fact."
                )
                schema = choice_schema(self.case["allowed_letters"])
            messages = aggregator_messages(
                self.case["question"],
                self.images.descriptor(self.images.images["original"]),
                final_history,
                direct.get("content") or "",
                system,
            )
            response, valid = read_final(
                self, "aggregator", messages, tokens[1], self.final_answer_mode, schema=schema
            )
            if valid is False:
                return None
            return parse_final_answer(
                response.get("content"), self.case["allowed_letters"], self.final_answer_mode
            )
        finally:
            self.transport.budget.release_unused(tokens)

    def finalization_allowed(self):
        return True

    def execute(self):
        start = time.perf_counter()
        prediction = None
        try:
            try:
                self.explore()
            except IntegrityError:
                raise
            except BudgetStop as error:
                self.stop = str(error)
                if any(word in self.stop for word in ["CUMULATIVE", "CONCURRENT", "GLOBAL"]):
                    self.global_stop_reason = self.stop
            except UnknownCharge:
                raise
            except Exception as error:
                if self.transport.config.get("strict_harness") and not isinstance(error, (ValueError, urllib.error.URLError)):
                    raise HarnessError(f"exploration:{type(error).__name__}:{error}") from error
                self.error("exploration", error)
                self.stop = "FORMAT_OR_EXECUTION_ERROR"
            if not self.global_stop_reason and self.finalization_allowed():
                prediction = self.finish()
        except HarnessError:
            raise
        except IntegrityError as error:
            self.transport.budget.halt()
            self.global_stop_reason = str(error)
            self.stop = "INTEGRITY_GLOBAL_STOP"
            self.error("integrity", error)
        except UnknownCharge as error:
            self.global_stop_reason = str(error)
            self.stop = "UNKNOWN_CHARGE_GLOBAL_STOP"
            self.error("billing", error)
        except Exception as error:
            if self.transport.config.get("strict_harness") and not isinstance(error, (ValueError, BudgetStop, urllib.error.URLError)):
                raise HarnessError(f"final_reader:{type(error).__name__}:{error}") from error
            if isinstance(error, BudgetStop) and any(
                word in str(error) for word in ["CUMULATIVE", "CONCURRENT", "GLOBAL"]
            ):
                self.global_stop_reason = str(error)
            self.error("final_reader", error)
            self.stop = "FINAL_READER_ERROR"
        consumed = [
            r
            for r in self.transport.records
            if r["case"] == self.case["key"] and r["arm"] == self.arm
        ]
        all_receipts = [
            dict(
                stage=r["stage"],
                receipt_id=r.get("receipt_id"),
                logical_usd=r.get("cost_usd"),
                cached=r.get("cached", False),
                request_hash=r["request_hash"],
                finish=r.get("finish_reason"),
                status=r.get("status", "cached"),
            )
            for r in consumed
        ]
        result = dict(
            protocol="crop_only",
            answer_parser=parser_name(self.final_answer_mode),
            final_answer_mode=self.final_answer_mode,
            direct_answer_mode=self.direct_answer_mode,
            reader_diagnostics=self.reader_diagnostics,
            real_api_allowed=bool(getattr(self.transport, "approval_bound", False)),
            crop_probe_protocol=self.crop_protocol,
            native_selector_enabled=self.crop_protocol["multi_candidate_selectors_enabled"],
            path_terminal=not self.global_stop_reason
            and not read(self.transport.budget.path)["stopped"],
            global_stop_reason=self.global_stop_reason,
            case_key=self.case["key"],
            dataset=self.case["dataset"],
            category=self.case["category"],
            arm=self.arm,
            prediction=prediction,
            status="ok" if prediction else "invalid_answer",
            stop_reason=self.stop,
            planner_turns=self.turn,
            crop_rounds=len(self.rounds),
            probe_calls=sum(r["stage"].startswith("probe_") for r in all_receipts),
            calls=len(all_receipts),
            logical_usd=sum(r["logical_usd"] for r in all_receipts if r["logical_usd"] is not None),
            logical_unknown_reserve_usd=sum(
                r.get("reserve_usd", self.transport.config["request_upper_usd"])
                for r in consumed
                if r.get("cost_usd") is None
            ),
            unknown_charges=sum(r["logical_usd"] is None for r in all_receipts),
            seconds=time.perf_counter() - start,
            receipts=all_receipts,
            errors=self.errors,
            rounds=self.rounds,
            history=self.history,
            final_history=self.final_history,
            events=self.events,
            no_novel_crop_rejections=self.no_novel_rejections,
            views=list(self.images.views.values()),
        )
        successful = [r for r in all_receipts if r["status"] in {"ok", "cached"}]
        result.update(
            attempts_including_errors=len(all_receipts),
            failed_transport_attempts=len(all_receipts) - len(successful),
            calls=len(successful),
            probe_calls=sum(r["stage"].startswith("probe_") for r in successful),
        )
        save(self.output / "result.json", result)
        return result
