"""One-time visual region planning followed by local graph search control."""

import json
from copy import deepcopy

from ..common import canonical, digest
from .search import CropSearch

INSTRUCTION = (
    "Build the requested visual search plan in English using the required tool. "
    "Use question/choice text as tasks, not as observed facts. Preserve referent "
    "qualifiers and dependencies. Regions are tentative original-image 0..1000 "
    "x1,y1,x2,y2 search boxes, not evidence. Python will reuse the accepted plan "
    "and choose subsequent crops and termination without further model planning. "
    "Never answer the question here. Correct the specified field after rejection."
)


def controller_mode(config):
    mode = config.get("graph_controller", "model")
    if mode not in {"python", "model"}:
        raise ValueError("INVALID_GRAPH_CONTROLLER")
    return mode


def plan_messages(run, tool):
    if tool == "plan_regions":
        packet = dict(
            required_tool=tool,
            targets=[
                {k: deepcopy(n[k]) for k in ["id", "task", "targets", "depends_on"]}
                for n in run.search.targets(run.tasks)
            ],
        )
    else:
        packet = dict(required_tool=tool, limits=run.tasks.plan_limits)
    if run.latest_plan_error:
        packet["correction"] = {
            k: deepcopy(run.latest_plan_error[k])
            for k in [
                "error",
                "identical_invalid_attempts",
                "step_index_map",
                "rejected_arguments",
                "dependency_example",
                "repair_indices",
                "instruction",
            ]
            if k in run.latest_plan_error
        }
    run.packets.append(dict(turn=run.turn, packet=dict(phase=tool, **packet)))
    return [
        dict(role="system", content=INSTRUCTION),
        dict(
            role="user",
            content=[
                run.images.block("original"),
                dict(type="text", text=run.tasks.question),
                dict(type="text", text=canonical(packet)),
            ],
        ),
    ]


def plan_step(run):
    phase = run.tasks.phase
    tool = run.planner_tools()[0] if phase != "search" else run.search.tool(run.tasks)
    name = tool["function"]["name"]
    token = run.group([f"planner_{run.turn}"])[0]
    message, _ = run.ask(
        f"planner_{run.turn}",
        plan_messages(run, name),
        token,
        tools=[tool],
        tool_choice=dict(type="function", function=dict(name=name)),
    )
    calls = message.get("tool_calls") or []
    valid_call = (
        isinstance(calls, list)
        and len(calls) == 1
        and isinstance(calls[0], dict)
        and isinstance(calls[0].get("function"), dict)
        and calls[0]["function"].get("name") == name
        and isinstance(calls[0]["function"].get("arguments"), str)
    )
    if name != "plan_regions":
        if not valid_call:
            run.reject_plan_action(ValueError("ONE_REQUIRED_PLAN_TOOL_CALL"))
        else:
            run.control_call(calls[0])
        return False
    try:
        if not valid_call:
            raise ValueError("ONE_PLAN_REGIONS_CALL_REQUIRED")
        regions = json.loads(calls[0]["function"]["arguments"])
        run.search.accept(run.tasks, regions)
    except (KeyError, ValueError, TypeError) as error:
        run.error("region_plan", error)
        payload = (
            calls[0]["function"]["arguments"]
            if valid_call
            else (
                [c.get("function") if isinstance(c, dict) else c for c in calls]
                if isinstance(calls, list)
                else calls
            )
        )
        signature = digest([str(error), payload])
        run.invalid_plans[signature] += 1
        count = run.invalid_plans[signature]
        run.plan_stalled = count >= run.transport.config.get("task_policy", {}).get(
            "max_identical_invalid_plans", 3
        )
        run.latest_plan_error = dict(
            error=dict(code="INVALID_REGION_PLAN", expected=str(error)),
            identical_invalid_attempts=count,
        )
        return False
    run.latest_plan_error = None
    run.events.append(
        dict(event="initial_region_plan_fixed", turn=run.turn, regions=deepcopy(run.search.regions))
    )
    return True


def explore(run):
    run.search = CropSearch(**run.transport.config.get("graph_search_limits", {}))
    for turn in range(1, 11):
        run.turn = turn
        reason = run.exploration_stop_reason()
        if reason:
            run.stop = reason
            return
        if turn == 10 and run.tasks.phase == "search" and run.search.regions is not None:
            run.stop = "TURN_LIMIT"
            return
        if run.tasks.phase != "search" or run.search.regions is None:
            if not plan_step(run):
                continue
        if turn == 10:
            run.stop = "TURN_LIMIT"
            return
        decision = run.search.next(
            run.tasks, run.images, run.graph_enabled, run.crop_protocol["mode"] == "single"
        )
        run.events.append(dict(event="python_search_decision", turn=turn, **deepcopy(decision)))
        if decision.get("stop"):
            run.stop = decision["stop"]
            return
        candidates = run.crop_candidates(decision["proposal"])
        selected, evidence = run.probes(candidates, 0)
        run.record_selected(
            "Python search proposal only: "
            + canonical({k: decision[k] for k in ["focus_id", "reason"]}),
            selected,
            evidence,
            f"python_crop_{turn}",
        )
        run.tasks.focus_override = None
    reason = run.exploration_stop_reason()
    run.turn = 10
    run.stop = reason or "TURN_LIMIT"
