"""Draft role/evidence controller; model, cropping and metering are injected.

No geometry preprocessing or shared mutable episode state. A reported support
state is a model claim, not verified visual truth. See docs/adaptive-design.md.
"""

from __future__ import annotations

import heapq
import math
from collections.abc import Awaitable, Callable, Sequence
from copy import deepcopy
from dataclasses import dataclass, field, replace
from typing import Literal, Protocol

Status = Literal["unresolved", "partial", "supported", "conflicting"]
Box = tuple[float, float, float, float]


class CropError(RuntimeError):
    """Expected crop/tool failure; transport and accounting errors must propagate."""


@dataclass(frozen=True)
class Role:
    id: str
    role: str
    expected_evidence: str
    requires_for_binding: tuple[str, ...] = ()
    importance: float = 1.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "requires_for_binding", tuple(self.requires_for_binding))


@dataclass(frozen=True)
class Claim:
    role_id: str
    text: str
    status: Status = "partial"
    resolves_conflict: bool = False


@dataclass(frozen=True)
class Plan:
    roles: tuple[Role, ...]
    goals: tuple[str, ...]
    original_claims: tuple[Claim, ...] = ()

    def __post_init__(self) -> None:
        for name in ("roles", "goals", "original_claims"):
            object.__setattr__(self, name, tuple(getattr(self, name)))


@dataclass(frozen=True)
class Candidate:
    id: str
    image_key: str
    bbox: Box
    role_ids: tuple[str, ...]
    expected_evidence: str
    branch_id: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "bbox", tuple(self.bbox))
        object.__setattr__(self, "role_ids", tuple(self.role_ids))


@dataclass(frozen=True)
class Report:
    claims: tuple[Claim, ...] = ()
    new_candidates: tuple[Candidate, ...] = ()
    plan_update: Plan | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "claims", tuple(self.claims))
        object.__setattr__(self, "new_candidates", tuple(self.new_candidates))


@dataclass(frozen=True)
class Observation:
    candidate: Candidate | None
    image_key: str
    report: Report
    error: str | None = None
    plan_version: int = 0


MAX_INTERACTION_TURNS = 10


@dataclass(frozen=True)
class Limits:
    max_steps: int = 10
    max_model_calls: int = 24  # Includes one reserved final-reader call.
    max_candidates: int = 32
    branch_steps: int = 1
    progress_bonus: int = 1
    no_progress_steps: int = 1
    max_visits: int = 2

    def __post_init__(self) -> None:
        if (
            any(
                type(v) is not int or v < 1
                for v in (
                    self.max_steps,
                    self.max_candidates,
                    self.branch_steps,
                    self.no_progress_steps,
                    self.max_visits,
                )
            )
            or type(self.max_model_calls) is not int
            or self.max_model_calls < 2
        ):
            raise ValueError("Invalid search/call limits")
        if type(self.progress_bonus) is not int or self.progress_bonus < 0:
            raise ValueError("Invalid progress bonus")


@dataclass
class CaseState:
    question: str
    original_image_key: str
    plan: Plan
    plans: list[Plan] = field(default_factory=list)
    statuses: dict[str, Status] = field(default_factory=dict)
    candidates: dict[str, Candidate] = field(default_factory=dict)
    observations: list[Observation] = field(default_factory=list)
    image_keys: set[str] = field(default_factory=set)
    visits: dict[str, int] = field(default_factory=dict)
    candidate_branches: dict[str, str] = field(default_factory=dict)
    image_branches: dict[str, str] = field(default_factory=dict)
    candidate_steps: dict[str, int] = field(default_factory=dict)
    stale_candidates: set[str] = field(default_factory=set)
    branch_steps: dict[str, int] = field(default_factory=dict)
    branch_allowances: dict[str, int] = field(default_factory=dict)
    branch_stalls: dict[str, int] = field(default_factory=dict)
    steps: int = 0
    model_calls: int = 0
    stop_reason: str = ""
    evidence: dict[str, dict] = field(default_factory=dict)


@dataclass(frozen=True)
class Result:
    answer: str
    state: CaseState


class Backend(Protocol):
    """Each callback represents one logical model call.

    The harness must reserve final-reader dollars and meter physical requests;
    any transport/format retries must be logged and billed separately. The
    controller's logical call/step limits do not themselves enforce a USD cap.
    """

    async def plan(self, question: str, original_image_key: str) -> Plan: ...
    async def propose(self, state: CaseState) -> Sequence[Candidate]: ...
    async def observe(
        self, state: CaseState, candidate: Candidate, cropped_image_key: str
    ) -> Report: ...
    async def finalize(self, state: CaseState) -> str: ...


CropExecutor = Callable[[Candidate], Awaitable[str]]


def _roles(plan: Plan) -> dict[str, Role]:
    roles = {r.id: r for r in plan.roles}
    if (
        not roles
        or len(roles) != len(plan.roles)
        or any(not isinstance(k, str) or not k.strip() for k in roles)
    ):
        raise ValueError("Role IDs must be unique and nonempty")
    if not plan.goals or not set(plan.goals) <= roles.keys():
        raise ValueError("Goals must refer to roles")
    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(key: str) -> None:
        if key in visiting:
            raise ValueError("Cyclic binding dependencies")
        if key in visited:
            return
        visiting.add(key)
        role = roles[key]
        if any(
            not isinstance(x, str) or not x.strip() for x in (role.role, role.expected_evidence)
        ):
            raise ValueError("Roles require a description and expected evidence")
        if (
            isinstance(role.importance, bool)
            or not isinstance(role.importance, (int, float))
            or not math.isfinite(role.importance)
            or role.importance < 0
        ):
            raise ValueError("Invalid role importance")
        for parent in role.requires_for_binding:
            if parent not in roles:
                raise ValueError("Unknown binding dependency")
            visit(parent)
        visiting.remove(key)
        visited.add(key)

    for key in roles:
        visit(key)
    return roles


def supported(state: CaseState, key: str) -> bool:
    roles = {r.id: r for r in state.plan.roles}
    return state.statuses.get(key) == "supported" and all(
        supported(state, parent) for parent in roles[key].requires_for_binding
    )


def answer_ready(state: CaseState) -> bool:
    return all(supported(state, key) for key in state.plan.goals)


def _set_plan(state: CaseState, plan: Plan) -> None:
    roles = _roles(plan)
    old = {r.id: r for r in state.plan.roles} if state.plans else {}
    changed = {key for key, role in roles.items() if old.get(key) != role}
    # A revised referent invalidates downstream bindings as well.
    while True:
        expanded = changed | {
            key for key, role in roles.items() if set(role.requires_for_binding) & changed
        }
        if expanded == changed:
            break
        changed = expanded
    state.statuses = {
        key: "unresolved" if key in changed else state.statuses.get(key, "unresolved")
        for key in roles
    }
    for key in changed | (set(state.evidence) - set(roles)):
        state.evidence.pop(key, None)
    state.stale_candidates.update(
        key
        for key, candidate in state.candidates.items()
        if set(candidate.role_ids) & changed or not set(candidate.role_ids) <= roles.keys()
    )
    state.plan = plan
    state.plans.append(plan)


def _add_candidates(state: CaseState, items: Sequence[Candidate], limits: Limits) -> None:
    roles = _roles(state.plan)
    staged = dict(state.candidates)
    branches = dict(state.candidate_branches)
    for candidate in items:
        if (
            not isinstance(candidate.id, str)
            or not candidate.id.strip()
            or candidate.image_key not in state.image_keys
        ):
            raise ValueError("Candidate has an unknown image reference or empty ID")
        if not candidate.role_ids or not set(candidate.role_ids) <= roles.keys():
            raise ValueError("Candidate has unknown target roles")
        box = candidate.bbox
        if (
            len(box) != 4
            or any(
                isinstance(x, bool)
                or not isinstance(x, (int, float))
                or not math.isfinite(x)
                or not 0 <= x <= 1000
                for x in box
            )
            or box[2] <= box[0]
            or box[3] <= box[1]
        ):
            raise ValueError("Invalid normalized crop coordinates")
        if candidate.id in staged and staged[candidate.id] != candidate:
            raise ValueError("Candidate IDs cannot be rebound to another proposal")
        inherited = state.image_branches.get(candidate.image_key)
        if inherited and candidate.branch_id and candidate.branch_id != inherited:
            raise ValueError("A nested crop cannot reset its source branch budget")
        branch = inherited or candidate.branch_id or candidate.id
        if not isinstance(branch, str) or not branch.strip():
            raise ValueError("Invalid branch ID")
        branches[candidate.id] = branch
        staged[candidate.id] = candidate
    if len(staged) > limits.max_candidates:
        raise ValueError("Candidate limit exceeded")
    state.candidates = staged
    state.candidate_branches = branches
    state.stale_candidates.difference_update(candidate.id for candidate in items)


def _record(state: CaseState, observation: Observation, limits: Limits) -> bool:
    report = observation.report
    if report.plan_update is not None:
        _set_plan(state, report.plan_update)
        observation = replace(observation, plan_version=len(state.plans) - 1)
    roles = _roles(state.plan)
    claims = {c.role_id: c for c in report.claims}
    if len(claims) != len(report.claims) or not claims.keys() <= roles.keys():
        raise ValueError("Duplicate or unknown claim roles")
    if any(
        c.status not in {"unresolved", "partial", "supported", "conflicting"}
        or not isinstance(c.text, str)
        or not c.text.strip()
        for c in claims.values()
    ):
        raise ValueError("Invalid evidence report")
    before = dict(state.statuses)
    # Rebinding an observed answer invalidates its dependent conclusions, even
    # when the model keeps the same role definitions. Re-observed descendants
    # in this report can subsequently become supported again.
    changed = {key for key, claim in claims.items()
               if key in state.evidence and
               (state.evidence[key]["answer"] != claim.text or claim.status != "supported")}
    affected = set(changed)
    while True:
        expanded = affected | {key for key, role in roles.items()
                               if set(role.requires_for_binding) & affected}
        if expanded == affected:
            break
        affected = expanded
    for key in affected - changed:
        state.statuses[key] = "unresolved"
        state.evidence.pop(key, None)
    pending = set(claims)
    # Process prerequisites first even when the model reports claims out of order.
    while pending:
        for key in sorted(pending):
            if set(roles[key].requires_for_binding) & pending:
                continue
            claim = claims[key]
            status = claim.status
            if before.get(key) == "conflicting" and not claim.resolves_conflict:
                status = "conflicting"
            elif status == "supported" and not all(
                supported(state, parent) for parent in roles[key].requires_for_binding
            ):
                # An early dependent observation remains usable, but is not
                # automatically confirmed when an anchor is discovered later.
                status = "partial"
            state.statuses[key] = status
            state.evidence[key] = dict(
                answer=claim.text, status=status, image_key=observation.image_key,
                observation_index=len(state.observations),
                uncertainty="model_claim" if status == "supported" else status,
            )
            pending.remove(key)
            break
    state.observations.append(observation)
    _add_candidates(state, report.new_candidates, limits)
    levels = {"unresolved": 0, "conflicting": 0, "partial": 1, "supported": 2}
    return any(
        levels[value] > levels[before.get(key, "unresolved")]
        for key, value in state.statuses.items()
    )


def goal_closure(state: CaseState) -> set[str]:
    roles = _roles(state.plan)
    needed = set(state.plan.goals)
    pending = list(needed)
    while pending:
        for parent in roles[pending.pop()].requires_for_binding:
            if parent not in needed:
                needed.add(parent)
                pending.append(parent)
    return needed


def evidence_summary(state: CaseState) -> dict:
    """Grounded partial answers, not expected-evidence tasks or truth labels."""
    needed = goal_closure(state)
    roles = _roles(state.plan)
    depth = {}
    def level(key):
        if key not in depth:
            depth[key] = 1 + max((level(p) for p in roles[key].requires_for_binding), default=0)
        return depth[key]
    return dict(goals=list(state.plan.goals),
                dependency_depth=max(level(k) for k in state.plan.goals),
                nodes={k: dict(question=roles[k].expected_evidence,
                               inputs=list(roles[k].requires_for_binding),
                               status=state.statuses.get(k, "unresolved"),
                               evidence=state.evidence.get(k)) for k in sorted(needed)})


def _priority(state: CaseState, candidate: Candidate) -> float:
    roles = _roles(state.plan)
    needed = goal_closure(state)
    weights = {"unresolved": 1.0, "partial": 0.5, "conflicting": 1.5, "supported": 0.0}
    benefit = sum(
        roles[key].importance * weights.get(state.statuses.get(key, "unresolved"), 1.0)
        + sum(key in roles[other].requires_for_binding and not supported(state, other)
              for other in needed)
        for key in candidate.role_ids if key in needed and not supported(state, key)
    )
    # A deterministic information-value proxy; not calibrated probabilities or dollars.
    return benefit / (1 + state.candidate_steps.get(candidate.id, 0))


class AdaptivePolicy:
    """Sequential controller, with all mutable state scoped to one run()."""

    def __init__(self, limits: Limits | None = None) -> None:
        self.limits = limits or Limits()

    async def run(
        self, question: str, original_image_key: str, backend: Backend, executor: CropExecutor
    ) -> Result:
        limits = self.limits
        if not isinstance(original_image_key, str) or not original_image_key.strip():
            raise ValueError("Original image reference is required")
        plan = await backend.plan(question, original_image_key)
        state = CaseState(question, original_image_key, plan, model_calls=1)
        state.image_keys.add(original_image_key)
        _set_plan(state, plan)
        _record(state, Observation(None, original_image_key, Report(plan.original_claims)), limits)
        active: str | None = None
        spent = state.branch_steps
        stalls = state.branch_stalls
        allowances = state.branch_allowances
        cooldown: str | None = None

        while state.steps < min(limits.max_steps, MAX_INTERACTION_TURNS) and not answer_ready(state):
            if state.model_calls >= limits.max_model_calls - 1:
                state.stop_reason = "model_call_limit"
                break
            eligible = [
                c
                for c in state.candidates.values()
                if set(c.role_ids) <= state.statuses.keys()
                and c.id not in state.stale_candidates
                and (
                    state.candidate_branches[c.id] == active
                    or state.visits.get(state.candidate_branches[c.id], 0) < limits.max_visits
                )
                and _priority(state, c) > 0
            ]
            if not eligible:
                state.model_calls += 1
                proposals = await backend.propose(deepcopy(state))
                _add_candidates(state, proposals, limits)
                eligible = [
                    c
                    for c in proposals
                    if state.visits.get(state.candidate_branches[c.id], 0) < limits.max_visits
                    and _priority(state, c) > 0
                ]
                if not eligible:
                    state.stop_reason = "no_candidates"
                    break
            # Rebuild after every observation; dependency edges do not prohibit
            # looking at a dependent candidate before its anchor is grounded.
            available_branches = {state.candidate_branches[c.id] for c in eligible}
            heap = [
                (-_priority(state, c), state.candidate_steps.get(c.id, 0), c.id)
                for c in eligible
                if state.candidate_branches[c.id] != cooldown or len(available_branches) == 1
            ]
            heapq.heapify(heap)
            chosen = state.candidates[heapq.heappop(heap)[2]]
            cooldown = None
            branch = state.candidate_branches[chosen.id]
            if branch != active:
                active = branch
                state.visits[active] = state.visits.get(active, 0) + 1
                spent[active] = stalls[active] = 0
                allowances[active] = limits.branch_steps
            if state.model_calls >= limits.max_model_calls - 1:
                state.stop_reason = "model_call_limit"
                break
            state.steps += 1
            state.candidate_steps[chosen.id] = state.candidate_steps.get(chosen.id, 0) + 1
            spent[active] += 1
            try:
                cropped_key = await executor(chosen)
                if (
                    not isinstance(cropped_key, str)
                    or not cropped_key.strip()
                    or cropped_key in state.image_keys
                ):
                    raise CropError("Executor must return a new, nonempty image key")
            except CropError as error:
                state.observations.append(
                    Observation(
                        chosen, chosen.image_key, Report(), str(error), len(state.plans) - 1
                    )
                )
                progress = False
            else:
                state.image_keys.add(cropped_key)
                state.image_branches[cropped_key] = branch
                state.model_calls += 1
                report = await backend.observe(deepcopy(state), chosen, cropped_key)
                progress = _record(
                    state,
                    Observation(chosen, cropped_key, report, plan_version=len(state.plans) - 1),
                    limits,
                )
            stalls[active] = 0 if progress else stalls[active] + 1
            important = any(r.importance > 1 for r in state.plan.roles
                            if r.id in chosen.role_ids and r.id in goal_closure(state))
            if progress and important:
                allowances[active] = min(
                    limits.branch_steps + limits.progress_bonus, allowances[active] + 1
                )
            if stalls[active] >= limits.no_progress_steps or spent[active] >= allowances[active]:
                cooldown, active = active, None

        if not state.stop_reason:
            state.stop_reason = "evidence_ready" if answer_ready(state) else "step_limit"
        state.model_calls += 1
        answer = await backend.finalize(deepcopy(state))
        return Result(answer, state)
