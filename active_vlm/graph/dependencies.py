"""Bounded dependency-only correction; task text and targets stay immutable."""

from copy import deepcopy

from ..baselines import prompts
from .planning import KINDS, PlanError


class DependencyRepair:
    def __init__(self, steps, indices):
        self.steps = deepcopy(steps)
        self.indices = indices

    @classmethod
    def for_plan(cls, steps, limits):
        """Offer edge repair only for otherwise well-shaped, bounded tasks."""
        if not isinstance(steps, list) or not 1 <= len(steps) <= limits["max_steps"]:
            return None
        indices, text_chars = [], 0
        for i, step in enumerate(steps):
            if (
                not isinstance(step, dict)
                or set(step) - {"kind", "task", "targets", "requires"}
                or not {"kind", "task", "targets"} <= set(step)
                or step["kind"] not in KINDS
                or not isinstance(step["task"], str)
                or len(step["task"].strip()) < 5
                or not isinstance(step["targets"], list)
                or not 1 <= len(step["targets"]) <= limits["max_targets"]
                or any(not isinstance(t, str) or not t.strip() for t in step["targets"])
            ):
                return None
            text_chars += len(step["task"]) + sum(map(len, step["targets"]))
            deps = step.get("requires")
            if (
                not isinstance(deps, list)
                or any(type(d) is not int or not 0 <= d < i for d in deps)
                or (step["kind"] != "locate" and not deps)
            ):
                if i == 0 and step["kind"] != "locate":
                    return None  # No possible earlier localization: replan.
                indices.append(i)
        if not indices or text_chars > limits["max_text_chars"]:
            return None
        return cls(steps, indices)

    def tool(self):
        fields = {}
        for i in self.indices:
            fields[str(i)] = prompts.array(
                dict(type="integer", minimum=0, maximum=max(0, i - 1)),
                0 if self.steps[i]["kind"] == "locate" else 1,
                i,
            )
            fields[str(i)]["uniqueItems"] = True
        return dict(
            type="function",
            function=dict(
                name="repair_dependencies",
                description=(
                    "Correct only the listed zero-based requires fields of the rejected plan. "
                    "Use earlier integer step indices; never self/forward indices. "
                    "Task text and targets remain fixed. This adds no visual evidence."
                ),
                parameters=prompts.obj(dict(requires=prompts.obj(fields))),
            ),
        )

    def apply(self, patch):
        expected = {str(i) for i in self.indices}
        if not isinstance(patch, dict) or set(patch) != expected:
            raise PlanError("requires", "INVALID_DEPENDENCY_PATCH", sorted(expected))
        steps = deepcopy(self.steps)
        for i in self.indices:
            deps = patch[str(i)]
            if (
                not isinstance(deps, list)
                or len(deps) > i
                or any(type(d) is not int or not 0 <= d < i for d in deps)
                or len(set(deps)) != len(deps)
                or (steps[i]["kind"] != "locate" and not deps)
            ):
                raise PlanError(
                    f"requires.{i}",
                    "INVALID_DEPENDENCY_PATCH",
                    f"unique earlier integer indices 0..{i - 1}; non-locate needs a dependency",
                )
            steps[i]["requires"] = list(deps)
        return steps
