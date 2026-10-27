"""Bayesian experimental design over a finite visual task DAG.

Likelihoods are VLM estimates, not calibrated truth. All arithmetic, dependencies,
coordinates and rewards are owned by Python; words/attributes have no dictionaries.
"""

import math
from copy import deepcopy

from ..imaging import validate_seed


def probability(x):
    if type(x) not in {int, float} or not math.isfinite(x) or not 0 <= x <= 1:
        raise ValueError("PROBABILITY_MUST_BE_FINITE_IN_0_TO_1")
    return float(x)


def vector(xs, n):
    if not isinstance(xs, list) or len(xs) != n:
        raise ValueError("LIKELIHOOD_DIMENSION_MUST_MATCH_CHOICES")
    return [probability(x) for x in xs]


def posterior(prior, likelihood):
    weights = [p * likelihood_value for p, likelihood_value in zip(prior, likelihood, strict=True)]
    z = math.fsum(weights)
    if z <= 0:
        raise ValueError("OBSERVATION_IMPOSSIBLE_UNDER_CURRENT_MODEL")
    return [w / z for w in weights]


def voi(prior, likelihood):
    """Expected reduction of Bayes 0-1 risk for a binary experiment.

    P(o) max_y P(y|o) = max_y P(y) P(o|y), including zero-probability outcomes.
    This also proves VOI >= 0 by convexity, without simulated future responses.
    """
    yes = max(p * likelihood_value for p, likelihood_value in zip(prior, likelihood, strict=True))
    no = max(
        p * (1 - likelihood_value) for p, likelihood_value in zip(prior, likelihood, strict=True)
    )
    return max(0.0, yes + no - max(prior))


class EvidenceDAG:
    def __init__(self, choices, cost_weight=1.0, expected_pair_usd=0.003):
        if type(choices) is not int or not 2 <= choices <= 11:
            raise ValueError("INVALID_CHOICE_COUNT")
        if any(
            type(x) not in {int, float} or not math.isfinite(x) or x <= 0
            for x in [cost_weight, expected_pair_usd]
        ):
            raise ValueError("INVALID_ECONOMIC_PARAMETERS")
        self.belief = [1 / choices] * choices
        self.cost_weight = cost_weight
        self.expected_pair_usd = expected_pair_usd
        self.nodes = []
        self.revision = 0
        self.observations = []
        self.retired = []
        self.paid_usd = 0.0

    def accept(self, steps, validate_box=None):
        if not isinstance(steps, list) or not 1 <= len(steps) <= 12:
            raise ValueError("PLAN_REQUIRES_1_TO_12_STEPS")
        nodes, text_size = [], 0
        for i, raw in enumerate(steps):
            if not isinstance(raw, dict) or set(raw) != {
                "test",
                "requires",
                "box",
                "positive_if_answer",
                "visibility",
            }:
                raise ValueError(f"STEP_{i}_FIELDS_INVALID")
            if not isinstance(raw["test"], str) or len(raw["test"].strip()) < 5:
                raise ValueError(f"STEP_{i}_MEASUREMENT_MUST_BE_NONEMPTY")
            text_size += len(raw["test"])
            deps = raw["requires"]
            if (
                not isinstance(deps, list)
                or len(set(map(str, deps))) != len(deps)
                or any(type(d) is not int or not 0 <= d < i for d in deps)
            ):
                raise ValueError(f"STEP_{i}_DEPENDENCIES_MUST_BE_UNIQUE_EARLIER_INTEGERS")
            if not isinstance(raw["box"], list) or len(raw["box"]) != 4:
                raise ValueError(f"STEP_{i}_BOX_MUST_HAVE_FOUR_ORIGINAL_NORMALIZED_COORDINATES")
            box = validate_seed(raw["box"])
            if validate_box is not None:
                validate_box(box)
            nodes.append(
                dict(
                    id=f"r{self.revision + 1}g{i + 1}",
                    index=i,
                    test=raw["test"].strip(),
                    requires=list(deps),
                    box=list(box),
                    likelihood=vector(raw["positive_if_answer"], len(self.belief)),
                    visibility=probability(raw["visibility"]),
                    status="pending",
                    citations=[],
                )
            )
        if text_size > 12000 or not any(max(n["likelihood"]) > min(n["likelihood"]) for n in nodes):
            raise ValueError("PLAN_MUST_BE_BOUNDED_AND_CONTAIN_AN_ANSWER_DISCRIMINATING_TEST")
        # Atomic commit: even an invalid late node cannot modify prior graph/state.
        self.retired.extend(deepcopy(self.nodes))
        self.nodes = nodes
        self.revision += 1

    def ancestors(self, index):
        visited = set()

        def visit(i):
            if i in visited:
                return
            for parent in self.nodes[i]["requires"]:
                visit(parent)
            visited.add(i)

        visit(index)
        return sorted(visited)

    def select(self):
        """Score a complete dependency macro-action, execute its first open node.

        A localization test may carry zero immediate answer information; its
        downstream leaf VOI still pays for the prerequisite. No support-count bonus.
        """
        proposals = []
        for leaf in self.nodes:
            if leaf["status"] != "pending":
                continue
            chain = [self.nodes[i] for i in self.ancestors(leaf["index"])]
            if any(n["status"] == "unobserved" for n in chain):
                continue
            pending = [n for n in chain if n["status"] == "pending"]
            visibility = math.prod(n["visibility"] for n in pending)
            information = voi(self.belief, leaf["likelihood"])
            cost = len(pending) * self.expected_pair_usd
            utility = visibility * information - self.cost_weight * cost
            proposals.append(
                dict(
                    leaf=leaf["id"],
                    focus=pending[0]["index"],
                    expected_gain=visibility * information,
                    expected_cost_usd=cost,
                    utility=utility,
                    path=[n["id"] for n in pending],
                )
            )
        best = max(proposals, key=lambda x: (x["utility"], -x["expected_cost_usd"]), default=None)
        return (best if best and best["utility"] > 0 else None), proposals

    def observe(self, index, report, receipt, crop_id, box, actual_cost_usd):
        expected = {
            "outcome",
            "confidence",
            "source",
            "observation",
            "target_description",
            "needs_repair",
            "reason",
        }
        if not isinstance(report, dict) or set(report) != expected:
            raise ValueError("OBSERVATION_FIELDS_INVALID")
        if (
            not isinstance(report["outcome"], str)
            or not isinstance(report["source"], str)
            or report["outcome"] not in {"positive", "negative", "unobserved"}
            or report["source"] not in {"crop", "original", "question", "inference"}
        ):
            raise ValueError("OBSERVATION_ENUM_INVALID")
        confidence = probability(report["confidence"])
        if type(report["needs_repair"]) is not bool or any(
            not isinstance(report[k], str) for k in ["observation", "target_description", "reason"]
        ):
            raise ValueError("OBSERVATION_TYPES_INVALID")
        if sum(len(report[k]) for k in ["observation", "target_description", "reason"]) > 4000:
            raise ValueError("OBSERVATION_TEXT_BUDGET")
        if type(index) is not int or not 0 <= index < len(self.nodes):
            raise ValueError("OBSERVATION_INDEX_INVALID")
        if (
            type(actual_cost_usd) not in {int, float}
            or not math.isfinite(actual_cost_usd)
            or actual_cost_usd < 0
        ):
            raise ValueError("OBSERVATION_COST_INVALID")
        node = self.nodes[index]
        if node["status"] != "pending" or any(
            x["crop_id"] == crop_id and x["test"] == node["test"] for x in self.observations
        ):
            raise ValueError("OBSERVATION_CANNOT_BE_COUNTED_TWICE")
        before = max(self.belief)
        observed = (
            report["source"] == "crop"
            and report["outcome"] != "unobserved"
            and not report["needs_repair"]
            and confidence > 0.5
            and bool(report["observation"].strip())
            and bool(report["target_description"].strip())
        )
        revised = self.belief
        if observed:
            # Confidence tempers the measurement rather than creating a free
            # reward. q=.5 is uninformative; negative is a complementary event.
            likelihood = [
                confidence * value + (1 - confidence) * (1 - value) for value in node["likelihood"]
            ]
            if report["outcome"] == "negative":
                likelihood = [1 - value for value in likelihood]
            try:
                revised = posterior(self.belief, likelihood)
            except ValueError:
                observed = False  # Requires model revision; never divide by zero.
        record = dict(
            node_id=node["id"],
            test=node["test"],
            crop_id=crop_id,
            original_xyxy=list(box),
            receipt_id=receipt,
            reported=deepcopy(report),
            observed=observed,
            belief_before=list(self.belief),
            belief_after=list(revised),
            realized_gain=max(revised) - before,
            actual_cost_usd=actual_cost_usd,
            reward=max(revised) - before - self.cost_weight * actual_cost_usd,
            semantic_truth_verified=False,
        )
        self.belief = list(revised)
        self.paid_usd += actual_cost_usd
        node["status"] = "observed" if observed else "unobserved"
        node["citations"].append(record)
        self.observations.append(record)
        return record
