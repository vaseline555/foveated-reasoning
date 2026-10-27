"""The user question is the query; visual facts form its prerequisites.

No semantic label, option likelihood or visual evidence is manufactured here.
The final query copies the question, and depends on every terminal planned fact.
"""

from copy import deepcopy


def dependency_errors(steps):
    """Describe invalid edges; never delete or invent one on the model's behalf."""
    if not isinstance(steps, list) or not 1 <= len(steps) <= 11:
        return {}
    errors = {}
    for i, step in enumerate(steps):
        if not isinstance(step, dict):
            return {}
        deps = step.get("requires")
        if (
            not isinstance(deps, list)
            or any(type(d) is not int or not 0 <= d < i for d in deps)
            or len(set(deps)) != len(deps)
        ):
            valid = (
                [d for d in deps if type(d) is int and 0 <= d < i] if isinstance(deps, list) else []
            )
            errors[f"g{i}"] = dict(
                index=i, supplied=deps, allowed=list(range(i)), preserve=sorted(set(valid))
            )
    return errors


def repair_dependencies(steps, replacements):
    """Apply only the requested edge patches to a private copy, then revalidate."""
    errors = dependency_errors(steps)
    if not errors or not isinstance(replacements, dict) or set(replacements) != set(errors):
        raise ValueError("DEPENDENCY_REPAIR_KEYS")
    result = deepcopy(steps)
    for key, error in errors.items():
        deps = replacements[key]
        if (
            not isinstance(deps, list)
            or any(type(d) is not int or d not in error["allowed"] for d in deps)
            or len(set(deps)) != len(deps)
        ):
            raise ValueError("DEPENDENCY_REPAIR_INVALID_EDGES")
        if not set(error["preserve"]).issubset(deps):
            raise ValueError("DEPENDENCY_REPAIR_MUST_PRESERVE_VALID_EDGES")
        result[error["index"]]["requires"] = deps
    return result


def query_region_patch(dag):
    """Refresh the derived query view after a prerequisite region was repaired.

    The query is Python-owned. This geometry follows its terminal prerequisites,
    and supplies no observation, identity confirmation or answer evidence.
    """
    query = dag.nodes[-1]
    if query["status"] in {"observed", "observed_unmapped"}:
        return []
    boxes = [dag.nodes[i]["box"] for i in query["requires"]]
    box = [
        min(b[0] for b in boxes),
        min(b[1] for b in boxes),
        max(b[2] for b in boxes),
        max(b[3] for b in boxes),
    ]
    if box == query["box"]:
        return []
    return [
        dict(
            node_index=query["index"],
            box=box,
            reason="Python-derived union of repaired terminal prerequisite regions; no evidence.",
        )
    ]


def question_plan(steps, question, letters):
    if not isinstance(steps, list) or not 1 <= len(steps) <= 11:
        raise ValueError("QUESTION_REQUIRES_1_TO_11_VISUAL_STEPS")
    if not isinstance(question, str) or not question.strip():
        raise ValueError("QUESTION_QUERY_TEXT_REQUIRED")
    facts = []
    for step in steps:
        if not isinstance(step, dict) or set(step) != {"task", "target", "requires", "box"}:
            raise ValueError("VISUAL_STEP_FIELDS")
        facts.append(dict(deepcopy(step), kind="locate"))
    # Validate untrusted dependencies/geometry before deriving any structural edge.
    from .partition import PartitionDAG

    validator = PartitionDAG(letters)
    validated = validator.validated_nodes(facts, lambda _: None, require_measurement=False)
    for raw, valid in zip(facts, validated, strict=True):
        raw["box"] = valid["box"]
    parents = {i for n in facts for i in n["requires"]}
    sinks = [i for i in range(len(facts)) if i not in parents]
    boxes = [facts[i]["box"] for i in sinks]
    query = dict(
        task=question.strip(),
        target="; ".join(dict.fromkeys(facts[i]["target"] for i in sinks)),
        kind="measure",
        requires=sinks,
        box=[
            min(b[0] for b in boxes),
            min(b[1] for b in boxes),
            max(b[2] for b in boxes),
            max(b[3] for b in boxes),
        ],
    )
    return facts + [query]
