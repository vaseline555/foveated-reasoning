"""Question-derived categorical tests; no attribute dictionaries or likelihoods."""

from ..baselines.prompts import array, obj
from .prompts import tool

PLAN = """Construct the main search path from question_spine FIRST:
identify the questioned referents -> resolve their identifying relations/order ->
locate the relevant object/part -> measure the property/relation the QUESTION asks.
Then use option_hypotheses ONLY to enrich the final discrimination step, preserving
that referent and the question's meaning. For example, candidate color names tell
you which colors to distinguish ON THAT OBJECT, not which colored objects to find
elsewhere. Numbers/units, materials, actions and relations work in exactly the same
question-derived way; no fixed vocabulary is required.
Prefer one shared crop and one joint observation to compare alternatives. Add a
further task/crop only when unresolved alternatives require a different visible
part, scale, or relationship context. Never add a paid crop per option by default.
Option mentions are hypotheses, never observed facts, identity evidence or gain.
If options clarify an ambiguous question term, use their meaning to define the
requested measurement while preserving the target. Preserve the reference frame:
image-relative position, object-facing direction and physical size are different.
If options enumerate the objects to compare, locating all of them is necessary;
otherwise the question determines the objects, and choices only refine the test.
If options offer no additional visual distinction, keep the question-only path.
For embedded questions, first read the question/options, then apply this sequence.
Preserve exact referents (clothing, ownership, ordinal and frame of reference).
Plan the concrete visible facts needed to answer the question: identify the target,
then inspect its requested property or relation. All steps are visual prerequisites;
Python appends the final query copied from the question and supplies its choices.
Do NOT generate node kinds, answer partitions, probabilities, option groups or
one yes/no task per choice. Include only necessary identity/context/property facts.
requires contains earlier zero-based indices; intermediate identity/relationship
steps are prerequisites. Keep only steps necessary to answer this question.
The first step has index0 and MUST have requires=[]. A node cannot require itself.
For node index i, use only earlier integers0..i-1, or[] when independent.
Each target is a full stable referring expression. A nearby object is not that target.
box gives two opposite ORIGINAL-image corners in0..1000 (Python orders corners).
For small distant objects propose a tight magnified crop including identifying
context; a whole-image crop does not increase visibility. Related steps may share
one crop and be observed together. Distinct people must not silently exchange roles.
For image-embedded questions first read the question/options into reading; for
ordinary questions reading=null. Return the tool only."""

OBSERVE = """Read the bound tasks using ONLY the CURRENT CROP as visual evidence.
Only this crop is attached. The native FOVEA report and past observations are fallible
text, not proof that a target is visible here. Original-image coordinates provide
a reference frame, not unseen pixels or evidence about objects outside this crop.
For each bound node, first state concrete visible identity_evidence and observation,
then compare that observed fact against the supplied option_hypotheses. Check the
alternatives jointly on the QUESTION'S bound object, not globally in the scene.
If two alternatives remain visually indistinguishable, use unresolved and explain
what part/scale/context is missing so the next view can target that gap.
Use crop_geometry to relate a visible target to the ORIGINAL image reference
frame when the question/options require it; geometry alone proves no identity.
A cropped object filling the view does not establish it is largest in the scene.
The mere presence of any candidate word/color elsewhere cannot support this node.
Resolve the exact specified person/object and any distinguishing clothing/ownership/
ordinal before measuring its property. Missing target or identifying context means
target_visible=false,resolved=false; it NEVER means a negative answer.
All nodes use the SAME fact fields. Never put group, answer or node_index inside
a node. Separately set top-level answer to an allowed letter ONLY if the bound
question-query is resolved by concrete facts; otherwise answer=null. Python binds
that letter only to the final query, never to identity/location nodes. Read units exactly.
Do not choose a letter because a target exists. Do not assume a failed test excludes
an answer. A clear observation of another object is not evidence about this target.
You may resolve a descendant in the same crop if its dependencies are visibly resolved
in this batch. Return a keyed object g<index> for the bound nodes; do not regenerate node IDs. Source is crop only if seen
in this crop. Use short phrases: identity <=192 characters, observation <=384,
reason <=192. State only values actually needed to resolve the question/options;
never extrapolate a numeric series or transcribe an entire table. Preserve units
and distinguishing labels. A nearby/similar label is not the requested label.
Saying values are readable is not a reading: state the actual visible value, or
resolved=false and identify the missing row/column/context. Return no long reasoning."""

REPAIR = """Repair search regions for the FIXED unresolved tasks and referents.
Do not change the question, options, tasks, dependencies, or object identity. Use
actual previous observations/failure reasons. Return new tight ORIGINAL-image
opposite-corner boxes in0..1000 for the needed unresolved nodes. If a target was too
small, zoom its likely region while retaining identifying context. Do not repeat
the whole image or an already tested region. A correct object observation must not
be replaced by a neighboring object. Region proposals are hints, never evidence.
Return only region patches and the concrete reason for each change."""


def plan_tool(embedded):
    from .prompts import plan_tool as old_tool

    reading = old_tool(embedded)["function"]["parameters"]["properties"]["reading"]
    step = obj(
        dict(
            task=dict(type="string"),
            target=dict(type="string"),
            requires=dict(
                array(dict(type="integer", minimum=0, maximum=10), 0, 10),
                description="Earlier zero-based step indices only. First step=[]; never self.",
            ),
            box=array(dict(type="number", minimum=0, maximum=1000), 4, 4),
        )
    )
    return tool("design_partition", dict(reading=reading, steps=array(step, 1, 11)))


DEPENDENCY_REPAIR = """Repair only the invalid dependency lists in the supplied plan.
The plan's task text, targets, geometry and valid edges are fixed. Each g<index>
lists its supplied invalid edges, allowed earlier indices, and edges to preserve.
Return only the requested g<index> lists. The first node has no earlier node, so
its list must be[]. Never require self or a later node; do not guess visual facts.
The image is not needed for this structural repair. No extra explanation."""


def dependency_tool(errors):
    fields = {}
    for key, error in errors.items():
        item = (
            dict(type="integer", enum=error["allowed"])
            if error["allowed"]
            else dict(type="integer")
        )
        fields[key] = array(item, 0, len(error["allowed"]))
    return tool("repair_plan_dependencies", dict(replacements=obj(fields)))


def observation_fields():
    return dict(
        target_visible=dict(type="boolean"),
        identity_evidence=dict(type="string", maxLength=192),
        observation=dict(type="string", maxLength=384),
        source=dict(type="string", enum=["crop", "original", "inference", "question"]),
        resolved=dict(type="boolean"),
        reason=dict(type="string", maxLength=192),
    )


def observe_tool(nodes, letters):
    # A common fact schema avoids per-node ID/kind/answer-format generation.
    return tool(
        "observe_partition",
        dict(
            observations=obj({f"g{n['index']}": obj(observation_fields()) for n in nodes}),
            answer=dict(anyOf=[dict(type="string", enum=list(letters)), dict(type="null")]),
        ),
    )


def bind_observations(data, nodes, answer=None, letters=None):
    expected = {f"g{n['index']}": n for n in nodes}
    if not isinstance(data, dict) or set(data) - set(expected):
        raise ValueError("OBSERVATION_BOUND_KEYS")
    if letters is None:
        letters = [g[0] for n in nodes if n["kind"] == "measure" for g in n["groups"]]
    # The auxiliary choice is optional. Invalid choices cannot supply a constraint
    # and must not discard otherwise well-formed observations of the actual crop.
    answer = answer if isinstance(answer, str) and answer in letters else None
    rows = []
    for key, node in expected.items():
        if key not in data:
            row = dict(
                target_visible=False,
                identity_evidence="",
                observation="",
                source="inference",
                resolved=False,
                reason="No observation returned for this bound node.",
            )
        else:
            row = data[key]
            if not isinstance(row, dict) or set(row) != set(observation_fields()):
                raise ValueError("OBSERVATION_FACT_FIELDS")
        # Choice mapping is independent of facts. An absent answer remains None;
        # never infer it by scanning possibly ungrounded natural-language prose.
        group = letters.index(answer) if node["kind"] == "measure" and answer is not None else None
        rows.append(dict(row, node_index=node["index"], group=group))
    return rows


def repair_tool(indices):
    patch = obj(
        dict(
            node_index=dict(type="integer", enum=indices),
            box=array(dict(type="number", minimum=0, maximum=1000), 4, 4),
            reason=dict(type="string"),
        )
    )
    return tool("repair_regions", dict(patches=array(patch, 1, len(indices))))
