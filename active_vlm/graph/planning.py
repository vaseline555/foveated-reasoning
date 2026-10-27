"""Small question compiler; unsupported text requires explicit visual tasks."""

import re

from .contracts import question_values
from .evidence import clean_target, conditional

KINDS = ("locate", "condition", "attribute", "frame", "relation", "count")
MAX_TARGETS = 2


def plan_limits(question, multiple_targets=False):
    """Allow localization/condition/value per choice, plus comparison context."""
    choices = len(question_values(question)["alternatives"])
    maximum = min(40, max(16, 3 * choices + 4))
    return dict(
        max_steps=maximum,
        # Comparisons may involve several objects in one observation. Bound
        # them by the same question-sized resource budget as the whole plan,
        # rather than imposing a binary relation on every visual task.
        max_targets=maximum if multiple_targets else MAX_TARGETS,
        # Preserve identifying qualifiers verbatim. The total text budget is a
        # resource guard, not a semantic limit on one object description.
        max_text_chars=min(32768, max(4096, len(question) * 8 + maximum * 256)),
    )


class PlanError(ValueError):
    def __init__(self, path, code, expected):
        self.detail = dict(path=path, code=code, expected=expected)
        super().__init__(f"{path}: {code}; expected {expected}")


def compile_question(question, embedded=False, open_questions=False):
    nodes = []

    def add(kind, task, targets=(), deps=(), **extra):
        node = dict(
            id=f"g{len(nodes) + 1}",
            kind=kind,
            task=task,
            targets=list(targets),
            depends_on=list(deps),
            status="unresolved",
            citations=[],
            **extra,
        )
        nodes.append(node)
        return node["id"]

    def locate(target):
        target = clean_target(target)
        loc = add("locate", f"Locate {target}; inspect a crop that actually contains it.", [target])
        if conditional(target):
            return add(
                "condition",
                f"Confirm the identifying clothing/ownership condition of {target}, rather than a nearby object.",
                [target],
                [loc],
            )
        return loc

    if embedded:
        add(
            "read_task",
            "Read the actual question and choices in the original image using read_question before defining visual targets.",
        )
        return nodes
    text = re.sub(
        r"\s*(?:Options|Choices)\s*:\s*$", "", question.split("\n")[0], flags=re.I
    ).strip()
    # A compound ordinal attribute needs a model plan: the attribute regex below
    # must not swallow its population/filter/order as one long object name.
    if re.search(
        r"\b(?:first|second|third|fourth|fifth|\d+(?:st|nd|rd|th))\b.*\bfrom (?:the )?(?:left|right)\b",
        text,
        re.I,
    ):
        add(
            "inspect",
            "Define the eligible population, requested order and attribute of the selected object as separate dependent goals.",
        )
        return nodes
    # V* and ordinary attribute questions: target localization precedes reading.
    match = re.match(r"What (?:is|are) the (color|pose|material) of (.+?)[?]?$", text, re.I)
    if not match:
        match = re.match(r"What (color|pose|material) (?:is|are) (.+?)[?]?$", text, re.I)
    if not match:
        match = re.match(
            r"Is the (color) of (.+?) (?:red|blue|yellow|green|black|white) or .+?[?]?$", text, re.I
        )
    if match:
        attr, target = match[1].lower(), clean_target(match[2])
        root = locate(target)
        add(
            "attribute",
            f"Read the {attr} of that same identified target.",
            [target],
            [root],
            attribute=attr,
        )
        return nodes
    match = re.match(
        r"Is (.+?) (?:on the left or right side of|above or below) (.+?)[?]?$", text, re.I
    )
    if match:
        targets = [clean_target(match[1]), clean_target(match[2])]
        deps = [locate(t) for t in targets]
        add(
            "relation",
            "Compare both identified targets in a common view; use the question coordinate frame.",
            targets,
            deps,
        )
        return nodes
    match = re.match(r"Which one is closer to the camera, (.+?) or (.+?)[?]?$", text, re.I)
    if match:
        targets = [clean_target(match[1]), clean_target(match[2])]
        deps = [locate(t) for t in targets]
        add("relation", "Compare depth to the camera for both identified targets.", targets, deps)
        return nodes
    # Read-image question after read_question may use this ordinary phrasing.
    match = re.search(r"(?:relative position|position) of (.+?) to (.+?)[?]?$", text, re.I)
    if match:
        targets = [clean_target(match[1]), clean_target(match[2])]
        deps = [locate(t) for t in targets]
        add("relation", "Compare the two located targets in a common view.", targets, deps)
        return nodes
    match = re.match(r"From the (?:viewpoint|perspective) of (.+?), (.+)$", text, re.I)
    if match:
        ref = clean_target(match[1])
        rest = match[2]
        target = re.search(r"other (child|person),? who is (.+?)[?]?$", rest, re.I)
        if target:
            target = f"{target[1]} {target[2]}"
        else:
            found = re.search(r"(?:is|of) (the .+?) (?:located|standing)[?]?$", rest, re.I)
            target = found[1] if found else None
        if target:
            target = clean_target(target)
            ref_id = locate(ref)
            target_id = locate(target)
            frame = add(
                "frame",
                f"Establish the facing direction of {ref}; image right is not automatically the subject right.",
                [ref],
                [ref_id],
            )
            add(
                "relation",
                "Transform both positions into the reference subject frame, then compare.",
                [ref, target],
                [frame, target_id],
                perspective=True,
            )
            return nodes
    match = re.match(r"What is the number of (.+?) in the image", text, re.I)
    if match:
        target = clean_target(match[1])
        loc = locate(target)
        add(
            "count",
            f"Inspect all relevant regions and count {target}; one crop does not establish scene-wide absence.",
            [target],
            [loc],
        )
        return nodes
    ordinal = re.search(r"from (?:the )?(left|right)", text, re.I)
    qualifier = re.search(r"(?:is|are) ((?:wearing|carrying|holding) .+?)[?]?$", text, re.I)
    if ordinal and qualifier and re.search(r"\b(?:person|people|man|woman)\b", text, re.I):
        loc = locate("people")
        eligible = clean_target("person " + qualifier[1])
        condition = add(
            "condition", f"Identify the person satisfying {qualifier[1]}.", [eligible], [loc]
        )
        frame = add("frame", "Use the left/right order specified by the question.")
        nodes[-1]["status"] = "specified_by_question"
        add(
            "count",
            "Enumerate the eligible objects in order before returning an ordinal.",
            [eligible],
            [loc, condition, frame],
            ordinal=True,
            direction=ordinal[1].lower(),
        )
        return nodes
    if open_questions:
        # These are grammatical decompositions of the supplied question, not
        # object/value dictionaries or visual observations. Keep the full
        # question in the judgement task so preambles and qualifiers survive.
        relation = re.search(
            r"What (?:is|are) the relative position of (.+?) and (.+?)[?]?$", text, re.I
        )
        where = re.search(r"Where (?:is|are) (.+?)(?: in the (?:image|picture))?[?]?$", text, re.I)
        attribute = re.search(r"What (?:is|are) the (.+?) of (.+?)[?]?$", text, re.I)
        if relation:
            targets = [clean_target(relation[1]), clean_target(relation[2])]
            deps = [locate(t) for t in targets]
            add("relation", text, targets, deps)
            return nodes
        if where or attribute:
            target = clean_target(where[1] if where else attribute[2])
            root = locate(target)
            add(
                "attribute",
                text,
                [target],
                [root],
                attribute="position" if where else attribute[1],
            )
            return nodes
    add(
        "inspect",
        "Call define_goals to create a valid visual task graph before cropping or answering. This placeholder is not a plan.",
    )
    return nodes


def normalize_root_fields(steps):
    """An omitted locate dependency means an independent original-image search.

    This default applies to every locate step, not just step zero. Explicit
    dependencies are preserved. Judgement omissions and invalid edges require
    model correction; no target, dependency edge or visual fact is invented.
    """
    from copy import deepcopy

    result, changes = deepcopy(steps), []
    if isinstance(result, list):
        for i, step in enumerate(result):
            if (
                isinstance(step, dict)
                and set(step) == {"kind", "task", "targets"}
                and step["kind"] == "locate"
            ):
                step["requires"] = []
                changes.append(dict(path=f"steps[{i}].requires", value=[]))
    return result, changes
