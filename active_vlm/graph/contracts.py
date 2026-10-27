"""Question-defined value alternatives, without attribute/value dictionaries.

Alternatives are language hypotheses, never observations. This conservative
lexical checker deliberately leaves paraphrases it cannot align unresolved.
"""

import re
from collections import Counter
from decimal import Decimal

FUNCTION_WORDS = set(
    "a an the this that these those is are was were be been being it its they their there he she her his him them whose both only each all any of in on at to for from with and or as by into which what who s image picture region crop cropped candidate question option answer side same used use using made make constructed construction material primary color pose number year determine identify read compare likely corresponds matches contains shows showing visible clearly directly can could would should".split()
)
NEGATIVE = re.compile(r"\b(?:no|not|without|neither|never)\b", re.I)
EXTREMA = {
    "maximum": r"\b(?:highest|tallest|largest|greatest|maximum|peak)\b",
    "minimum": r"\b(?:lowest|shortest|smallest|least|minimum)\b",
}


def tokens(text):
    text = re.sub(r"(?<=\d),(?=\d)", "", text.lower())
    return [
        format(Decimal(t).normalize(), "f") if re.fullmatch(r"-?\d+(?:\.\d+)?", t) else t
        for t in re.findall(r"[a-z]+|-?\d+(?:\.\d+)?", text)
    ]


def forms(token):
    values = {token}
    if len(token) > 4:
        if token.endswith("ing"):
            values |= {token[:-3], token[:-3] + "e"}
        if token.endswith("ed"):
            values |= {token[:-2], token[:-1]}
        if token.endswith("ly"):
            values.add(token[:-2])
        if token.endswith("s"):
            values.add(token[:-1])
    return values


def content(text):
    return (
        set().union(*(forms(t) for t in tokens(text) if t not in FUNCTION_WORDS)) if text else set()
    )


def question_values(question):
    alternatives = []
    for line in question.splitlines():
        m = re.match(r"\s*(?:\(([A-K])\)|([A-K])[.)])\s*(.+)", line)
        if m:
            text = m[3].strip()
            alternatives.append(dict(label=m[1] or m[2], text=text))
    frequency = Counter(
        t
        for a in alternatives
        for t in content(a["text"]) - {"no", "not", "without", "neither", "never"}
    )
    for a in alternatives:
        a["cues"] = sorted(
            t for t in content(a["text"]) if frequency[t] == 1 and not NEGATIVE.fullmatch(t)
        )
        a["negative"] = bool(NEGATIVE.search(a["text"]))
        a["affirmative_cues"] = (
            sorted(content(NEGATIVE.split(a["text"], maxsplit=1)[0]) & set(a["cues"]))
            if a["negative"]
            else a["cues"]
        )
        # "touching" / "not touching" differ by polarity, not a unique noun.
        a["polarity_cues"] = sorted(content(a["text"]) - set(NEGATIVE.findall(a["text"].lower())))
        a["global_absence"] = bool(
            re.search(
                r"(?:image|picture).*(?:not|doesn't|without).*(?:feature|contain|show)|(?:no|not).*present",
                a["text"],
                re.I,
            )
        )
    question_text = question.splitlines()[0] if question.splitlines() else ""
    constraints = [
        name for name, pattern in EXTREMA.items() if re.search(pattern, question_text, re.I)
    ]
    return dict(source="question_and_choices", alternatives=alternatives, constraints=constraints)


def bind_values(nodes, question):
    spec = question_values(question)
    for node in nodes:
        if node["kind"] in {"attribute", "relation", "count"} and len(spec["alternatives"]) >= 2:
            node["value_spec"] = spec
        if node["kind"] == "attribute" and not node.get("attribute"):
            node["attribute"] = node["task"]  # Open task text, not a whitelist category.


def value_signature(node, claim):
    spec = node.get("value_spec")
    if not spec:
        return explicit_open_value(node, claim)
    # A copied choice or answer letter is not a physical observation.
    if re.match(r"\s*(?:this |it )?(?:matches|corresponds|aligns).*\boption\b", claim, re.I):
        return None
    claim = re.split(
        r",?\s*which (?:is|matches) (?:the )?(?:(?:exact|central) )?subject of the question",
        claim,
        maxsplit=1,
        flags=re.I,
    )[0]
    observed = re.split(
        r"\b(?:corresponds? to|matches? option|eliminating|therefore|which is (?:the )?correct answer)\b",
        claim,
        maxsplit=1,
        flags=re.I,
    )[0]
    if any(not re.search(EXTREMA[name], observed, re.I) for name in spec["constraints"]):
        return None
    # A highest visible bar is not necessarily the highest bar in the chart.
    if spec["constraints"] and re.search(
        r"visible (?:portion|part)|among.*visible|only.*(?:few|some) bars", observed, re.I
    ):
        return None
    observed = re.split(r",\s*not\b", observed, maxsplit=1, flags=re.I)[0]
    # Main predicates dominate incidental parentheses (e.g. right side, near the
    # bottom-right). If the value is only parenthesized, retain it (FY 2024).
    main = re.sub(r"\([^)]*\)", "", observed)
    matches = matching_alternatives(spec, main, node)
    if not matches:
        matches = matching_alternatives(spec, observed, node)
    if len(matches) != 1:
        return None
    return ("question_choice", re.sub(r"\s+", " ", matches[0]["text"].lower()))


def matching_alternatives(spec, claim, node):
    # Identifying words (e.g. the van's yellow plate) cannot vote for an
    # alternative about a yellow shop awning.
    # A camera viewpoint is not the object's position in the image. Strip the
    # grammatical viewpoint phrase, rather than special-casing a direction.
    claim = re.sub(
        r"\b(?:from\s+(?:a|the)\s+|its\s+)[^,;.]*\b(?:perspective|viewpoint|view)\b",
        "",
        claim,
        flags=re.I,
    )
    identity = set().union(*(content(t) for t in node["targets"]))
    actual = content(claim) - identity
    negative = bool(NEGATIVE.search(claim))
    matches = []
    for a in spec["alternatives"]:
        if a["global_absence"]:
            continue  # One crop cannot certify scene-wide absence.
        if negative and not a["negative"]:
            continue
        cues = set(a["cues"]) - identity
        if a["negative"] and not negative:
            cues &= set(a["affirmative_cues"])
        polarity_match = (
            not cues
            and negative == a["negative"]
            and bool((set(a["polarity_cues"]) - identity) & actual)
        )
        if actual & cues or polarity_match:
            matches.append(a)
    return matches


def explicit_open_value(node, claim):
    """No choices: accept explicit property predicates, not arbitrary adjectives."""
    if node["kind"] != "attribute":
        return None
    measure = node.get("attribute") or node["task"]
    pattern = re.escape(measure) + r"\s+(?:is|are)\s+(.+?)(?:[.;]|$)"
    found = re.search(pattern, claim, re.I)
    if not found:
        found = re.search(r"\b(?:made of|constructed from)\s+(.+?)(?:[.;]|$)", claim, re.I)
    if found and not NEGATIVE.search(found[1]):
        return ("literal", re.sub(r"\s+", " ", found[1].lower()).strip())
    if re.search(r"\b(?:pose|motion|state|action)\b", measure, re.I):
        values = {
            t
            for t in tokens(claim)
            if t.endswith("ing") and t not in {"showing", "wearing", "carrying", "holding"}
        }
        return ("literal", next(iter(values))) if len(values) == 1 else None
    return None
