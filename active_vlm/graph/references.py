"""Resolve explicit crop references; a question definition is never evidence."""

import re

STOP = set(
    "a an the of in on with wearing dressed who is are was has have to from side image picture other and or what which one s".split()
)


ALIASES = dict(
    women="woman",
    ladies="woman",
    lady="woman",
    men="man",
    children="child",
    girl="child",
    boy="child",
    people="person",
    persons="person",
    buses="bus",
    vehicles="vehicle",
    beanies="beanie",
    patterned="",
    short="",
    sleeved="",
    carrying="",
    wears="",
    cloth="clothing",
)


def words(text):
    tokens = re.findall(r"[a-z]+|\d+(?:\.\d+)?", text.lower().replace("-", " "))
    return (
        {ALIASES.get(t, t[:-1] if t.endswith("s") and len(t) > 4 else t) for t in tokens}
        - STOP
        - {""}
    )


def conditional(target):
    return bool(re.search(r"\b(?:with|wearing|carrying|dressed|holding)\b", target, re.I))


def target_matches(target, claim, condition=False):
    target_tokens = words(target)
    if not target_tokens or not target_tokens <= words(claim):
        return False
    if condition and re.search(r"\b(?:with|carrying|holding)\b", target, re.I):
        if not re.search(
            r"\b(?:wearing|carrying|holding|with|dressed|has|backpack on)\b", claim, re.I
        ):
            return False
        # A nearby bag is not a bag worn by this person.
        if re.search(r"\b(?:next to|beside|near|separate|another)\b", claim, re.I):
            return False
    return True


def target_head(target):
    phrase = re.split(
        r"\b(?:with|wearing|dressed|holding|carrying|in)\b", target, maxsplit=1, flags=re.I
    )[0]
    tokens = re.findall(r"[a-z]+", phrase.lower())
    return tokens[-1] if tokens else ""


def other_target(claim, targets):
    # "Not on a different part of the image" is alignment metadata, not a
    # different referent. Only a changed target noun invalidates its binding.
    return any(
        re.search(
            r"\b(?:another|different)\s+(?:\w+\s+){0,3}" + re.escape(target_head(t)) + r"\b",
            claim,
            re.I,
        )
        for t in targets
    )


def absent_target(claim, targets):
    return any(
        re.search(
            r"\b(?:no|without)\s+(?:\w+\s+){0,3}" + re.escape(target_head(t)) + r"\b"
            r"|\b(?:not|cannot|can't)\s+(?:\w+\s+){0,3}(?:show|contain|include)\s+(?:\w+\s+){0,3}"
            + re.escape(target_head(t))
            + r"\b|\b"
            + re.escape(target_head(t))
            + r"\b[^.;]{0,20}\b(?:not|never)\s+(?:visible|present)\b",
            claim,
            re.I,
        )
        for t in targets
    )


def source_references(node, text, claim, inventory):
    """Return exact prior definition quotes for unambiguous short references.

    Definitions only identify the referent. The caller still requires an actual
    candidate-local observation, distinct probe receipts and a measured value.
    No clothing/ownership condition is inferred from "this person".
    """
    if node["kind"] in {"condition", "frame"}:
        return None
    prefix = text[: text.find(claim)]
    sources = []
    for target in node["targets"]:
        if target_matches(target, claim, condition=node["kind"] != "locate"):
            continue
        if conditional(target):
            return None
        head = target_head(target)
        meanings = {frozenset(words(t)) for t in inventory if target_head(t) == head}
        if not head or len(meanings) != 1:
            return None
        reference = re.search(
            r"\b(?:this(?: specific| exact)?|that|same|such|these|the(?: relevant)?|one such)\s+"
            + re.escape(head)
            + r"\b",
            claim,
            re.I,
        )
        if not reference:
            return None
        # Explicit candidate qualifiers must confirm the question's identity.
        # Derive them from the target, rather than a colour/clothing dictionary.
        qualifier = re.match(
            r"\s+(?:in|wearing|with|carrying|holding|dressed in)\s+([^,;.]+)",
            claim[reference.end() :],
            re.I,
        )
        if qualifier and re.search(
            r"\b(?:in|wearing|with|carrying|holding|dressed)\b", target, re.I
        ):
            expected = words(target) - {head}
            if not re.match(r"question\b", qualifier[1], re.I) and not expected <= words(
                qualifier[1]
            ):
                return None
        source = next(
            (
                sentence.strip()
                for sentence in re.split(r"(?<=[.!?])\s+|\n+", prefix)
                if target_matches(target, sentence)
                and re.search(r"\b(?:question|original (?:image|scene|view))\b", sentence, re.I)
                and not re.search(r"\b(?:agent|option|answer)\b", sentence, re.I)
            ),
            None,
        )
        if source is None:
            return None
        sources.append(source)
    return list(dict.fromkeys(sources)) if sources else None
