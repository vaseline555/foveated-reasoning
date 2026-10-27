"""Conservative, cited observations from unchanged native FOVEA probe text."""

import re

from .contracts import value_signature as question_value
from .references import (
    absent_target,
    conditional,
    other_target,
    source_references,
    target_head,
    target_matches,
    words,
)


def clean_target(text):
    return re.sub(r"\s+", " ", re.sub(r"[?.]+$", "", text)).strip(" ,")


def overlap(a, b):
    intersection = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(
        0, min(a[3], b[3]) - max(a[1], b[1])
    )

    def area(c):
        return max(0, c[2] - c[0]) * max(0, c[3] - c[1])

    return intersection / (area(a) + area(b) - intersection) if intersection else 0.0


def scoped_claims(text, targets=()):
    """Keep affirmative candidate-crop clauses; a question mention is not evidence.

    Final No may mean the crop alone cannot answer the WHOLE question. It does
    not erase a preceding explicit observation of one required object.
    """
    local = False
    for sentence in re.split(r"(?<=[.!?])\s+|\n+", text or ""):
        for clause in re.split(r";\s*|\b(?:but|however|whereas|although)\b", sentence, flags=re.I):
            # A purpose/relevance statement is not a clothing, identity or
            # spatial observation. Preserve the concrete leading assertion as
            # an exact substring, never its inferred downstream conclusion.
            clause = re.split(
                r",?\s+\b(?:allowing|enabling)\b|[,;]?\s+\bwhich (?:allows?|enables?)\b|[,;]?\s+\band thus\b"
                r"|,\s+which\b(?=[^.;]*\b(?:needed|required|sufficient|necessary)\b[^.;]*\bto (?:identify|determine|infer|answer)\b)",
                clause,
                maxsplit=1,
                flags=re.I,
            )[0]
            lower = clause.lower()
            if re.search(r"(?:agent[’']s|tool args|parent image)", lower) and not re.search(
                r"candidate (?:cropped |crop |region)", lower
            ):
                local = False
                yield None
                continue
            original = re.search(r"\boriginal (?:image|scene|view)\b", lower)
            crop = re.search(
                r"(?:the )?(?:candidate )?(?:cropped|zoomed)(?:[- ]in)? (?:region|image|view|area|section)",
                lower,
            )
            if original:
                local = False
                yield None
                if not crop:
                    continue
                if original.start() < crop.start():
                    # Do not relabel an original-image claim as crop evidence
                    # merely because the same sentence later mentions a crop.
                    clause = clause[crop.start() :]
                    lower = clause.lower()
                else:
                    # "The crop is in the original image's upper left,
                    # while X is elsewhere" never says X is in the crop.
                    clause = re.split(
                        r",?\s+\b(?:while|whereas)\b", clause, maxsplit=1, flags=re.I
                    )[0]
                    lower = clause.lower()
            explicit = bool(
                re.search(
                    r"(?:cropped|zoomed)(?:[- ]in)? (?:region|image|view|area|section)", lower
                )
            )
            if explicit:
                local = local or bool(
                    re.search(
                        r"\b(?:show(?:s|ing)?|contains?|includes?|features?|depicts?|isolates?|displays?|see|visible|focuses?)\b|\bis a (?:close[- ]up|zoomed[- ]in view)\b|\b(?:is|are) (?:located|positioned|facing)\b",
                        lower,
                    )
                )
            if not (
                explicit
                and local
                or local
                and re.match(r"\s*(?:the |this |he |she |it |from )", lower)
            ):
                continue
            if re.search(
                r"\b(?:would|could|may|might|maybe|likely|possibly|uncertain|unclear|appears?)\b",
                lower,
            ):
                yield None
                continue
            if re.search(
                r"\b(?:too blurry|lacks? (?:any )?(?:clear|discernible|useful) (?:visual )?(?:features?|information))\b",
                lower,
            ):
                yield None
                continue
            if re.search(
                r"\b(?:question (?:asks?|requires?|is)|to answer|requires?|need)\b", lower
            ):
                continue
            if re.search(
                r"\b(?:information|sufficient|relevant|relevance)\b", lower
            ) and not re.search(r"\b(?:show(?:s|ing)?|isolates?|displays?|features?)\b", lower):
                continue
            if re.search(
                r"\b(?:allows?|enables?|helps?|possible|able)\b.{0,45}\b(?:identify|determine|infer|recognize|establish|answer)\b|\b(?:to identify|to determine|to infer)\b",
                lower,
            ):
                continue
            if re.search(
                r"\b(?:not|cannot|can't|without|missing|absent)\b.{0,35}\b(?:show|include|contain|visible|present)\b|\b(?:not|never) visible\b",
                lower,
            ):
                if targets and not absent_target(clause, targets):
                    continue  # Missing irrelevant areas/body details is not target absence.
                local = False
                yield None
                continue
            if re.search(r"\b(?:suggests?|infer|inferred|hypothetically|assuming)\b", lower):
                yield None
                continue
            yield clause.strip()


def local_claims(text):
    return [claim for claim in scoped_claims(text) if claim is not None]


def supports(node, claim):
    if not all(
        target_matches(t, claim, condition=node["kind"] != "locate") for t in node["targets"]
    ):
        return False
    kind = node["kind"]
    if re.search(r"\b(?:no|not|never|without)\b", claim, re.I):
        if kind == "frame":
            return False
        if kind in {"locate", "condition"}:
            affirmative = re.split(r"\b(?:no|not|never|without)\b", claim, maxsplit=1, flags=re.I)[
                0
            ]
            if not all(
                target_matches(t, affirmative, condition=kind == "condition")
                for t in node["targets"]
            ):
                return False
    if kind == "locate":
        return True
    if kind == "condition":
        return any(conditional(t) for t in node["targets"]) and bool(
            re.search(
                r"\b(?:wearing|carrying|holding|with|dressed|has|in pink|in white|in blue)\b",
                claim,
                re.I,
            )
        )
    if kind == "attribute":
        return question_value(node, claim) is not None
    if kind == "frame":
        return bool(re.search(r"\b(?:facing|looking|turned|orientation)\b", claim, re.I))
    if kind == "relation":
        if node.get("perspective") and not re.search(
            r"\b(?:perspective|viewpoint|facing)\b", claim, re.I
        ):
            return False
        if node.get("value_spec"):
            return question_value(node, claim) is not None
        return bool(
            re.search(
                r"\b(?:left|right|above|below|closer|farther|farthest|behind|front|inside)\b",
                claim,
                re.I,
            )
        )
    if kind == "count":
        if node.get("ordinal"):
            return bool(
                re.search(
                    r"\b(?:first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth|\d+(?:st|nd|rd|th))\b",
                    claim,
                    re.I,
                )
            ) and bool(re.search(r"\bfrom (?:the )?" + node["direction"] + r"\b", claim, re.I))
        if not re.search(r"\b(?:all|total|entire|complete)\b", claim, re.I):
            return False  # A local count is not the full-scene count.
        if node.get("value_spec"):
            return question_value(node, claim) is not None
        return bool(
            re.search(r"\b(?:\d+|one|two|three|four|five|six|seven|eight|nine|ten)\b", claim, re.I)
        )
    return False  # Unknown/reading tasks never become completed from relevance Yes.


def value_signature(node, claim):
    """Require agreement on the observation, not just two affirmative sentences."""
    kind = node["kind"]
    if kind == "attribute":
        return question_value(node, claim)
    if kind == "frame":
        values = set(
            re.findall(r"\b(?:camera|viewer|forward|away|left|right|backward)\b", claim.lower())
        )
        values = {"camera" if v == "viewer" else v for v in values}
        return next(iter(values)) if len(values) == 1 else None
    if kind == "relation":
        if node.get("value_spec"):
            return question_value(node, claim)
        values = set(
            re.findall(
                r"\b(?:left|right|above|below|closer|farther|farthest|behind|front|inside)\b",
                claim.lower(),
            )
        )
        if {"left", "right"} <= values or {"above", "below"} <= values:
            return None
        return tuple(sorted(values)) if values else None
    if kind == "count":
        if node.get("value_spec") and not node.get("ordinal"):
            return question_value(node, claim)
        if node.get("ordinal"):
            values = set(
                re.findall(
                    r"\b(?:first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth|\d+(?:st|nd|rd|th))\b",
                    claim.lower(),
                )
            )
            return next(iter(values)) if len(values) == 1 else None
        values = set(
            re.findall(
                r"\b(?:\d+|one|two|three|four|five|six|seven|eight|nine|ten)\b", claim.lower()
            )
        )
        return next(iter(values)) if len(values) == 1 else None
    return "target_reported" if kind in {"locate", "condition"} else None


def evidence_units(node, text, inventory=None):
    """Bind short references to a preceding explicit crop-local target claim.

    Both the value quote and its target context remain verbatim, separately
    cited. A different explicit subject clears the context instead of borrowing
    another object's attribute.
    """
    context, context_sources = None, []
    inventory = inventory or node["targets"]
    heads = {target_head(t) for t in node["targets"]} - {""}
    task_terms = words(node["task"]) - set().union(*(words(t) for t in node["targets"]))
    claims = list(scoped_claims(text, node["targets"]))
    coverage = next(
        (
            c
            for c in claims
            if c
            and re.search(r"\b(?:all|entire|complete|full)\b", c, re.I)
            and (heads | task_terms) & words(c)
            and not re.search(r"\b(?:not|missing|necessary|sufficient)\b", c, re.I)
        ),
        None,
    )
    for claim in claims:
        if claim is None:
            context, context_sources = None, []
            continue
        if other_target(claim, node["targets"]) or absent_target(claim, node["targets"]):
            context, context_sources = None, []
            continue
        coreferenced = False
        direct = all(
            target_matches(t, claim, condition=node["kind"] != "locate") for t in node["targets"]
        )
        if direct:
            context, context_sources = claim, []
        elif sources := source_references(node, text, claim, inventory):
            context, context_sources = claim, sources
            direct = coreferenced = True
        elif re.search(r"(?:cropped|zoomed).*(?:shows?|contains?|includes?)", claim, re.I):
            if not (
                context
                and (
                    heads & words(claim)
                    or re.search(
                        r"\b(?:subject|question|information|detail|material)\b", claim, re.I
                    )
                )
            ):
                context = None
        short = bool(re.match(r"\s*(?:the |this |it |its |he |she |from )", claim, re.I))
        reference = bool(
            (heads | (task_terms if node["kind"] == "attribute" else set())) & words(claim)
        ) or bool(re.match(r"\s*(?:this |it |its |he |she |the visible )", claim, re.I))
        if not direct and not (context and short and reference):
            if re.match(r"\s*(?:the |a |an |there )", claim, re.I):
                context = None
            continue
        if node["kind"] in {"locate", "condition", "frame"}:
            if supports(node, claim) or (
                node["kind"] == "locate"
                and coreferenced
                and not re.search(r"\b(?:no|not|never|without)\b", claim, re.I)
            ):
                yield dict(
                    signature=value_signature(node, claim),
                    quote=claim,
                    context_quote=None,
                    source_quote=context_sources[0] if context_sources else None,
                    reference_quotes=context_sources,
                )
            continue
        if node.get("perspective") and not re.search(
            r"\b(?:perspective|viewpoint|facing)\b", claim, re.I
        ):
            continue
        if (
            node["kind"] == "count"
            and not node.get("ordinal")
            and not re.search(r"\b(?:all|total|entire|complete)\b", claim, re.I)
        ):
            continue
        # Attribute evidence must not come from a neighbouring object's tail.
        value_clause = (
            re.split(r"\b(?:next to|beside|in the background)\b", claim, maxsplit=1, flags=re.I)[0]
            if node["kind"] == "attribute"
            else claim
        )
        signature = value_signature(node, value_clause)
        if node.get("value_spec", {}).get("constraints") and not coverage:
            continue  # A local maximum is not a maximum over the full comparison set.
        relation_view = relation_coverage(node, signature, claims)
        if relation_view is False:
            continue
        if signature is not None:
            yield dict(
                signature=signature,
                quote=claim,
                context_quote=context if context != claim else None,
                source_quote=context_sources[0] if context_sources else None,
                reference_quotes=context_sources,
                coverage_quote=coverage
                if node.get("value_spec", {}).get("constraints")
                else relation_view,
            )


def relation_coverage(node, signature, claims):
    """A partial body crop cannot certify a whole-body absence of contact."""
    if node["kind"] != "relation" or not signature or signature[0] != "question_choice":
        return None
    alternative = next(
        (
            a
            for a in node.get("value_spec", {}).get("alternatives", [])
            if re.sub(r"\s+", " ", a["text"].lower()) == signature[1]
        ),
        None,
    )
    if not (
        alternative
        and alternative["negative"]
        and re.search(r"\b(?:at all|any part|anywhere)\b", alternative["text"], re.I)
        and re.search(r"\bbody\b", node["task"], re.I)
    ):
        return None
    affirmative = [c for c in claims if c]
    # Explicit partial views dominate contradictory "full body from waist down".
    if any(
        re.search(r"\b(?:waist down|lower (?:part|body)|only.*legs)\b", c, re.I)
        for c in affirmative
    ):
        return False
    return next(
        (
            c
            for c in affirmative
            if re.search(
                r"\b(?:(?:full|entire|whole) body|head to (?:toe|feet)|top of her head to her feet)\b",
                c,
                re.I,
            )
        ),
        False,
    )
