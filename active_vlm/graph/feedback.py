"""Cited hints and bounded repetition checks over actually observed crops.

A whole-question No is not object absence. Neither hints nor search geometry
become evidence. Native crop/probe requests stay intact.
"""

import re

from .evidence import local_claims, words

MIN_NEW_AREA = 0.1
MAX_QUOTE_CHARS = 320


def _local_sentences(text):
    local = False
    for sentence in re.split(r"(?<=[.!?])\s+|\n+", text or ""):
        explicit = re.search(
            r"(?:candidate )?(?:cropped|zoomed)(?:[- ]in)?(?: candidate)? (?:region|image|view|area|section)",
            sentence,
            re.I,
        )
        if re.search(r"\boriginal (?:image|scene|view)\b", sentence, re.I) and not explicit:
            local = False
        if explicit:
            local = True
        if local and not re.search(r"\bquestion (?:asks?|requires?|is)\b", sentence, re.I):
            yield sentence.strip()


def summarize_probe(candidate, reports, nodes, newly_supported):
    target_words = set().union(*(words(t) for n in nodes for t in n["targets"]))
    observations, limitations, seen = [], [], set()
    for report in reports:
        if report["receipt_id"] in seen:
            continue
        seen.add(report["receipt_id"])
        text = report["message"].get("content") or ""
        observations.extend((s, report["receipt_id"]) for s in local_claims(text))
        limitations.extend(
            (s, report["receipt_id"])
            for s in _local_sentences(text)
            if re.search(
                r"\b(?:no|not|cannot|can't|unclear|uncertain|insufficient|blurry)\b|too (?:small|narrow)",
                s,
                re.I,
            )
        )

    def quotes(items):
        # Prefer concrete observations/target limitations over a generic relevance verdict.
        def priority(item):
            text = item[0]
            return (
                int(
                    bool(
                        re.search(
                            r"\b(?:person|people|helmet|wall|bus|light|shows?|contains?|wearing)\b",
                            text,
                            re.I,
                        )
                    )
                ),
                len(words(text) & target_words),
                -len(text),
            )

        output, used = [], set()
        for text, receipt in sorted(items, key=priority, reverse=True):
            key = re.sub(r"\W+", " ", text.lower()).strip()
            if key in used:
                continue
            used.add(key)
            output.append(
                dict(
                    quote=text[:MAX_QUOTE_CHARS],
                    probe_receipt_id=receipt,
                    truncated=len(text) > MAX_QUOTE_CHARS,
                )
            )
            if len(output) == 2:
                break
        return output

    return dict(
        crop_id=candidate["id"],
        original_xyxy=list(candidate["original_xyxy"]),
        observations=quotes(observations),
        limitations=quotes(limitations),
        assessment="new_goal_support" if newly_supported else "no_new_goal_support",
        newly_supported=list(newly_supported),
        evidence_status="quoted_model_reports_only_not_verified_visual_truth",
    )


def pool_hints(candidates, reports, nodes):
    """Retain tentative local mentions from non-winning candidates as hints only.

    K3 agreement/identity rules remain in TaskGraph.evidence. A minority or
    uncertain observation can suggest a closer look without supplying gain.
    """
    targets = [words(t) for n in nodes for t in n["targets"]]
    qualifiers = [
        words(parts[1])
        for n in nodes
        for target in n["targets"]
        if len(
            parts := re.split(
                r"\b(?:with|wearing|carrying|holding|dressed in)\b", target, maxsplit=1, flags=re.I
            )
        )
        == 2
    ]
    output, seen = [], set()
    for candidate, raw in zip(candidates, reports, strict=True):
        for report in raw:
            text = report["message"].get("content") or ""
            concrete = set(local_claims(text))
            for sentence in _local_sentences(text):
                uncertain_target = bool(
                    re.search(
                        r"\b(?:likely|might|maybe|possibly|appears?|uncertain)\b", sentence, re.I
                    )
                ) and any(t and t <= words(sentence) for t in targets)
                # A group member can visibly satisfy a question qualifier
                # without identifying the exact target (e.g. "one of whom").
                # Retain that local observation as a search hint only. It
                # supplies neither identity, dependency support nor gain.
                group_member = (
                    sentence in concrete
                    and bool(re.search(r"\bone of (?:whom|them|the)\b", sentence, re.I))
                    and any(q and q <= words(sentence) for q in qualifiers)
                )
                if not (uncertain_target or group_member):
                    continue
                key = re.sub(r"\W+", " ", sentence.lower()).strip()
                if key in seen:
                    continue
                seen.add(key)
                output.append(
                    dict(
                        crop_id=candidate["id"],
                        original_xyxy=list(candidate["original_xyxy"]),
                        probe_receipt_id=report["receipt_id"],
                        quote=sentence[:MAX_QUOTE_CHARS],
                        truncated=len(sentence) > MAX_QUOTE_CHARS,
                        evidence_status="unconfirmed_search_hint_only_no_gain",
                    )
                )
                if len(output) == 2:
                    return output
    return output


def _area(box):
    return max(0, box[2] - box[0]) * max(0, box[3] - box[1])


def _covered_fraction(box, regions):
    """Exact rectangle-union coverage; overlapping history is never counted twice."""
    clipped = [
        [max(box[0], r[0]), max(box[1], r[1]), min(box[2], r[2]), min(box[3], r[3])]
        for r in regions
    ]
    clipped = [r for r in clipped if _area(r)]
    xs = sorted({x for r in clipped for x in [r[0], r[2]]})
    covered = 0
    for left, right in zip(xs, xs[1:]):
        intervals = sorted((r[1], r[3]) for r in clipped if r[0] <= left and r[2] >= right)
        length, end = 0, float("-inf")
        for low, high in intervals:
            length += max(0, high - max(low, end))
            end = max(end, high)
        covered += (right - left) * length
    return min(1.0, covered / _area(box)) if _area(box) else 1.0


def _failed(attempts, focus_id):
    return [
        a
        for a in attempts
        if focus_id in a.get("unsupported_pool_goal_ids", a.get("unsupported_goal_ids", []))
    ]


def repeat_decision(attempts, focus, nodes, candidates, partition_size=None):
    if focus is None:
        return dict(
            reject=True,
            reason="no_unresolved_focus",
            candidate_checks=[],
            instruction="Do not crop without a ready unresolved goal. If all goals are supported, proceed to the common final readers; otherwise retain the unresolved state.",
        )
    failed = _failed(attempts, focus["id"])
    zero = [a for a in failed if not a["newly_supported"]]
    # Only measured candidate regions count. Padding a failed crop invents
    # unseen coverage and used to suppress useful adjacent regions.
    regions = [b for a in zero for b in a.get("search_tested_boxes", a["candidate_pool_boxes"])]
    checks = [
        dict(crop_id=c["id"], new_area_fraction=1 - _covered_fraction(c["original_xyxy"], regions))
        for c in candidates
    ]
    exact = any({c["id"] for c in candidates} <= set(a["candidate_pool_ids"]) for a in failed)
    stale = bool(checks) and all(c["new_area_fraction"] < MIN_NEW_AREA for c in checks)
    if exact:
        return dict(
            reject=True,
            reason="same_failed_pool",
            focus_id=focus["id"],
            candidate_checks=checks,
            instruction="Do not resubmit the identical failed candidate pool. Choose a new region, wider context or a substantially closer detail view.",
        )
    if not stale:
        return dict(
            reject=False,
            reason="new_region_or_context",
            focus_id=focus["id"],
            candidate_checks=checks,
        )

    # A coarse whole-image No cannot exclude smaller targets in its quadrants.
    # Python control admits only four fixed original-frame partitions, not
    # arbitrary shifted boxes. Exact failed pools were already rejected above.
    if partition_size and focus["kind"] == "locate" and len(candidates) == 1:
        w, h = partition_size
        box = candidates[0]["original_xyxy"]
        partitions = [[x, y, x + w / 2, y + h / 2] for y in (0, h / 2) for x in (0, w / 2)]
        fixed = any(max(abs(a - b) for a, b in zip(box, p, strict=True)) <= 1 for p in partitions)
        closer = any(
            _area(box) <= 0.5 * _area(old) and _covered_fraction(box, [old]) >= 0.8
            for old in regions
        )
        if fixed and closer:
            return dict(
                reject=False,
                reason="partition_detail",
                focus_id=focus["id"],
                candidate_checks=checks,
            )

    # One substantial detail zoom per focus may revisit an observed area even
    # when the target is not yet located. A global relevance No cannot rule out
    # a small object. This exception never supplies identity or evidence.
    parents = [
        n
        for n in nodes
        if n["id"] in focus["depends_on"]
        and n["kind"] in {"locate", "condition"}
        and n["status"] == "supported_by_reports"
        and n["citations"]
    ]
    used = any(a.get("search_refinement") and focus["id"] in a["focus_ids"] for a in attempts)
    if focus["kind"] in {"locate", "attribute", "condition", "frame"} and not used:
        if candidates:
            # Compare complete pool scales, not a new 0.8x variant to an old
            # 1.5x variant: that mistake would admit ordinary micro-jitter.
            box = max((c["original_xyxy"] for c in candidates), key=_area)
            smaller = any(
                _area(box) <= 0.5 * _area(old) and _covered_fraction(box, [old]) >= 0.8
                for a in zero
                for old in [max(a.get("search_tested_boxes", a["candidate_pool_boxes"]), key=_area)]
            )
            connected = any(
                _covered_fraction(box, [c["original_xyxy"]]) >= 0.8
                for n in parents
                for c in n["citations"]
            )
            if smaller:
                return dict(
                    reject=False,
                    reason="target_detail_refinement"
                    if connected
                    else "unconfirmed_detail_refinement",
                    focus_id=focus["id"],
                    candidate_checks=checks,
                )
    return dict(
        reject=True,
        reason="zero_progress_observed_area_repeat",
        focus_id=focus["id"],
        candidate_checks=checks,
        instruction="This pool adds less than 10% unobserved area per candidate. Search another region or expand context. One substantial detail zoom per focus is allowed; small shifts and repeated zooms are not. Coverage is not evidence that an object is absent.",
    )


def search_coverage(attempts, focus, image_size=None):
    if focus is None:
        return []
    output = []
    for attempt in _failed(attempts, focus["id"])[-3:]:
        if attempt["newly_supported"]:
            continue
        boxes = attempt.get("search_tested_boxes", attempt["candidate_pool_boxes"])
        item = dict(
            goal_id=focus["id"],
            tested_original_xyxy=[list(box) for box in boxes],
            interpretation="observed_coverage_only_not_object_absence_or_an_exclusion_zone",
            minimum_new_area_fraction=MIN_NEW_AREA,
            detail_refinement_available=not any(
                a.get("search_refinement") and focus["id"] in a["focus_ids"] for a in attempts
            ),
        )
        if image_size:
            item["tested_normalized_1000"] = [
                [
                    round(max(0, min(image_size[i % 2], v)) / image_size[i % 2] * 1000)
                    for i, v in enumerate(box)
                ]
                for box in boxes
            ]
        output.append(item)
    return output
