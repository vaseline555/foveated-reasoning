"""Reuse one region plan; choose bounded crops and stops without model calls.

Regions and search coverage are proposals, never object detections or evidence.
All geometry is in the original image's normalized 0..1000 frame.
"""

import re
from copy import deepcopy

from ..baselines import prompts
from ..imaging import validate_seed
from .evidence import local_claims
from .references import absent_target, target_matches, words


class CropSearch:
    def __init__(self, max_probes=6, max_zero_gain=2):
        if any(type(x) is not int or x < 1 for x in (max_probes, max_zero_gain)):
            raise ValueError("SEARCH_LIMITS_MUST_BE_POSITIVE_INTEGERS")
        self.max_probes, self.max_zero_gain = max_probes, max_zero_gain
        self.regions = None
        self.tried = set()
        self.decisions = []

    @staticmethod
    def history(tasks, focus):
        return [a for a in tasks.attempts if focus["id"] in a["focus_ids"]]

    def stalled(self, tasks, focus):
        zero = 0
        for attempt in reversed(self.history(tasks, focus)):
            if attempt["newly_supported"]:
                break
            zero += 1
        return zero >= self.max_zero_gain

    def context_recovery(self, tasks, focus):
        history = self.history(tasks, focus)
        return (
            len(history) >= self.max_zero_gain
            and len(history) < self.max_zero_gain + 1
            and bool(self.refinements(tasks, focus))
        )

    def refinements(self, tasks, focus):
        """Spend a closer/wider view only when an observed target can benefit."""
        history = self.history(tasks, focus)
        local_target = any(
            all(target_matches(t, q["quote"]) for t in focus["targets"])
            and not absent_target(q["quote"], focus["targets"])
            and local_claims(q["quote"])
            for a in history
            for q in a.get("probe_feedback", {}).get("observations", [])
        )
        hinted = any(
            all(target_matches(t, q["quote"]) for t in focus["targets"])
            for a in history
            for q in a.get("probe_feedback", {}).get("unconfirmed_pool_hints", [])
        )
        # A visible object head may justify ONE closer search even when its
        # identifying qualifiers are unconfirmed. This hint never supports a
        # node; the unchanged full-target evidence checks still apply.
        phrases = [
            re.split(
                r"\b(?:on|in|with|wearing|carrying|holding|that|who|next to|beside|near)\b",
                t,
                maxsplit=1,
                flags=re.I,
            )[0].split()
            for t in focus["targets"]
        ]
        heads = [words(p[-1]) if p else set() for p in phrases]
        hinted = hinted or any(
            heads and all(h & words(q["quote"]) for h in heads) and local_claims(q["quote"])
            for a in history
            for q in a.get("probe_feedback", {}).get("observations", [])
        )
        identified = local_target or hinted or focus["kind"] != "locate"
        if not identified:
            return []
        limitations = " ".join(
            q["quote"]
            for a in history[-1:]
            for q in a.get("probe_feedback", {}).get("limitations", [])
        )
        if focus["kind"] != "locate" and re.search(
            r"\b(?:context|partial|outside|omits?|cut off|not include)\b", limitations, re.I
        ):
            return [("context", 1.5), ("detail", 0.65)]
        return [("detail", 0.65), ("context", 1.5)]

    @staticmethod
    def targets(tasks):
        return [n for n in tasks.nodes if n["kind"] == "locate"]

    def tool(self, tasks):
        return dict(
            type="function",
            function=dict(
                name="plan_regions",
                description=(
                    "Propose one initial search region per fixed localization goal. "
                    "Coordinates are original-image x1,y1,x2,y2 in 0..1000. "
                    "Keep target qualifiers. Regions are tentative, not evidence. "
                    "Python will select all later crops and decide when to stop."
                ),
                parameters=prompts.obj(
                    {n["id"]: deepcopy(prompts.BOX) for n in self.targets(tasks)}
                ),
            ),
        )

    def accept(self, tasks, regions):
        tasks.require_plan()
        if self.regions is not None:
            raise ValueError("INITIAL_REGION_PLAN_ALREADY_FIXED")
        expected = {n["id"] for n in self.targets(tasks)}
        if not expected or not isinstance(regions, dict) or set(regions) != expected:
            raise ValueError("REGION_KEYS_MUST_MATCH_FIXED_LOCALIZATION_GOALS")
        # Validate every entry before committing any part of the plan.
        checked = {}
        for key, box in regions.items():
            try:
                checked[key] = list(validate_seed(box))
            except (TypeError, ValueError) as error:
                raise ValueError(f"{key}: {error}") from error
        self.regions = checked

    @staticmethod
    def transform(box, scale):
        x1, y1, x2, y2 = box
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        w, h = (x2 - x1) * scale, (y2 - y1) * scale
        return [
            max(0, cx - w / 2),
            max(0, cy - h / 2),
            min(1000, cx + w / 2),
            min(1000, cy + h / 2),
        ]

    def boxes(self, tasks, focus):
        """Evidence-linked views first; new original-image regions when stalled."""
        anchors = []
        if focus["id"] in self.regions:
            anchors.append(self.regions[focus["id"]])
        parents = [n for n in tasks.nodes if n["id"] in focus["depends_on"]]
        w, h = tasks.image_size
        for parent in parents:
            for citation in parent["citations"]:
                b = citation["original_xyxy"]
                anchors.append([1000 * b[0] / w, 1000 * b[1] / h, 1000 * b[2] / w, 1000 * b[3] / h])
        if len(anchors) > 1:
            # A relation/count may require several targets in the same view.
            yield (
                "dependency_context",
                [
                    min(b[0] for b in anchors),
                    min(b[1] for b in anchors),
                    max(b[2] for b in anchors),
                    max(b[3] for b in anchors),
                ],
            )
        for box in anchors:
            yield "initial_region" if focus["kind"] == "locate" else "dependency_region", box
            for reason, scale in self.refinements(tasks, focus):
                yield reason, self.transform(box, scale)
        # Fixed, non-jittered tiles prevent repeatedly shifting a zero-gain box.
        # Dependencies/target binding are checked before these views are used.
        if focus["kind"] == "locate":
            for y in (0, 500):
                for x in (0, 500):
                    yield "new_region", [x, y, x + 500, y + 500]
            yield "whole_context", [0, 0, 1000, 1000]

    def next(self, tasks, images, graph=True, single=True):
        tasks.require_plan()
        tasks.focus_override = None
        if tasks.complete:
            return dict(stop="ALL_GOALS_SUPPORTED")
        if len(tasks.attempts) >= self.max_probes:
            return dict(stop="PROBE_BUDGET_EXHAUSTED", unresolved=True)
        focus = tasks.focus(graph)
        if focus is None:
            return dict(stop="NO_READY_VISUAL_GOAL")
        if self.regions is None:
            raise ValueError("INITIAL_REGION_PLAN_REQUIRED")
        counts = {
            n["id"]: sum(n["id"] in a["focus_ids"] for a in tasks.attempts)
            for n in tasks.ready(graph)
        }
        ready = sorted(tasks.ready(graph), key=lambda n: (counts[n["id"]], int(n["id"][1:])))
        rejected = []
        used_ids = {a["crop_id"] for a in tasks.attempts}
        for focus in ready:
            stalled = self.stalled(tasks, focus)
            recovery = stalled and self.context_recovery(tasks, focus)
            if stalled and not recovery:
                rejected.append(dict(focus_id=focus["id"], reason="zero_gain_limit"))
                continue
            tasks.focus_override = focus["id"]
            for reason, box in self.boxes(tasks, focus):
                if recovery and reason != "context":
                    continue  # One hinted context rescue, never more blind tiles.
                try:
                    box = list(validate_seed(box))
                    key = (focus["id"], tuple(box))
                    if key in self.tried:
                        continue
                    proposal = dict(
                        anchor=dict(parent="original", box=box, target_nodes=[]), branches=[]
                    )
                    candidates = (
                        [images.candidate("original", box)]
                        if single
                        else images.pool("A", proposal)
                    )
                    if all(c["id"] in used_ids for c in candidates):
                        rejected.append(
                            dict(focus_id=focus["id"], box=box, reason="already_probed_view")
                        )
                        self.tried.add(key)
                        continue
                    decision = tasks.repeat_decision(candidates, graph)
                except (ValueError, TypeError, KeyError):
                    continue
                self.tried.add(key)
                if decision["reject"]:
                    rejected.append(dict(focus_id=focus["id"], box=box, reason=decision["reason"]))
                    continue
                audit = dict(
                    focus_id=focus["id"],
                    reason=reason,
                    proposal=proposal,
                    skipped=rejected,
                    context_recovery=recovery,
                )
                self.decisions.append(deepcopy(audit))
                return audit
        # A focus with no admissible views must not block another ready root.
        # Focus rotation belongs to this controller, not to evidence/support state.
        tasks.focus_override = None
        return dict(
            stop="NO_PRODUCTIVE_SEARCH"
            if any(r["reason"] == "zero_gain_limit" for r in rejected)
            else "SEARCH_VIEWS_EXHAUSTED",
            skipped=rejected,
            unresolved=True,
        )
