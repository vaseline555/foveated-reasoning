"""Deterministic controller checks; no model credentials or optional SDKs needed."""

import asyncio
import importlib.util
import sys
import unittest
from pathlib import Path

MODULE_PATH = Path(__file__).parents[1] / "src/cv_agent/policies/adaptive.py"
SPEC = importlib.util.spec_from_file_location("adaptive_under_test", MODULE_PATH)
adaptive = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = adaptive
SPEC.loader.exec_module(adaptive)


def plan(importance=1.0):
    return adaptive.Plan(
        (adaptive.Role("target", "target", "visible detail", importance=importance),), ("target",)
    )


def candidate(key="a", image="original", roles=("target",), box=(0, 0, 1000, 1000)):
    return adaptive.Candidate(key, image, box, roles, "inspect the requested detail")


class FakeBackend:
    def __init__(self, initial=None, proposals=None, reports=None):
        self.initial = initial or plan()
        self.proposals = proposals or (candidate(),)
        self.reports = list(reports or ())
        self.calls = []
        self.seen = []

    async def plan(self, question, image):
        self.calls.append("plan")
        await asyncio.sleep(0)
        return self.initial

    async def propose(self, state):
        self.calls.append("propose")
        await asyncio.sleep(0)
        return self.proposals

    async def observe(self, state, proposal, image):
        self.calls.append("observe")
        self.seen.append((state.question, proposal.id, proposal.image_key, image))
        await asyncio.sleep(0)
        return self.reports.pop(0) if self.reports else adaptive.Report()

    async def finalize(self, state):
        self.calls.append("finalize")
        await asyncio.sleep(0)
        return state.question


class Crops:
    def __init__(self):
        self.actions = []

    async def __call__(self, proposal):
        self.actions.append(proposal)
        await asyncio.sleep(0)
        return f"crop-{len(self.actions)}"


class StateChecks(unittest.TestCase):
    def state(self, initial=None):
        initial = initial or plan()
        state = adaptive.CaseState("question", "original", initial)
        state.image_keys.add("original")
        adaptive._set_plan(state, initial)
        return state

    def test_invalid_dag_and_unknown_references(self):
        role = adaptive.Role
        invalid = (
            adaptive.Plan((role("a", "a", "a", ("b",)), role("b", "b", "b", ("a",))), ("a",)),
            adaptive.Plan((role("a", "a", "a", ("missing",)),), ("a",)),
            adaptive.Plan((role("a", "a", "a"),), ("missing",)),
            adaptive.Plan((role("a", "a", "a"), role("a", "duplicate", "a")), ("a",)),
        )
        for value in invalid:
            with self.subTest(value=value), self.assertRaises(ValueError):
                adaptive._roles(value)

    def test_no_area_overlap_or_distance_preprocessing(self):
        state = self.state()
        proposals = (candidate("whole"), candidate("same"), candidate("tiny", box=(10, 10, 11, 11)))
        adaptive._add_candidates(state, proposals, adaptive.Limits())
        self.assertEqual(state.candidates, {item.id: item for item in proposals})

    def test_invalid_candidates_rejected_without_partial_commit(self):
        for invalid in (
            candidate(image="unknown"),
            candidate(roles=("missing",)),
            candidate(box=(0, 0, 1001, 1000)),
            candidate(box=(10, 0, 0, 1000)),
        ):
            state = self.state()
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                adaptive._add_candidates(state, (candidate("valid"), invalid), adaptive.Limits())
            self.assertEqual(state.candidates, {})

    def test_early_support_is_partial_preserves_raw_report_and_needs_rebinding(self):
        initial = adaptive.Plan(
            (
                adaptive.Role("anchor", "person", "identity"),
                adaptive.Role("detail", "watch", "color", ("anchor",)),
            ),
            ("detail",),
        )
        state = self.state(initial)
        raw = adaptive.Report((adaptive.Claim("detail", "blue watch", "supported"),))
        observation = adaptive.Observation(candidate(roles=("detail",)), "crop-early", raw)
        adaptive._record(state, observation, adaptive.Limits())
        self.assertEqual(state.statuses["detail"], "partial")
        self.assertIs(state.observations[0], observation)
        self.assertEqual(state.observations[0].report.claims[0].status, "supported")
        self.assertEqual(state.observations[0].image_key, "crop-early")
        adaptive._record(
            state,
            adaptive.Observation(
                None,
                "original",
                adaptive.Report((adaptive.Claim("anchor", "person identified", "supported"),)),
            ),
            adaptive.Limits(),
        )
        self.assertEqual(state.statuses["detail"], "partial")
        self.assertFalse(adaptive.answer_ready(state))

    def test_plan_revision_invalidates_changed_descendants_only(self):
        initial = adaptive.Plan(
            (
                adaptive.Role("anchor", "person", "red bag"),
                adaptive.Role("detail", "watch", "color", ("anchor",)),
                adaptive.Role("other", "sign", "text"),
            ),
            ("detail",),
        )
        state = self.state(initial)
        state.statuses = dict.fromkeys(state.statuses, "supported")
        old_observation = adaptive.Observation(
            None,
            "original",
            adaptive.Report((adaptive.Claim("detail", "blue watch", "supported"),)),
        )
        state.observations.append(old_observation)
        revised = adaptive.Plan(
            (adaptive.Role("anchor", "person", "green bag"), initial.roles[1], initial.roles[2]),
            ("detail",),
        )
        adaptive._set_plan(state, revised)
        self.assertEqual(
            state.statuses, {"anchor": "unresolved", "detail": "unresolved", "other": "supported"}
        )
        self.assertEqual(state.plans, [initial, revised])
        self.assertIs(state.observations[0], old_observation)

    def test_revised_role_requires_candidate_revalidation(self):
        state = self.state()
        proposal = candidate()
        adaptive._add_candidates(state, (proposal,), adaptive.Limits())
        revised = adaptive.Plan(
            (adaptive.Role("target", "target", "different evidence"),), ("target",)
        )
        adaptive._set_plan(state, revised)
        self.assertEqual(state.stale_candidates, {proposal.id})
        adaptive._add_candidates(state, (proposal,), adaptive.Limits())
        self.assertEqual(state.stale_candidates, set())

    def test_conflict_requires_explicit_resolution(self):
        state = self.state()
        for claim in (
            adaptive.Claim("target", "conflicting reports", "conflicting"),
            adaptive.Claim("target", "a later assertion", "supported"),
        ):
            adaptive._record(
                state,
                adaptive.Observation(None, "original", adaptive.Report((claim,))),
                adaptive.Limits(),
            )
        self.assertEqual(state.statuses["target"], "conflicting")
        adaptive._record(
            state,
            adaptive.Observation(
                None,
                "original",
                adaptive.Report(
                    (adaptive.Claim("target", "explicit resolution report", "supported", True),)
                ),
            ),
            adaptive.Limits(),
        )
        self.assertTrue(adaptive.answer_ready(state))
        self.assertEqual(len(state.observations), 3)


class ControllerChecks(unittest.IsolatedAsyncioTestCase):
    async def test_original_evidence_can_skip_crops(self):
        initial = adaptive.Plan(
            plan().roles,
            ("target",),
            (adaptive.Claim("target", "visible in original", "supported"),),
        )
        backend, crops = FakeBackend(initial), Crops()
        result = await adaptive.AdaptivePolicy().run("question", "original", backend, crops)
        self.assertEqual(backend.calls, ["plan", "finalize"])
        self.assertEqual(result.state.stop_reason, "evidence_ready")
        self.assertEqual(crops.actions, [])

    async def test_concurrent_cases_are_isolated(self):
        policy = adaptive.AdaptivePolicy(adaptive.Limits(max_steps=1))
        left, right = (
            FakeBackend(proposals=(candidate("left"),)),
            FakeBackend(proposals=(candidate("right"),)),
        )
        results = await asyncio.gather(
            policy.run("left question", "original", left, Crops()),
            policy.run("right question", "original", right, Crops()),
        )
        self.assertEqual([result.answer for result in results], ["left question", "right question"])
        self.assertEqual(
            [set(result.state.candidates) for result in results], [{"left"}, {"right"}]
        )
        self.assertIsNot(results[0].state.observations, results[1].state.observations)

    async def test_binding_dependencies_do_not_gate_exploration(self):
        initial = adaptive.Plan(
            (
                adaptive.Role("anchor", "person", "identity"),
                adaptive.Role("detail", "watch", "color", ("anchor",)),
            ),
            ("detail",),
        )
        backend = FakeBackend(
            initial,
            (candidate(roles=("detail",)),),
            (adaptive.Report((adaptive.Claim("detail", "blue watch", "supported"),)),),
        )
        crops = Crops()
        result = await adaptive.AdaptivePolicy(adaptive.Limits(max_steps=1)).run(
            "question", "original", backend, crops
        )
        self.assertEqual(len(crops.actions), 1)
        self.assertEqual(result.state.statuses["detail"], "partial")

    async def test_no_progress_switches_and_exhausts_visits(self):
        backend = FakeBackend(proposals=(candidate("a"), candidate("b")))
        crops = Crops()
        result = await adaptive.AdaptivePolicy(adaptive.Limits(max_steps=8, max_visits=1)).run(
            "question", "original", backend, crops
        )
        self.assertEqual([item.id for item in crops.actions], ["a", "b"])
        self.assertEqual(result.state.visits, {"a": 1, "b": 1})
        self.assertEqual(result.state.stop_reason, "no_candidates")

    async def test_branch_extension_requires_progress_and_importance(self):
        for importance, progress, expected in ((2.0, True, 2), (1.0, True, 1), (2.0, False, 1)):
            with self.subTest(importance=importance, progress=progress):
                reports = (
                    (adaptive.Report((adaptive.Claim("target", "some detail", "partial"),)),)
                    if progress
                    else ()
                )
                backend = FakeBackend(plan(importance), reports=reports)
                crops = Crops()
                await adaptive.AdaptivePolicy(
                    adaptive.Limits(max_steps=5, branch_steps=1, progress_bonus=1, max_visits=1)
                ).run("question", "original", backend, crops)
                self.assertEqual(len(crops.actions), expected)

    async def test_nested_candidate_inherits_branch_budget(self):
        child = candidate("child", image="crop-1")
        grandchild = candidate("grandchild", image="crop-2")
        reports = (
            adaptive.Report((adaptive.Claim("target", "partial detail", "partial"),), (child,)),
            adaptive.Report(new_candidates=(grandchild,)),
        )
        backend, crops = FakeBackend(plan(2.0), reports=reports), Crops()
        result = await adaptive.AdaptivePolicy(
            adaptive.Limits(max_steps=8, branch_steps=1, progress_bonus=1, max_visits=1)
        ).run("question", "original", backend, crops)
        self.assertEqual([item.id for item in crops.actions], ["a", "child"])
        self.assertEqual(
            result.state.candidate_branches, {"a": "a", "child": "a", "grandchild": "a"}
        )
        self.assertEqual(result.state.visits, {"a": 1})
        self.assertEqual(result.state.branch_steps, {"a": 2})

    async def test_nested_candidate_cannot_override_inherited_branch(self):
        override = adaptive.Candidate(
            "child", "crop-1", (0, 0, 1000, 1000), ("target",), "detail", branch_id="new-budget"
        )
        backend = FakeBackend(reports=(adaptive.Report(new_candidates=(override,)),))
        with self.assertRaisesRegex(ValueError, "cannot reset"):
            await adaptive.AdaptivePolicy().run("question", "original", backend, Crops())

    async def test_final_model_call_is_reserved(self):
        backend, crops = FakeBackend(), Crops()
        result = await adaptive.AdaptivePolicy(adaptive.Limits(max_model_calls=3)).run(
            "question", "original", backend, crops
        )
        self.assertEqual(backend.calls, ["plan", "propose", "finalize"])
        self.assertEqual(result.state.model_calls, 3)
        self.assertEqual(crops.actions, [])

    async def test_api_and_meter_errors_propagate(self):
        class ApiBackend(FakeBackend):
            async def observe(self, *args):
                raise RuntimeError("API unavailable")

        async def meter_failure(proposal):
            raise RuntimeError("meter reservation rejected")

        policy = adaptive.AdaptivePolicy()
        with self.assertRaisesRegex(RuntimeError, "API unavailable"):
            await policy.run("question", "original", ApiBackend(), Crops())
        with self.assertRaisesRegex(RuntimeError, "meter reservation rejected"):
            await policy.run("question", "original", FakeBackend(), meter_failure)

    async def test_crop_error_is_logged_with_parent_and_does_not_become_evidence(self):
        async def bad_crop(proposal):
            raise adaptive.CropError("crop tool rejected coordinates")

        result = await adaptive.AdaptivePolicy(adaptive.Limits(max_steps=1)).run(
            "question", "original", FakeBackend(), bad_crop
        )
        observation = result.state.observations[-1]
        self.assertIn("crop tool rejected", observation.error)
        self.assertEqual(observation.image_key, "original")
        self.assertEqual(observation.report.claims, ())
        self.assertEqual(result.state.statuses["target"], "unresolved")

    async def test_original_parent_reference_survives_later_crops(self):
        backend = FakeBackend(proposals=(candidate("a"), candidate("b")))
        crops = Crops()
        result = await adaptive.AdaptivePolicy(adaptive.Limits(max_steps=2, max_visits=1)).run(
            "question", "original", backend, crops
        )
        self.assertEqual([item.image_key for item in crops.actions], ["original", "original"])
        self.assertEqual(
            [item.image_key for item in result.state.observations], ["original", "crop-1", "crop-2"]
        )

    async def test_backend_cannot_mutate_controller_state(self):
        class MutatingBackend(FakeBackend):
            async def propose(self, state):
                state.question = "changed"
                state.image_keys.clear()
                state.statuses.clear()
                return self.proposals

            async def observe(self, state, proposal, image):
                state.question = "changed"
                state.observations.clear()
                state.steps = 999
                return adaptive.Report()

            async def finalize(self, state):
                state.question = "changed"
                state.observations.clear()
                return "answer"

        result = await adaptive.AdaptivePolicy(adaptive.Limits(max_steps=1)).run(
            "original question", "original", MutatingBackend(), Crops()
        )
        self.assertEqual(result.state.question, "original question")
        self.assertEqual(result.state.steps, 1)
        self.assertEqual(result.state.statuses, {"target": "unresolved"})
        self.assertEqual(len(result.state.observations), 2)
        self.assertEqual(result.state.image_keys, {"original", "crop-1"})


if __name__ == "__main__":
    unittest.main()
