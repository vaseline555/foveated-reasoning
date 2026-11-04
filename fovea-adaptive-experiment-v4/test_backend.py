"""Offline contract tests; the fake client makes no HTTP requests."""

from __future__ import annotations

import asyncio
import copy
import hashlib
import io
import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

from PIL import Image

import backend

a = backend.adaptive


# Exact native tool responses from the V3 smoke; no evaluation labels.
V3_OBSERVATION_FAILURES = [
    {
        "role": "assistant",
        "content": None,
        "refusal": None,
        "reasoning": None,
        "tool_calls": [
            {
                "type": "function",
                "index": 0,
                "id": "chatcmpl-tool-b12416167a63ea1b",
                "function": {
                    "name": "submit_report",
                    "arguments": '{"claims": [{"role_id": "glove_material", "text": "the glove appears to be made of a thin, flexible, blue material consistent with disposable nitrile or latex gloves", "status": "supported"}], "new_candidates": [{"id": "candidate_2", "image_key": "original", "bbox": [278, 93, 362, 201], "role_ids": ["glove_material"], "expected_evidence": "a close-up view of the blue glove to identify its material"}, {"id": "candidate_3", "image_key": "original", "bbox": [278, 93, 362, 201], "role_ids": ["glove_material"], "expected_evidence": "a close-up view of the blue glove to identify its material"}, {"id": "candidate_4", "image_key": "original", "bbox": [278, 93, 362, 201], "role_ids": ["glove_material"], "expected_evidence": "a close-up view of the blue glove to identify its material"}], "plan_update": {"roles": [{"id": "glove_material", "expected_evidence": "a close-up view of the blue glove to identify its material", "importance": 2, "requires_for_binding": ["glove"]}, {"id": "glove", "expected_evidence": "a glove worn by a person in the image", "importance": 1, "requires_for_binding": []}, {"id": "glove_color", "expected_evidence": "the color of the glove", "importance": 1, "requires_for_binding": ["glove"]}], "goals": ["glove_material"]}}',
                },
            }
        ],
    },
    {
        "role": "assistant",
        "content": None,
        "refusal": None,
        "reasoning": None,
        "tool_calls": [
            {
                "type": "function",
                "index": 0,
                "id": "chatcmpl-tool-93ba2ac18541ace4",
                "function": {
                    "name": "submit_report",
                    "arguments": '{"claims": [{"role_id": "glove_material", "text": "the glove appears to be made of a thin, flexible, blue material consistent with disposable nitrile or latex gloves", "status": "supported"}], "new_candidates": [], "plan_update": {"roles": [{"id": "glove_material", "expected_evidence": "a close-up view of the blue glove to identify its material", "importance": 2, "requires_for_binding": ["glove"]}, {"id": "glove", "expected_evidence": "a glove worn by a person in the image", "importance": 1, "requires_for_binding": []}, {"id": "glove_color", "expected_evidence": "the color of the glove", "importance": 1, "requires_for_binding": ["glove"]}], "goals": ["glove_material"]}}',
                },
            }
        ],
    },
]


def plan_value(supported=False):
    return {
        "roles": [
            {
                "id": "target",
                "role": "target object",
                "expected_evidence": "its color",
                "requires_for_binding": [],
                "importance": 1,
            }
        ],
        "goals": ["target"],
        "original_claims": [
            {
                "role_id": "target",
                "text": "The color is visible" if supported else "Color is unclear",
                "status": "supported" if supported else "unresolved",
            }
        ],
    }


def candidate_value(**changes):
    value = {
        "id": "target_1",
        "image_key": "original",
        "bbox": [100, 200, 700, 800],
        "role_ids": ["target"],
        "expected_evidence": "read the color",
        "branch_id": "b1",
    }
    value.update(changes)
    return value


def content(value):
    return {"role": "assistant", "content": json.dumps(value)}


class RequestFailure(RuntimeError):
    pass


class FakeClient:
    def __init__(self, responses, cached=False):
        self.responses = list(responses)
        self.calls = []
        self.cached = cached

    def call(self, messages, **kwargs):
        self.calls.append({"messages": copy.deepcopy(messages), **kwargs})
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        if not self.cached and not kwargs["validator"](response):
            raise RequestFailure("primary: OUTPUT_VALIDATION_FAILED")
        return response, {
            "id": str(len(self.calls)),
            "model": backend.MODEL,
            "stage": kwargs["stage"],
            "cache_hit": self.cached,
        }


class BackendTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)
        self.image = self.root / "source.png"
        image = Image.new("RGB", (101, 79))
        image.putdata(
            [(x % 256, y % 256, (x + y) % 256) for y in range(79) for x in range(101)]
        )
        image.save(self.image)
        self.case = {
            "key": "vstar_offline",
            "question": "What color is the object? (A) Red (B) Blue",
            "image_path": str(self.image),
            "label": "NEVER_INCLUDE",
        }
        self.baseline = {
            "status": "complete",
            "models": [backend.MODEL],
            "message": {"content": "Independent Direct analysis; <answer>A</answer>"},
        }

    def tearDown(self):
        self.directory.cleanup()

    def build(self, responses, cached=False):
        client = FakeClient(responses, cached)
        instance = backend.Backend(
            self.case, self.root / "shared", self.root / "case", client, self.baseline
        )
        return instance, client

    def state(self):
        plan = backend.parse_plan(plan_value())
        state = a.CaseState(self.case["question"], "original", plan)
        state.image_keys.add("original")
        a._set_plan(state, plan)
        a._record(
            state,
            a.Observation(None, "original", a.Report(plan.original_claims)),
            a.Limits(),
        )
        return state

    def test_zero_crop_policy_and_final_no_answer_remains_model_outcome(self):
        instance, client = self.build(
            [
                content(plan_value(True)),
                {"role": "assistant", "content": "The color is unclear."},
            ]
        )
        result = asyncio.run(
            a.AdaptivePolicy().run(
                self.case["question"], "original", instance, instance.executor
            )
        )
        self.assertEqual(result.state.stop_reason, "evidence_ready")
        self.assertEqual(result.state.steps, 0)
        self.assertEqual(result.answer, "The color is unclear.")
        self.assertEqual([x["stage"] for x in client.calls], ["plan", "aggregator"])
        for call in client.calls:
            self.assertEqual(call["method"], "ADAPTIVE_V4")
            self.assertEqual(call["force_model"], "primary")
            self.assertEqual(call["max_tokens"], 2048)
            self.assertNotIn("NEVER_INCLUDE", json.dumps(call["messages"]))

    def test_actual_v3_reports_name_every_missing_nested_role_field(self):
        for message in V3_OBSERVATION_FAILURES:
            value = backend._response_object(message, "submit_report")
            prior = copy.deepcopy(value)
            with self.assertRaises(ValueError) as failure:
                backend.validate_schema(value, backend.REPORT_SCHEMA)
            diagnostic = str(failure.exception)
            for index in range(3):
                self.assertIn(
                    f"$.plan_update.roles[{index}].role: required field is missing",
                    diagnostic,
                )
            self.assertNotIn("is not of type 'null'", diagnostic)
            self.assertEqual(value, prior)

    def test_actual_v3_observation_repair_gets_paths_and_omits_unchanged_plan(self):
        bad = V3_OBSERVATION_FAILURES[0]
        value = backend._response_object(bad, "submit_report")
        fixed = {"claims": value["claims"], "new_candidates": []}
        instance, client = self.build([copy.deepcopy(bad), content(fixed)])
        result = instance._call(
            "observe_1",
            instance._messages(backend.OBSERVE_INSTRUCTION),
            backend.parse_report,
            "submit_report",
            backend.REPORT_SCHEMA,
        )
        self.assertIsNone(result.plan_update)
        self.assertEqual(len(client.calls), 2)
        repair = client.calls[1]["messages"][-1]["content"]
        for index in range(3):
            self.assertIn(f"$.plan_update.roles[{index}].role", repair)
        saved = json.loads(
            next((self.root / "case" / "format_feedback").glob("*.json")).read_text()
        )
        self.assertEqual(saved["invalid_response"], bad)
        self.assertNotIn("role", value["plan_update"]["roles"][0])

    def test_schema_enum_type_and_unexpected_fields_have_paths(self):
        value = {
            "claims": [{"role_id": True, "text": "detail", "status": "resolved"}],
            "unexpected": 1,
        }
        with self.assertRaises(ValueError) as failure:
            backend.validate_schema(value, backend.REPORT_SCHEMA)
        diagnostic = str(failure.exception)
        self.assertIn("$.claims[0].role_id", diagnostic)
        self.assertIn("$.claims[0].status", diagnostic)
        self.assertIn("unexpected", diagnostic)

    def test_cached_schema_failure_reconstructs_same_nested_diagnostics(self):
        bad = V3_OBSERVATION_FAILURES[1]
        value = backend._response_object(bad, "submit_report")
        fixed = {"claims": value["claims"]}
        instance, client = self.build(
            [RequestFailure("primary: cached OUTPUT_VALIDATION_FAILED"), content(fixed)]
        )
        client.models = {"primary": {"id": backend.MODEL}}
        database = self.root / "ledger.sqlite"

        def connect():
            result = sqlite3.connect(database)
            result.row_factory = sqlite3.Row
            return result

        client.connect = connect
        messages = instance._messages(backend.OBSERVE_INSTRUCTION)
        stage = "observe_1"
        seed = int(
            hashlib.sha256(
                f"{self.case['key']}/{backend.METHOD}/{stage}".encode()
            ).hexdigest()[:8],
            16,
        )
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "submit_report",
                    "description": "Submit the requested structured visual evidence response.",
                    "parameters": backend.REPORT_SCHEMA,
                },
            }
        ]
        logical = {
            "messages": messages,
            "stage": stage,
            "case": self.case["key"],
            "method": backend.METHOD,
            "tools": tools,
            "max_tokens": 2048,
            "seed": seed,
            "models": client.models,
            "force_model": "primary",
        }
        key = hashlib.sha256(backend._json(logical).encode()).hexdigest()
        raw = self.root / "schema-invalid.json"
        raw.write_text(
            json.dumps({"choices": [{"finish_reason": "tool_calls", "message": bad}]})
        )
        with connect() as db:
            db.execute(
                "CREATE TABLE attempts (logical_key,status,finished,response_path)"
            )
            db.execute(
                "INSERT INTO attempts VALUES(?,?,?,?)", (key, "invalid", 1, str(raw))
            )
        result = instance._call(
            stage,
            messages,
            backend.parse_report,
            "submit_report",
            backend.REPORT_SCHEMA,
        )
        self.assertIsNone(result.plan_update)
        repair = client.calls[1]["messages"][-1]["content"]
        for index in range(3):
            self.assertIn(f"$.plan_update.roles[{index}].role", repair)

    def test_native_tool_plan_and_fenced_content(self):
        value = plan_value()
        tool = {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {"function": {"name": "submit_plan", "arguments": json.dumps(value)}}
            ],
        }
        instance, _ = self.build([tool])
        plan = asyncio.run(instance.plan(self.case["question"], "original"))
        self.assertEqual(plan.goals, ("target",))
        self.assertEqual(
            backend._response_object(
                {"content": "```json\n" + json.dumps(value) + "\n```"}, "submit_plan"
            ),
            value,
        )

    def test_cyclic_plan_rejected_before_transport_marks_valid_and_retry_is_bounded(
        self,
    ):
        bad = plan_value()
        bad["roles"][0]["requires_for_binding"] = ["target"]
        instance, client = self.build([content(bad), content(plan_value())])
        result = asyncio.run(instance.plan(self.case["question"], "original"))
        self.assertEqual(result.goals, ("target",))
        self.assertEqual(
            [x["stage"] for x in client.calls], ["plan", "plan_format_retry1"]
        )
        self.assertEqual(
            client.calls[1]["seed"], (client.calls[0]["seed"] + 1) % (2**32)
        )
        self.assertEqual(client.calls[0]["messages"], client.calls[1]["messages"][:-1])
        self.assertIn(
            "Cyclic binding dependencies", client.calls[1]["messages"][-1]["content"]
        )
        self.assertTrue(
            any(x["event"] == "output_validation_failed" for x in instance.events)
        )

    def test_natural_language_goal_is_rejected_with_exact_reference_diagnostic(self):
        bad = plan_value()
        bad["goals"] = ["material of the glove"]
        with self.assertRaisesRegex(
            ValueError,
            "unknown goals=.*material of the glove.*allowed role IDs=.*target",
        ):
            backend.parse_plan(bad)
        instance, client = self.build([content(bad), content(plan_value())])
        result = asyncio.run(instance.plan(self.case["question"], "original"))
        self.assertEqual(result.goals, ("target",))
        repair = client.calls[1]["messages"][-1]["content"]
        self.assertIn("exact roles[].id references", repair)
        self.assertIn("material of the glove", repair)
        self.assertIn("invalid_response", repair)
        self.assertIn("No new observations were made", repair)
        self.assertEqual(client.calls[0]["messages"], client.calls[1]["messages"][:-1])

    def test_cached_invalid_resume_reuses_identical_repair_prompt_and_saved_feedback(
        self,
    ):
        bad = plan_value()
        bad["goals"] = ["material of the glove"]
        instance, client = self.build([content(bad), RequestFailure("HTTP 400")])
        with self.assertRaisesRegex(RequestFailure, "HTTP 400"):
            asyncio.run(instance.plan(self.case["question"], "original"))
        first_repair = client.calls[1]["messages"]
        files = list((self.root / "case" / "format_feedback").glob("*.json"))
        self.assertEqual(len(files), 1)
        original_feedback = files[0].read_bytes()
        resumed, replay = self.build(
            [
                RequestFailure("primary: cached OUTPUT_VALIDATION_FAILED"),
                content(plan_value()),
            ]
        )
        result = asyncio.run(resumed.plan(self.case["question"], "original"))
        self.assertEqual(result.goals, ("target",))
        self.assertEqual(replay.calls[1]["messages"], first_repair)
        self.assertEqual(files[0].read_bytes(), original_feedback)

    def test_cached_truncation_reconstructs_exact_ledger_draft_without_validator(self):
        instance, client = self.build(
            [
                RequestFailure("primary: cached OUTPUT_VALIDATION_FAILED"),
                content(plan_value()),
            ]
        )
        client.models = {"primary": {"id": backend.MODEL}}
        database = self.root / "ledger.sqlite"

        def connect():
            result = sqlite3.connect(database)
            result.row_factory = sqlite3.Row
            return result

        client.connect = connect
        messages = instance._messages(backend.PLAN_INSTRUCTION)
        seed = int(
            hashlib.sha256(
                f"{self.case['key']}/{backend.METHOD}/plan".encode()
            ).hexdigest()[:8],
            16,
        )
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "submit_plan",
                    "description": "Submit the requested structured visual evidence response.",
                    "parameters": backend.PLAN_SCHEMA,
                },
            }
        ]
        logical = {
            "messages": messages,
            "stage": "plan",
            "case": self.case["key"],
            "method": backend.METHOD,
            "tools": tools,
            "max_tokens": 2048,
            "seed": seed,
            "models": client.models,
            "force_model": "primary",
        }
        key = hashlib.sha256(backend._json(logical).encode()).hexdigest()
        raw = self.root / "truncated.json"
        raw.write_text(
            json.dumps(
                {
                    "choices": [
                        {
                            "finish_reason": "length",
                            "message": {
                                "role": "assistant",
                                "content": '{"roles": [unfinished draft',
                            },
                        }
                    ]
                }
            )
        )
        with connect() as db:
            db.execute(
                "CREATE TABLE attempts (logical_key,status,finished,response_path)"
            )
            db.execute(
                "INSERT INTO attempts VALUES(?,?,?,?)", (key, "invalid", 1, str(raw))
            )
        result = asyncio.run(instance.plan(self.case["question"], "original"))
        self.assertEqual(result.goals, ("target",))
        self.assertIn(
            "finish_reason=length", client.calls[1]["messages"][-1]["content"]
        )
        self.assertIn("unfinished draft", client.calls[1]["messages"][-1]["content"])

    def test_invalid_proposal_references_and_bbox_rejected_without_state_mutation(self):
        state = self.state()
        original = copy.deepcopy(state)
        instance, client = self.build(
            [
                content({"candidates": [candidate_value(image_key="invented")]}),
                content({"candidates": [candidate_value()]}),
            ]
        )
        candidates = asyncio.run(instance.propose(state))
        self.assertEqual(candidates[0].id, "target_1")
        self.assertEqual(state, original)
        self.assertEqual(len(client.calls), 2)
        for box in ([0, 0, 0, 100], [0, 0, 1001, 100], [False, 0, 10, 10]):
            with self.subTest(box=box), self.assertRaises(ValueError):
                a._add_candidates(
                    copy.deepcopy(state),
                    [backend.parse_candidate(candidate_value(bbox=box))],
                    a.Limits(),
                )

    def test_crop_matches_official_parent_pixels_floor_and_jpeg90_without_min50(self):
        instance, _ = self.build([])
        proposal = backend.parse_candidate(candidate_value(bbox=[123, 234, 456, 678]))
        key = asyncio.run(instance.executor(proposal))
        absolute = backend.official.absolute_coords(
            proposal.bbox, instance.images["original"].size
        )
        expected = io.BytesIO()
        instance.images["original"].crop(absolute).save(
            expected, format="JPEG", quality=90, optimize=True
        )
        stored = Path(instance.image_registry[key]["path"]).read_bytes()
        self.assertEqual(stored, expected.getvalue())
        self.assertLess(min(instance.images[key].size), 50)
        self.assertEqual(
            instance.image_registry[key]["sha256"], hashlib.sha256(stored).hexdigest()
        )
        self.assertEqual(instance.image_registry[key]["parent"], "original")
        self.assertEqual(instance.image_registry[key]["absolute_bbox"], absolute)

    def test_floor_to_empty_crop_is_expected_crop_error(self):
        instance, client = self.build([])
        proposal = backend.parse_candidate(candidate_value(bbox=[0, 0, 1, 1]))
        with self.assertRaisesRegex(a.CropError, "Empty crop"):
            asyncio.run(instance.executor(proposal))
        self.assertEqual(client.calls, [])
        self.assertTrue(
            any(event["event"] == "crop_error" for event in instance.events)
        )

    def test_observation_validation_does_not_mutate_state_and_keeps_crop_provenance(
        self,
    ):
        report = {
            "claims": [
                {
                    "role_id": "target",
                    "text": "The object looks red",
                    "status": "supported",
                }
            ],
            "new_candidates": [],
        }
        instance, client = self.build([content(report)])
        state = self.state()
        candidate = backend.parse_candidate(candidate_value())
        a._add_candidates(state, [candidate], a.Limits())
        key = asyncio.run(instance.executor(candidate))
        state.image_keys.add(key)
        state.image_branches[key] = "b1"
        prior = copy.deepcopy(state)
        actual = asyncio.run(instance.observe(state, candidate, key))
        self.assertEqual(actual.claims[0].status, "supported")
        self.assertEqual(state, prior)
        prompt = json.dumps(client.calls[0]["messages"])
        self.assertIn("newly_executed_crop", prompt)
        self.assertIn("crop_1", prompt)
        self.assertIn("Color is unclear", prompt)

    def test_report_conflict_flag_requires_real_boolean(self):
        for value in ("true", "false", 1, None):
            with self.subTest(value=value), self.assertRaises(ValueError):
                backend.parse_claim(
                    {
                        "role_id": "target",
                        "text": "detail",
                        "status": "supported",
                        "resolves_conflict": value,
                    }
                )

    def test_observation_bad_nested_branch_is_rejected_before_valid(self):
        state = self.state()
        candidate = backend.parse_candidate(candidate_value())
        instance, client = self.build(
            [
                content(
                    {
                        "new_candidates": [
                            candidate_value(
                                id="child", image_key="crop_1", branch_id="reset"
                            )
                        ]
                    }
                ),
                content(
                    {
                        "claims": [
                            {
                                "role_id": "target",
                                "text": "unclear",
                                "status": "partial",
                            }
                        ]
                    }
                ),
            ]
        )
        a._add_candidates(state, [candidate], a.Limits())
        key = asyncio.run(instance.executor(candidate))
        state.image_keys.add(key)
        state.image_branches[key] = "b1"
        report = asyncio.run(instance.observe(state, candidate, key))
        self.assertEqual(report.claims[0].status, "partial")
        self.assertEqual(len(client.calls), 2)

    def test_final_reader_sees_all_executed_observations_original_and_direct(self):
        instance, client = self.build(
            [{"role": "assistant", "content": "<answer>B</answer>"}]
        )
        state = self.state()
        for index in range(2):
            candidate = backend.parse_candidate(candidate_value(id=f"p{index}"))
            key = asyncio.run(instance.executor(candidate))
            state.image_keys.add(key)
            a._record(
                state,
                a.Observation(
                    candidate,
                    key,
                    a.Report(
                        (a.Claim("target", f"Actual observation {index}", "partial"),)
                    ),
                ),
                a.Limits(),
            )
        self.assertEqual(asyncio.run(instance.finalize(state)), "<answer>B</answer>")
        messages = client.calls[0]["messages"]
        image_count = sum(
            block.get("type") == "image_url"
            for msg in messages
            if isinstance(msg.get("content"), list)
            for block in msg["content"]
        )
        self.assertEqual(image_count, 3)
        text = json.dumps(messages)
        for expected in (
            "Actual observation 0",
            "Actual observation 1",
            "Independent Direct analysis",
            "support statuses are model claims",
        ):
            self.assertIn(expected, text)

    def test_transport_exception_propagates_without_a_second_request(self):
        instance, client = self.build([RequestFailure("HTTP 400")])
        with self.assertRaisesRegex(RequestFailure, "HTTP 400"):
            asyncio.run(instance.plan(self.case["question"], "original"))
        self.assertEqual(len(client.calls), 1)

    def test_two_invalid_responses_stop_and_preserve_both_events(self):
        instance, client = self.build([content({"bad": True}), content({"bad": True})])
        with self.assertRaisesRegex(RequestFailure, "OUTPUT_VALIDATION_FAILED"):
            asyncio.run(instance.plan(self.case["question"], "original"))
        self.assertEqual(len(client.calls), 2)
        self.assertEqual(
            sum(x["event"] == "output_validation_failed" for x in instance.events), 2
        )

    def test_cached_valid_response_reparsed_without_new_transport_validation(self):
        instance, _ = self.build([content(plan_value())], cached=True)
        self.assertEqual(
            asyncio.run(instance.plan(self.case["question"], "original")).goals,
            ("target",),
        )

    def test_context_is_deterministic_and_snapshot_events_preserve_old_files(self):
        instance, _ = self.build([])
        state = self.state()
        before = instance._context(state)
        self.assertEqual(before, instance._context(copy.deepcopy(state)))
        first_files = {
            path: path.read_bytes()
            for path in (self.root / "case" / "events").glob("*.json")
        }
        backend.Backend(
            self.case,
            self.root / "shared",
            self.root / "case",
            FakeClient([]),
            self.baseline,
        )
        self.assertTrue(
            all(path.read_bytes() == raw for path, raw in first_files.items())
        )

    def test_non8b_direct_baseline_is_rejected(self):
        self.baseline["models"] = ["qwen/qwen3-vl-30b-a3b-instruct"]
        with self.assertRaisesRegex(ValueError, "pure-8B"):
            self.build([])


if __name__ == "__main__":
    unittest.main()
