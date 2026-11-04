"""Runner contracts checked offline; no API or production-ledger calls."""

from __future__ import annotations

import asyncio
import json
import sqlite3
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import run as runner


class RunnerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.output = self.root / "adaptive"
        self.output.mkdir()
        self.case = {"key": "vstar_test", "question": "Which object? (A) One (B) Two"}

    def tearDown(self):
        self.temp.cleanup()

    def test_answer_parsing_scores_no_valid_choice_as_model_outcome(self):
        for text, expected in [
            ("<answer>B</answer>", "B"),
            ("The answer is (C).", "C"),
            ("The object cannot be identified.", None),
            ("", None),
            ("<answer>FAILED</answer>", None),
        ]:
            with self.subTest(text=text):
                self.assertEqual(runner.parse_answer(text), expected)

    def test_overlay_precedes_original_and_requires_pure_8b(self):
        shared = self.root / "shared"
        for folder, answer in [("rows", "A"), ("result_overlays", "B")]:
            path = shared / folder / "Direct" / "vstar_test.json"
            runner.save(
                path,
                {
                    "status": "complete",
                    "models": ["qwen/qwen3-vl-8b-instruct"],
                    "message": {"content": answer},
                },
            )
        with patch.object(runner, "SHARED", shared):
            row, path = runner.direct_result(self.case)
            self.assertEqual(row["message"]["content"], "B")
            self.assertIn("result_overlays", str(path))
            row["models"] = ["qwen/qwen3-vl-30b-a3b-instruct"]
            runner.save(path, row)
            with self.assertRaises(ValueError):
                runner.direct_result(self.case)

    def test_existing_terminal_row_never_repeats_paid_work(self):
        path = self.output / "rows" / "vstar_test.json"
        expected = {"case": "vstar_test", "status": "execution_failure"}
        runner.save(path, expected)
        client = Mock()
        self.assertEqual(
            asyncio.run(runner.execute_case(self.case, self.output, client, {})),
            expected,
        )
        client.reserve_final.assert_not_called()

    def test_final_hold_released_when_backend_fails(self):
        direct = self.root / "direct.json"
        runner.save(direct, {"status": "complete"})
        client = Mock()
        with (
            patch.object(runner, "SHARED", self.root / "shared"),
            patch.object(runner, "direct_result", return_value=({}, direct)),
            patch.object(
                runner, "Backend", side_effect=ValueError("bad backend input")
            ),
            self.assertRaisesRegex(ValueError, "bad backend input"),
        ):
            asyncio.run(runner.execute_case(self.case, self.output, client, {}))
        client.reserve_final.assert_called_once_with("vstar_test")
        client.release_final.assert_called_once_with("vstar_test")
        self.assertFalse((self.output / "rows" / "vstar_test.json").exists())

    def test_completed_no_answer_preserves_response_and_final_hold_cleanup(self):
        direct = self.root / "direct.json"
        runner.save(direct, {"status": "complete"})
        state = SimpleNamespace(steps=0, model_calls=2, stop_reason="evidence_ready")

        class Policy:
            def __init__(self, limits):
                pass

            async def run(self, *args):
                return SimpleNamespace(
                    answer="Insufficient visual evidence.", state=state
                )

        backend = SimpleNamespace(
            original_image_key="original",
            executor=Mock(),
            provenance={},
            image_registry={"original": {}},
        )
        client = Mock()
        # Real policy state is a dataclass; use a minimal matching serializable substitute.
        from dataclasses import make_dataclass

        state_type = make_dataclass(
            "State", [("steps", int), ("model_calls", int), ("stop_reason", str)]
        )
        state = state_type(0, 2, "evidence_ready")
        with (
            patch.object(runner, "SHARED", self.root / "shared"),
            patch.object(runner, "direct_result", return_value=({}, direct)),
            patch.object(runner, "Backend", return_value=backend),
            patch.object(runner.adaptive, "AdaptivePolicy", Policy),
        ):
            row = asyncio.run(
                runner.execute_case(self.case, self.output, client, {"limits": {}})
            )
        self.assertEqual(row["status"], "complete")
        self.assertIsNone(row["prediction"])
        self.assertEqual(row["grade"], "NO_ANSWER")
        self.assertEqual(row["answer"], "Insufficient visual evidence.")
        client.release_final.assert_called_once_with("vstar_test")

    def test_summary_keeps_nonphysical_reservations_out_of_request_counts(self):
        prep = self.root / "prep"
        runner.save(prep / "labels.evaluation-only.json", {"vstar_test": "A"})
        runner.save(
            self.output / "rows" / "vstar_test.json",
            {
                "case": "vstar_test",
                "status": "complete",
                "prediction": None,
                "crops": 0,
            },
        )
        database = self.root / "ledger.sqlite"
        connection = sqlite3.connect(database)
        connection.execute(
            "CREATE TABLE attempts(method,case_id,stage,status,error,model,response_path,cost,reserved,started)"
        )
        for status, cost, reserve in [
            ("reserved_final", None, 0.07),
            ("cancelled_reservation", 0.0, 0.07),
            ("valid", 0.01, 0.03),
            ("rejected", 0.0, 0.03),
        ]:
            connection.execute(
                "INSERT INTO attempts VALUES(?,?,?,?,?,?,?,?,?,?)",
                (
                    runner.METHOD,
                    "vstar_test",
                    "plan",
                    status,
                    None,
                    "qwen/qwen3-vl-8b-instruct",
                    None,
                    cost,
                    reserve,
                    1,
                ),
            )
        connection.commit()
        connection.close()

        class Client:
            def connect(self):
                result = sqlite3.connect(database)
                result.row_factory = sqlite3.Row
                return result

            def stats(self):
                return {"confirmed_usd": 1.01, "held_usd": 0.07}

            def final_reservation_stats(self):
                return {"pending_final_reservations": 1, "pending_final_usd": 0.07}

        with patch.object(runner, "PREP", prep):
            report = runner.summarize(self.output, Client())
        self.assertEqual(report["physical"]["attempts"], 2)
        self.assertEqual(report["request_error_events"], 1)
        self.assertEqual(report["no_answer"], 1)
        self.assertEqual(report["failed"], 0)
        self.assertEqual(report["completed"], 1)
        self.assertIsNone(report["accuracy_full191"])
        self.assertEqual(report["shared_campaign"]["confirmed_usd"], 1.01)
        errors = json.loads((self.output / "harness_errors.json").read_text())
        self.assertEqual(errors["failed_results"], [])

        # A failed smoke has no result row, but must still consume one case in
        # the failure denominator and retain work executed before its failure.
        runner.save(
            self.output / "smoke_attempts" / "failure.json",
            {
                "case": "vstar_other",
                "status": "execution_failure",
                "completed_at": 1,
                "crops": 2,
                "crop_attempts": 3,
            },
        )
        with patch.object(runner, "PREP", prep):
            report = runner.summarize(self.output, Client())
        self.assertEqual(report["failed"], 1)
        self.assertEqual(report["failed_result_rows"], 0)
        self.assertEqual(report["failed_smoke_cases"], 1)
        self.assertEqual(report["pending"], 189)
        self.assertEqual(report["not_attempted"], 189)
        self.assertEqual(report["crops"], 2)
        self.assertEqual(report["crop_attempts"], 3)

        # An in-flight case has no terminal row yet; it is already attempted.
        with sqlite3.connect(database) as connection:
            connection.execute(
                "INSERT INTO attempts VALUES(?,?,?,?,?,?,?,?,?,?)",
                (
                    runner.METHOD,
                    "vstar_inflight",
                    "plan",
                    "dispatched",
                    None,
                    "qwen/qwen3-vl-8b-instruct",
                    None,
                    None,
                    0.01,
                    2,
                ),
            )
        with patch.object(runner, "PREP", prep):
            report = runner.summarize(self.output, Client())
        self.assertEqual(report["pending"], 189)
        self.assertEqual(report["not_attempted"], 188)
        self.assertEqual(report["partial_attempted_cases"], 1)
        self.assertEqual(report["physical"]["attempts"], 3)

    def test_health_requires_three_fixed_cases_and_an_actual_observation(self):
        runner.save(
            self.output / "config.json",
            {"health_cases": ["vstar_0", "vstar_115", "vstar_1"]},
        )
        runner.save(
            self.output / "rows" / "vstar_0.json",
            {
                "case": "vstar_0",
                "status": "complete",
                "crops": 1,
                "observed_crops": 1,
            },
        )
        self.assertFalse(runner.refresh_health(self.output)["ready"])
        runner.save(
            self.output / "rows" / "vstar_115.json",
            {
                "case": "vstar_115",
                "status": "complete",
                "crops": 0,
                "observed_crops": 0,
            },
        )
        self.assertFalse(runner.refresh_health(self.output)["ready"])
        runner.save(
            self.output / "rows" / "vstar_1.json",
            {"case": "vstar_1", "status": "complete", "crops": 0, "observed_crops": 0},
        )
        self.assertTrue(runner.refresh_health(self.output)["ready"])
        runner.save(
            self.output / "smoke_attempts" / "failure.json",
            {
                "case": "vstar_2",
                "status": "execution_failure",
                "completed_at": 1,
            },
        )
        self.assertFalse(runner.refresh_health(self.output)["ready"])

    def test_smoke_failure_publishes_attention_and_an_immutable_attempt(self):
        prep = self.root / "prep"
        runner.save(prep / "manifest.json", {"cases": [self.case]})
        runner.save(self.output / "config.json", {"source_hashes": {}})
        with (
            patch.object(runner, "PREP", prep),
            patch.object(runner, "SHARED", self.root / "shared"),
            patch.object(runner, "client_for", return_value=Mock()),
            patch.object(
                runner,
                "execute_case",
                AsyncMock(side_effect=runner.RequestFailure("invalid goals")),
            ),
            patch.object(
                runner, "summarize", return_value={"completed": 0, "failed": 1}
            ),
            patch("builtins.print"),
        ):
            self.assertEqual(runner.smoke(self.output, "vstar_test"), 1)
        state = runner.read(self.output / "state.json")
        self.assertEqual(state["status"], "attention_required")
        self.assertEqual(state["case"], "vstar_test")
        self.assertEqual(len(runner.smoke_records(self.output)), 1)
        self.assertEqual(
            runner.smoke_records(self.output)[0]["status"], "execution_failure"
        )
        self.assertFalse((self.output / "rows" / "vstar_test.json").exists())

    def test_explicit_smoke_case_is_selected_without_evaluation_labels(self):
        prep = self.root / "prep"
        other = dict(self.case, key="vstar_other")
        runner.save(prep / "manifest.json", {"cases": [self.case, other]})
        runner.save(self.output / "config.json", {"source_hashes": {}})
        execute = AsyncMock(return_value={"case": "vstar_other", "status": "complete"})
        with (
            patch.object(runner, "PREP", prep),
            patch.object(runner, "SHARED", self.root / "shared"),
            patch.object(runner, "client_for", return_value=Mock()),
            patch.object(runner, "execute_case", execute),
            patch.object(runner, "summarize", return_value={"completed": 1}),
            patch("builtins.print"),
        ):
            self.assertEqual(runner.smoke(self.output, "vstar_other"), 0)
        self.assertEqual(execute.call_args.args[0]["key"], "vstar_other")
        self.assertFalse((prep / "labels.evaluation-only.json").exists())

    def test_full_run_stops_after_three_consecutive_contract_failures(self):
        prep = self.root / "prep"
        cases = [dict(self.case, key=f"vstar_{index}") for index in range(6)]
        runner.save(prep / "manifest.json", {"cases": cases})
        runner.save(
            self.output / "config.json",
            {
                "source_hashes": {},
                "seed": 42,
                "max_consecutive_structured_failures": 3,
            },
        )
        execute = AsyncMock(side_effect=runner.RequestFailure("invalid goals"))
        with (
            patch.object(runner, "PREP", prep),
            patch.object(runner, "SHARED", self.root / "shared"),
            patch.object(runner, "client_for", return_value=Mock()),
            patch.object(runner, "refresh_health", return_value={"ready": True}),
            patch.object(runner, "execute_case", execute),
            patch.object(
                runner, "summarize", return_value={"completed": 0, "failed": 3}
            ),
            patch("builtins.print"),
        ):
            runner.run(self.output)
        self.assertEqual(execute.call_count, 3)
        self.assertEqual(len(list((self.output / "rows").glob("*.json"))), 3)
        state = runner.read(self.output / "state.json")
        self.assertEqual(state["status"], "attention_required")
        self.assertIn("3 consecutive", state["reason"])


if __name__ == "__main__":
    unittest.main()
