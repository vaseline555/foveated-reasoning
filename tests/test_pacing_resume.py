"""Adaptive scheduling and exact cache replay preserve inputs and financial caps."""

import io
import urllib.error
from copy import deepcopy
from email.utils import formatdate

import pytest

from active_vlm.baselines.runner import FoveaRun
from active_vlm.budget import BudgetStore
from active_vlm.common import read
from active_vlm.pacing import AdaptivePacer, retry_after_seconds
from active_vlm.transport import Transport

from .replay import no_dispatch
from .test_execution import SelectorResponder, config, make_case, response


def test_successes_reduce_interval_and_429_increases_it():
    cfg = dict(
        config(),
        min_dispatch_interval_seconds=15,
        retry_cooldown_seconds=60,
        pacing=dict(
            mode="adaptive",
            minimum_seconds=0.5,
            maximum_seconds=120,
            success_window=3,
            decrease_factor=0.8,
            jitter_fraction=0,
        ),
    )
    clock = [1000.0]
    pacer = AdaptivePacer(cfg, clock=lambda: clock[0])
    for _ in range(3):
        pacer.succeeded()
    assert pacer.state["interval_seconds"] == 12
    assert pacer.failed(429, "90") == 90
    assert pacer.state["interval_seconds"] == 24
    assert pacer.delay() == 90
    assert pacer.failed(429, "1") == 120
    assert pacer.state["interval_seconds"] == 48
    clock[0] += 120
    assert pacer.delay() == 0
    pacer.succeeded()
    assert pacer.state["consecutive_errors"] == 0
    for _ in range(100):
        pacer.succeeded()
    assert pacer.state["interval_seconds"] == 0.5


def test_retry_after_dates_and_server_deadlines_are_respected():
    now = 1000
    assert retry_after_seconds(formatdate(1600, usegmt=True), now) == 600
    assert retry_after_seconds("NaN", now) == 0
    assert retry_after_seconds("broken", now) == 0
    pacer = AdaptivePacer(
        dict(config(), retry_cooldown_seconds=60, pacing=dict(mode="adaptive", jitter_fraction=0)),
        clock=lambda: now,
    )
    assert pacer.failed(503, "1200") == 1200
    assert pacer.delay() == 1200


def test_resume_replays_paid_prefix_without_new_charge_and_retains_failed_hold(tmp_path):
    with no_dispatch():
        case = make_case(tmp_path)
        cfg = config()
        budget = BudgetStore(tmp_path / "ledger.json", cfg, 0, 30, [case["key"] + "/A"])
        base = SelectorResponder()
        failed = []

        def responder(request, meta):
            if not failed:
                failed.append(True)
                raise urllib.error.HTTPError("offline", 429, "rate limit", {}, io.BytesIO(b"{}"))
            return base(request, meta)

        class PauseTransport(Transport):
            def call(self, *args, **kwargs):
                if self.physical_successes == 4:
                    raise KeyboardInterrupt
                return super().call(*args, **kwargs)

        first = PauseTransport(tmp_path, budget, cfg, responder=responder)
        with pytest.raises(KeyboardInterrupt):
            FoveaRun(case, "A", first, tmp_path / "trajectory").execute()
        original = {p.name: p.read_bytes() for p in (tmp_path / "requests").glob("*.json")}
        before = read(budget.path)
        assert float(before["physical_paid"]) == 0.004
        budget.prepare_resume(set())
        resumed_responder = SelectorResponder()
        resumed = Transport(tmp_path, budget, cfg, responder=resumed_responder, resume=True)
        result = FoveaRun(case, "A", resumed, tmp_path / "trajectory").execute()
        assert result["prediction"] == "A" and result["calls"] == 13
        assert result["failed_transport_attempts"] == 1
        assert len(resumed_responder.inputs) == 9
        after = read(budget.path)
        assert float(after["physical_paid"]) == 0.013
        assert float(after["paths"][case["key"] + "/A"]["paid"]) == 0.013
        assert len(after["held"]) == 1
        assert next(iter(after["held"].values()))["status"] == "authorized_unknown"
        assert not after["final_pool"]
        assert all(
            (tmp_path / "requests" / name).read_bytes() == body for name, body in original.items()
        )


def test_resume_input_divergence_blocks_before_dispatch(tmp_path):
    with no_dispatch():
        cfg = config()
        budget = BudgetStore(tmp_path / "ledger.json", cfg, 0, 30, ["case/A"])
        first = Transport(tmp_path, budget, cfg, responder=lambda request, meta: response(meta))
        token = budget.reserve("case/A", ["planner_1"])[0]
        first.call("case", "A", "planner_1", [], token)
        budget.prepare_resume(set())
        dispatched = []
        second = Transport(
            tmp_path,
            budget,
            deepcopy(cfg),
            responder=lambda *args: dispatched.append(args),
            resume=True,
        )
        token = budget.reserve("case/A", ["planner_1"])[0]
        with pytest.raises(RuntimeError, match="REPLAY_DIVERGED"):
            second.call("case", "A", "planner_1", [dict(role="user", content="changed")], token)
        assert not dispatched


def test_http_200_api_timeout_body_retries_with_the_unknown_reservation_kept(tmp_path):
    with no_dispatch():
        cfg = config()
        budget = BudgetStore(tmp_path / "ledger.json", cfg, 0, 30, ["case/A"])
        calls = []

        def responder(request, meta):
            calls.append(meta["request_hash"])
            if len(calls) == 1:
                return dict(
                    id="captured-timeout",
                    error=dict(
                        message="A Timeout Occurred", code=504, metadata=dict(error_type="timeout")
                    ),
                )
            return response(meta)

        transport = Transport(tmp_path, budget, cfg, responder=responder)
        token = budget.reserve("case/A", ["planner_1"])[0]
        _, receipt = transport.call("case", "A", "planner_1", [], token)
        ledger = read(budget.path)
        assert calls[0] == calls[1] and receipt["cost_usd"] == 0.001
        assert transport.physical_successes == transport.physical_failures == 1
        assert len(ledger["held"]) == 1 and not ledger["stopped"]
        assert next(iter(ledger["held"].values()))["status"] == "authorized_unknown"
        assert float(ledger["physical_paid"]) == 0.001
        failed = next((tmp_path / "requests").glob("*.failed_attempt_1.json"))
        assert read(failed)["http_status"] == 504 and read(failed)["cost_usd"] is None
