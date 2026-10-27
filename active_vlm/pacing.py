"""Serial adaptive pacing, independent of model inputs and sampling randomness."""

import math
import random
import time
from datetime import UTC
from email.utils import parsedate_to_datetime


def retry_after_seconds(value, now=None):
    if not value:
        return 0.0
    try:
        result = float(value)
    except (ValueError, TypeError):
        try:
            date = parsedate_to_datetime(value)
            if date.tzinfo is None:
                date = date.replace(tzinfo=UTC)
            result = date.timestamp() - (time.time() if now is None else now)
        except (ValueError, TypeError, OverflowError):
            return 0.0
    return max(0.0, result) if math.isfinite(result) else 0.0


class AdaptivePacer:
    def __init__(self, config, state=None, clock=time.time, uniform=None):
        policy = config.get("pacing", {})
        self.enabled = policy.get("mode") == "adaptive"
        self.minimum = float(policy.get("minimum_seconds", 0.5))
        self.maximum = float(policy.get("maximum_seconds", 120))
        self.success_window = int(policy.get("success_window", 3))
        self.decrease = float(policy.get("decrease_factor", 0.8))
        self.base_retry = float(config.get("retry_cooldown_seconds", 60))
        self.max_retry = float(policy.get("maximum_backoff_seconds", 300))
        self.jitter = float(policy.get("jitter_fraction", 0.1))
        self.clock = clock
        self.uniform = uniform or random.SystemRandom().uniform
        self.state = dict(
            interval_seconds=float(config["min_dispatch_interval_seconds"]),
            consecutive_successes=0,
            consecutive_errors=0,
            last_dispatch_unix=0.0,
            not_before_unix=0.0,
            rate_limits=0,
        )
        if state:
            self.state.update(state)
        if self.enabled and not (
            0 < self.minimum <= self.maximum
            and 0 < self.decrease < 1
            and self.success_window > 0
            and 0 <= self.jitter <= 1
        ):
            raise ValueError("INVALID_ADAPTIVE_PACING_POLICY")

    def delay(self):
        return max(
            0.0,
            self.state["not_before_unix"] - self.clock(),
            self.state["last_dispatch_unix"] + self.state["interval_seconds"] - self.clock(),
        )

    def dispatched(self):
        self.state["last_dispatch_unix"] = self.clock()

    def succeeded(self):
        self.state["consecutive_errors"] = 0
        self.state["consecutive_successes"] += 1
        if self.enabled and self.state["consecutive_successes"] >= self.success_window:
            self.state["interval_seconds"] = max(
                self.minimum, self.state["interval_seconds"] * self.decrease
            )
            self.state["consecutive_successes"] = 0

    def failed(self, http_status=None, retry_after=None):
        self.state["consecutive_successes"] = 0
        self.state["consecutive_errors"] += 1
        if self.enabled and http_status == 429:
            self.state["rate_limits"] += 1
            self.state["interval_seconds"] = min(
                self.maximum, max(self.minimum, self.state["interval_seconds"]) * 2
            )
        backoff = self.base_retry
        if self.enabled:
            exponent = min(self.state["consecutive_errors"] - 1, 10)
            backoff = min(self.max_retry, backoff * 2**exponent)
        # Server deadlines take precedence over the local backoff ceiling.
        backoff = max(backoff, retry_after_seconds(retry_after, self.clock()))
        if self.enabled:
            backoff += self.uniform(0, backoff * self.jitter)
        self.state["not_before_unix"] = self.clock() + backoff
        return backoff
