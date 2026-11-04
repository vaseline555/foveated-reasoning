"""Adaptive's 8B transport, using the baseline's ledger and dispatch gate.

A final-reader hold is an undispatched future attempt, visible to the unmodified
baseline's SUM(COALESCE(cost,reserved)). It is never a fabricated API response.
Only the first aggregator dispatch consumes that hold; physical retries receive
ordinary new reservations. Unknown or dispatched attempts are never released.
"""

from __future__ import annotations

import hashlib
import importlib.util
import math
import time
import uuid
from pathlib import Path

_CLIENT_PATH = Path("/workspace/fovea-reproduction-8b-recovery-v2/client.py")
_spec = importlib.util.spec_from_file_location("adaptive_baseline_client", _CLIENT_PATH)
if _spec is None or _spec.loader is None:
    raise ImportError(f"Cannot load baseline transport: {_CLIENT_PATH}")
_baseline = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_baseline)
Client = _baseline.Client
BudgetStop = _baseline.BudgetStop
UnknownCharge = _baseline.UnknownCharge
RequestFailure = _baseline.RequestFailure
RateLimitPause = _baseline.RateLimitPause

METHOD = "ADAPTIVE_V4"
MODEL = "qwen/qwen3-vl-8b-instruct"
FINAL_STAGE = "aggregator"
HOLD_STAGE = "aggregator_reserved"
HOLD_STATUS = "reserved_final"
FINAL_MAX_TOKENS = 2048


class AdaptiveClient(Client):
    """Same constructor as Client; defaults retain the campaign's 3s pacing."""

    def __init__(
        self,
        run_dir,
        models,
        cap=15,
        historical_usd=0,
        stop_callback=None,
        min_request_interval=3.0,
        rate_retry_limit=36,
    ):
        for role in ("primary", "fallback"):
            model = models.get(role, {})
            provider = model.get("provider", {})
            if (
                model.get("id") != MODEL
                or model.get("prompt_price") != 0.25
                or model.get("completion_price") != 0.75
                or model.get("context_length") != 262144
                or provider.get("only") != ["Parasail"]
                or provider.get("allow_fallbacks") is not False
                or provider.get("require_parameters") is not True
                or model.get("temperature", 1.0) != 1.0
            ):
                raise ValueError(
                    "Adaptive requires the frozen 8B Parasail model settings"
                )
        if min_request_interval < 3.0:
            raise ValueError(
                "Adaptive must share a dispatch interval of at least 3 seconds"
            )
        super().__init__(
            run_dir,
            models,
            cap,
            historical_usd,
            stop_callback,
            min_request_interval,
            rate_retry_limit,
        )
        model = self.models["primary"]
        self.final_upper = (
            (model["context_length"] - FINAL_MAX_TOKENS) * model["prompt_price"]
            + FINAL_MAX_TOKENS * model["completion_price"]
        ) / 1e6
        self.final_upper = self.final_upper * 1.02 + 0.000001

    def reserve_final(self, case):
        """Reserve one final reader before any paid work on this case.

        Returning an existing pending ID permits a safe resume. A final request
        already in flight or with an unknown charge requires reconciliation.
        """
        case = str(case)
        token = hashlib.sha256(case.encode()).hexdigest()
        key = METHOD + ":final-reservation:" + token
        with self.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            if connection.execute(
                "SELECT 1 FROM attempts WHERE status='unknown' LIMIT 1"
            ).fetchone():
                raise UnknownCharge(
                    "Unresolved request charge; final reservation not created"
                )
            rows = connection.execute(
                "SELECT * FROM attempts WHERE method=? AND case_id=? AND cost IS NULL "
                "AND stage IN (?,?)",
                (METHOD, case, HOLD_STAGE, FINAL_STAGE),
            ).fetchall()
            if rows:
                if (
                    len(rows) == 1
                    and rows[0]["status"] == HOLD_STATUS
                    and rows[0]["stage"] == HOLD_STAGE
                ):
                    return rows[0]["id"]
                raise UnknownCharge(
                    "Final reader is already reserved/dispatched; inspect ledger"
                )
            total = (
                self.historical_usd
                + connection.execute(
                    "SELECT COALESCE(SUM(COALESCE(cost,reserved)),0) FROM attempts"
                ).fetchone()[0]
            )
            if total + self.final_upper > self.cap + 1e-10:
                raise BudgetStop(
                    f"Campaign cap cannot reserve final reader ${self.final_upper:.8f}"
                )
            attempt = uuid.uuid4().hex
            artifact = self.artifacts / "reservations" / (attempt + ".json")
            _baseline.save(
                artifact,
                {
                    "kind": "undispatched_final_reservation",
                    "attempt_id": attempt,
                    "case_id": case,
                    "method": METHOD,
                    "model": MODEL,
                    "upper_usd": self.final_upper,
                    "created_at": time.time(),
                    "note": "Reservation provenance only; no API request was sent.",
                },
            )
            connection.execute(
                """INSERT INTO attempts
                (id,logical_key,model,role,case_id,method,stage,started,status,reserved,request_path)
                VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    attempt,
                    key,
                    MODEL,
                    "primary",
                    case,
                    METHOD,
                    HOLD_STAGE,
                    time.time(),
                    HOLD_STATUS,
                    self.final_upper,
                    str(artifact),
                ),
            )
        return attempt

    def release_final(self, case):
        """Cancel only an undispatched hold; preserve unknown/in-flight charges."""
        with self.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            result = connection.execute(
                "UPDATE attempts SET status='cancelled_reservation',cost=0,finished=?,error=? "
                "WHERE method=? AND case_id=? AND stage=? AND status=? AND cost IS NULL",
                (
                    time.time(),
                    "Undispatched final-reader hold released; no physical request",
                    METHOD,
                    str(case),
                    HOLD_STAGE,
                    HOLD_STATUS,
                ),
            )
            return result.rowcount

    def _reserve(self, key, model, role, case, method, stage, upper, path):
        if method != METHOD or role != "primary" or model["id"] != MODEL:
            raise ValueError(
                "Adaptive transport accepts only ADAPTIVE_V4 primary 8B calls"
            )
        if not math.isfinite(upper) or upper < 0:
            raise ValueError("Invalid request reservation")
        if stage == FINAL_STAGE:
            with self.connect() as connection:
                connection.execute("BEGIN IMMEDIATE")
                duplicate = connection.execute(
                    "SELECT 1 FROM attempts WHERE logical_key=? AND cost IS NULL LIMIT 1",
                    (key,),
                ).fetchone()
                unknown = connection.execute(
                    "SELECT 1 FROM attempts WHERE status='unknown' LIMIT 1"
                ).fetchone()
                if duplicate or unknown:
                    raise UnknownCharge(
                        "Unresolved request charge; inspect ledger before resume"
                    )
                holds = connection.execute(
                    "SELECT * FROM attempts WHERE method=? AND case_id=? AND stage=? "
                    "AND status=? AND cost IS NULL",
                    (METHOD, str(case), HOLD_STAGE, HOLD_STATUS),
                ).fetchall()
                if len(holds) > 1:
                    raise UnknownCharge("Multiple final-reader holds; inspect ledger")
                if holds:
                    hold = holds[0]
                    if upper > hold["reserved"] + 1e-10:
                        raise BudgetStop(
                            "Final reader exceeds its reserved full-context upper"
                        )
                    connection.execute(
                        "UPDATE attempts SET logical_key=?,model=?,role=?,stage=?,started=?,"
                        "status='reserved',reserved=?,request_path=? WHERE id=?",
                        (
                            key,
                            model["id"],
                            role,
                            stage,
                            time.time(),
                            upper,
                            str(path),
                            hold["id"],
                        ),
                    )
                    return hold["id"]
        return super()._reserve(key, model, role, case, method, stage, upper, path)

    def call(
        self,
        messages,
        stage,
        case,
        method,
        tools=None,
        max_tokens=2048,
        seed=42,
        validator=None,
        force_model=None,
    ):
        if method != METHOD or force_model not in (None, "primary"):
            raise ValueError("Adaptive calls must use ADAPTIVE_V4 and primary 8B")
        if not isinstance(max_tokens, int) or not 1 <= max_tokens <= FINAL_MAX_TOKENS:
            raise ValueError("Adaptive output cap must be 1..2048 tokens")
        result = super().call(
            messages,
            stage,
            case,
            method,
            tools,
            max_tokens,
            seed,
            validator,
            force_model="primary",
        )
        if stage == FINAL_STAGE and result[1].get("cache_hit"):
            self.release_final(case)
        return result

    def final_reservation_stats(self):
        """Nonphysical ledger rows, separate from ordinary physical attempts."""
        with self.connect() as connection:
            return dict(
                connection.execute(
                    """SELECT
                COALESCE(SUM(status='reserved_final'),0) pending_final_reservations,
                COALESCE(SUM(status='cancelled_reservation'),0) cancelled_final_reservations,
                COALESCE(SUM(CASE WHEN status='reserved_final' THEN reserved ELSE 0 END),0)
                    pending_final_usd,
                COALESCE(SUM(status NOT IN ('reserved_final','cancelled_reservation')),0)
                    physical_attempt_rows
                FROM attempts WHERE method=?""",
                    (METHOD,),
                ).fetchone()
            )
