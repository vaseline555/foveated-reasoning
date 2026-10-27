"""Atomic physical reservations, logical caps, shared final readers, and retries."""

import fcntl
import uuid
from contextlib import contextmanager
from decimal import Decimal
from pathlib import Path

from .common import read, save


class BudgetStop(RuntimeError):
    pass


class UnknownCharge(RuntimeError):
    pass


def dec(x):
    return Decimal(str(x))


class BudgetStore:
    def __init__(self, path, config, historical, cumulative_cap, paths):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.lock_path = self.path.with_suffix(".lock")
        with self.locked(initial=True):
            if not self.path.exists():
                final = dec(config["request_upper_usd"]) * 2
                save(
                    self.path,
                    dict(
                        physical_paid="0",
                        historical=str(historical),
                        cumulative_cap=str(cumulative_cap),
                        stopped=False,
                        upper=str(config["request_upper_usd"]),
                        paths={
                            p: dict(
                                paid="0",
                                final=str(final),
                                cap=str(config["per_case_usd_cap"]),
                                calls=0,
                                probes=0,
                                planners=0,
                            )
                            for p in paths
                        },
                        held={},
                        receipts={},
                        retried_requests={},
                        reservation_schema="shared_direct",
                        shared_direct_receipts={},
                        final_pool={
                            **{
                                p.rsplit("/", 1)[0] + "/direct": config["request_upper_usd"]
                                for p in paths
                            },
                            **{p + "/aggregator": config["request_upper_usd"] for p in paths},
                        },
                    ),
                )
            else:
                existing = read(self.path)
                if any(h["status"] in {"dispatched", "unknown"} for h in existing["held"].values()):
                    existing["stopped"] = True
                    save(self.path, existing)
        with self.locked() as state:
            if set(paths) != set(state["paths"]) or dec(cumulative_cap) != dec(
                state["cumulative_cap"]
            ):
                raise BudgetStop("BUDGET_SCOPE_CHANGED")
            if self.total(state) > dec(cumulative_cap):
                raise BudgetStop("INITIAL_FINAL_RESERVES_EXCEED_GLOBAL_CAP")

    @contextmanager
    def locked(self, initial=False):
        with self.lock_path.open("a+") as handle:
            fcntl.flock(handle, fcntl.LOCK_EX)
            try:
                yield (None if initial else read(self.path))
            finally:
                fcntl.flock(handle, fcntl.LOCK_UN)

    @staticmethod
    def total(state):
        return (
            dec(state["historical"])
            + dec(state["physical_paid"])
            + sum(
                (dec(h["upper"]) for h in state["held"].values() if not h.get("physical_external")),
                dec(0),
            )
            + sum((dec(v) for v in state["final_pool"].values()), dec(0))
        )

    def _reserve_exploration(self, path, stages, final=False):
        with self.locked() as s:
            p = s["paths"][path]
            if p.get("finished"):
                raise BudgetStop("PATH_ALREADY_TERMINAL")
            if s["stopped"]:
                raise UnknownCharge("GLOBAL_UNKNOWN_OR_UPPER_BOUND_STOP")
            upper = dec(s["upper"])
            n = len(stages)
            probes = sum(x.startswith("probe_") for x in stages)
            planners = sum(x.startswith("planner_") for x in stages)
            if (
                p["calls"] + n > (102 if final else 100)
                or p["probes"] + probes > 90
                or p["planners"] + planners > 10
            ):
                raise BudgetStop("CALL_OR_TURN_CAP")
            held_path = sum(dec(x["upper"]) for x in s["held"].values() if x["path"] == path)
            floor = dec("0") if final else dec(p["final"])
            if dec(p["paid"]) + held_path + upper * n + floor > dec(p["cap"]):
                raise BudgetStop("LOGICAL_USD_CAP")
            if final:
                if n != 2 or p["final"] == "0":
                    raise BudgetStop("INVALID_FINAL_RESERVATION")
                p["final"] = "0"
            if self.total(s) + upper * n > dec(s["cumulative_cap"]):
                raise BudgetStop("CUMULATIVE_PHYSICAL_CAP")
            tokens = []
            for stage in stages:
                token = uuid.uuid4().hex
                s["held"][token] = dict(path=path, stage=stage, upper=str(upper), status="reserved")
                tokens.append(token)
            p["calls"] += n
            p["probes"] += probes
            p["planners"] += planners
            save(self.path, s)
            return tokens

    def reserve(self, path, stages, final=False):
        if not final:
            return self._reserve_exploration(path, stages, final=False)
        if stages != ["direct", "aggregator"]:
            raise BudgetStop("INVALID_FINAL_STAGES")
        with self.locked() as state:
            if state["stopped"]:
                raise UnknownCharge("GLOBAL_UNKNOWN_OR_UPPER_BOUND_STOP")
            p = state["paths"][path]
            if p.get("finished"):
                raise BudgetStop("PATH_ALREADY_TERMINAL")
            upper = dec(state["upper"])
            if p["calls"] + 2 > 102 or dec(p["final"]) != 2 * upper:
                raise BudgetStop("INVALID_OR_ALREADY_CONSUMED_FINAL_RESERVATION")
            held = sum(
                (dec(h["upper"]) for h in state["held"].values() if h["path"] == path), dec(0)
            )
            if dec(p["paid"]) + held + 2 * upper > dec(p["cap"]):
                raise BudgetStop("LOGICAL_USD_CAP")
            case, _ = path.rsplit("/", 1)
            direct_key = case + "/direct"
            aggregate_key = path + "/aggregator"
            cached = state["shared_direct_receipts"].get(case)
            if aggregate_key not in state["final_pool"]:
                raise BudgetStop("AGGREGATOR_FINAL_RESERVE_MISSING")
            if direct_key not in state["final_pool"] and (not cached):
                raise BudgetStop("SHARED_DIRECT_PENDING_OR_UNAVAILABLE")
            tokens = []
            for stage, key in [("direct", direct_key), ("aggregator", aggregate_key)]:
                token = uuid.uuid4().hex
                cache_only = stage == "direct" and bool(cached)
                if not cache_only:
                    del state["final_pool"][key]
                state["held"][token] = dict(
                    path=path,
                    stage=stage,
                    upper=str(upper),
                    status="reserved",
                    final_slot=key,
                    physical_external=cache_only,
                    cache_only=cache_only,
                    expected_receipt=cached if cache_only else None,
                )
                tokens.append(token)
            p["final"] = "0"
            p["calls"] += 2
            assert self.total(state) <= dec(state["cumulative_cap"])
            save(self.path, state)
            return tokens

    def cache_expectation(self, token):
        with self.locked() as state:
            return state["held"][token].get("expected_receipt")

    def dispatched(self, token):
        with self.locked() as state:
            if state["stopped"]:
                raise UnknownCharge("GLOBAL_STOP")
            if state["held"][token].get("cache_only"):
                raise BudgetStop("SHARED_DIRECT_CACHE_ONLY_MUST_NOT_DISPATCH")
            state["held"][token]["status"] = "dispatched"
            save(self.path, state)

    def _settle_receipt(self, token, actual, receipt=None, cached=False):
        with self.locked() as s:
            held = s["held"][token]
            if actual is None:
                held["status"] = "unknown"
                s["stopped"] = True
                save(self.path, s)
                raise UnknownCharge("UNKNOWN_COST_NO_RETRY")
            charge = dec(actual)
            if charge < 0:
                raise ValueError("NEGATIVE_COST")
            p = s["paths"][held["path"]]
            p["paid"] = str(dec(p["paid"]) + charge)
            if not cached:
                if not receipt:
                    raise ValueError("MISSING_PHYSICAL_RECEIPT")
                if receipt in s["receipts"]:
                    if dec(s["receipts"][receipt]) != charge:
                        raise ValueError("RECEIPT_COST_CHANGED")
                else:
                    s["receipts"][receipt] = str(charge)
                    s["physical_paid"] = str(dec(s["physical_paid"]) + charge)
            del s["held"][token]
            if charge > dec(held["upper"]) or dec(p["paid"]) > dec(p["cap"]):
                s["stopped"] = True
                save(self.path, s)
                raise UnknownCharge("COST_UPPER_BOUND_VIOLATION")
            save(self.path, s)

    def settle(self, token, actual, receipt=None, cached=False):
        with self.locked() as state:
            held = dict(state["held"][token])
            expected = held.get("expected_receipt")
            if expected and (
                actual is None
                or not cached
                or receipt != expected["receipt_id"]
                or (dec(actual) != dec(expected["cost_usd"]))
            ):
                raise BudgetStop("SHARED_DIRECT_CACHE_RECEIPT_MISMATCH")
        self._settle_receipt(token, actual, receipt, cached)
        if actual is not None and held["stage"] == "direct":
            case, _ = held["path"].rsplit("/", 1)
            with self.locked() as state:
                state["shared_direct_receipts"][case] = dict(
                    receipt_id=receipt, cost_usd=str(actual)
                )
                save(self.path, state)

    def release_unused(self, tokens):
        with self.locked() as state:
            for token in tokens:
                held = state["held"].get(token)
                if not held or held["status"] != "reserved":
                    continue
                p = state["paths"][held["path"]]
                if held.get("final_slot"):
                    if not held.get("cache_only"):
                        state["final_pool"][held["final_slot"]] = held["upper"]
                    p["final"] = str(dec(p["final"]) + dec(held["upper"]))
                del state["held"][token]
                p["calls"] -= 1
                p["probes"] -= held["stage"].startswith("probe_")
                p["planners"] -= held["stage"].startswith("planner_")
            save(self.path, state)

    def halt(self):
        with self.locked() as s:
            s["stopped"] = True
            save(self.path, s)

    def finish_path(self, path):
        """Release unused final-reader protection, never payments or error holds."""
        with self.locked() as state:
            if state["stopped"] or any(
                h["path"] == path and h["status"] in {"reserved", "dispatched", "unknown"}
                for h in state["held"].values()
            ):
                raise BudgetStop("CANNOT_FINISH_UNSETTLED_PATH")
            state["paths"][path].update(finished=True, final="0")
            state["final_pool"].pop(path + "/aggregator", None)
            case = path.rsplit("/", 1)[0]
            if all(
                v.get("finished") for p, v in state["paths"].items() if p.rsplit("/", 1)[0] == case
            ):
                state["final_pool"].pop(case + "/direct", None)
            save(self.path, state)

    def retry_token(self, token, request_hash):
        """Retain failed money reservation; retry the same one logical slot."""
        with self.locked() as state:
            old = state["held"][token]
            if old["status"] != "unknown" or old.get("cache_only"):
                raise BudgetStop("RETRY_REQUIRES_A_LOGGED_UNKNOWN_TRANSPORT_ATTEMPT")
            if any(
                (
                    h["status"] in {"unknown", "dispatched"}
                    for k, h in state["held"].items()
                    if k != token
                )
            ):
                raise BudgetStop("UNRELATED_UNRESOLVED_REQUEST")
            upper = dec(state["upper"])
            p = state["paths"][old["path"]]
            held = sum(
                (dec(h["upper"]) for h in state["held"].values() if h["path"] == old["path"]),
                dec(0),
            )
            if self.total(state) + upper > dec(state["cumulative_cap"]):
                raise BudgetStop("RETRY_CUMULATIVE_PHYSICAL_CAP")
            if dec(p["paid"]) + held + upper + dec(p["final"]) > dec(p["cap"]):
                raise BudgetStop("RETRY_LOGICAL_USD_CAP")
            old.update(status="authorized_unknown", excluded_from_success_count=True)
            new = uuid.uuid4().hex
            state["held"][new] = dict(
                path=old["path"], stage=old["stage"], upper=str(upper), status="reserved"
            )
            state.setdefault("retry_attempts", []).append(
                dict(request_hash=request_hash, failed_token=token, retry_token=new)
            )
            state["stopped"] = False
            save(self.path, state)
            return new

    def refresh_historical(self, amount):
        with self.locked() as s:
            s["historical"] = str(max(dec(s["historical"]), dec(amount)))
            if self.total(s) > dec(s["cumulative_cap"]):
                s["stopped"] = True
                save(self.path, s)
                raise BudgetStop("CONCURRENT_SPEND_CUMULATIVE_CAP")
            save(self.path, s)

    def prepare_resume(self, completed):
        """Replay unfinished logical paths; retain every physical charge and hold."""
        with self.locked() as state:
            if state["stopped"] or any(
                h["status"] in {"unknown", "dispatched"} for h in state["held"].values()
            ):
                raise BudgetStop("RECONCILE_UNRESOLVED_DISPATCH_BEFORE_RESUMING")
            state["held"] = {k: h for k, h in state["held"].items() if h["status"] != "reserved"}
            state["final_pool"] = {}
            upper = state["upper"]
            for path, values in state["paths"].items():
                if path in completed:
                    continue
                values.update(paid="0", calls=0, probes=0, planners=0, final=str(dec(upper) * 2))
                case = path.rsplit("/", 1)[0]
                if case not in state["shared_direct_receipts"]:
                    state["final_pool"][case + "/direct"] = upper
                state["final_pool"][path + "/aggregator"] = upper
            if self.total(state) > dec(state["cumulative_cap"]):
                raise BudgetStop("RESUME_FINAL_RESERVES_EXCEED_CAP")
            save(self.path, state)
