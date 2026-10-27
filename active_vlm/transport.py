"""OpenRouter HTTP, exact request cache, and auditable transient retries."""

import base64
import io
import json
import math
import re
import time
import urllib.error
import urllib.request
from copy import deepcopy
from pathlib import Path

from .answers import answer_mode, direct_mode
from .budget import BudgetStop, UnknownCharge
from .common import MODEL, PROVIDER, canonical, digest, read, save, seed_for
from .errors import IntegrityError
from .experiment.accounting import approval, finished, snapshot
from .pacing import AdaptivePacer
from .readers import RETRY_STAGES, recovery_policy


def semantic_request(request):
    out = deepcopy(request)
    for m in out["messages"]:
        if isinstance(m.get("content"), list):
            for b in m["content"]:
                if b.get("type") == "image_url" and isinstance(b["image_url"]["url"], dict):
                    b["image_url"]["url"].pop("path", None)
    return out


def wire_request(request):
    out = deepcopy(request)
    images = []
    for mi, m in enumerate(out["messages"]):
        if isinstance(m.get("content"), list):
            for bi, b in enumerate(m["content"]):
                if b.get("type") == "image_url":
                    view = b["image_url"]["url"]
                    data = Path(view["path"]).read_bytes()
                    if digest(data) != view["sha256"]:
                        raise IntegrityError("WIRE_IMAGE_CHANGED")
                    b["image_url"]["url"] = (
                        "data:image/jpeg;base64," + base64.b64encode(data).decode()
                    )
                    images.append(dict(message=mi, block=bi, **view))
    return (out, images)


class Transport:
    def __init__(self, root, budget, config, api_key="", responder=None, resume=False):
        # Ordinary planner/probe/readers keep their frozen output settings.
        tokens = config.get("max_tokens", 2048)
        temperature = config.get("temperature", 0.7)
        self.final_answer_mode = answer_mode(config)
        self.direct_answer_mode = direct_mode(config)
        if type(tokens) is not int or not 1 <= tokens <= 2048:
            raise ValueError("OUTPUT_TOKEN_CAP_MUST_BE_AN_INTEGER_IN_1_TO_2048")
        self.reader_recovery = recovery_policy(config)
        if self.reader_recovery["max_length_retries"]:
            required = (
                262144 * config["input_price_per_million"]
                + max(tokens, self.reader_recovery["max_tokens"])
                * config["output_price_per_million"]
            ) / 1e6
            if config["request_upper_usd"] + 1e-12 < required:
                raise ValueError("NATIVE_READER_RECOVERY_NOT_COVERED_BY_RESERVATION")
        if "choice" in {self.final_answer_mode, self.direct_answer_mode} and tokens < 16:
            raise ValueError("CHOICE_OUTPUT_REQUIRES_AT_LEAST_16_TOKENS")
        if (
            type(temperature) not in {int, float}
            or not math.isfinite(temperature)
            or not 0 <= temperature <= 2
        ):
            raise ValueError("TEMPERATURE_MUST_BE_FINITE_IN_0_TO_2")
        if responder is None:
            approval(root)
        if responder is None and config != read(root / "config.json"):
            raise RuntimeError("APPROVED_CONFIG_MISMATCH")
        self.output = root / "requests"
        self.output.mkdir(parents=True, exist_ok=True)
        self.budget, self.config = (budget, config)
        self.responder = responder
        self.records = []
        self.last_dispatch = 0.0
        self.root = root
        self.api_key = api_key
        self.approval_bound = responder is None
        self.offline_only = responder is not None
        self.pacer = AdaptivePacer(
            config,
            read(root / "pacing_state.json") if (root / "pacing_state.json").exists() else None,
        )
        previous = [read(p) for p in self.output.glob("*.json")] if resume else []
        self.replay_expected = {}
        self.physical_successes = sum(r.get("status") == "ok" for r in previous)
        self.physical_failures = sum(r.get("status") == "error" for r in previous)
        if resume:
            completed = {
                p.parent.parent.name + "/" + p.parent.name
                for p in (root / "trajectories").glob("*/*/result.json")
                if finished(read(p))
            }
            for record in previous:
                path = record["case"] + "/" + record["arm"]
                if path in completed:
                    continue
                if record.get("status") == "ok":
                    self.replay_expected.setdefault(path, set()).add(record["request_hash"])
                elif record.get("status") == "error":
                    self.records.append(record)
        if responder is None and (not api_key):
            raise RuntimeError("OPENROUTER_API_KEY is required")
        if responder is None and any(
            read(p).get("backend") == "mock" for p in self.output.glob("*.json")
        ):
            raise RuntimeError("Mock records cannot be used in a paid run")

    def request(self, case, arm, stage, messages, tools=None, schema=None, tool_choice=None):
        base_stage = RETRY_STAGES.get(stage, stage)
        recovery = stage in RETRY_STAGES
        if recovery and (
            not self.reader_recovery["max_length_retries"]
            or tools is not None
            or schema is not None
            or (base_stage == "direct" and self.direct_answer_mode != "native")
            or (base_stage == "aggregator" and self.final_answer_mode != "native")
        ):
            raise ValueError("INVALID_NATIVE_READER_RECOVERY_REQUEST")
        seed_arm = self.config.get("seed_arm_aliases", {}).get(arm, arm)
        seed_stage = "direct" if base_stage == "direct" else seed_arm + "/" + base_stage
        request = dict(
            model=self.config.get("model", MODEL),
            provider=PROVIDER,
            messages=deepcopy(messages),
            temperature=self.config.get("temperature", 0.7),
            max_tokens=(
                self.reader_recovery["max_tokens"]
                if recovery
                else min(self.config.get("max_tokens", 2048), 64)
                if (stage == "aggregator" and self.final_answer_mode == "choice")
                or (stage == "direct" and self.direct_answer_mode == "choice")
                else self.config.get("max_tokens", 2048)
            ),
            seed=seed_for(case, seed_stage, seed=self.config.get("seed", 0)),
        )
        if tools:
            request.update(tools=tools, tool_choice=tool_choice or "auto")
        if schema:
            request["response_format"] = dict(
                type="json_schema",
                json_schema=dict(
                    name="plan" if stage.startswith("planner_") else "report",
                    strict=True,
                    schema=schema,
                ),
            )
        request["provider"] = deepcopy(self.config.get("provider", PROVIDER))
        request["provider"]["max_price"] = dict(
            prompt=self.config["input_price_per_million"],
            completion=self.config["output_price_per_million"],
            request=0,
            image=0,
        )
        if re.search(r"[\uac00-\ud7a3\u1100-\u11ff]", canonical(semantic_request(request))):
            raise ValueError("API_TEXT_MUST_BE_ENGLISH_NO_HANGUL")
        return request

    def call(self, case, arm, stage, messages, token, tools=None, schema=None, tool_choice=None):
        while True:
            self.check_deadline()
            if (self.root / "stop_requested.json").exists():
                raise KeyboardInterrupt
            try:
                return self._call_once(
                    case, arm, stage, messages, token, tools, schema, tool_choice
                )
            except (urllib.error.URLError, TimeoutError, ConnectionError) as error:
                if isinstance(error, urllib.error.HTTPError) and error.code not in {
                    408,
                    429,
                    500,
                    502,
                    503,
                    504,
                }:
                    raise
                request = self.request(case, arm, stage, messages, tools, schema, tool_choice)
                key = digest(semantic_request(request))
                target = self.output / (key + ".json")
                record = read(target)
                if record["status"] != "error" or record.get("cost_usd") is not None:
                    raise
                n = 1
                while (self.output / (key + f".failed_attempt_{n}.json")).exists():
                    n += 1
                target.rename(self.output / (key + f".failed_attempt_{n}.json"))
                token = self.budget.retry_token(token, key)
                cooldown = self.pacer.failed(record.get("http_status"), record.get("retry_after"))
                save(self.root / "pacing_state.json", self.pacer.state)
                self.update_progress(retry_wait_seconds=cooldown)
                print(
                    f"  API retry {n}: {case}/{arm}/{stage}; {type(error).__name__}; waiting {cooldown}s",
                    flush=True,
                )
                self.wait(cooldown)

    def wait(self, seconds):
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            self.check_deadline()
            if (self.root / "stop_requested.json").exists():
                raise KeyboardInterrupt
            time.sleep(min(1, max(0, deadline - time.monotonic())))

    def check_deadline(self):
        if self.config.get("deadline_unix") is not None and time.time() >= self.config["deadline_unix"]:
            raise BudgetStop("GLOBAL_EXPERIMENT_DEADLINE")

    def update_progress(self, **changes):
        from .experiment.process import progress

        state = read(self.budget.path)
        held = sum(
            float(h["upper"])
            for h in state["held"].values()
            if h["status"] in {"unknown", "authorized_unknown", "dispatched"}
            and not h.get("physical_external")
        )
        progress(
            self.root,
            successful_physical_requests=self.physical_successes,
            failed_attempts=self.physical_failures,
            paid_api_calls=self.physical_successes + self.physical_failures,
            known_usd=float(state["physical_paid"]),
            held_usd=held,
            adaptive_interval_seconds=self.pacer.state["interval_seconds"],
            **changes,
        )

    def _dispatch(self, wire):
        req = urllib.request.Request(
            "https://openrouter.ai/api/v1/chat/completions",
            data=json.dumps(wire, allow_nan=False).encode(),
            headers={"Content-Type": "application/json", "Authorization": "Bearer " + self.api_key},
        )
        self.last_dispatch = time.monotonic()
        self.pacer.dispatched()
        self.dispatch_started_unix = self.pacer.state["last_dispatch_unix"]
        save(self.root / "pacing_state.json", self.pacer.state)
        http_start = time.perf_counter()
        try:
            with urllib.request.urlopen(req, timeout=180) as response:
                return json.load(response)
        finally:
            self.http_seconds = time.perf_counter() - http_start

    def _call_once(
        self, case, arm, stage, messages, token, tools=None, schema=None, tool_choice=None
    ):
        req = self.request(case, arm, stage, messages, tools, schema, tool_choice)
        key = digest(semantic_request(req))
        target = self.output / (key + ".json")
        meta = dict(
            case=case,
            arm=arm,
            stage=stage,
            request_hash=key,
            request=req,
        )
        expected = self.budget.cache_expectation(token)
        if target.exists():
            r = read(target)
            if digest(semantic_request(r["request"])) != key:
                raise IntegrityError("CACHE_REQUEST_HASH_MISMATCH")
            if r.get("imported_from"):
                source = read(r["imported_from"])
                if (
                    digest(__import__("pathlib").Path(r["imported_from"]).read_bytes())
                    != r["imported_sha256"]
                ):
                    raise IntegrityError("Imported source receipt changed")
                if semantic_request(r["request"]) != semantic_request(source["request"]) or any(
                    r.get(k) != source.get(k)
                    for k in ["message", "receipt_id", "cost_usd", "finish_reason"]
                ):
                    raise IntegrityError("Imported receipt copy changed")
            if r["status"] != "ok":
                raise UnknownCharge("FAILED_REQUEST_NOT_AUTOMATICALLY_RETRIED")
            if expected and (
                r["receipt_id"] != expected["receipt_id"]
                or str(r["cost_usd"]) != expected["cost_usd"]
            ):
                raise UnknownCharge("SHARED_DIRECT_CACHE_MISMATCH")
            self.budget.settle(token, r["cost_usd"], r["receipt_id"], cached=True)
            rec = dict(
                **meta,
                cached=True,
                cost_usd=r["cost_usd"],
                receipt_id=r["receipt_id"],
                finish_reason=r["finish_reason"],
            )
            self.records.append(rec)
            self.replay_expected.get(case + "/" + arm, set()).discard(key)
            return (deepcopy(r["message"]), rec)
        if expected:
            raise UnknownCharge("SHARED_DIRECT_CACHE_MISSING")
        if self.replay_expected.get(case + "/" + arm):
            raise IntegrityError("RESUME_REPLAY_DIVERGED_BEFORE_NEW_API_CALL")
        if self.responder is None:
            try:
                approval(self.root)
                history = snapshot(exclude=self.root, account_root=self.config.get("account_root"))[
                    "known_or_reserved_usd"
                ]
            except (OSError, ValueError, RuntimeError) as error:
                raise IntegrityError("PRE_DISPATCH_INTEGRITY_CHECK_FAILED") from error
            self.budget.refresh_historical(history)
        wire, images = wire_request(req)
        meta.update(wire_images=images, wire_sha256=digest(wire))
        if len(json.dumps(wire).encode()) > 64000000:
            raise ValueError("WIRE_SIZE_CAP")
        # Wait/check BEFORE recording a dispatched charge. A deadline reached
        # while pacing must not fabricate a failed physical request or hold.
        pause = self.pacer.delay() if self.responder is None else 0
        self.wait(pause)
        self.check_deadline()
        rec = dict(
            **meta,
            status="pending",
            reserve_usd=self.config["request_upper_usd"],
            started_unix=time.time(),
            cached=False,
            backend="mock" if self.responder else "api",
        )
        save(target, rec)
        self.budget.dispatched(token)
        start = time.perf_counter()
        self.throttle_wait = pause
        self.http_seconds = 0
        try:
            body = self.responder(req, meta) if self.responder else self._dispatch(wire)
            amount = body.get("usage", {}).get("cost")
            receipt = body.get("id")
            rec.update(response=body, cost_usd=amount, receipt_id=receipt)
            if isinstance(body.get("error"), dict):
                error = body["error"]
                code = error.get("code")
                if isinstance(code, int) and 400 <= code <= 599:
                    # Gateways can return HTTP 200 with an API-level 504/429 body.
                    # Use the same retained-reservation retry policy as HTTP errors.
                    raise urllib.error.HTTPError(
                        "https://openrouter.ai/api/v1/chat/completions",
                        code,
                        str(error.get("message", "API error")),
                        None,
                        io.BytesIO(json.dumps(body).encode()),
                    )
            if (
                type(amount) not in {int, float}
                or not math.isfinite(amount)
                or amount < 0
                or not receipt
            ):
                rec["cost_usd"] = None
                self.budget.settle(token, None)
            self.budget.settle(token, amount, receipt)
            if body.get("model") != self.config["model"] or body.get("provider") != "Parasail":
                self.budget.halt()
                raise UnknownCharge("PINNED_MODEL_PROVIDER_MISMATCH")
            choice = body["choices"][0]
            if choice.get("finish_reason") not in {"stop", "tool_calls", "length", "end_turn"}:
                raise RuntimeError("INTERRUPTED_RESPONSE")
            rec.update(
                status="ok", message=choice["message"], finish_reason=choice["finish_reason"]
            )
            self.pacer.succeeded()
            save(self.root / "pacing_state.json", self.pacer.state)
            return (deepcopy(rec["message"]), rec)
        except Exception as error:
            rec.update(status="error", error_type=type(error).__name__)
            if isinstance(error, urllib.error.HTTPError):
                rec["http_status"] = error.code
                rec["retry_after"] = error.headers.get("Retry-After") if error.headers else None
                try:
                    info = json.loads(error.read(65536)).get("error", {})
                    rec["api_error_message"] = str(info.get("message", ""))[:1000].replace(
                        self.api_key, "[REDACTED]"
                    )
                except (ValueError, TypeError):
                    pass
            if rec.get("cost_usd") is None:
                try:
                    self.budget.settle(token, None)
                except UnknownCharge:
                    pass
            else:
                self.budget.halt()
            raise
        finally:
            rec.update(
                seconds=time.perf_counter() - start,
                throttle_wait_seconds=self.throttle_wait,
                http_seconds=self.http_seconds,
                dispatched_unix=getattr(self, "dispatch_started_unix", None),
            )
            save(target, rec)
            self.records.append(rec)
            self.physical_successes += rec["status"] == "ok"
            self.physical_failures += rec["status"] == "error"
            self.update_progress(current_stage=stage, retry_wait_seconds=0)
