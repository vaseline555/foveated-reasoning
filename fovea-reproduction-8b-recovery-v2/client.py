"""Metered OpenRouter transport; no model calls occur at import time.

Prices are USD per million tokens. Every dispatch reserves a conservative upper
bound before sending. Uncertain charges retain that reservation and stop paid
work. Request/response artifacts never contain authentication headers.
"""
from __future__ import annotations

import base64
from contextlib import contextmanager
import hashlib
import io
import json
import math
import os
from pathlib import Path
import sqlite3
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from email.utils import parsedate_to_datetime


API = "https://openrouter.ai/api/v1"
MAX_IMAGE_PIXELS = 64_000_000
IMAGE_TOKEN_UPPER = 32768


class BudgetStop(RuntimeError):
    pass


class UnknownCharge(RuntimeError):
    pass


class RequestFailure(RuntimeError):
    pass


class RateLimitPause(RuntimeError):
    """Persistent upstream rate limiting; preserve the case for later resume."""
    pass


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    tmp.replace(path)


def input_upper(payload):
    """UTF8 byte upper for non-image JSON plus <=32768 tokens/image.

    Probe images use official max 16384 32x32 patches. Full original images
    remain intact. Their reservation also covers a pessimistic 16x16 patch
    encoding, while small/probe images retain a 32768-token minimum bound.
    The complete JSON byte count additionally covers tool schemas and wrappers.
    """
    from PIL import Image

    request = json.loads(json.dumps(payload))
    image_tokens = 0
    for message in request["messages"]:
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for block in content:
            if block.get("type") != "image_url":
                continue
            url = block["image_url"]["url"]
            if not isinstance(url, str) or not url.startswith("data:image/"):
                raise ValueError("Metered image requests require local base64 data URLs")
            raw = base64.b64decode(url.split(",", 1)[1], validate=True)
            with Image.open(io.BytesIO(raw)) as im:
                if im.width * im.height > MAX_IMAGE_PIXELS:
                    raise ValueError("Image exceeds 64M-pixel transport validation limit")
                image_tokens += max(IMAGE_TOKEN_UPPER, (im.width * im.height + 255)//256)
                im.verify()
            block["image_url"]["url"] = "image-placeholder"
    return len(canonical(request).encode("utf-8")) + image_tokens + 256


class Client:
    def __init__(self, run_dir, models, cap=15, historical_usd=0,
                 stop_callback=None, min_request_interval=1.0, rate_retry_limit=36):
        self.root = Path(run_dir)
        self.root.mkdir(parents=True, exist_ok=True)
        self.artifacts = self.root / "api"
        self.artifacts.mkdir(exist_ok=True)
        self.db = self.root / "ledger.sqlite"
        self.models = json.loads(json.dumps(models))
        self.cap = float(cap)
        self.historical_usd = float(historical_usd)
        self.min_request_interval = float(min_request_interval)
        self.rate_retry_limit = int(rate_retry_limit)
        self.stop_callback = stop_callback
        if not math.isfinite(self.cap) or self.cap <= 0 or self.cap > 15:
            raise ValueError("Campaign cap must be positive and <= $15")
        if not math.isfinite(self.historical_usd) or not 0 <= self.historical_usd <= self.cap:
            raise ValueError("Invalid prior campaign cost")
        if not math.isfinite(self.min_request_interval) or self.min_request_interval < 0:
            raise ValueError("Invalid minimum dispatch interval")
        if self.rate_retry_limit < 1:
            raise ValueError("Invalid rate-limit retry count")
        for role in ("primary", "fallback"):
            model = self.models[role]
            for key in ("id", "prompt_price", "completion_price", "context_length"):
                if key not in model:
                    raise ValueError("Missing model metadata: " + key)
            if min(float(model["prompt_price"]), float(model["completion_price"])) < 0:
                raise ValueError("Negative model price")
        with self.connect() as connection:
            connection.execute("CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY,value TEXT)")
            connection.execute("""CREATE TABLE IF NOT EXISTS attempts (
                id TEXT PRIMARY KEY, logical_key TEXT, model TEXT, role TEXT,
                case_id TEXT, method TEXT, stage TEXT, started REAL, finished REAL,
                status TEXT, reserved REAL, cost REAL, request_path TEXT,
                response_path TEXT, error TEXT, generation_id TEXT)""")
            connection.execute("CREATE INDEX IF NOT EXISTS logical_idx ON attempts(logical_key)")
            expected = canonical({"models": self.models, "cap": self.cap,
                                  "historical_usd": self.historical_usd,
                                  "min_request_interval": self.min_request_interval,
                                  "rate_retry_limit": self.rate_retry_limit,
                                  "image_token_upper": IMAGE_TOKEN_UPPER,
                                  "max_image_pixels": MAX_IMAGE_PIXELS})
            old = connection.execute("SELECT value FROM metadata WHERE key='scope'").fetchone()
            if old and old[0] != expected:
                raise ValueError("Frozen model/cap scope changed")
            connection.execute("INSERT OR IGNORE INTO metadata VALUES ('scope',?)", (expected,))
            connection.execute("""CREATE TABLE IF NOT EXISTS api_health (
                singleton INTEGER PRIMARY KEY CHECK(singleton=1),
                cooldown_until REAL NOT NULL DEFAULT 0,
                next_dispatch_at REAL NOT NULL DEFAULT 0,
                consecutive_429 INTEGER NOT NULL DEFAULT 0,
                total_429 INTEGER NOT NULL DEFAULT 0,
                paused INTEGER NOT NULL DEFAULT 0)""")
            connection.execute("INSERT OR IGNORE INTO api_health(singleton) VALUES (1)")

    @contextmanager
    def connect(self):
        connection = sqlite3.connect(self.db, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA busy_timeout=30000")
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    def _save_request(self, path, payload):
        """Store repeated images once; only the wire payload uses base64."""
        artifact = json.loads(json.dumps(payload))
        assets = self.artifacts / "assets"
        for message in artifact["messages"]:
            content = message.get("content")
            if not isinstance(content,list):
                continue
            for block in content:
                if block.get("type") != "image_url":
                    continue
                url = block["image_url"]["url"]
                header, encoded = url.split(",",1)
                raw = base64.b64decode(encoded,validate=True)
                sha = hashlib.sha256(raw).hexdigest()
                mime = header.split(";",1)[0].removeprefix("data:")
                suffix = ".png" if mime == "image/png" else ".jpg"
                target = assets / (sha + suffix)
                if not target.exists():
                    assets.mkdir(exist_ok=True)
                    temporary=target.with_name(target.name + "." + uuid.uuid4().hex + ".tmp")
                    temporary.write_bytes(raw)
                    temporary.replace(target)
                block["image_url"]["url"] = {"asset_path":str(target.resolve()),
                    "sha256":sha,"mime_type":mime}
        save(path,artifact)

    def stats(self):
        with self.connect() as connection:
            row = connection.execute("""SELECT COUNT(*) attempts,
                COALESCE(SUM(cost),0) confirmed_usd,
                COALESCE(SUM(CASE WHEN cost IS NULL THEN reserved ELSE 0 END),0) held_usd,
                SUM(status='valid') valid, SUM(role='fallback') fallback_attempts
                FROM attempts""").fetchone()
            result = dict(row)
            result['run_confirmed_usd'] = result['confirmed_usd']
            result['historical_confirmed_usd'] = self.historical_usd
            result['confirmed_usd'] += self.historical_usd
            result['remaining_usd'] = self.cap-result['confirmed_usd']-result['held_usd']
            return result | {"cap_usd": self.cap}

    def _check_stop(self):
        if (self.root/'STOP').exists() or self.stop_callback and self.stop_callback():
            raise BudgetStop('STOP requested during API wait')

    def _wait(self, seconds):
        deadline = time.time() + max(0, seconds)
        while True:
            self._check_stop()
            remaining = deadline-time.time()
            if remaining <= 0:
                return
            time.sleep(min(1, remaining))

    def _wait_for_dispatch(self, case=None, method=None, stage=None):
        """All threads share one SQLite dispatch gate and cooldown clock."""
        while True:
            self._check_stop()
            with self.connect() as connection:
                connection.execute('BEGIN IMMEDIATE')
                health = dict(connection.execute('SELECT * FROM api_health WHERE singleton=1').fetchone())
                if health['paused']:
                    raise RateLimitPause('Persistent HTTP429; paid dispatch is paused')
                now = time.time()
                ready_at = max(health['cooldown_until'], health['next_dispatch_at'])
                wait = ready_at-now
                if wait <= 0:
                    connection.execute('UPDATE api_health SET next_dispatch_at=? WHERE singleton=1',
                                       (now+self.min_request_interval,))
                    return
            # Check the shared gate again after every <=1s wait: another thread
            # can extend the global cooldown while this caller is waiting.
            if health['cooldown_until'] > now:
                save(self.root/'api_wait.json', dict(status='waiting', reason='HTTP429_global_cooldown',
                    next_retry_at=ready_at, case=case, method=method, stage=stage,
                    consecutive_429=health['consecutive_429'], total_429=health['total_429']))
            self._wait(min(1, wait))

    def _rate_limited(self, retry_after, case, method, stage):
        try:
            header_wait = max(0, float(retry_after))
        except (TypeError, ValueError):
            try:
                header_wait = max(0, parsedate_to_datetime(retry_after).timestamp()-time.time())
            except (TypeError, ValueError, OverflowError):
                header_wait = 0
        if not math.isfinite(header_wait):
            header_wait = 0
        with self.connect() as connection:
            connection.execute('BEGIN IMMEDIATE')
            health = dict(connection.execute('SELECT * FROM api_health WHERE singleton=1').fetchone())
            consecutive = health['consecutive_429']+1
            # The exponential part caps at 300s; a longer Retry-After header is
            # respected in full rather than silently shortened.
            delay = max(min(300, 30*2**min(consecutive-1, 8)), header_wait)
            retry_at = max(health['cooldown_until'], time.time()+delay)
            total = health['total_429']+1
            connection.execute('UPDATE api_health SET cooldown_until=?,consecutive_429=?,total_429=? WHERE singleton=1',
                               (retry_at,consecutive,total))
        save(self.root/'api_wait.json', dict(status='waiting', reason='HTTP429_global_cooldown',
            next_retry_at=retry_at, case=case, method=method, stage=stage,
            consecutive_429=consecutive, total_429=total))

    def _rate_limit_pause(self, case, method, stage):
        with self.connect() as connection:
            connection.execute('UPDATE api_health SET paused=1 WHERE singleton=1')
        save(self.root/'api_wait.json', dict(status='paused',reason='HTTP429_retry_limit',
            case=case,method=method,stage=stage,at=time.time(),retry_limit=self.rate_retry_limit))
        raise RateLimitPause('Persistent HTTP429; current case preserved for resume')

    def _request_succeeded(self):
        # Successful earlier in-flight calls must not cancel an active cooldown.
        with self.connect() as connection:
            now=time.time()
            health=dict(connection.execute('SELECT * FROM api_health WHERE singleton=1').fetchone())
            if now >= health['cooldown_until']:
                connection.execute('UPDATE api_health SET consecutive_429=0 WHERE singleton=1')
                save(self.root/'api_wait.json', dict(status='ready',at=now,
                    consecutive_429=0,total_429=health['total_429']))

    def _cached(self, key):
        with self.connect() as connection:
            row = connection.execute(
                "SELECT * FROM attempts WHERE logical_key=? AND status='valid' ORDER BY finished DESC LIMIT 1",
                (key,),
            ).fetchone()
        if row:
            data = json.loads(Path(row["response_path"]).read_text())
            return data["choices"][0]["message"], dict(row) | {"cache_hit": True}
        return None

    def _reserve(self, key, model, role, case, method, stage, upper, path):
        attempt = uuid.uuid4().hex
        with self.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            # 'dispatched' may be an active concurrent call. A resumed call for
            # the same logical request cannot safely repeat it.
            duplicate = connection.execute(
                "SELECT COUNT(*) FROM attempts WHERE logical_key=? AND cost IS NULL", (key,)
            ).fetchone()[0]
            unknown = connection.execute("SELECT COUNT(*) FROM attempts WHERE status='unknown'").fetchone()[0]
            if duplicate or unknown:
                raise UnknownCharge("Unresolved request charge; inspect ledger before resume")
            total = self.historical_usd + connection.execute("SELECT COALESCE(SUM(COALESCE(cost,reserved)),0) FROM attempts").fetchone()[0]
            if total + upper > self.cap + 1e-10:
                raise BudgetStop(f"Campaign cap: committed+held ${total:.6f}, next upper ${upper:.6f}")
            connection.execute("""INSERT INTO attempts
                (id,logical_key,model,role,case_id,method,stage,started,status,reserved,request_path)
                VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
                (attempt,key,model["id"],role,str(case),method,stage,time.time(),"reserved",upper,str(path)))
        return attempt

    def _update(self, attempt, **fields):
        with self.connect() as connection:
            sql = ",".join(key + "=?" for key in fields)
            connection.execute("UPDATE attempts SET " + sql + " WHERE id=?", (*fields.values(),attempt))

    def _cost(self, body, key):
        usage = body.get("usage") or {}
        cost = usage.get("cost")
        if isinstance(cost, (int, float)) and cost >= 0:
            return float(cost)
        generation = body.get("id")
        if not generation:
            return None
        for delay in (0, 1, 2):
            if delay:
                time.sleep(delay)
            request = urllib.request.Request(
                API + "/generation?" + urllib.parse.urlencode({"id": generation}),
                headers={"Authorization": "Bearer " + key},
            )
            try:
                with urllib.request.urlopen(request, timeout=30) as response:
                    detail = json.load(response)
                save(self.artifacts / (generation.replace("/", "_") + ".generation.json"), detail)
                cost = (detail.get("data") or {}).get("total_cost")
                if isinstance(cost, (int,float)) and cost >= 0:
                    return float(cost)
            except (urllib.error.URLError, TimeoutError, ValueError):
                pass
        return None

    def call(self, messages, stage, case, method, tools=None, max_tokens=2048,
             seed=42, validator=None, force_model=None):
        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            raise RequestFailure("OPENROUTER_API_KEY is not configured")
        logical = {"messages": messages, "stage": stage, "case": case, "method": method,
                   "tools": tools, "max_tokens": max_tokens,"seed": seed,"models": self.models,
                   "force_model":force_model}
        key = hashlib.sha256(canonical(logical).encode()).hexdigest()
        cached = self._cached(key)
        if cached:
            return cached
        errors = []
        if force_model not in (None,"primary","fallback"):
            raise ValueError("force_model must be primary/fallback/None")
        roles = (force_model,) if force_model else ("primary", "fallback")
        for role in roles:
            model = self.models[role]
            with self.connect() as connection:
                invalid = connection.execute(
                    "SELECT id FROM attempts WHERE logical_key=? AND role=? AND status='invalid' LIMIT 1",
                    (key,role)).fetchone()
            if invalid:
                errors.append(role + ": cached OUTPUT_VALIDATION_FAILED")
                continue
            payload = {"model":model["id"], "messages":messages,"max_tokens":max_tokens,
                       "temperature":model.get("temperature",1.0),"seed":seed,
                       "usage":{"include":True},
                       "provider": dict(model.get("provider") or {}) | {
                           "max_price":{"prompt":model["prompt_price"],
                                        "completion":model["completion_price"],
                                        "request":0,"image":0}}}
            if tools:
                payload["tools"] = tools
                payload["tool_choice"] = "auto"
            context = int(model["context_length"])
            if not 1 <= max_tokens < context:
                raise ValueError("Invalid output token cap")
            # A conservative patch bound can exceed context for long visual
            # histories. Successful inference cannot consume more input than
            # the endpoint context minus its requested output reservation.
            input_tokens = min(input_upper(payload), context-max_tokens)
            upper = (input_tokens*float(model["prompt_price"])
                     + max_tokens*float(model["completion_price"]))/1e6
            # Account for pricing rounding while enforcing server-side max_price.
            upper = upper*1.02 + 0.000001
            request_path = self.artifacts / (key + "." + role + ".request.json")
            self._save_request(request_path,payload)
            rate_retries = 0
            while True:
                self._wait_for_dispatch(case,method,stage)
                attempt = self._reserve(key,model,role,case,method,stage,upper,request_path)
                response_path = self.artifacts / (attempt + ".response.json")
                self._update(attempt,status="dispatched")
                request = urllib.request.Request(API + "/chat/completions",
                    data=json.dumps(payload).encode(),headers={
                        "Authorization":"Bearer " + api_key,"Content-Type":"application/json"})
                try:
                    with urllib.request.urlopen(request,timeout=180) as response:
                        body = json.load(response)
                except urllib.error.HTTPError as exc:
                    raw = exc.read().decode("utf-8",errors="replace")
                    save(response_path,{"http_status":exc.code,"error_body":raw})
                    explicit_rejection = exc.code in {400,401,402,403,404,405,413,415,422,429}
                    self._update(attempt,status="rejected" if explicit_rejection else "unknown",
                        cost=0.0 if explicit_rejection else None,error=f"HTTP {exc.code}: {raw[:2000]}",
                        response_path=str(response_path),finished=time.time())
                    if not explicit_rejection:
                        raise UnknownCharge(f"HTTP {exc.code}; reservation retained") from None
                    if exc.code == 429:
                        rate_retries += 1
                        self._rate_limited(exc.headers.get('Retry-After'),case,method,stage)
                        if rate_retries >= self.rate_retry_limit:
                            self._rate_limit_pause(case,method,stage)
                        continue
                    model_specific = exc.code in {400,404,422} and any(
                        word in raw.lower() for word in ("model","provider","endpoint","tool","parameter","context"))
                    if model_specific and role == "primary":
                        errors.append(f"{role}: HTTP {exc.code}")
                        break
                    raise RequestFailure(f"OpenRouter HTTP {exc.code}: {raw[:300]}") from None
                except (urllib.error.URLError,TimeoutError,ConnectionError,ValueError,OSError) as exc:
                    self._update(attempt,status="unknown",error=type(exc).__name__,finished=time.time())
                    raise UnknownCharge("Network/response failure; reservation retained: " + type(exc).__name__) from None
                save(response_path,body)
                self._request_succeeded()
                cost = self._cost(body,api_key)
                if cost is None:
                    self._update(attempt,status="unknown",response_path=str(response_path),
                                 generation_id=body.get("id"),finished=time.time())
                    raise UnknownCharge("Response cost unavailable; reservation retained")
                if cost > upper + 1e-9:
                    self._update(attempt,status="unknown",cost=cost,response_path=str(response_path),
                                 error="COST_EXCEEDED_RESERVATION",finished=time.time())
                    raise UnknownCharge("Actual cost exceeded conservative upper; paid work stopped")
                choices = body.get("choices") or []
                message = choices[0].get("message") if choices else None
                valid = isinstance(message,dict) and choices[0].get("finish_reason") != "length"
                if valid and validator is not None:
                    try:
                        valid = bool(validator(message))
                    except (ValueError,TypeError,KeyError):
                        valid = False
                self._update(attempt,status="valid" if valid else "invalid",cost=cost,
                    response_path=str(response_path),generation_id=body.get("id"),finished=time.time(),
                    error=None if valid else "OUTPUT_VALIDATION_FAILED")
                if valid:
                    with self.connect() as connection:
                        record = dict(connection.execute("SELECT * FROM attempts WHERE id=?",(attempt,)).fetchone())
                    return message, record | {"cache_hit":False,"fallback_reason":errors or None}
                errors.append(role + ": OUTPUT_VALIDATION_FAILED")
                break
        raise RequestFailure("Requested backbone output validation failed: " + "; ".join(errors))
