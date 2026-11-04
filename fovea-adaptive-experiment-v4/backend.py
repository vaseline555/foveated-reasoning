"""Metered 8B adapter for the role/evidence Adaptive controller.

This module never creates a transport or reads evaluation labels. The supplied
client owns API requests, the shared ledger, rate limiting and USD reservations.
"""

from __future__ import annotations

import base64
import hashlib
import importlib.util
import io
import json
import re
import sys
from copy import deepcopy
from dataclasses import asdict, is_dataclass
from importlib.metadata import version
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator
from PIL import Image

POLICY_PATH = Path("/workspace/fovea-official/src/cv_agent/policies/adaptive.py")
ADAPTER_PATH = Path("/workspace/fovea-reproduction-8b-recovery-v2/official_adapter.py")
METHOD = "ADAPTIVE_V4"
MODEL = "qwen/qwen3-vl-8b-instruct"
SCHEMA_VALIDATOR_VERSION = version("jsonschema")


def _load(name: str, path: Path):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(str(path))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


adaptive = _load("fovea_adaptive_controller_v1", POLICY_PATH)
official = _load("fovea_adaptive_official_adapter_v1", ADAPTER_PATH)


def serializable(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        value = asdict(value)
    if isinstance(value, dict):
        return {key: serializable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [serializable(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return sorted(serializable(item) for item in value)
    return value


def _json(value: Any) -> str:
    return json.dumps(
        serializable(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _save_new(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as target:
        target.write(
            json.dumps(
                serializable(value), ensure_ascii=False, indent=2, allow_nan=False
            )
            + "\n"
        )


def _object(value: Any, required: set[str], optional: set[str] = frozenset()) -> dict:
    if not isinstance(value, dict):
        raise TypeError("Expected an object")
    if not required <= value.keys():
        raise ValueError(
            "Missing required object fields: "
            + ", ".join(sorted(required - value.keys()))
        )
    if value.keys() - required - optional:
        raise ValueError(
            "Unknown object fields: "
            + ", ".join(sorted(value.keys() - required - optional))
        )
    return value


def _schema_path(parts) -> str:
    return "$" + "".join(
        f"[{part}]" if isinstance(part, int) else "." + part for part in parts
    )


def validate_schema(value: Any, schema: dict) -> None:
    """Report exact contract paths without modifying the model's response."""

    def leaves(error):
        if error.context:
            # plan_update accepts an object or null. For a malformed object,
            # the null branch is irrelevant and hides useful nested errors.
            meaningful = [
                e
                for e in error.context
                if not (
                    e.validator == "type"
                    and e.schema.get("type") == "null"
                    and e.instance is not None
                )
            ]
            for child in meaningful or error.context:
                yield from leaves(child)
        else:
            yield error

    diagnostics = set()
    for outer in Draft202012Validator(schema).iter_errors(value):
        for error in leaves(outer):
            path = list(error.absolute_path)
            if error.validator == "required" and isinstance(error.instance, dict):
                for missing in set(error.validator_value) - error.instance.keys():
                    diagnostics.add(
                        _schema_path([*path, missing]) + ": required field is missing"
                    )
            else:
                diagnostics.add(_schema_path(path) + ": " + error.message)
    if diagnostics:
        raise ValueError(
            "Structured response violates schema: " + "; ".join(sorted(diagnostics))
        )


def _strings(value: Any) -> tuple[str, ...]:
    if not isinstance(value, list) or any(not isinstance(x, str) for x in value):
        raise ValueError("Expected a list of strings")
    return tuple(value)


def _list(value: Any) -> list:
    if not isinstance(value, list):
        raise TypeError("Expected a list")
    return value


def parse_claim(value: Any):
    item = _object(value, {"role_id", "text", "status"}, {"resolves_conflict"})
    if not isinstance(item["role_id"], str) or not isinstance(item["text"], str):
        raise TypeError("Claim role and text must be strings")
    if item["status"] not in ("unresolved", "partial", "supported", "conflicting"):
        raise ValueError("Unknown evidence status")
    resolved = item.get("resolves_conflict", False)
    if type(resolved) is not bool:
        raise ValueError("resolves_conflict must be a boolean")
    return adaptive.Claim(item["role_id"], item["text"], item["status"], resolved)


def parse_plan(value: Any):
    item = _object(value, {"roles", "goals"}, {"original_claims"})
    roles = []
    for raw in _list(item["roles"]):
        role = _object(
            raw,
            {"id", "role", "expected_evidence"},
            {"requires_for_binding", "importance"},
        )
        roles.append(
            adaptive.Role(
                role["id"],
                role["role"],
                role["expected_evidence"],
                _strings(role.get("requires_for_binding", [])),
                role.get("importance", 1.0),
            )
        )
    if len(roles) > 8:
        raise ValueError("At most eight roles per plan")
    plan = adaptive.Plan(
        tuple(roles),
        _strings(item["goals"]),
        tuple(parse_claim(x) for x in _list(item.get("original_claims", []))),
    )
    ids = [role.id for role in roles]
    if all(isinstance(key, str) for key in ids):
        allowed = sorted(set(ids))
        unknown = sorted(set(plan.goals) - set(ids))
        if not plan.goals or unknown:
            raise ValueError(
                "goals must be a nonempty array of exact roles[].id references; "
                f"unknown goals={unknown!r}; allowed role IDs={allowed!r}"
            )
        for role in roles:
            missing = sorted(set(role.requires_for_binding) - set(ids))
            if missing:
                raise ValueError(
                    f"Role {role.id!r} has unknown requires_for_binding IDs={missing!r}; "
                    f"allowed role IDs={allowed!r}"
                )
    adaptive._roles(plan)
    return plan


def parse_candidate(value: Any):
    item = _object(
        value,
        {"id", "image_key", "bbox", "role_ids", "expected_evidence"},
        {"branch_id"},
    )
    if any(
        not isinstance(item[key], str) or not item[key].strip()
        for key in ("id", "image_key", "expected_evidence")
    ):
        raise ValueError("Candidate strings must be nonempty")
    branch = item.get("branch_id", "")
    if not isinstance(branch, str):
        raise TypeError("branch_id must be a string")
    return adaptive.Candidate(
        item["id"],
        item["image_key"],
        tuple(_list(item["bbox"])),
        _strings(item["role_ids"]),
        item["expected_evidence"],
        branch,
    )


def parse_report(value: Any):
    item = _object(value, set(), {"claims", "new_candidates", "plan_update"})
    update = item.get("plan_update")
    return adaptive.Report(
        tuple(parse_claim(x) for x in _list(item.get("claims", []))),
        tuple(parse_candidate(x) for x in _list(item.get("new_candidates", []))),
        None if update is None else parse_plan(update),
    )


def _response_object(message: dict, tool_name: str) -> dict:
    calls = message.get("tool_calls") or []
    if calls:
        if len(calls) != 1 or calls[0].get("function", {}).get("name") != tool_name:
            raise ValueError("Expected exactly one structured response tool call")
        value = calls[0]["function"]["arguments"]
    else:
        value = message.get("content")
        if not isinstance(value, str) or not value.strip():
            raise ValueError("Empty structured response")
        value = value.strip()
        fenced = re.fullmatch(r"```(?:json)?\s*([\s\S]*?)\s*```", value, re.IGNORECASE)
        if fenced:
            value = fenced.group(1)
    parsed = json.loads(value) if isinstance(value, str) else value
    if not isinstance(parsed, dict):
        raise TypeError("Structured response must be a JSON object")
    return parsed


STRING = {"type": "string"}
STRINGS = {"type": "array", "items": STRING}
CLAIM_SCHEMA = {
    "type": "object",
    "required": ["role_id", "text", "status"],
    "properties": {
        "role_id": {
            "type": "string",
            "description": "Exact existing role ID; at most one claim per role_id in this array.",
        },
        "text": STRING,
        "status": {
            "type": "string",
            "enum": ["unresolved", "partial", "supported", "conflicting"],
        },
        "resolves_conflict": {"type": "boolean"},
    },
    "additionalProperties": False,
}
PLAN_SCHEMA = {
    "type": "object",
    "required": ["roles", "goals"],
    "properties": {
        "roles": {
            "type": "array",
            "minItems": 1,
            "maxItems": 8,
            "items": {
                "type": "object",
                "required": ["id", "role", "expected_evidence"],
                "properties": {
                    "id": {
                        "type": "string",
                        "description": "Unique short reference ID, e.g. answer_color.",
                    },
                    "role": {
                        "type": "string",
                        "description": "Required human-readable role label, also required in every plan_update.roles item; copy it from the current plan when unchanged.",
                    },
                    "expected_evidence": STRING,
                    "requires_for_binding": {
                        "type": "array",
                        "uniqueItems": True,
                        "items": {
                            "type": "string",
                            "description": "Exact ID from this plan's roles[].id.",
                        },
                    },
                    "importance": {"type": "number", "minimum": 0},
                },
                "additionalProperties": False,
            },
        },
        "goals": {
            "type": "array",
            "minItems": 1,
            "uniqueItems": True,
            "description": "References only: copy exact IDs of answer-evidence roles from roles[].id. Never put task descriptions here.",
            "items": {
                "type": "string",
                "description": "An exact roles[].id value, e.g. answer_color (not 'color of the object').",
            },
        },
        "original_claims": {"type": "array", "items": CLAIM_SCHEMA},
    },
    "additionalProperties": False,
}
CANDIDATE_SCHEMA = {
    "type": "object",
    "required": ["id", "image_key", "bbox", "role_ids", "expected_evidence"],
    "properties": {
        "id": STRING,
        "image_key": STRING,
        "bbox": {
            "type": "array",
            "items": {"type": "number", "minimum": 0, "maximum": 1000},
            "minItems": 4,
            "maxItems": 4,
        },
        "role_ids": STRINGS,
        "expected_evidence": STRING,
        "branch_id": STRING,
    },
    "additionalProperties": False,
}
PROPOSAL_SCHEMA = {
    "type": "object",
    "required": ["candidates"],
    "properties": {
        "candidates": {"type": "array", "maxItems": 4, "items": CANDIDATE_SCHEMA}
    },
    "additionalProperties": False,
}
REPORT_SCHEMA = {
    "type": "object",
    "properties": {
        "claims": {"type": "array", "items": CLAIM_SCHEMA},
        "new_candidates": {"type": "array", "maxItems": 4, "items": CANDIDATE_SCHEMA},
        "plan_update": {"anyOf": [PLAN_SCHEMA, {"type": "null"}]},
    },
    "additionalProperties": False,
}

SYSTEM = (
    "You are a visual evidence planner for a multiple-choice question. Use only the question, "
    "images and recorded observations. Expected evidence is a task, never an observed fact. "
    "A supported status is your current visual claim, not verified truth. Do not claim support "
    "when the requested detail or referent is unclear. Keep descriptions brief. Return exactly "
    "one structured tool call, or the equivalent JSON object without commentary. "
    "In each original_claims or claims array, include at most one entry per role_id. Combine "
    "that role's observations into one text and one overall status for its complete "
    "expected_evidence. Supported means the whole required evidence is satisfied; incidental "
    "visible attributes do not establish another requested attribute."
)
PLAN_INSTRUCTION = (
    "Decompose the question and options into at most 8 necessary object/relation/evidence roles. "
    "Each role has id, role, expected_evidence, requires_for_binding (role IDs needed to bind its "
    "identity/conclusion, not to permit looking), and importance (normally 1; key anchor 2). "
    "Use an acyclic dependency graph. goals is a nonempty list of EXACT roles[].id strings, "
    "never natural-language tasks. For example, if roles contain IDs 'object' and 'answer_color', "
    'use {"goals":["answer_color"]}, NOT {"goals":["color of the object"]}. Every requires_for_binding '
    "and claim role_id must also copy an exact declared role ID. Independently visible objects "
    "can be independent; a relation goal depends on its objects. Goals must cover the actual "
    "attribute, relation, count or comparison needed to answer the question; finding the object "
    "alone does not meet an attribute goal. Use evidence roles, not one role per answer choice. "
    "For example, a color question needs an object-identity role and a color-evidence goal. "
    "original_claims must describe "
    "only what you can actually see in the original. Mark every goal supported only if the "
    "original already supplies the required visual evidence; this permits zero-crop stopping. "
    "Otherwise mark uncertainty partial or unresolved. Do not create crop proposals yet."
)
PROPOSE_INSTRUCTION = (
    "Propose up to 4 promising crop candidates for unresolved evidence roles, including a "
    "different plausible location when a prior branch made no progress. Role count need not "
    "equal candidate count. Use a new candidate id, an available image_key, bbox [x1,y1,x2,y2] "
    "normalized 0..1000 relative to THAT image, nonempty role_ids, expected_evidence, and "
    "branch_id. Reuse a branch_id for refinements of the same hypothesis; nested crops inherit "
    "their source branch. Never redefine an existing candidate ID. Return candidates:[] only "
    "when no useful alternative exists. No overlap/area/distance preprocessing is applied."
)
OBSERVE_INSTRUCTION = (
    "Inspect the newly executed crop and the question in context. Report claims only for "
    "evidence actually visible, with role_id, concise text and status unresolved/partial/"
    "supported/conflicting. Preserve uncertainty and distinguish missing target from an "
    "unreadable target. Resolve a previous conflict only with explicit new evidence and "
    "resolves_conflict:true. Propose up to 4 new_candidates for useful refinement or a different "
    "hypothesis using the same candidate schema. Revise a mistaken role/referent with optional "
    "plan_update only when the role definitions, referent, goals or dependencies actually change. "
    "New visual evidence belongs in top-level claims, not plan_update.original_claims; "
    "if roles are unchanged, omit plan_update or use null. "
    "A plan_update must be the COMPLETE revised plan, including every role's id, role, and "
    "expected_evidence fields (copy unchanged role labels from the current plan), goals, and "
    "any dependencies. A new plan invalidates changed "
    "roles and downstream bindings. Do not repeat expected_evidence as if observed."
)


class Backend:
    """One case's images, structured callbacks and immutable audit events."""

    original_image_key = "original"

    def __init__(
        self,
        case: dict,
        shared_root: Path,
        outputdir: Path,
        client: Any,
        direct_baseline: dict,
    ):
        self.case = {key: case[key] for key in ("key", "question", "image_path")}
        self.shared_root = Path(shared_root)
        self.outputdir = Path(outputdir)
        self.client = client
        self.direct_baseline = direct_baseline
        if direct_baseline.get("status") != "complete" or direct_baseline.get(
            "models"
        ) != [MODEL]:
            raise ValueError("A completed pure-8B Direct baseline is required")
        self.limits = adaptive.Limits(max_steps=10, max_model_calls=24)
        self.records: list[dict] = []
        self.events: list[dict] = []
        self.images: dict[str, Image.Image] = {}
        self.image_urls: dict[str, str] = {}
        self.image_registry: dict[str, dict] = {}
        self._proposals = 0
        self._observations = 0
        self.outputdir.mkdir(parents=True, exist_ok=True)
        existing = [
            int(path.stem)
            for path in (self.outputdir / "events").glob("*.json")
            if path.stem.isdigit()
        ]
        self._event_index = max(existing, default=0)
        source = Path(case["image_path"])
        with Image.open(source) as original:
            original_url = official.jpeg_url(original.convert("RGB"))
        raw = base64.b64decode(original_url.split(",", 1)[1], validate=True)
        self._register(
            "original",
            raw,
            self.outputdir / "images" / "original.jpg",
            source_path=str(source),
            source_sha256=_sha(source),
            encoding="JPEG85",
        )
        self.provenance = {
            "method": METHOD,
            "model": MODEL,
            "schema_validator": {
                "package": "jsonschema",
                "version": SCHEMA_VALIDATOR_VERSION,
                "draft": "2020-12",
            },
            "backend_sha256": _sha(Path(__file__)),
            "policy_path": str(POLICY_PATH),
            "policy_sha256": _sha(POLICY_PATH),
            "official": official.provenance(),
            "prompt_sha256": {
                name: hashlib.sha256(text.encode()).hexdigest()
                for name, text in {
                    "system": SYSTEM,
                    "plan": PLAN_INSTRUCTION,
                    "propose": PROPOSE_INSTRUCTION,
                    "observe": OBSERVE_INSTRUCTION,
                }.items()
            },
            "limits": serializable(self.limits),
            "crop_geometry": "parent-relative 0..1000; floor to pixels; positive box only",
            "direct_source": direct_baseline.get("reused_from"),
        }
        self._event(
            "initialized", provenance=self.provenance, images=self.image_registry
        )

    def _event(self, event: str, **details: Any) -> None:
        self._event_index += 1
        value = dict(event=event, **serializable(details))
        _save_new(self.outputdir / "events" / f"{self._event_index:04d}.json", value)
        self.events.append(value)

    def _register(self, key: str, raw: bytes, path: Path, **details: Any) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            if path.read_bytes() != raw:
                raise ValueError(
                    "Refusing to overwrite a different crop/original image"
                )
        else:
            with path.open("xb") as target:
                target.write(raw)
        decoded = Image.open(io.BytesIO(raw))
        decoded.load()
        self.images[key] = decoded
        self.image_urls[key] = (
            "data:image/jpeg;base64," + base64.b64encode(raw).decode()
        )
        self.image_registry[key] = dict(
            path=str(path),
            sha256=hashlib.sha256(raw).hexdigest(),
            size=list(decoded.size),
            **details,
        )

    def _context(self, state: Any) -> str:
        images = {
            key: {
                k: v
                for k, v in self.image_registry[key].items()
                if k not in {"path", "sha256", "source_path", "source_sha256"}
            }
            for key in sorted(state.image_keys)
        }
        return _json(
            {
                "state": state,
                "available_images": images,
                "evidence_note": "All reports/statuses are model claims with recorded provenance.",
            }
        )

    def _messages(
        self, instruction: str, state: Any = None, extra: Any = None
    ) -> list[dict]:
        text = instruction + "\nQuestion and options:\n" + self.case["question"]
        if state is not None:
            text += "\nRecorded state:\n" + self._context(state)
        if extra is not None:
            text += "\nCurrent task:\n" + _json(extra)
        keys = ["original"] + sorted(
            (state.image_keys if state is not None else set()) - {"original"}
        )
        content = [{"type": "text", "text": text}]
        for key in keys:
            content.extend(
                [
                    {"type": "text", "text": "Image key: " + key},
                    {"type": "image_url", "image_url": {"url": self.image_urls[key]}},
                ]
            )
        return [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": content},
        ]

    def _call(
        self,
        stage: str,
        messages: list[dict],
        parser: Any,
        tool_name: str | None = None,
        schema: dict | None = None,
    ):
        parsed = []
        prompt_sha = hashlib.sha256(_json(messages).encode()).hexdigest()
        feedback_path = (
            self.outputdir / "format_feedback" / f"{stage}-{prompt_sha}.json"
        )
        feedback = None

        def parse_response(message):
            if not tool_name:
                return parser(message)
            value = _response_object(message, tool_name)
            if schema is not None:
                validate_schema(value, schema)
            return parser(value)

        def remember_feedback(diagnostic, message):
            nonlocal feedback
            feedback = {
                "stage": stage,
                "prompt_sha256": prompt_sha,
                "diagnostic": diagnostic,
                "invalid_response": message,
            }
            if feedback_path.exists():
                if _json(json.loads(feedback_path.read_text())) != _json(feedback):
                    raise ValueError("Conflicting immutable format-retry feedback")
            else:
                _save_new(feedback_path, feedback)

        def validator(message):
            try:
                value = parse_response(message)
            except (ValueError, TypeError, KeyError, IndexError) as error:
                if attempt == 0:
                    remember_feedback(str(error), message)
                self._event(
                    "output_validation_failed",
                    stage=physical_stage,
                    error=str(error),
                    message=message,
                )
                return False
            parsed[:] = [value]
            return True

        seed = int(
            hashlib.sha256(f"{self.case['key']}/{METHOD}/{stage}".encode()).hexdigest()[
                :8
            ],
            16,
        )
        tools = (
            None
            if not tool_name
            else [
                {
                    "type": "function",
                    "function": {
                        "name": tool_name,
                        "description": "Submit the requested structured visual evidence response.",
                        "parameters": schema,
                    },
                }
            ]
        )
        for attempt in range(2):
            physical_stage = stage if attempt == 0 else stage + "_format_retry1"
            request_messages = messages
            if attempt:
                if feedback is None and feedback_path.exists():
                    feedback = json.loads(feedback_path.read_text())
                    if (
                        feedback.get("stage") != stage
                        or feedback.get("prompt_sha256") != prompt_sha
                    ):
                        raise ValueError(
                            "Format-retry feedback does not match its prompt"
                        )
                if feedback is None:
                    # Length failures do not invoke the transport's validator.
                    # Reconstruct their raw draft by the exact cached logical key.
                    if hasattr(self.client, "connect") and hasattr(
                        self.client, "models"
                    ):
                        logical = {
                            "messages": messages,
                            "stage": stage,
                            "case": self.case["key"],
                            "method": METHOD,
                            "tools": tools,
                            "max_tokens": 2048,
                            "seed": seed % (2**32),
                            "models": self.client.models,
                            "force_model": "primary",
                        }
                        logical_key = hashlib.sha256(
                            _json(logical).encode()
                        ).hexdigest()
                        with self.client.connect() as db:
                            row = db.execute(
                                "SELECT response_path FROM attempts WHERE logical_key=? "
                                "AND status='invalid' ORDER BY finished DESC LIMIT 1",
                                (logical_key,),
                            ).fetchone()
                        if row and row["response_path"]:
                            body = json.loads(Path(row["response_path"]).read_text())
                            choice = (body.get("choices") or [{}])[0]
                            draft = choice.get("message")
                            diagnostic = "Response failed output validation"
                            if choice.get("finish_reason") == "length":
                                diagnostic = "Response was truncated (finish_reason=length); keep the complete response concise"
                            elif isinstance(draft, dict):
                                try:
                                    parse_response(draft)
                                except (
                                    ValueError,
                                    TypeError,
                                    KeyError,
                                    IndexError,
                                ) as error:
                                    diagnostic = str(error)
                            remember_feedback(diagnostic, draft)
                    if feedback is None:
                        remember_feedback(
                            "Previous request failed output validation; no draft was available to the adapter",
                            None,
                        )
                request_messages = messages + [
                    {
                        "role": "user",
                        "content": "The previous response failed validation. Correct the complete response using "
                        "the original schema and the diagnostic below; do not explain the repair. "
                        "Reference fields (goals, requires_for_binding, role_id/role_ids) must use exact "
                        "declared role IDs, with at most one claim per role. No new observations were made.\n"
                        + _json(feedback),
                    }
                ]
            self._event(
                "call_start",
                stage=physical_stage,
                prompt_sha256=hashlib.sha256(
                    _json(request_messages).encode()
                ).hexdigest(),
            )
            try:
                message, record = self.client.call(
                    request_messages,
                    stage=physical_stage,
                    case=self.case["key"],
                    method=METHOD,
                    tools=tools,
                    seed=(seed + attempt) % (2**32),
                    max_tokens=2048,
                    validator=validator,
                    force_model="primary",
                )
            except Exception as error:
                self._event(
                    "call_error",
                    stage=physical_stage,
                    error_type=type(error).__name__,
                    error=str(error),
                )
                if (
                    attempt == 0
                    and type(error).__name__ == "RequestFailure"
                    and "OUTPUT_VALIDATION_FAILED" in str(error)
                ):
                    continue
                raise
            if attempt:
                record = dict(record, format_retry_of=stage)
            self.records.append(record)
            # The shared client returns cached valid responses without rerunning
            # its validator. Reparse those without changing the saved response.
            if not parsed and not validator(message):
                raise ValueError(
                    "Cached response failed the current structured validator"
                )
            self._event(
                "call_complete",
                stage=physical_stage,
                record=record,
                message=message,
                parsed=parsed[0],
            )
            return parsed[0]
        raise AssertionError("Unreachable")

    async def plan(self, question: str, original_image_key: str):
        if question != self.case["question"] or original_image_key != "original":
            raise ValueError("Mismatched case or original image key")

        def validate(value):
            plan = parse_plan(value)
            temporary = adaptive.CaseState(question, original_image_key, plan)
            temporary.image_keys.add(original_image_key)
            adaptive._set_plan(temporary, plan)
            adaptive._record(
                temporary,
                adaptive.Observation(
                    None, original_image_key, adaptive.Report(plan.original_claims)
                ),
                self.limits,
            )
            return plan

        return self._call(
            "plan",
            self._messages(PLAN_INSTRUCTION),
            validate,
            "submit_plan",
            PLAN_SCHEMA,
        )

    async def propose(self, state):
        self._proposals += 1
        self._event("state_before_propose", state=state)

        def validate(value):
            item = _object(value, {"candidates"})
            candidates = tuple(parse_candidate(x) for x in _list(item["candidates"]))
            if len(candidates) > 4:
                raise ValueError("At most four proposals per call")
            adaptive._add_candidates(deepcopy(state), candidates, self.limits)
            return candidates

        return self._call(
            f"propose_{self._proposals}",
            self._messages(PROPOSE_INSTRUCTION, state),
            validate,
            "submit_candidates",
            PROPOSAL_SCHEMA,
        )

    async def executor(self, candidate) -> str:
        if candidate.image_key not in self.images:
            raise adaptive.CropError("Unknown parent image key")
        # Validate public executor calls as well as candidates emitted by the model.
        bbox = candidate.bbox
        if (
            len(bbox) != 4
            or any(
                isinstance(x, bool)
                or not isinstance(x, (int, float))
                or not 0 <= x <= 1000
                for x in bbox
            )
            or bbox[0] >= bbox[2]
            or bbox[1] >= bbox[3]
        ):
            raise adaptive.CropError("Invalid normalized crop coordinates")
        parent = self.images[candidate.image_key]
        absolute = official.absolute_coords(bbox, parent.size)
        if absolute[0] >= absolute[2] or absolute[1] >= absolute[3]:
            self._event(
                "crop_error",
                candidate=candidate,
                absolute=absolute,
                error="Empty crop after floor-to-pixel conversion",
            )
            raise adaptive.CropError("Empty crop after floor-to-pixel conversion")
        try:
            cropped = parent.crop(absolute)
            buffer = io.BytesIO()
            cropped.save(buffer, format="JPEG", quality=90, optimize=True)
        except (OSError, ValueError) as error:
            self._event(
                "crop_error", candidate=candidate, absolute=absolute, error=str(error)
            )
            raise adaptive.CropError(str(error)) from error
        number = len(self.images)
        key = f"crop_{number}"
        self._register(
            key,
            buffer.getvalue(),
            self.outputdir / "images" / f"{key}.jpg",
            parent=candidate.image_key,
            candidate_id=candidate.id,
            normalized_bbox=list(bbox),
            absolute_bbox=absolute,
            encoding="JPEG90",
        )
        self._event(
            "crop_complete",
            image_key=key,
            candidate=candidate,
            provenance=self.image_registry[key],
        )
        return key

    async def observe(self, state, candidate, cropped_image_key):
        self._observations += 1
        if (
            cropped_image_key not in self.images
            or cropped_image_key not in state.image_keys
        ):
            raise ValueError("Observation must refer to an executed, registered crop")
        self._event(
            "state_before_observe",
            state=state,
            candidate=candidate,
            cropped_image_key=cropped_image_key,
        )

        def validate(value):
            report = parse_report(value)
            if len(report.new_candidates) > 4:
                raise ValueError("At most four follow-up proposals per call")
            adaptive._record(
                deepcopy(state),
                adaptive.Observation(
                    candidate,
                    cropped_image_key,
                    report,
                    plan_version=len(state.plans) - 1,
                ),
                self.limits,
            )
            return report

        messages = self._messages(
            OBSERVE_INSTRUCTION,
            state,
            {"candidate": candidate, "newly_executed_crop": cropped_image_key},
        )
        return self._call(
            f"observe_{self._observations}",
            messages,
            validate,
            "submit_report",
            REPORT_SCHEMA,
        )

    def _aggregator_history(self, state) -> list[dict]:
        history = [
            {"role": "system", "content": "Adaptive visual evidence report."},
            {"role": "user", "content": self.case["question"]},
        ]
        for index, observation in enumerate(state.observations):
            description = _json(observation)
            if observation.candidate is None or observation.error:
                history.append({"role": "assistant", "content": description})
                continue
            call_id = f"adaptive_observation_{index}"
            history.extend(
                [
                    {
                        "role": "assistant",
                        "content": "Executed proposal (expected evidence is a task): "
                        + _json(observation.candidate),
                        "tool_calls": [{"id": call_id}],
                    },
                    {"role": "tool", "tool_call_id": call_id, "content": description},
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": "Observed image key: " + observation.image_key,
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": self.image_urls[observation.image_key]
                                },
                            },
                        ],
                    },
                ]
            )
        history.append(
            {
                "role": "assistant",
                "content": "Adaptive role/evidence state; support statuses are model claims, not verified truth:\n"
                + self._context(state),
            }
        )
        return history

    async def finalize(self, state) -> str:
        self._event("state_before_finalize", state=state, images=self.image_registry)
        history = self._aggregator_history(state)
        messages = official.aggregator_messages(
            self.case["question"],
            self.image_urls["original"],
            history,
            self.direct_baseline["message"].get("content") or "",
        )

        def validate(message):
            text = message.get("content")
            if (
                message.get("tool_calls")
                or not isinstance(text, str)
                or not text.strip()
            ):
                raise ValueError(
                    "Final reader must return nonempty natural-language text"
                )
            # No valid MCQ choice is a model NO_ANSWER outcome, not an API error.
            return text

        return self._call("aggregator", messages, validate)
