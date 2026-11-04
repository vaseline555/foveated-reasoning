"""Execute pinned FOVEA pure functions without LangGraph, MCP, or MinIO.

The reasoning prompts, candidate geometry, Qwen image conversion, probe
composition/parser, output formatting, and aggregator composition come from the
official source via AST extraction. Only messages and local data-URI transport
replace service/framework objects. The committed image encoding used here is
the official storage JPEG85 function; the runner emits committed crops with
JPEG90, matching the bundled MCP server.
"""

from __future__ import annotations

import ast
import base64
import copy
import hashlib
import json
import math
import re
from io import BytesIO
from pathlib import Path

from PIL import Image
from pydantic import BaseModel, Field, ValidationInfo, field_validator


OFFICIAL_ROOT = Path(__file__).resolve().parent.parent / "fovea-official"
OFFICIAL_COMMIT = "0dc77fb69a3ae31467e2416f677fae1d21bbc498"
_SOURCES: dict[str, str] = {}


class _Message:
    role = ""

    def __init__(self, content=None, **kwargs):
        self.content = content
        for key, value in kwargs.items():
            setattr(self, key, value)


class _SystemMessage(_Message):
    role = "system"


class _HumanMessage(_Message):
    role = "user"


class _AIMessage(_Message):
    role = "assistant"

    def __init__(self, content=None, tool_calls=None, **kwargs):
        super().__init__(content, tool_calls=tool_calls or [], **kwargs)


class _ToolMessage(_Message):
    role = "tool"


class _Logger:
    def __getattr__(self, name):
        return lambda *args, **kwargs: None


_NS = dict(
    math=math, re=re, json=json, base64=base64, BytesIO=BytesIO,
    Image=Image, Bbox=tuple, AnyMessage=_Message, BaseMessage=_Message,
    SystemMessage=_SystemMessage, HumanMessage=_HumanMessage,
    AIMessage=_AIMessage, ToolMessage=_ToolMessage, logger=_Logger(),
    BaseModel=BaseModel, Field=Field, ValidationInfo=ValidationInfo,
    field_validator=field_validator,
)


def _source(relative):
    path = OFFICIAL_ROOT / relative
    data = path.read_bytes()
    _SOURCES[relative] = hashlib.sha256(data).hexdigest()
    return ast.parse(data.decode()), str(path)


def _extract(relative, names):
    tree, filename = _source(relative)
    found = {}
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names:
            item = copy.deepcopy(node)
            item.decorator_list = []
            found[item.name] = item
        elif isinstance(node, ast.Assign):
            ids = [t.id for t in node.targets if isinstance(t, ast.Name)]
            if any(name in names for name in ids):
                found[next(name for name in ids if name in names)] = copy.deepcopy(node)
    missing = set(names) - set(found)
    if missing:
        raise RuntimeError(f"Pinned official definitions missing: {sorted(missing)}")
    module = ast.Module(body=[found[name] for name in names], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), filename, "exec"), _NS)


_extract("src/cv_agent/utils/storage.py", [
    "image_to_jpeg_bytes", "encode_pil_image", "get_base64_url",
])
_extract("src/cv_agent/utils/qwen_vl.py", [
    "MAX_RATIO", "SPATIAL_MERGE_SIZE", "IMAGE_MIN_TOKEN_NUM", "IMAGE_MAX_TOKEN_NUM",
    "VIDEO_MIN_TOKEN_NUM", "VIDEO_MAX_TOKEN_NUM", "round_by_factor", "ceil_by_factor",
    "floor_by_factor", "smart_resize", "to_rgb", "convert_for_qwen_vl",
    "get_base64_url_for_qwen_vl", "rel_coords_to_abs",
])
_extract("src/cv_agent/policies/utils.py", ["scale_bbox_uniformly", "generate_candidates"])
_extract("src/cv_agent/agents/bed_nodes.py", ["validate_crop_coordinates"])
_extract("src/cv_agent/policies/boed.py", ["construct_input_for_confidence"])
_extract("src/cv_agent/policies/events.py", ["INVALID_CONFIDENCE_ASSESS"])
_extract("src/cv_agent/agents/aggregator_nodes.py", ["_build_aggregator_messages"])
_extract("src/cv_agent/agents/cv_agent_nodes.py", ["_post_process_tool_output"])
_extract("src/cv_agent/tools/cv_toolkit.py", ["CropImageInput"])

# Reuse the official synchronous counting tail, excluding the network abatch.
_tree, _filename = _source("src/cv_agent/policies/boed.py")
_evaluation = next(n for n in _tree.body if isinstance(n, ast.AsyncFunctionDef)
                   and n.name == "evaluate_candidate_confidence")
_start = next(i for i, n in enumerate(_evaluation.body)
              if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Tuple)
              and [a.id for a in n.targets[0].elts] == ["votes_yes", "total"])
_parser = ast.FunctionDef(
    name="_official_score_responses",
    args=ast.arguments(posonlyargs=[], args=[ast.arg(arg="responses")], vararg=None,
                       kwonlyargs=[], kw_defaults=[], kwarg=None, defaults=[]),
    body=copy.deepcopy(_evaluation.body[_start:]), decorator_list=[],
)
exec(compile(ast.fix_missing_locations(ast.Module(body=[_parser], type_ignores=[])),
             _filename, "exec"), _NS)


_TEMPLATES = {
    "planner": "planner_react.j2", "planner_react": "planner_react.j2",
    "direct": "direct_reasoning_system.j2", "direct_reasoning": "direct_reasoning_system.j2",
    "direct_reasoning_system": "direct_reasoning_system.j2",
    "aggregator": "aggregator_system.j2", "aggregator_system": "aggregator_system.j2",
    "resolvability": "boed_resolvability.md", "boed_resolvability": "boed_resolvability.md",
}


def template(name, question=None):
    filename = _TEMPLATES.get(name, name)
    if filename not in set(_TEMPLATES.values()):
        raise ValueError(f"Unknown official template: {name}")
    relative = "src/cv_agent/prompts/templates/" + filename
    path = OFFICIAL_ROOT / relative
    data = path.read_bytes()
    _SOURCES[relative] = hashlib.sha256(data).hexdigest()
    value = data.decode()
    # Official direct/aggregator templates contain only this Jinja placeholder.
    if "{{ question }}" in value:
        if question is None:
            raise ValueError("A question is required for this template")
        value = value.replace("{{ question }}", str(question))
    return value if filename.endswith(".md") else value.strip()


def generate_candidates(box, scales=(1.5, 1, 0.8)):
    return _NS["generate_candidates"](box, list(scales))


def validate_crop_coordinates(box):
    return _NS["validate_crop_coordinates"](box)


def jpeg_url(image, probe=False):
    fn = "get_base64_url_for_qwen_vl" if probe else "get_base64_url"
    return _NS[fn](image)


def _wire(messages):
    return [{"role": m.role, "content": m.content} for m in messages]


def probe_messages(question, original, parent_orNone, candidate):
    return _wire(_NS["construct_input_for_confidence"](
        question, original, parent_orNone, candidate, template("resolvability")))


def _message(value):
    role = value["role"]
    cls = {"system": _SystemMessage, "user": _HumanMessage,
           "assistant": _AIMessage, "tool": _ToolMessage}[role]
    # The official aggregator only tests presence of calls, not their schema.
    return cls(**{key: copy.deepcopy(val) for key, val in value.items() if key != "role"})


def aggregator_messages(question, original_url, history, direct_text):
    state = dict(question=question, original_figure_url=original_url,
                 messages=[_message(m) for m in history],
                 direct_reasoning_result=direct_text)
    return _wire(_NS["_build_aggregator_messages"](
        state, template("aggregator", question)))


def _remove_titles(value):
    if isinstance(value, dict):
        return {k: _remove_titles(v) for k, v in value.items() if k != "title"}
    if isinstance(value, list):
        return [_remove_titles(v) for v in value]
    return value


def crop_tool():
    # Mirrors LangChain's Pydantic conversion, including excluded default fields
    # that remain present in Pydantic's validation schema.
    schema = _remove_titles(_NS["CropImageInput"].model_json_schema())
    return dict(type="function", function=dict(
        name="cropping",
        description="Precise image cropping tool. Crops an image using coordinates [x1, y1, x2, y2].",
        parameters=schema,
    ))


def yes_score(contents):
    responses = [_Message(c.get("content") if isinstance(c, dict) else c) for c in contents]
    return _NS["_official_score_responses"](responses)["score"]


def absolute_coords(box, size, probe=False):
    if probe:
        return _NS["rel_coords_to_abs"](box, size)
    width, height = size
    sx, sy = width / 1000.0, height / 1000.0
    return [int(box[0] * sx), int(box[1] * sy), int(box[2] * sx), int(box[3] * sy)]


def process_output(data):
    return _NS["_post_process_tool_output"]("cropping", data)


def source_hashes():
    for filename in sorted(set(_TEMPLATES.values())):
        relative = "src/cv_agent/prompts/templates/" + filename
        _SOURCES[relative] = hashlib.sha256((OFFICIAL_ROOT / relative).read_bytes()).hexdigest()
    return dict(sorted(_SOURCES.items()))


def provenance():
    return dict(
        official_repository="https://github.com/iamlilAJ/active-vlm",
        official_commit=OFFICIAL_COMMIT,
        official_root=str(OFFICIAL_ROOT),
        source_hashes=source_hashes(),
        adapter_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        deviations=[
            "Pure official functions executed by AST; local message shims replace LangChain objects.",
            "Local PIL/data URIs replace MinIO and cropping MCP network services.",
            "JPEG helper uses official storage JPEG85; runner committed crops use bundled MCP JPEG90.",
            "Official yes-counting tail executes without its API batching and error suppression.",
        ],
    )
