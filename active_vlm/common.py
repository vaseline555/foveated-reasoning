"""Local, credential-free experiment utilities."""

import hashlib
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
ROOT = REPO.parent
MODEL = "qwen/qwen3-vl-8b-instruct"
PROVIDER = dict(
    only=["Parasail"], allow_fallbacks=False, require_parameters=True, quantizations=["bf16"]
)
SEED = 0


def canonical(value):
    return json.dumps(
        value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False
    )


def digest(value):
    return hashlib.sha256(
        value if isinstance(value, bytes) else canonical(value).encode()
    ).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False))
    temporary.replace(path)


def seed_for(case, stage, replicate=0, seed=SEED):
    # Arm-independent for shared Direct; probe stage includes turn/candidate/replicate.
    return int(digest([seed, case, stage, replicate])[:8], 16) % (2**31)
