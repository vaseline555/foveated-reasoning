"""Local benchmark input only. Ground-truth fields never enter returned cases."""

import base64
import csv
import json
import re
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageOps

from ..common import digest, read


def source_path(data, ds, name):
    path = data / ds / name
    return path.resolve() if path.exists() else None


def tree_question(row):
    question = row["question"].replace("\\n", "\n").strip()
    if not row.get("A", "").strip():
        return question, True, list("ABCD")
    raw = row.get("multi-choice options", "").replace("\\n", "\n").strip()
    if raw:
        pattern = re.compile(r"(?m)(?<!\w)([A-K])\.\s*|^\s*([A-K])\s+")
        matches = list(pattern.finditer(raw))
        labels = [m[1] or m[2] for m in matches]
        if not labels or labels != list("ABCDEFGHIJK"[: len(labels)]):
            raise ValueError("Invalid TreeBench options")
        choices = [
            f"({label}) "
            + raw[m.end() : matches[i + 1].start() if i + 1 < len(matches) else len(raw)].strip()
            for i, (label, m) in enumerate(zip(labels, matches, strict=True))
        ]
    else:
        labels = [k for k in "ABCDEFGHIJK" if row.get(k, "").strip()]
        choices = [f"({k}) " + row[k].strip() for k in labels]
    return question + "\n" + "\n".join(choices), False, labels


def catalog(data, benchmarks):
    cases = []
    sources = {}
    csv.field_size_limit(100_000_000)
    for ds in benchmarks:
        custom = data / (ds + ".jsonl")
        if custom.exists():
            rows = [json.loads(line) for line in custom.read_text().splitlines() if line.strip()]
            for row in rows:
                if {"answer", "label", "Ground truth", "gt_boxes", "prediction"} & row.keys():
                    raise ValueError(
                        "Use a label-free manifest; pass labels separately for scoring"
                    )
                row = dict(row, dataset=ds)
                p = Path(row["image_path"])
                row["image_path"] = str(p if p.is_absolute() else (custom.parent / p).resolve())
                cases.append(row)
            continue
        name = {
            "vstar": "test_questions.jsonl",
            "treebench": "TreeBench.tsv",
            "mme": "mme_questions.json",
        }[ds]
        path = source_path(data, ds, name)
        if path is None:
            raise FileNotFoundError(f"Put {ds}.jsonl or {ds}/{name} under {data}")
        sources[ds] = dict(path=str(path), sha256=digest(path.read_bytes()))
        if ds == "treebench":
            with path.open() as stream:
                for row in csv.DictReader(stream, delimiter="\t"):
                    q, embedded, letters = tree_question(row)
                    cases.append(
                        dict(
                            dataset=ds,
                            id=str(row["index"]),
                            question=q,
                            category=row.get("category", ""),
                            subcategory=row.get("l2-category", ""),
                            embedded_options=embedded,
                            allowed_letters=letters,
                            _tree_source=str(path),
                        )
                    )
            continue
        rows = (
            [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
            if ds == "vstar"
            else read(path)
        )
        for row in rows:
            if ds == "vstar":
                c = dict(
                    dataset=ds,
                    id=str(row["question_id"]),
                    question=row["text"],
                    category=row.get("category", ""),
                    relative=row["image"],
                )
                candidates = [data / ds / row["image"], path.parent / row["image"]]
            else:
                q = (
                    row["Text"]
                    + "\n"
                    + "\n".join(
                        re.sub(r"^([A-K])\.\s*", r"(\1) ", x) for x in row["Answer choices"]
                    )
                )
                c = dict(
                    dataset=ds,
                    id=str(row["Question_id"]),
                    question=q,
                    category=row["Task"] + "/" + row["Subtask"],
                    subcategory=row.get("Category", ""),
                    question_type=row.get("Question Type", ""),
                    relative=row["Image"],
                )
                basename = Path(row["Image"]).name
                candidates = [
                    data / ds / "images" / basename,
                    data / ds / row["Image"],
                ]
            c["image_path"] = str(
                next((p.resolve() for p in candidates if p.exists()), candidates[0])
            )
            cases.append(c)
    return cases, sources


def balanced_sample(rows, count, seed, fields=("category", "subcategory")):
    """Equal quotas across available strata, then random sampling within each.

    Small strata contribute all their rows; unused quota moves to other strata.
    Secondary strata diversify question characteristics within each main task.
    No label, model output, or image availability participates in selection.
    """
    if count > len(rows):
        raise ValueError("Requested more samples than the population")
    if not fields:
        return sorted(rows, key=lambda c: digest([seed, "within-stratum", c["id"]]))[:count]
    groups = defaultdict(list)
    for row in rows:
        groups[str(row.get(fields[0], ""))].append(row)
    order = sorted(groups, key=lambda key: digest([seed, fields[0], key]))
    quotas = dict.fromkeys(order, 0)
    while sum(quotas.values()) < count:
        for key in order:
            if quotas[key] < len(groups[key]):
                quotas[key] += 1
                if sum(quotas.values()) == count:
                    break
    selected = {
        key: balanced_sample(
            groups[key], quotas[key], int(digest([seed, fields[0], key])[:8], 16), fields[1:]
        )
        for key in order
    }
    return [
        selected[key][i]
        for i in range(max(quotas.values(), default=0))
        for key in order
        if i < quotas[key]
    ]


def select(cases, benchmarks, count, seed, full=False, preserve=False, stratified=False):
    if preserve:
        return [dict(c) for c in cases if c["dataset"] in benchmarks]
    groups = {
        ds: sorted(
            (dict(c) for c in cases if c["dataset"] == ds),
            key=lambda c: digest([seed, "uniform-question-priority", ds, str(c["id"])]),
        )
        for ds in benchmarks
    }
    if not full:
        for ds, rows in groups.items():
            if len(rows) < count:
                raise ValueError(f"{ds}: requested {count}, available {len(rows)}")
        groups = {
            ds: balanced_sample(rows, count, seed) if stratified else rows[:count]
            for ds, rows in groups.items()
        }
    return [
        groups[ds][i]
        for i in range(max(map(len, groups.values()), default=0))
        for ds in benchmarks
        if i < len(groups[ds])
    ]


def materialize(cases, root):
    images = root / "images"
    images.mkdir(parents=True, exist_ok=True)
    selected = {c["id"]: c for c in cases if c.get("_tree_source")}
    for path in {c["_tree_source"] for c in selected.values()}:
        with Path(path).open() as stream:
            for row in csv.DictReader(stream, delimiter="\t"):
                if row["index"] in selected:
                    p = images / ("treebench_" + row["index"] + ".jpg")
                    p.write_bytes(base64.b64decode(row["image"]))
                    selected[row["index"]]["image_path"] = str(p.resolve())
    normalized = []
    fields = {
        "dataset",
        "id",
        "category",
        "subcategory",
        "question_type",
        "question",
        "image_path",
        "key",
        "embedded_options",
        "embedded_question",
        "allowed_letters",
        "source_repository",
        "source_revision",
    }
    for i, c in enumerate(cases):
        row = {k: v for k, v in c.items() if k in fields}
        row.setdefault("category", "")
        row["id"] = str(row["id"])
        row.setdefault("key", row["dataset"] + "_" + digest(row["id"])[:12])
        path = Path(row["image_path"]).resolve()
        if not path.exists():
            raise FileNotFoundError(f"Image missing; sample was NOT replaced: {path}")
        row["image_path"] = str(path)
        row["image_sha256"] = digest(path.read_bytes())
        with Image.open(path) as image:
            rgb = ImageOps.exif_transpose(image).convert("RGB")
            row.update(original_size=list(rgb.size), pixel_sha256=digest(rgb.tobytes()))
        row.setdefault(
            "embedded_question",
            bool(
                re.search(
                    r"question(?: and (?:options|choices))? in the image", row["question"], re.I
                )
            ),
        )
        row.setdefault("embedded_options", False)
        row.setdefault(
            "allowed_letters", re.findall(r"^\(([A-K])\)", row["question"], re.M) or list("ABCD")
        )
        row.update(execution_order=i, arm_order=list("AMLB")[i % 4 :] + list("AMLB")[: i % 4])
        normalized.append(row)
    if len({c["key"] for c in normalized}) != len(normalized):
        raise ValueError("Duplicate case key")
    return normalized
