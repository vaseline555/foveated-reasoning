"""Read benchmark labels only after a run is terminal."""

import csv
from pathlib import Path

from ..common import digest, read, save


def labels_after_terminal(root, cases):
    sources = read(root / "dataset_sources.json")["sources"]
    labels = {}
    for dataset, source in sources.items():
        targets = {c["id"]: c["key"] for c in cases if c["dataset"] == dataset}
        if not targets:
            continue
        path = Path(source["path"])
        if digest(path.read_bytes()) != source["sha256"]:
            raise RuntimeError("DATASET_METADATA_CHANGED")
        if dataset == "treebench":
            csv.field_size_limit(100000000)
            with path.open() as stream:
                for row in csv.DictReader(stream, delimiter="\t"):
                    if row["index"] in targets:
                        labels[targets[row["index"]]] = row["answer"]
        else:
            rows = (
                [__import__("json").loads(line) for line in path.read_text().splitlines()]
                if dataset == "vstar"
                else read(path)
            )
            idkey, goldkey = (
                ("question_id", "label") if dataset == "vstar" else ("Question_id", "Ground truth")
            )
            for row in rows:
                if str(row[idkey]) in targets:
                    labels[targets[str(row[idkey])]] = row[goldkey]
    assert len(labels) == len(cases)
    save(root / "terminal_labels_post_inference.json", dict(labels=labels, sources=sources))
    return labels
