#!/usr/bin/env python3
"""Download pinned public benchmarks; never call an inference API.

Use the runner's exact label-free sample selection before fetching images.
TreeBench embeds images in its TSV; MME images are fetched as ZIP ranges.
"""

import argparse
import hashlib
import json
import shutil
import sys
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path, PurePosixPath

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from active_vlm.experiment.data import catalog, select
from scripts.range_zip import RangeFile

SOURCES = {
    "vstar": {
        "repo": "craigwu/vstar_bench",
        "revision": "d9ae62c903da0c98336e85c5ee89cd863b04b4da",
        "metadata": "test_questions.jsonl",
        "sha256": "d7e1ca9d1beb9a1f86b0cb309bb4a73091e8cb062f75eda81c2fda5304923ae9",
    },
    "treebench": {
        "repo": "HaochenWang/TreeBench",
        "revision": "a4eb437ab71d14f6ab1a4cf05213d0f72f831b71",
        "metadata": "TreeBench.tsv",
        "sha256": "128b4d82dfeafad45b1576c9eef83e3faced8c08b00c87f58ed9207bda5ff80d",
    },
    "mme": {
        "repo": "yifanzhang114/MME-RealWorld-Lite",
        "revision": "0e378f64aea87bb2c30dae8bc600f99fae966711",
        "archive": "data.zip",
        "metadata_member": "data/MME-RealWorld-Lite.json",
    },
}


def source_url(dataset, filename):
    source = SOURCES[dataset]
    return (
        f"https://huggingface.co/datasets/{source['repo']}/resolve/"
        f"{source['revision']}/{urllib.parse.quote(filename, safe='/')}"
    )


def relative_path(name):
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts or "\\" in name or not path.name:
        raise ValueError(f"Invalid dataset-relative path: {name!r}")
    return path


def checksum(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def verify_image(path):
    with Image.open(path) as image:
        image.verify()


def download(url, path, sha256=None, image=False):
    if path.exists():
        if sha256 and checksum(path) != sha256:
            raise ValueError(f"Existing metadata checksum mismatch: {path}")
        if image:
            verify_image(path)
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    try:
        with urllib.request.urlopen(url, timeout=90) as response, temporary.open("wb") as out:
            shutil.copyfileobj(response, out, length=1 << 20)
        if sha256 and checksum(temporary) != sha256:
            raise ValueError(f"Downloaded metadata checksum mismatch: {path}")
        if image:
            verify_image(temporary)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def open_mme(data):
    return RangeFile(source_url("mme", "data.zip"), data / ".downloads" / "mme")


def metadata(dataset, data):
    source = SOURCES[dataset]
    root = data / dataset
    root.mkdir(parents=True, exist_ok=True)
    if dataset != "mme":
        download(source_url(dataset, source["metadata"]), root / source["metadata"], source["sha256"])
        return
    # Read the metadata from the pinned archive even on a repeated invocation.
    with open_mme(data) as stream, zipfile.ZipFile(stream) as archive:
        rows = json.loads(archive.read(source["metadata_member"]))
    path = root / "mme_questions.json"
    body = json.dumps(rows, ensure_ascii=False).encode()
    if path.exists() and json.loads(path.read_bytes()) != rows:
        raise ValueError(f"Existing MME metadata differs from the pinned archive: {path}")
    temporary = path.with_suffix(".json.part")
    temporary.write_bytes(body)
    temporary.replace(path)


def download_images(dataset, cases, data):
    if dataset == "treebench":
        return  # Embedded images are materialized by the experiment runner.
    if dataset == "vstar":
        names = sorted({str(relative_path(c["relative"])) for c in cases})
        for i, name in enumerate(names, 1):
            path = data / "vstar" / name
            if not path.resolve().is_relative_to(data.resolve()):
                raise ValueError("Image destination escapes the data directory")
            download(source_url(dataset, name), path, image=True)
            print(f"vstar: {i}/{len(names)} images ready", flush=True)
        return
    names = sorted({relative_path(c["relative"]).name for c in cases})
    with open_mme(data) as stream, zipfile.ZipFile(stream) as archive:
        for i, name in enumerate(names, 1):
            path = data / "mme" / "images" / name
            if not path.resolve().is_relative_to(data.resolve()):
                raise ValueError("Image destination escapes the data directory")
            if not path.exists():
                path.parent.mkdir(parents=True, exist_ok=True)
                temporary = path.with_suffix(path.suffix + ".part")
                try:
                    with archive.open("data/imgs/" + name) as src, temporary.open("wb") as dst:
                        shutil.copyfileobj(src, dst, length=1 << 20)
                    verify_image(temporary)
                    temporary.replace(path)
                finally:
                    temporary.unlink(missing_ok=True)
            verify_image(path)
            print(f"mme: {i}/{len(names)} images ready", flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmarks", nargs="+", choices=list(SOURCES), default=list(SOURCES))
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--samples", type=int, help="Questions per benchmark; default: all images")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--stratified", action="store_true")
    parser.add_argument("--metadata-only", action="store_true")
    args = parser.parse_args(argv)
    if args.samples is not None and args.samples <= 0:
        parser.error("--samples must be positive")
    for dataset in dict.fromkeys(args.benchmarks):
        metadata(dataset, args.data_dir)
        rows, _ = catalog(args.data_dir, [dataset])
        chosen = select(rows, [dataset], args.samples or 1, args.seed,
                        full=args.samples is None, stratified=args.stratified)
        if not args.metadata_only:
            download_images(dataset, chosen, args.data_dir)
        record = dict(SOURCES[dataset], selected_ids=[c["id"] for c in chosen],
                      samples=args.samples, seed=args.seed, stratified=args.stratified,
                      metadata_only=args.metadata_only)
        (args.data_dir / dataset / "download_manifest.json").write_text(
            json.dumps(record, indent=2, ensure_ascii=False) + "\n"
        )
        print(f"{dataset}: metadata ready; {len(chosen)} selected questions", flush=True)


if __name__ == "__main__":
    main()
