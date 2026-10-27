"""Verified native crops and explicitly recorded encoder image layouts."""

import io
import math
from copy import deepcopy
from pathlib import Path

from PIL import Image, ImageOps

from .baselines import geometry
from .common import digest


def validate_seed(box, min_span=1):
    if len(box) != 4 or any(
        type(v) not in {int, float} or not math.isfinite(v) or (not 0 <= v <= 1000) for v in box
    ):
        raise ValueError("INVALID_SEED")
    box = tuple(map(round, box))
    if box[2] - box[0] < min_span or box[3] - box[1] < min_span:
        raise ValueError("SEED_SPAN_BELOW_MINIMUM")
    return box


class Images:
    def __init__(self, case, output):
        self.output = Path(output)
        source = Path(case["image_path"])
        if digest(source.read_bytes()) != case["image_sha256"]:
            raise ValueError("SOURCE_IMAGE_CHANGED")
        with Image.open(source) as im:
            original = ImageOps.exif_transpose(im).convert("RGB")
        if list(original.size) != case["original_size"]:
            raise ValueError("IMAGE_SIZE_CHANGED")
        if digest(original.tobytes()) != case["pixel_sha256"]:
            raise ValueError("IMAGE_PIXELS_CHANGED")
        self.images = {"original": original}
        self.boxes = {"original": [0, 0, *original.size]}
        self.views = {}

    def descriptor(self, image, probe=False):
        encoded = geometry.convert_for_qwen_vl(image) if probe else image
        stream = io.BytesIO()
        encoded.save(stream, format="JPEG", quality=85, optimize=True)
        body = stream.getvalue()
        sha = digest(body)
        path = self.output / "views" / (sha + ".jpg")
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            path.write_bytes(body)
        view = dict(
            path=str(path.resolve()),
            sha256=sha,
            size=list(encoded.size),
            native_size=list(image.size),
            mime="image/jpeg",
            probe_resize=probe,
            bytes=len(body),
        )
        self.views[sha] = view
        return deepcopy(view)

    def block(self, key):
        return dict(type="image_url", image_url=dict(url=self.descriptor(self.images[key])))

    def candidate(self, parent, box):
        source = self.images[parent]
        pixel = geometry.rel_coords_to_abs(box, source.size)
        if pixel[0] >= pixel[2] or pixel[1] >= pixel[3]:
            raise ValueError("EMPTY_NATIVE_CROP")
        image = source.crop(pixel)
        pb = self.boxes[parent]
        absolute = [
            pb[i % 2] + pixel[i] * (pb[2 + i % 2] - pb[i % 2]) / source.size[i % 2]
            for i in range(4)
        ]
        identity = digest(
            dict(original_xyxy=absolute, pixels=digest(image.tobytes()), size=list(image.size))
        )
        return dict(
            id="crop_" + identity[:20],
            parent=parent,
            normalized=list(box),
            pixel_xyxy=pixel,
            original_xyxy=absolute,
            image=image,
        )

    def pool(self, arm, proposal):
        anchor = proposal["anchor"]
        base = validate_seed(anchor["box"])
        if anchor["parent"] not in self.images:
            raise ValueError("UNKNOWN_PARENT")
        if arm == "A":
            seeds = [
                dict(
                    parent=anchor["parent"],
                    box=geometry.scale_bbox_uniformly(base, s),
                    target_nodes=[],
                )
                for s in [1.5, 1.0, 0.8]
            ]
        else:
            if len(proposal["branches"]) > 2:
                raise ValueError("TOO_MANY_BRANCHES")
            seeds = [anchor, *proposal["branches"]]
            seeds = [dict(s, box=validate_seed(s["box"])) for s in seeds]
            seeds += [dict(anchor, box=geometry.scale_bbox_uniformly(base, s)) for s in [1.5, 0.8]]
        candidates = []
        seen = set()
        for seed in seeds:
            candidate = self.candidate(seed["parent"], seed["box"])
            candidate["target_nodes"] = seed["target_nodes"]
            if arm == "A" or candidate["id"] not in seen:
                candidates.append(candidate)
                seen.add(candidate["id"])
            if len(candidates) == 3:
                break
        return candidates

    def commit(self, candidate):
        self.images[candidate["id"]] = candidate["image"]
        self.boxes[candidate["id"]] = candidate["original_xyxy"]

    def audit(self, candidate):
        return {
            **{k: v for k, v in candidate.items() if k != "image"},
            "native_view": self.descriptor(candidate["image"]),
            "probe_view": self.descriptor(candidate["image"], True),
        }
