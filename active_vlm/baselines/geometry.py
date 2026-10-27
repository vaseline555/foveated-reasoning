"""FOVEA upstream image and candidate geometry (MIT; see NOTICE)."""

from __future__ import annotations

import math

from PIL import Image

MAX_RATIO = 200
SPATIAL_MERGE_SIZE = 2
IMAGE_MIN_TOKEN_NUM = 4
IMAGE_MAX_TOKEN_NUM = 16384
VIDEO_MIN_TOKEN_NUM = 128
VIDEO_MAX_TOKEN_NUM = 768


def round_by_factor(number, factor):
    """Returns the closest integer to 'number' that is divisible by 'factor'."""
    return round(number / factor) * factor


def ceil_by_factor(number, factor):
    """Returns the smallest integer greater than or equal to 'number' that is divisible by
    'factor'."""
    return math.ceil(number / factor) * factor


def floor_by_factor(number, factor):
    """Returns the largest integer less than or equal to 'number' that is divisible by 'factor'."""
    return math.floor(number / factor) * factor


def smart_resize(height, width, factor=32, min_pixels=None, max_pixels=None):
    """
    Rescales the image so that the following conditions are met:

    1. Both dimensions (height and width) are divisible by 'factor'.
    2. The total number of pixels is within the range ['min_pixels', 'max_pixels'].
    3. The aspect ratio of the image is maintained as closely as possible.
    """
    max_pixels = max_pixels if max_pixels is not None else IMAGE_MAX_TOKEN_NUM * factor**2
    min_pixels = min_pixels if min_pixels is not None else IMAGE_MIN_TOKEN_NUM * factor**2
    assert max_pixels >= min_pixels, (
        "The max_pixels of image must be greater than or equal to min_pixels."
    )
    if max(height, width) / min(height, width) > MAX_RATIO:
        raise ValueError(
            f"absolute aspect ratio must be smaller than {MAX_RATIO}, got {max(height, width) / min(height, width)}"
        )
    h_bar = max(factor, round_by_factor(height, factor))
    w_bar = max(factor, round_by_factor(width, factor))
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt(height * width / max_pixels)
        h_bar = floor_by_factor(height / beta, factor)
        w_bar = floor_by_factor(width / beta, factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = ceil_by_factor(height * beta, factor)
        w_bar = ceil_by_factor(width * beta, factor)
    return (h_bar, w_bar)


def to_rgb(image):
    if image.mode == "RGBA":
        white_background = Image.new("RGB", image.size, (255, 255, 255))
        white_background.paste(image, mask=image.split()[3])
        return white_background
    else:
        return image.convert("RGB")


def convert_for_qwen_vl(image):
    resized_height, resized_width = smart_resize(image.height, image.width)
    return to_rgb(image.resize((resized_width, resized_height)))


def rel_coords_to_abs(coords, real_resolution):
    lx, uy, rx, by = coords
    width, height = real_resolution
    scale_x = width / 1000
    scale_y = height / 1000
    return list(map(round, [lx * scale_x, uy * scale_y, rx * scale_x, by * scale_y]))


def generate_candidates(bbox, scaling_factors):
    return [scale_bbox_uniformly(bbox, factor) for factor in scaling_factors]


def scale_bbox_uniformly(bbox, scale_factor):

    def avg(*args):
        return sum(args) / len(args)

    def half(n):
        return n / 2.0

    x1, y1, x2, y2 = bbox
    center_x = avg(x1, x2)
    center_y = avg(y1, y2)
    width = x2 - x1
    height = y2 - y1
    new_w = width * scale_factor
    new_h = height * scale_factor
    new_x1 = center_x - half(new_w)
    new_y1 = center_y - half(new_h)
    new_x2 = center_x + half(new_w)
    new_y2 = center_y + half(new_h)
    res = tuple(map(round, (max(0, new_x1), max(0, new_y1), min(1000, new_x2), min(1000, new_y2))))
    return res
