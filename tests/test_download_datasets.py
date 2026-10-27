"""Offline checks for bounded archive reads and benchmark layout/selection."""

import io
import json
import zipfile
from pathlib import Path
from unittest.mock import patch

import pytest
from PIL import Image

from active_vlm.experiment.data import catalog
from scripts.download_datasets import download_images, metadata, relative_path
from scripts.range_zip import RangeFile


def fixture_zip():
    image = io.BytesIO()
    Image.new("RGB", (10, 10), "red").save(image, format="JPEG")
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w") as archive:
        archive.writestr("data/MME-RealWorld-Lite.json", json.dumps([{
            "Question_id": "1", "Text": "Which color?", "Answer choices": ["A. Red", "B. Blue"],
            "Ground truth": "A", "Task": "Perception", "Subtask": "Color", "Image": "imgs/x.jpg",
        }]))
        archive.writestr("data/imgs/x.jpg", image.getvalue())
    return stream.getvalue()


def opener(body, requests):
    def open_request(request, **kwargs):
        value = request.headers["Range"]
        requests.append(value)
        start, end = map(int, value.removeprefix("bytes=").split("-"))
        response = io.BytesIO(body[start:end + 1])
        response.status = 206
        response.headers = {"Content-Range": f"bytes {start}-{end}/{len(body)}"}
        return response
    return open_request


def test_pinned_mme_metadata_and_selected_image_match_native_loader(tmp_path):
    requests = []
    with patch("urllib.request.urlopen", opener(fixture_zip(), requests)):
        metadata("mme", tmp_path)
        cases, _ = catalog(tmp_path, ["mme"])
        assert cases[0]["question"].endswith("(B) Blue")
        assert "Ground truth" not in cases[0]
        download_images("mme", cases, tmp_path)
        assert Path(cases[0]["image_path"]).is_file()
        count = len(requests)
        download_images("mme", cases, tmp_path)
        assert len(requests) == count + 1  # Archive header check; data blocks reused.


def test_range_reader_seeking_and_cached_blocks(tmp_path):
    body = fixture_zip()
    calls = []
    with patch("urllib.request.urlopen", opener(body, calls)):
        with RangeFile("https://example.test/pinned.zip", tmp_path) as stream:
            assert stream.read(2) == b"PK"
            stream.seek(-8, 2)
            assert stream.read() == body[-8:]
            stream.seek(0)
            assert stream.read(8) == body[:8]
            with pytest.raises(ValueError):
                stream.seek(-1)
        assert len(calls) == 2


def test_no_silent_full_archive_fallback(tmp_path):
    response = io.BytesIO(b"PK")
    response.status = 200
    response.headers = {}
    with patch("urllib.request.urlopen", return_value=response), pytest.raises(RuntimeError):
        RangeFile("https://example.test/pinned.zip", tmp_path)


@pytest.mark.parametrize("name", ["../x.jpg", "/x.jpg", "a/../../x.jpg", "a\\x.jpg"])
def test_metadata_paths_cannot_escape_data_root(name):
    with pytest.raises(ValueError):
        relative_path(name)
