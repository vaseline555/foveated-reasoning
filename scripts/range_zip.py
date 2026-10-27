"""Seekable, bounded HTTP range reader for the pinned MME image archive."""

import hashlib
import io
import re
import urllib.request
from pathlib import Path


class RangeFile(io.RawIOBase):
    block_size = 1 << 20

    def __init__(self, url, cache):
        super().__init__()
        self.url, self.pos = url, 0
        self.cache = Path(cache) / hashlib.sha256(url.encode()).hexdigest()
        self.cache.mkdir(parents=True, exist_ok=True)
        request = urllib.request.Request(url, headers={"Range": "bytes=0-0"})
        with urllib.request.urlopen(request, timeout=90) as response:
            match = re.fullmatch(r"bytes 0-0/(\d+)", response.headers.get("Content-Range", ""))
            if response.status != 206 or not match or response.read(2) != b"P":
                raise RuntimeError("Invalid ZIP range response; full-archive fallback is disabled")
            self.length = int(match[1])

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, offset, whence=0):
        if whence not in (0, 1, 2):
            raise ValueError("Invalid seek origin")
        position = (0, self.pos, self.length)[whence] + offset
        if position < 0:
            raise ValueError("Negative seek position")
        self.pos = position
        return position

    def read(self, size=-1):
        remaining = max(0, self.length - self.pos)
        size = remaining if size is None or size < 0 else min(size, remaining)
        output = bytearray()
        while size:
            start = self.pos // self.block_size * self.block_size
            end = min(start + self.block_size, self.length) - 1
            path = self.cache / f"{start}-{end}.bin"
            expected = end - start + 1
            if not path.exists() or path.stat().st_size != expected:
                request = urllib.request.Request(
                    self.url, headers={"Range": f"bytes={start}-{end}"}
                )
                with urllib.request.urlopen(request, timeout=90) as response:
                    if response.status != 206 or response.headers.get("Content-Range") != (
                        f"bytes {start}-{end}/{self.length}"
                    ):
                        raise RuntimeError("Incorrect HTTP range; full-archive fallback is disabled")
                    body = response.read(expected + 1)
                if len(body) != expected:
                    raise RuntimeError("Incomplete HTTP range")
                temporary = path.with_suffix(".part")
                temporary.write_bytes(body)
                temporary.replace(path)
            block = path.read_bytes()
            offset = self.pos - start
            take = min(size, len(block) - offset)
            output.extend(block[offset:offset + take])
            self.pos += take
            size -= take
        return bytes(output)
