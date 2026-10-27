"""Tests must never dispatch network requests."""

import socket
import urllib.request
from contextlib import contextmanager
from unittest.mock import patch


@contextmanager
def no_dispatch():
    def blocked(*args, **kwargs):
        raise AssertionError("Network access forbidden in offline tests")

    with (
        patch.object(socket.socket, "connect", blocked),
        patch.object(urllib.request, "urlopen", blocked),
    ):
        yield
