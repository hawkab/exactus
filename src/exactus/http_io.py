"""One cancellable absolute deadline for headers, bodies and streamed lines."""

from contextlib import contextmanager
from contextvars import ContextVar
from http.client import HTTPConnection, HTTPSConnection, HTTPException
import socket
import threading
import time
from urllib.parse import urlsplit
from .cancellation import current as current_cancellation

deadline = ContextVar("exactus_http_deadline", default=None)
MAX_REPLY = 1048576


@contextmanager
def open_response(url, body=None, *, method="POST", headers=None, timeout=60.0):
    target = urlsplit(url)
    if (
        target.scheme not in ("http", "https")
        or not target.hostname
        or target.username
        or target.password
    ):
        raise ValueError("an http(s) endpoint without URL credentials is required")
    end = min(time.monotonic() + timeout, deadline.get() or float("inf"))
    remaining = end - time.monotonic()
    if remaining <= 0:
        raise TimeoutError("upstream deadline exceeded")
    cls = HTTPSConnection if target.scheme == "https" else HTTPConnection
    connection = cls(target.hostname, target.port, timeout=remaining)
    active = [None]

    def interrupt():
        sock = active[0] or connection.sock
        if sock is not None:
            try:
                sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
        connection.close()

    timer = threading.Timer(remaining, interrupt)
    timer.daemon = True
    timer.start()
    cancellation = current_cancellation.get()
    try:
        if cancellation is not None:
            cancellation.register(interrupt)
        connection.connect()
        connection.auto_open = (
            0  # cancellation must never trigger an automatic reconnect
        )
        active[0] = connection.sock
        if cancellation is not None:
            cancellation.check()
        if time.monotonic() >= end:
            raise TimeoutError("upstream deadline exceeded")
        path = (target.path or "/") + ("?" + target.query if target.query else "")
        connection.request(method, path, body=body, headers=headers or {})
        reply = connection.getresponse()
        yield reply, active[0], end
    except (OSError, ValueError, HTTPException) as exc:
        if time.monotonic() >= end:
            raise TimeoutError("upstream deadline exceeded") from exc
        raise
    finally:
        if cancellation is not None:
            cancellation.discard(interrupt)
        timer.cancel()
        interrupt()


def _remaining(sock, end):
    remaining = end - time.monotonic()
    if remaining <= 0:
        raise TimeoutError("upstream deadline exceeded")
    sock.settimeout(remaining)


def request_bytes(
    url, body=None, *, method="POST", headers=None, timeout=60.0, limit=MAX_REPLY
):
    with open_response(url, body, method=method, headers=headers, timeout=timeout) as (
        reply,
        sock,
        end,
    ):
        if reply.length is not None and reply.length > limit:
            raise ValueError("upstream response exceeds the byte limit")
        chunks, size = [], 0
        while reply.length != 0:
            if reply.isclosed():
                if reply.length not in (None, 0):
                    raise OSError("incomplete upstream response")
                break
            _remaining(sock, end)
            chunk = reply.read1(min(65536, limit + 1 - size))
            if not chunk:
                if reply.length not in (None, 0):
                    raise OSError("incomplete upstream response")
                break
            size += len(chunk)
            if size > limit:
                raise ValueError("upstream response exceeds the byte limit")
            chunks.append(chunk)
        return reply.status, b"".join(chunks)


def stream_lines(
    url, body, *, headers=None, timeout=300.0, limit=2097152, line_limit=65536
):
    with open_response(url, body, headers=headers, timeout=timeout) as (
        reply,
        sock,
        end,
    ):
        if reply.status != 200:
            raise OSError(f"stream endpoint returned HTTP {reply.status}")
        size = 0
        while reply.length != 0:
            if reply.isclosed():
                if reply.length not in (None, 0):
                    raise OSError("incomplete upstream stream")
                break
            _remaining(sock, end)
            line = reply.readline(line_limit + 1)
            if not line:
                if reply.length not in (None, 0):
                    raise OSError("incomplete upstream stream")
                break
            size += len(line)
            if len(line) > line_limit or size > limit:
                raise ValueError("SSE line or response exceeds the byte limit")
            yield line
