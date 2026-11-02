"""ASGI response ownership, bounded serialization and typed error classification."""

__author__ = "Olshansky Grigory"

import asyncio
import json
import time

from ..domain.errors import ExactusError, WorkerError
from ..proxy_modes import RequestError
from ..request_errors import as_request_error
from .engine_client import EngineError

try:
    from starlette.responses import Response
except ImportError:
    Response = object

MAX_RESPONSE = 1048576


class Lease:
    """Event-loop-only owners: transport and actual computation."""

    __slots__ = ("_release", "_owners")

    def __init__(self, release):
        """Initialize Lease state and dependencies."""
        self._release, self._owners = release, 2

    def release(self):
        """Release one owner and return admission only after the final owner finishes."""
        self._owners -= 1
        if self._owners == 0:
            self._release()


def encode(payload):
    """Encode a JSON response and enforce the transport response size contract."""
    body = json.dumps(
        payload, ensure_ascii=False, allow_nan=False, separators=(",", ":")
    ).encode()
    if len(body) > MAX_RESPONSE:
        raise RequestError("response exceeds 1 MiB", "response_too_large", 502)
    return body


def failure(exc):
    """Translate request, engine, and domain failures into HTTP status and JSON data."""
    if isinstance(exc, RequestError):
        return exc.status, {**exc.body(), "retryable": exc.status >= 500}
    if isinstance(exc, (EngineError, WorkerError)):
        return 503, {
            "error": str(exc),
            "error_code": "engine_unavailable"
            if isinstance(exc, EngineError)
            else "worker_unavailable",
            "retryable": True,
        }
    if isinstance(exc, ExactusError):
        error = as_request_error(exc)
        return error.status, {**error.body(), "retryable": False}
    return 500, {
        "error": "internal error",
        "error_code": "internal_error",
        "retryable": False,
    }


class OwnedResponse(Response):
    """Retain the admission lease until the response has finished or been cancelled."""

    def __init__(self, body, status, deadline, lease):
        """Initialize Owned Response state and dependencies."""
        self.deadline, self.lease = deadline, lease
        super().__init__(body, status_code=status, media_type="application/json")

    async def __call__(self, scope, receive, send):
        """Send the response within its deadline and always release transport ownership."""
        try:
            remaining = self.deadline - time.monotonic()
            if self.status_code >= 400:
                remaining = max(0.1, remaining)
            if remaining <= 0:
                raise asyncio.TimeoutError("response deadline exceeded")
            await asyncio.wait_for(super().__call__(scope, receive, send), remaining)
        finally:
            self.lease.release()


response = OwnedResponse
