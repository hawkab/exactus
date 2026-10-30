"""ASGI response ownership, bounded serialization and typed error classification."""

import asyncio
import json
import time

from ..domain.errors import ExactusError, WorkerError
from ..proxy_modes import RequestError
from .engine_client import EngineError

try:
    from starlette.responses import Response
except ImportError:
    Response = object  # optional server dependency, not required by calculator CLI

MAX_RESPONSE = 1048576


class Lease:
    """Event-loop-only owners: transport and actual computation."""

    __slots__ = ("_release", "_owners")

    def __init__(self, release):
        self._release, self._owners = release, 2

    def release(self):
        self._owners -= 1
        if self._owners == 0:
            self._release()


def encode(payload):
    body = json.dumps(
        payload, ensure_ascii=False, allow_nan=False, separators=(",", ":")
    ).encode()
    if len(body) > MAX_RESPONSE:
        raise RequestError("response exceeds 1 MiB", "response_too_large", 502)
    return body


def failure(exc):
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
        codes = {
            "GrammarError": "grammar_error",
            "BudgetError": "budget_error",
            "DataError": "data_error",
            "FormatError": "format_error",
        }
        return 422, {
            "error": str(exc),
            "error_code": getattr(
                exc, "error_code", codes.get(type(exc).__name__, "compute_error")
            ),
            "retryable": False,
        }
    return 500, {
        "error": "internal error",
        "error_code": "internal_error",
        "retryable": False,
    }


class OwnedResponse(Response):
    def __init__(self, body, status, deadline, lease):
        self.deadline, self.lease = deadline, lease
        super().__init__(body, status_code=status, media_type="application/json")

    async def __call__(self, scope, receive, send):
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
