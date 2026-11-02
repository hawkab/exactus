"""HTTP route adapter exposing the exact-span endpoint on a FastAPI app."""

from __future__ import annotations

import asyncio
import time

from ..cancellation import Cancellation
from ..cancellation import current as current_cancellation
from ..http_io import deadline as http_deadline
from ..proxy_modes import RequestError
from .async_work import blocking
from .endpoint_adapter import ExactusCoordinator
from .endpoint_body import _load as _load
from .endpoint_http import Lease, encode, failure, response
from .endpoint_request import _run as _run
from .endpoint_request import coordinator_prompt as coordinator_prompt
from .endpoint_response import (
    _serialize as _serialize,
)
from .endpoint_response import (
    _serialize_encoded as _serialize_encoded,
)

try:
    from starlette.requests import Request
except ImportError:
    Request = None

__author__ = "Olshansky Grigory"

_DEADLINE_S = 600.0


def attach(
    app,
    coordinator: ExactusCoordinator,
    models: tuple[str, ...],
    *,
    deadline_s: float = _DEADLINE_S,
    max_concurrent: int = 16,
) -> None:
    """Register the exact-compute route inside the app's auth stack."""
    from starlette.responses import JSONResponse

    if max_concurrent < 1 or deadline_s <= 0:
        raise ValueError("positive concurrency and deadline required")
    if any(
        getattr(route, "path", None)
        in ("/v1/exact/completions", "/v1/exact/external-computers", "/v1/exact/plans")
        and "POST" in (getattr(route, "methods", None) or ())
        for route in app.routes
    ):
        raise ValueError("Exactus route collision: enable exactly one provider")

    admission = asyncio.Semaphore(max_concurrent)

    async def execute(request, expires, cancellation):
        """Dispatch a decoded request while propagating deadlines and disconnection."""
        http_deadline.set(expires)
        current_cancellation.set(cancellation)
        payload = await _load(request)
        owner = asyncio.current_task()

        async def disconnect():
            """Cancel in-flight computation when the ASGI client disconnects."""
            while True:
                message = await request.receive()
                if message["type"] == "http.disconnect":
                    cancellation.cancel()
                    owner.cancel()
                    return
                await asyncio.sleep(0.05)

        watcher = asyncio.create_task(disconnect())
        try:
            if request.url.path == "/v1/exact/plans":
                if "plan" not in payload or set(payload) - {"plan", "inputs", "checks"}:
                    raise RequestError("expected plan/inputs/checks", "invalid_request")
                result = await coordinator.run_plan(
                    payload["plan"],
                    inputs=payload.get("inputs"),
                    checks=payload.get("checks"),
                )
                return await blocking(encode, result)
            if request.url.path == "/v1/exact/external-computers":
                if set(payload) != {"tasks"}:
                    raise RequestError("expected tasks object only", "invalid_request")
                result = await coordinator.run_external_computers(payload["tasks"])
                return await blocking(encode, result)
            return await _run(coordinator, models, payload, deadline_s)
        finally:
            watcher.cancel()
            await asyncio.gather(watcher, return_exceptions=True)

    async def exact_completions(request: Request):
        """Run one exact-compute request under a strict schema."""
        if admission.locked():
            return JSONResponse(
                {"error": "server is at capacity", "error_code": "overloaded"},
                status_code=503,
            )
        await admission.acquire()
        lease = Lease(admission.release)
        deadline = time.monotonic() + deadline_s
        cancellation = Cancellation()
        work = asyncio.create_task(execute(request, deadline, cancellation))

        def release(task):
            """Release computation ownership and retrieve any task exception."""
            lease.release()
            if not task.cancelled():
                task.exception()

        work.add_done_callback(release)
        try:
            result = await asyncio.wait_for(
                asyncio.shield(work), max(0, deadline - time.monotonic())
            )
        except asyncio.TimeoutError:
            cancellation.cancel()
            work.cancel()
            return response(
                encode(
                    {
                        "error": "request deadline exceeded",
                        "error_code": "deadline_exceeded",
                        "retryable": True,
                    }
                ),
                504,
                deadline,
                lease,
            )
        except asyncio.CancelledError:
            cancellation.cancel()
            work.cancel()
            lease.release()
            raise
        except Exception as exc:
            status, body = failure(exc)
            return response(encode(body), status, deadline, lease)
        return response(result, 200, deadline, lease)

    exact_completions.__annotations__ = {"request": Request}
    app.add_api_route("/v1/exact/completions", exact_completions, methods=["POST"])
    app.add_api_route(
        "/v1/exact/external-computers", exact_completions, methods=["POST"]
    )
    app.add_api_route("/v1/exact/plans", exact_completions, methods=["POST"])
