"""HTTP route adapter exposing the exact-span endpoint on a FastAPI app.

Author: Olshansky Grigory
"""

from __future__ import annotations

import asyncio
import json
import uuid
import time

from ..domain.finalization import finalize
from ..proxy_modes import RequestError, decode_data, resolve_mode
from .async_work import blocking
from .endpoint_adapter import ExactusCoordinator
from .endpoint_http import Lease, encode, failure, response
from ..http_io import deadline as http_deadline
from ..cancellation import Cancellation, current as current_cancellation

try:
    from starlette.requests import Request
except ImportError:
    Request = None

__author__ = "Olshansky Grigory"

_ALLOWED = {
    "model",
    "exact_compute",
    "prompt",
    "max_tokens",
    "require_final",
    "output_format",
    "data",
    "mode",
}
_MAX_BODY = 32768
_MAX_PROMPT = 16384
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
        http_deadline.set(expires)
        current_cancellation.set(cancellation)
        payload = await _load(request)
        owner = asyncio.current_task()

        async def disconnect():
            while True:
                message = await request.receive()
                if message["type"] == "http.disconnect":
                    cancellation.cancel()
                    owner.cancel()
                    return
                await asyncio.sleep(0.05)  # tolerate simplistic ASGI test receivers

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


async def coordinator_prompt(coordinator: ExactusCoordinator, prompt: str) -> list[int]:
    """Tokenize the prompt with the coordinator's tokenizer port."""
    tokenizer = coordinator._tokenizer
    encode = getattr(tokenizer, "encode_prompt", tokenizer.encode)
    return await blocking(encode, prompt)


async def _load(request) -> dict:
    """Read one request body under a hard size bound with strict JSON."""
    declared = request.headers.get("content-length")
    if declared is not None:
        try:
            length = int(declared)
            if length < 0:
                raise ValueError("negative length")
            too_large = length > _MAX_BODY
        except ValueError as exc:
            raise RequestError(
                "invalid Content-Length header", "invalid_request", 400
            ) from exc
        if too_large:
            raise RequestError(
                "request body exceeds the size limit", "body_too_large", 413
            )
    body = bytearray()
    async for chunk in request.stream():
        if len(body) + len(chunk) > _MAX_BODY:
            raise RequestError(
                "request body exceeds the size limit", "body_too_large", 413
            )
        body.extend(chunk)
    try:
        payload = json.loads(body or b"")
    except (ValueError, RecursionError) as exc:
        raise RequestError(
            "request body is not valid JSON", "invalid_json", 400
        ) from exc
    if not isinstance(payload, dict):
        raise RequestError("request body must be a JSON object", "invalid_request", 422)
    return payload


async def _run(
    coordinator: ExactusCoordinator,
    models: tuple[str, ...],
    payload: dict,
    deadline_s: float,
) -> dict:
    """Validate one request and run it under the explicit mode policy.

    The coordinator is always asked to preserve the text; the mode
    policy decides here what the client receives, so a missing final
    span is an explicit coded failure in every strict mode.
    """
    if set(payload) - _ALLOWED:
        raise RequestError("unsupported request fields", "invalid_request", 422)
    if payload.get("exact_compute") is not True:
        raise RequestError("exact_compute=true is required", "invalid_request", 422)
    prompt = payload.get("prompt")
    if (
        not isinstance(prompt, str)
        or not prompt.strip()
        or len(prompt.encode("utf-8")) > _MAX_PROMPT
    ):
        raise RequestError(
            "prompt must be non-empty text under 16 KiB", "invalid_request"
        )
    if payload.get("model", models[0]) not in models:
        raise RequestError("model switching is not allowed", "invalid_request", 422)
    max_tokens = payload.get("max_tokens", 2048)
    if type(max_tokens) is not int or max_tokens <= 0:
        raise RequestError(
            "max_tokens must be a positive integer", "invalid_request", 422
        )
    if "require_final" in payload and type(payload["require_final"]) is not bool:
        raise RequestError("require_final must be a boolean", "invalid_request", 422)
    mode = resolve_mode(
        payload.get("mode"), legacy_require_final=payload.get("require_final")
    )
    data = decode_data(payload.get("data"))
    prompt_ids = await coordinator_prompt(coordinator, prompt)
    request_id = str(uuid.uuid4())
    result = await coordinator.run(
        prompt_ids,
        data,
        max_tokens=max_tokens,
        require_final=False,
        output_format=payload.get("output_format", "dec"),
        request_id=request_id,
        mode=mode,
        task_prompt=prompt,
    )
    return await blocking(_serialize_encoded, result, mode, models[0])


def _serialize_encoded(result, mode, model):
    return encode({"model": model, "exact_compute": True, **_serialize(result, mode)})


def _serialize(result, mode):
    proof = result.proof
    if proof is None:  # third-party coordinator contract; still bounded/off-loop
        text, proof, exact = finalize(result.transcript, result.operations, mode)
        emitted = any(s.final and s.origin == "model_marker" for s in proof.spans)
    else:
        text, exact = result.text, result.exact_final
        emitted = result.model_final_emitted
    return {
        "text": text,
        "operations": [
            dict(expression=o.expression, value=o.value, final=o.final, format=o.format)
            for o in result.operations
        ],
        "engine_segments": result.engine_segments,
        "sampled_tokens": result.sampled_tokens,
        "known_tokens": result.known_tokens,
        "known_tokens_scope": "standalone_values_tokenizer_estimate_not_rendered_text_length",
        "prefill_tokens": result.prefill_tokens,
        "exact_final": exact,
        "elapsed_s": result.elapsed_s,
        "request_id": result.request_id,
        "parent_id": result.parent_id,
        "finish_reason": result.finish_reason,
        "audit": proof.as_dict(detail=mode == "audit"),
        **proof.verification(),
        "model_final_emitted": emitted,
        "protocol_complete": True,
        "generation_complete": result.finish_reason == "stop",
        "engine_wait_s": result.engine_wait_s,
        "cpu_stage_s": result.cpu_stage_s,
        "prompt_tokens": result.prompt_tokens,
        "sampled_tokens_exact": result.sampled_tokens_exact,
        "mode": mode,
    }
