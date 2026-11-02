"""Validate and execute exact-completion HTTP requests through coordinator ports."""

from __future__ import annotations

__author__ = "Olshansky Grigory"

import uuid

from ..proxy_modes import RequestError, decode_data, resolve_mode
from .async_work import blocking
from .endpoint_adapter import ExactusCoordinator
from .endpoint_response import _serialize_encoded

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
_MAX_PROMPT = 16384


async def coordinator_prompt(coordinator: ExactusCoordinator, prompt: str) -> list[int]:
    """Tokenize the prompt with the coordinator's tokenizer port."""
    tokenizer = coordinator.tokenizer
    encode = getattr(tokenizer, "encode_prompt", tokenizer.encode)
    return await blocking(encode, prompt)


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
