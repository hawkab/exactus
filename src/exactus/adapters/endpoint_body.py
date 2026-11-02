"""Decode streamed endpoint bodies and validate declared sizes."""

__author__ = "Olshansky Grigory"

import json

from ..proxy_modes import RequestError

_MAX_BODY = 32768


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
