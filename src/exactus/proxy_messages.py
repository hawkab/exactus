"""Read proxy bodies and preserve supported upstream HTTP headers."""

__author__ = "Olshansky Grigory"

import json
import time
from typing import Any

from .proxy_modes import RequestError

_MAX_BODY = 1 << 20
_FORWARDED = ("authorization", "content-type", "accept")


def read_body(handler, max_body: int = _MAX_BODY) -> bytes:
    """Read the declared request body under a hard size bound.

    The Content-Length header is checked before any body byte is read,
    so an oversized request is refused without being loaded into memory.
    """
    raw_length = handler.headers.get("Content-Length")
    if handler.headers.get("Transfer-Encoding"):
        raise RequestError(
            "chunked request bodies are not supported", "unsupported_transfer", 400
        )
    try:
        length = int(raw_length) if raw_length is not None else 0
    except ValueError as exc:
        raise RequestError(
            "invalid Content-Length header", "invalid_request", 400
        ) from exc
    if length < 0 or length > max_body:
        raise RequestError("request body exceeds the size limit", "body_too_large", 413)
    parts = []
    while length:
        if hasattr(handler, "_deadline"):
            remaining = handler._deadline - time.monotonic()
            if remaining <= 0:
                raise RequestError(
                    "request body deadline exceeded", "deadline_exceeded", 504
                )
            handler.connection.settimeout(remaining)
        chunk = handler.rfile.read1(min(length, 65536))
        if not chunk:
            raise RequestError("incomplete request body", "invalid_request", 400)
        parts.append(chunk)
        length -= len(chunk)
    return b"".join(parts)


def parse_body(raw: bytes) -> dict:
    """Decode one strict JSON request object."""
    try:
        payload = json.loads(raw or b"")
    except (ValueError, RecursionError) as exc:
        raise RequestError(
            "request body is not valid JSON", "invalid_json", 400
        ) from exc
    if not isinstance(payload, dict):
        raise RequestError("request body must be a JSON object", "invalid_request", 422)
    return payload


def select_headers(headers, *, has_body: bool) -> dict:
    """Return the client headers preserved for the upstream request."""
    selected: dict[str, str] = {}
    if headers is not None:
        for name, value in headers.items():
            lower = name.lower()
            if lower in _FORWARDED or lower.startswith("x-"):
                selected[name] = value
    if has_body and not any(name.lower() == "content-type" for name in selected):
        selected["Content-Type"] = "application/json"
    return selected


def decode_reply(status: int, raw: bytes) -> tuple[int, Any]:
    """Decode one upstream reply body into JSON-safe data.

    A non-JSON error body is wrapped into a valid JSON error object so
    clients never receive an unparseable failure payload.
    """
    try:
        parsed = json.loads(raw or b"{}")
    except ValueError:
        snippet = raw[:256].decode("utf-8", "replace") if raw else ""
        code = 502 if 200 <= status < 300 else status
        return code, {
            "error": "upstream returned a non-JSON body",
            "error_code": "upstream_error",
            "upstream_status": status,
            "body_snippet": snippet,
        }
    return status, parsed
