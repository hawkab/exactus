"""HTTP transport for the exactus proxy: limits, deadlines and admission.

Author: Olshansky Grigory
"""

from __future__ import annotations

import json
import threading
import socket
import time
from typing import Any

from .proxy_modes import RequestError
from .transport_work import invoke_with_deadline
from .transport_headers import DeadlineHeaders

__author__ = "Olshansky Grigory"

_MAX_BODY = 1 << 20
_MAX_CONCURRENT = 16
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


class Admission:
    """Bounds the number of concurrently handled requests."""

    def __init__(self, limit: int) -> None:
        """Create a semaphore admitting up to limit concurrent requests."""
        self._semaphore = threading.Semaphore(max(1, limit))

    def enter(self) -> bool:
        """Try to admit one request without blocking."""
        return self._semaphore.acquire(blocking=False)

    def exit(self) -> None:
        """Release one admitted request slot."""
        self._semaphore.release()


def make_handler(
    proxy,
    *,
    max_body: int = _MAX_BODY,
    max_concurrent: int = _MAX_CONCURRENT,
    deadline_s: float | None = None,
):
    """Return an HTTP request handler bound to one proxy."""
    admission = Admission(max_concurrent)
    timeout = (
        deadline_s if deadline_s is not None else getattr(proxy, "deadline_s", 600.0)
    )

    class Handler(DeadlineHeaders):
        """Transports one proxy over HTTP with strict JSON bodies."""

        _admission = admission
        _timeout = timeout

        def do_POST(self) -> None:
            """Process one JSON request and write the JSON reply."""
            self._respond("POST")

        def do_GET(self) -> None:
            """Forward one passthrough request and write the JSON reply."""
            self._respond("GET")

        def _respond(self, method: str) -> None:
            """Read, handle and answer one request under the limits."""
            self._header_timer.cancel()
            try:
                status, response, extra = self._outcome(method)
            except RequestError as exc:
                status, response, extra = exc.status, exc.body(), {}
            except (socket.timeout, TimeoutError):
                status, response, extra = (
                    504,
                    {
                        "error": "request deadline exceeded",
                        "error_code": "deadline_exceeded",
                    },
                    {},
                )
            except Exception:
                status, response, extra = (
                    500,
                    {"error": "internal error", "error_code": "internal_error"},
                    {},
                )
            self._write(status, response, extra)

        def _outcome(self, method: str):
            """Run one request to completion under the total deadline."""
            payload = parse_body(read_body(self, max_body)) if method == "POST" else {}
            remaining = self._deadline - time.monotonic()
            if remaining <= 0:
                raise RequestError(
                    "request deadline exceeded", "deadline_exceeded", 504
                )
            self._lease.hold()
            outcome = invoke_with_deadline(
                lambda: proxy.handle_ex(
                    self.path, payload, method=method, headers=self.headers
                ),
                remaining,
                self._lease.release,
                executor=getattr(self, "_executor", None),
                cancellation=getattr(self, "_cancellation", None),
            )
            if outcome is None:
                raise RequestError(
                    "request deadline exceeded", "deadline_exceeded", 504
                )
            return outcome

        def _write(self, status: int, response, extra: dict) -> None:
            """Write one JSON response with explicit length and extra headers."""
            # A bounded grace period allows the timeout response itself to be sent.
            self.connection.settimeout(
                max(0.1, min(1.0, self._deadline - time.monotonic()))
            )
            encoded = json.dumps(response, ensure_ascii=False).encode("utf-8")
            if len(encoded) > 1048576:
                status, encoded = (
                    502,
                    b'{"error":"response exceeds 1 MiB","error_code":"response_too_large"}',
                )
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(encoded)))
            for name, value in extra.items():
                self.send_header(name, value)
            self.end_headers()
            self.wfile.write(encoded)

        def log_message(self, *args: Any) -> None:
            """Silence the default access log."""

    return Handler


def build_server(proxy, host: str, port: int, **handler_options):
    """Create the threaded HTTP server for one proxy."""
    from .http_server import BoundedHTTPServer

    return BoundedHTTPServer(
        (host, port),
        make_handler(proxy, **handler_options),
        limit=handler_options.get("max_concurrent", _MAX_CONCURRENT),
    )
