"""HTTP transport for the exactus proxy: limits, deadlines and admission."""

from __future__ import annotations

__author__ = "Olshansky Grigory"

import json
import socket
import threading
import time
from typing import Any

from .proxy_messages import (
    decode_reply as decode_reply,
)
from .proxy_messages import (
    parse_body as parse_body,
)
from .proxy_messages import (
    read_body as read_body,
)
from .proxy_messages import (
    select_headers as select_headers,
)
from .proxy_modes import RequestError
from .transport_headers import DeadlineHeaders
from .transport_work import invoke_with_deadline

_MAX_BODY = 1 << 20
_MAX_CONCURRENT = 16


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
