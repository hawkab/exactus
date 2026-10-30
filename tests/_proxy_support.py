"""Shared test support: a fake OpenAI upstream and a proxy server harness.

Author: Olshansky Grigory
"""

from __future__ import annotations

import json
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from exactus.proxy_transport import make_handler

__author__ = "Olshansky Grigory"


class FakeUpstream:
    """Scriptable OpenAI-style upstream recording every received request."""

    def __init__(
        self,
        status: int = 200,
        body=None,
        content_type: str = "application/json",
        delay_s: float = 0.0,
    ) -> None:
        """Store the canned reply and start with an empty request log."""
        self.status = status
        self.body = body if body is not None else {}
        self.content_type = content_type
        self.delay_s = delay_s
        self.calls: list[dict] = []
        self._server = None

    def set_reply(
        self,
        status: int,
        body,
        content_type: str = "application/json",
        delay_s: float = 0.0,
    ) -> None:
        """Replace the canned reply for subsequent requests."""
        self.status = status
        self.body = body
        self.content_type = content_type
        self.delay_s = delay_s

    def start(self) -> str:
        """Bind an ephemeral port, serve in a daemon thread and return the URL."""
        upstream = self

        class Handler(BaseHTTPRequestHandler):
            """Records one request and writes the scripted reply."""

            def do_POST(self) -> None:
                """Record and answer one POST request."""
                self._answer()

            def do_GET(self) -> None:
                """Record and answer one GET request."""
                self._answer()

            def _answer(self) -> None:
                """Record the request and write the scripted reply."""
                length = int(self.headers.get("Content-Length") or 0)
                body = self.rfile.read(length)
                upstream.calls.append(
                    {
                        "method": self.command,
                        "path": self.path,
                        "headers": dict(self.headers),
                        "body": body,
                    }
                )
                if upstream.delay_s:
                    time.sleep(upstream.delay_s)
                payload = (
                    upstream.body
                    if isinstance(upstream.body, bytes)
                    else json.dumps(upstream.body).encode("utf-8")
                )
                self.send_response(upstream.status)
                self.send_header("Content-Type", upstream.content_type)
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                try:
                    self.wfile.write(payload)
                except OSError:
                    pass  # cap/deadline tests deliberately close the upstream socket

            def log_message(self, *args) -> None:
                """Silence the default access log."""

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=self._server.serve_forever, daemon=True).start()
        return f"http://127.0.0.1:{self._server.server_address[1]}"

    def stop(self) -> None:
        """Stop the upstream server."""
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
            self._server = None


def proxy_server(proxy, **options):
    """Serve one proxy in a daemon thread and return the base URL."""
    server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(proxy, **options))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, f"http://127.0.0.1:{server.server_address[1]}"


def call(url: str, *, method: str = "POST", body: bytes = b"", headers=None):
    """Send one real HTTP request and return status, raw body and headers."""
    request = urllib.request.Request(
        url,
        data=body if method == "POST" else None,
        method=method,
        headers=headers or {},
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as reply:
            return reply.status, reply.read(), dict(reply.headers)
    except urllib.error.HTTPError as exc:
        status, raw, headers = exc.code, exc.read(), dict(exc.headers)
        exc.close()
        return status, raw, headers


def call_json(url: str, payload=None, **kwargs):
    """Send one JSON request and return status, decoded body and headers."""
    body = b"" if payload is None else json.dumps(payload).encode("utf-8")
    status, raw, headers = call(url, body=body, **kwargs)
    return status, json.loads(raw or b"{}"), headers
