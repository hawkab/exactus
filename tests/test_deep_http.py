"""Actual loopback transport caps/deadlines and ASGI lifecycle regressions."""

import asyncio
import json
import threading
import time
import unittest
import socket
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from exactus.adapters.digest import HashlibHasher
from exactus.adapters.endpoint_adapter import ExactusCoordinator
from exactus.adapters.endpoint_route import attach
from exactus.adapters.engine_client import EngineError
from exactus.adapters.in_process import InProcessExternalComputer
from exactus.adapters.tokenizer import ByteTokenizer
from exactus.domain.ports import Segment
from exactus.http_io import request_bytes
from exactus.cancellation import Cancellation, current as current_cancellation
from exactus.proxy_transport import build_server
from exactus.proxy import ExactusProxy
from exactus.proxy_modes import RequestError
from tests._proxy_support import FakeUpstream
from tests.test_endpoint_limits import post, FastAPI


class DeepHTTP(unittest.TestCase):
    def external_computer(self):
        return InProcessExternalComputer(HashlibHasher())

    def test_cancellation_interrupts_blocked_upstream_read(self):
        ready, release = threading.Event(), threading.Event()

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                self.send_response(200)
                self.send_header("Content-Length", "100")
                self.end_headers()
                self.wfile.write(b"x")
                self.wfile.flush()
                ready.set()
                release.wait(2)

            def log_message(self, *args):
                pass

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        server.daemon_threads = True
        threading.Thread(target=server.serve_forever, daemon=True).start()
        token = Cancellation()
        errors = []

        def reader():
            current_cancellation.set(token)
            try:
                request_bytes(
                    f"http://127.0.0.1:{server.server_address[1]}", timeout=10
                )
            except Exception as exc:
                errors.append(exc)

        thread = threading.Thread(target=reader, daemon=True)
        thread.start()
        try:
            self.assertTrue(ready.wait(1))
            token.cancel()
            thread.join(1)
            self.assertFalse(thread.is_alive())
            self.assertEqual(len(errors), 1)
        finally:
            release.set()
            server.shutdown()
            server.server_close()

    def test_socket_limit_precedes_handler_thread_creation(self):
        server = build_server(
            ExactusProxy("unused", self.external_computer()),
            "127.0.0.1",
            0,
            max_concurrent=1,
            deadline_s=1,
        )
        threading.Thread(target=server.serve_forever, daemon=True).start()
        first = socket.create_connection(server.server_address)
        try:
            first.sendall(b"GET /v1/models HTTP/1.1\r\nHost:")
            for _ in range(100):
                if server._connections._value == 0:
                    break
                time.sleep(0.005)
            status, body = request_bytes(
                f"http://127.0.0.1:{server.server_address[1]}/v1/models",
                method="GET",
                timeout=1,
            )
            self.assertEqual(status, 503)
            self.assertIn(b"overloaded", body)
            self.assertEqual(len(server._executor._threads), 0)
        finally:
            first.close()
            server.shutdown()
            server.server_close()

    def test_chunked_response_closes_without_spurious_bad_fd(self):
        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def do_POST(self):
                self.send_response(200)
                self.send_header("Transfer-Encoding", "chunked")
                self.end_headers()
                self.wfile.write(b'7\r\n{"x":1}\r\n0\r\n\r\n')

            def log_message(self, *args):
                pass

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        server.daemon_threads = True
        threading.Thread(target=server.serve_forever, daemon=True).start()
        try:
            status, body = request_bytes(
                f"http://127.0.0.1:{server.server_address[1]}", timeout=1
            )
            self.assertEqual((status, body), (200, b'{"x":1}'))
        finally:
            server.shutdown()
            server.server_close()

    def test_finish_required_and_stale_logprobs_removed(self):
        proxy = ExactusProxy("unused", self.external_computer())
        reply = {
            "choices": [
                {
                    "message": {"content": "<calc-final>2+3</calc-final>"},
                    "logprobs": {"content": [{"token": "<calc-final>"}]},
                }
            ],
            "usage": {"completion_tokens": 15, "prompt_tokens": 7},
        }
        with self.assertRaises(RequestError):
            proxy._compute({"messages": []}, reply)
        reply["choices"][0]["finish_reason"] = "stop"
        result = proxy._compute({"messages": []}, reply)
        self.assertNotIn("logprobs", result["choices"][0])
        self.assertEqual(result["usage"], result["upstream_usage"])
        self.assertEqual(result["upstream_usage"]["completion_tokens"], 15)

    def test_error_body_is_capped_too(self):
        upstream = FakeUpstream(status=500, body={"error": "x" * 2097152})
        url = upstream.start()
        try:
            status, reply = ExactusProxy(url, self.external_computer())._forward(
                "/v1/models", None, method="GET", headers={}
            )
            self.assertEqual(status, 502)
            self.assertLess(len(str(reply)), 1000)
        finally:
            upstream.stop()

    def test_absolute_deadline_stops_drip_headers(self):
        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                try:
                    for byte in b"HTTP/1.0 200 OK\r\nContent-Length: 1\r\n\r\nx":
                        self.connection.sendall(bytes([byte]))
                        time.sleep(0.01)
                except OSError:
                    pass

            def log_message(self, *args):
                pass

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        server.daemon_threads = True
        threading.Thread(target=server.serve_forever, daemon=True).start()
        try:
            started = time.monotonic()
            with self.assertRaises((TimeoutError, OSError)):
                request_bytes(
                    f"http://127.0.0.1:{server.server_address[1]}",
                    method="GET",
                    timeout=0.06,
                )
            self.assertLess(time.monotonic() - started, 0.5)
        finally:
            server.shutdown()
            server.server_close()

    @unittest.skipUnless(FastAPI, "server extra missing")
    def test_transient_engine_error_is_503(self):
        class Engine:
            async def generate(self, *args):
                raise EngineError("temporarily unavailable")

        app = FastAPI()
        attach(
            app,
            ExactusCoordinator(Engine(), ByteTokenizer(), self.external_computer()),
            ("m",),
        )
        self.assertEqual(asyncio.run(post(app)), 503)

    @unittest.skipUnless(FastAPI, "server extra missing")
    def test_route_collision_fails_before_shadowing(self):
        app = FastAPI()
        coordinator = ExactusCoordinator(
            None, ByteTokenizer(), self.external_computer()
        )
        attach(app, coordinator, ("m",))
        with self.assertRaises(ValueError):
            attach(app, coordinator, ("m",))

    @unittest.skipUnless(FastAPI, "server extra missing")
    def test_strict_endpoint_checks_same_task(self):
        from tests import test_http

        case = test_http.TestExactRoute()
        app, _ = case._app(["<calc-final>43</calc-final>"])
        status, body = case._post(
            app, {"exact_compute": True, "prompt": "Вычисли: 7*6", "mode": "strict"}
        )
        self.assertEqual(status, 422)
        self.assertEqual(body["error_code"], "answer_mismatch")

    @unittest.skipUnless(FastAPI, "server extra missing")
    def test_slow_send_holds_slot_and_expires(self):
        class Engine:
            async def generate(self, *args):
                return Segment("done", (), 1, "stop")

        async def run():
            app = FastAPI()
            attach(
                app,
                ExactusCoordinator(Engine(), ByteTokenizer(), self.external_computer()),
                ("m",),
                deadline_s=0.08,
                max_concurrent=1,
            )
            sending = asyncio.Event()
            body = json.dumps({"exact_compute": True, "prompt": "p"}).encode()

            async def receive():
                return {"type": "http.request", "body": body, "more_body": False}

            async def send(msg):
                if msg["type"] == "http.response.body":
                    sending.set()
                    await asyncio.Event().wait()

            scope = {
                "type": "http",
                "asgi": {"version": "3.0"},
                "http_version": "1.1",
                "method": "POST",
                "path": "/v1/exact/completions",
                "raw_path": b"/v1/exact/completions",
                "query_string": b"",
                "headers": [],
                "scheme": "http",
                "root_path": "",
                "client": ("127.0.0.1", 1),
                "server": ("local", 80),
            }
            first = asyncio.create_task(app(scope, receive, send))
            await asyncio.wait_for(sending.wait(), 1)
            self.assertEqual(await post(app), 503)
            with self.assertRaises(asyncio.TimeoutError):
                await asyncio.wait_for(first, 1)
            self.assertEqual(await post(app), 200)

        asyncio.run(run())
