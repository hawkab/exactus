"""Actual loopback transport caps/deadlines and ASGI lifecycle regressions."""

__author__ = "Olshansky Grigory"


import socket
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from exactus.cancellation import Cancellation
from exactus.cancellation import current as current_cancellation
from exactus.http_io import request_bytes
from exactus.proxy import ExactusProxy
from exactus.proxy_transport import build_server
from tests._support_deep_http import DeepHTTPFixture


class DeepHTTP(DeepHTTPFixture):
    """Verify deep http behavior."""

    def test_cancellation_interrupts_blocked_upstream_read(self):
        """Verify cancellation interrupts blocked upstream read."""
        ready, release = threading.Event(), threading.Event()

        class Handler(BaseHTTPRequestHandler):
            """Exercise Handler behavior with isolated fixtures."""

            def do_POST(self):
                """Handle the POST request according to this transport regression fixture."""
                self.send_response(200)
                self.send_header("Content-Length", "100")
                self.end_headers()
                self.wfile.write(b"x")
                self.wfile.flush()
                ready.set()
                release.wait(2)

            def log_message(self, *args):
                """Suppress access logging for the isolated test server."""
                pass

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        server.daemon_threads = True
        threading.Thread(target=server.serve_forever, daemon=True).start()
        token = Cancellation()
        errors = []

        def reader():
            """Read the local response while recording cancellation or transport failure."""
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
        """Verify socket limit precedes handler thread creation."""
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
        """Verify chunked response closes without spurious bad fd."""

        class Handler(BaseHTTPRequestHandler):
            """Exercise Handler behavior with isolated fixtures."""

            protocol_version = "HTTP/1.1"

            def do_POST(self):
                """Handle the POST request according to this transport regression fixture."""
                self.send_response(200)
                self.send_header("Transfer-Encoding", "chunked")
                self.end_headers()
                self.wfile.write(b'7\r\n{"x":1}\r\n0\r\n\r\n')

            def log_message(self, *args):
                """Suppress access logging for the isolated test server."""
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

    def test_absolute_deadline_stops_drip_headers(self):
        """Verify absolute deadline stops drip headers."""

        class Handler(BaseHTTPRequestHandler):
            """Exercise Handler behavior with isolated fixtures."""

            def do_GET(self):
                """Handle the GET request according to this transport regression fixture."""
                try:
                    for byte in b"HTTP/1.0 200 OK\r\nContent-Length: 1\r\n\r\nx":
                        self.connection.sendall(bytes([byte]))
                        time.sleep(0.01)
                except OSError:
                    pass

            def log_message(self, *args):
                """Suppress access logging for the isolated test server."""
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
