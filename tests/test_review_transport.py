"""Admission survives response timeouts and upload deadlines are bounded."""

import socket
import threading
import time
import unittest

from exactus.adapters.digest import HashlibHasher
from exactus.adapters.in_process import InProcessExternalComputer
from exactus.proxy import ExactusProxy
from tests._proxy_support import FakeUpstream, call_json, proxy_server


class ReviewTransport(unittest.TestCase):
    def setUp(self):
        self.upstream = FakeUpstream(
            body={
                "choices": [
                    {
                        "message": {"content": "<calc>2+3</calc>"},
                        "finish_reason": "stop",
                    }
                ]
            }
        )
        self.url = self.upstream.start()
        self.servers = []

    def tearDown(self):
        for server in self.servers:
            server.shutdown()
            server.server_close()
        self.upstream.stop()

    def proxy(self, external_computer=None, deadline=0.05):
        proxy = ExactusProxy(
            self.url,
            external_computer or InProcessExternalComputer(HashlibHasher()),
            deadline_s=deadline,
        )
        server, base = proxy_server(proxy, max_concurrent=1)
        self.servers.append(server)
        return server, base

    def test_timed_out_work_keeps_lease(self):
        started, release = threading.Event(), threading.Event()

        class ExternalComputer:
            def evaluate(self, expression, data=None, **kw):
                started.set()
                release.wait(10)
                return InProcessExternalComputer(HashlibHasher()).evaluate(
                    expression, data, **kw
                )

        # The event, not a tiny scheduling margin, keeps work alive after 504.
        server, base = self.proxy(ExternalComputer(), deadline=0.25)
        try:
            status, _, _ = call_json(base + "/v1/chat/completions", {"messages": []})
            self.assertEqual(status, 504)
            self.assertTrue(started.is_set())
            for _ in range(3):
                status, body, _ = call_json(
                    base + "/v1/chat/completions", {"messages": []}
                )
                self.assertEqual(status, 503)
                self.assertEqual(body["error_code"], "overloaded")
            self.assertEqual(len(self.upstream.calls), 1)
        finally:
            release.set()

    def test_upload_deadline(self):
        server, _ = self.proxy(deadline=0.1)
        with socket.create_connection(server.server_address) as sock:
            sock.settimeout(1)
            sock.sendall(
                b"POST /v1/chat/completions HTTP/1.0\r\nHost: localhost\r\nContent-Length: 10\r\n\r\n{"
            )
            started = time.monotonic()
            data = b""
            while True:
                chunk = sock.recv(4096)
                if not chunk:
                    break
                data += chunk
            self.assertIn(b"504", data)
            self.assertIn(b"deadline_exceeded", data)
            self.assertLess(time.monotonic() - started, 0.8)
        self.assertEqual(self.upstream.calls, [])

    def test_incomplete_header_deadline(self):
        server, _ = self.proxy(deadline=0.1)
        with socket.create_connection(server.server_address) as sock:
            sock.settimeout(1)
            sock.sendall(b"POST /v1/chat/completions HTTP/1.1\r\nHost:")
            self.assertEqual(sock.recv(4096), b"")

    def test_chunked_rejected_explicitly(self):
        server, _ = self.proxy()
        with socket.create_connection(server.server_address) as sock:
            sock.settimeout(1)
            sock.sendall(
                b"POST /v1/chat/completions HTTP/1.0\r\nHost: localhost\r\nTransfer-Encoding: chunked\r\n\r\n0\r\n\r\n"
            )
            self.assertIn(b"400", sock.recv(4096))
