"""Limit and refusal tests for the exactus proxy HTTP transport.

Author: Olshansky Grigory
"""

import json
import threading
import time
import unittest

from exactus.adapters.digest import HashlibHasher
from exactus.adapters.in_process import InProcessExternalComputer
from exactus.proxy import ExactusProxy

from tests._proxy_support import FakeUpstream, call, call_json, proxy_server

__author__ = "Olshansky Grigory"


class SlowExternalComputer:
    """External computer stub that sleeps before delegating to the in-process one."""

    def __init__(self, delay_s: float) -> None:
        """Store the delay and build the delegate external computer."""
        self._delay_s = delay_s
        self._delegate = InProcessExternalComputer(HashlibHasher())

    def evaluate(self, expression, data=None, *, output_format="dec"):
        """Sleep, then evaluate one expression with the delegate."""
        time.sleep(self._delay_s)
        return self._delegate.evaluate(expression, data, output_format=output_format)


class TestProxyLimits(unittest.TestCase):
    """Verifies 413, 400, 504 and 503 refusal behaviour of the transport."""

    def setUp(self) -> None:
        """Start one fake upstream with a computed chat reply."""
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
        self._servers = []

    def tearDown(self) -> None:
        """Stop every proxy server and the fake upstream."""
        for server in self._servers:
            server.shutdown()
            server.server_close()
        self.upstream.stop()

    def _proxy(self, external_computer=None, **options) -> str:
        """Start one proxy server and return its base URL."""
        proxy = ExactusProxy(
            self.url,
            external_computer or InProcessExternalComputer(HashlibHasher()),
            deadline_s=options.pop("deadline_s", 600.0),
        )
        server, base = proxy_server(proxy, **options)
        self._servers.append(server)
        return base

    def test_oversized_body_is_413(self):
        """A body above the limit is refused before it is read."""
        base = self._proxy(max_body=4096)
        status, body, _ = call_json(base + "/v1/chat/completions", {"blob": "x" * 8192})
        self.assertEqual(status, 413)
        self.assertEqual(body["error_code"], "body_too_large")
        self.assertEqual(len(self.upstream.calls), 0)

    def test_invalid_json_is_400(self):
        """A malformed JSON body yields 400 instead of a silent empty payload."""
        base = self._proxy()
        status, raw, _ = call(base + "/v1/chat/completions", body=b"{not json")
        self.assertEqual(status, 400)
        self.assertEqual(json.loads(raw)["error_code"], "invalid_json")

    def test_deadline_exceeded_is_504(self):
        """A slow compute call is cut by the end-to-end deadline."""
        base = self._proxy(SlowExternalComputer(1.0), deadline_s=0.25)
        status, body, _ = call_json(
            base + "/v1/chat/completions", {"model": "m", "messages": []}
        )
        self.assertEqual(status, 504)
        self.assertEqual(body["error_code"], "deadline_exceeded")

    def test_admission_overflow_is_503(self):
        """Concurrent load beyond the admission limit yields 503."""
        base = self._proxy(SlowExternalComputer(0.6), max_concurrent=1)
        results = []

        def worker() -> None:
            """Send one blocking chat request and record its status."""
            results.append(
                call_json(
                    base + "/v1/chat/completions", {"model": "m", "messages": []}
                )[0]
            )

        first = threading.Thread(target=worker)
        first.start()
        time.sleep(0.2)
        status, body, _ = call_json(
            base + "/v1/chat/completions", {"model": "m", "messages": []}
        )
        first.join(5)
        self.assertEqual(status, 503)
        self.assertEqual(body["error_code"], "overloaded")
        self.assertIn(200, results)


if __name__ == "__main__":
    unittest.main()
