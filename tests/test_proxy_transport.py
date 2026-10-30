"""E2E HTTP tests for the exactus proxy transport and compute paths.

Author: Olshansky Grigory
"""

import hashlib
import json
import unittest

from exactus.adapters.digest import HashlibHasher
from exactus.adapters.in_process import InProcessExternalComputer
from exactus.proxy import ExactusProxy

from tests._proxy_support import FakeUpstream, call_json, proxy_server

__author__ = "Olshansky Grigory"

_ABC_SHA256 = str(int.from_bytes(hashlib.sha256(b"abc").digest(), "big"))


class TestProxyTransport(unittest.TestCase):
    """Exercises the proxy over real HTTP against a scripted upstream."""

    def setUp(self) -> None:
        """Start one fake upstream and one proxy server."""
        self.upstream = FakeUpstream(
            body={
                "choices": [
                    {
                        "message": {"role": "assistant", "content": "hello"},
                        "finish_reason": "stop",
                    }
                ]
            }
        )
        self.url = self.upstream.start()
        self.proxy = ExactusProxy(self.url, InProcessExternalComputer(HashlibHasher()))
        self.server, self.base = proxy_server(self.proxy)

    def tearDown(self) -> None:
        """Stop the proxy and the fake upstream."""
        self.server.shutdown()
        self.server.server_close()
        self.upstream.stop()

    def _chat(self, payload, headers=None):
        """POST one chat completion request through the proxy."""
        return call_json(self.base + "/v1/chat/completions", payload, headers=headers)

    def test_get_models_is_passthrough(self):
        """GET /v1/models reaches the upstream as a GET request."""
        self.upstream.set_reply(200, {"object": "list", "data": [{"id": "m1"}]})
        status, body, _ = call_json(self.base + "/v1/models", method="GET")
        self.assertEqual(status, 200)
        self.assertEqual(body, {"object": "list", "data": [{"id": "m1"}]})
        self.assertEqual(self.upstream.calls[-1]["method"], "GET")

    def test_client_headers_forwarded(self):
        """Authorization, Accept and X- headers reach the upstream."""
        self._chat(
            {"model": "m", "messages": []},
            headers={
                "Authorization": "Bearer audit-token",
                "Accept": "application/json",
                "X-Request-Id": "7",
            },
        )
        sent = {k.lower(): v for k, v in self.upstream.calls[-1]["headers"].items()}
        self.assertEqual(sent.get("authorization"), "Bearer audit-token")
        self.assertEqual(sent.get("x-request-id"), "7")
        self.assertEqual(sent.get("accept"), "application/json")

    def test_stream_is_rejected_honestly(self):
        """stream=true yields an explicit JSON 400, never a 502."""
        before = len(self.upstream.calls)
        status, body, _ = self._chat({"model": "m", "messages": [], "stream": True})
        self.assertEqual(status, 400)
        self.assertEqual(body["error_code"], "streaming_unsupported")
        self.assertIn("honest limited mode", body["error"])
        self.assertEqual(len(self.upstream.calls), before)

    def test_data_hash_computed_locally_and_stripped(self):
        """data is base64-decoded locally and never forwarded upstream."""
        self.upstream.set_reply(
            200,
            {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": "<calc-final>sha256(@m)</calc-final>",
                        },
                        "finish_reason": "stop",
                    }
                ]
            },
        )
        status, body, _ = self._chat(
            {"model": "m", "messages": [], "data": {"m": "YWJj"}}
        )
        self.assertEqual(status, 200)
        self.assertEqual(body["choices"][0]["message"]["content"], _ABC_SHA256)
        self.assertNotIn("data", json.loads(self.upstream.calls[-1]["body"]))
        self.assertTrue(body["exactus"]["exact_final"])

    def test_n_choices_all_processed(self):
        """Every choice of an n>1 response is computed, not just the first."""
        self.upstream.set_reply(
            200,
            {
                "choices": [
                    {
                        "message": {"content": "a <calc>2+3</calc>"},
                        "finish_reason": "stop",
                    },
                    {
                        "message": {"content": "<calc>10*10</calc> b"},
                        "finish_reason": "stop",
                    },
                ]
            },
        )
        status, body, _ = self._chat({"model": "m", "messages": []})
        self.assertEqual(status, 200)
        contents = [c["message"]["content"] for c in body["choices"]]
        self.assertEqual(contents, ["a 5", "100 b"])
        self.assertEqual(body["exactus"]["operations"], 2)

    def test_preserve_text_keeps_report(self):
        """The default mode inserts values in place instead of eating the report."""
        self.upstream.set_reply(
            200,
            {
                "choices": [
                    {
                        "message": {
                            "content": "Summary. Budget <calc-final>6*7</calc-final>. Recommendations follow."
                        },
                        "finish_reason": "stop",
                    }
                ]
            },
        )
        status, body, headers = self._chat({"model": "m", "messages": []})
        self.assertEqual(status, 200)
        self.assertEqual(
            body["choices"][0]["message"]["content"],
            "Summary. Budget 42. Recommendations follow.",
        )
        self.assertEqual(body["exactus"]["mode"], "preserve-text")
        self.assertEqual(body["exactus"]["computed"], ["42"])
        self.assertEqual(headers.get("X-Exactus-Ops"), "1")

    def test_numeric_final_returns_only_number(self):
        """numeric-final collapses to the final value by explicit contract."""
        self.upstream.set_reply(
            200,
            {
                "choices": [
                    {
                        "message": {"content": "Answer <calc-final>6*7</calc-final>"},
                        "finish_reason": "stop",
                    }
                ]
            },
        )
        status, body, _ = self._chat(
            {"model": "m", "messages": [], "mode": "numeric-final"}
        )
        self.assertEqual(status, 200)
        self.assertEqual(body["choices"][0]["message"]["content"], "42")

    def test_strict_mode_violation_is_machine_readable(self):
        """strict mode turns a missing final span into a coded 422."""
        self.upstream.set_reply(
            200,
            {
                "choices": [
                    {"message": {"content": "guessed 42"}, "finish_reason": "stop"}
                ]
            },
        )
        status, body, _ = self._chat({"model": "m", "messages": [], "mode": "strict"})
        self.assertEqual(status, 422)
        self.assertEqual(body["error_code"], "missing_final")

    def test_upstream_html_error_is_valid_json(self):
        """An HTML upstream error body still yields a valid JSON error."""
        self.upstream.set_reply(502, b"<html>boom</html>", "text/html")
        status, body, _ = self._chat({"model": "m", "messages": []})
        self.assertEqual(status, 502)
        self.assertIn("error", body)
        self.assertEqual(body["error_code"], "upstream_error")

    def test_completions_unsupported(self):
        """/v1/completions is refused with an explicit 501 subset notice."""
        status, body, _ = call_json(self.base + "/v1/completions", {"model": "m"})
        self.assertEqual(status, 501)
        self.assertEqual(body["error_code"], "unsupported_endpoint")
        self.assertIn("honest limited mode", body["error"])

    def test_invalid_data_payload_is_rejected(self):
        """Invalid base64 in data is refused with a coded 422 before forwarding."""
        before = len(self.upstream.calls)
        status, body, _ = self._chat(
            {"model": "m", "messages": [], "data": {"m": "!!!"}}
        )
        self.assertEqual(status, 422)
        self.assertEqual(body["error_code"], "invalid_data")
        self.assertEqual(len(self.upstream.calls), before)


if __name__ == "__main__":
    unittest.main()
