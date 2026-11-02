"""Actual loopback transport caps/deadlines and ASGI lifecycle regressions."""

__author__ = "Olshansky Grigory"

import asyncio
import json
import unittest

from exactus.adapters.endpoint_adapter import ExactusCoordinator
from exactus.adapters.endpoint_route import attach
from exactus.adapters.engine_client import EngineError
from exactus.adapters.tokenizer import ByteTokenizer
from exactus.domain.ports import Segment
from exactus.proxy import ExactusProxy
from exactus.proxy_modes import RequestError
from tests._proxy_support import FakeUpstream
from tests._support_deep_http import DeepHTTPFixture
from tests.test_endpoint_limits import FastAPI, post


class DeepHTTP(DeepHTTPFixture):
    """Verify http completion behavior."""

    def test_finish_required_and_stale_logprobs_removed(self):
        """Verify finish required and stale logprobs removed."""
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
        """Verify error body is capped too."""
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

    @unittest.skipUnless(FastAPI, "server extra missing")
    def test_transient_engine_error_is_503(self):
        """Verify transient engine error is 503."""

        class Engine:
            """Exercise Engine behavior with isolated fixtures."""

            async def generate(self, *args):
                """Yield the configured generation result for a correlated engine request."""
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
        """Verify route collision fails before shadowing."""
        app = FastAPI()
        coordinator = ExactusCoordinator(
            None, ByteTokenizer(), self.external_computer()
        )
        attach(app, coordinator, ("m",))
        with self.assertRaises(ValueError):
            attach(app, coordinator, ("m",))

    @unittest.skipUnless(FastAPI, "server extra missing")
    def test_strict_endpoint_checks_same_task(self):
        """Verify strict endpoint checks same task."""
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
        """Verify slow send holds slot and expires."""

        class Engine:
            """Exercise Engine behavior with isolated fixtures."""

            async def generate(self, *args):
                """Yield the configured generation result for a correlated engine request."""
                return Segment("done", (), 1, "stop")

        async def run():
            """Execute one synchronized workload case."""
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
                """Provide the next request or disconnect event to the ASGI application."""
                return {"type": "http.request", "body": body, "more_body": False}

            async def send(msg):
                """Capture ASGI response events for the transport assertions."""
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
