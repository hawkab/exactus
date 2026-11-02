"""ASGI deadlines retain ownership of non-cancellable CPU work."""

__author__ = "Olshansky Grigory"

import asyncio
import json
import threading
import time
import unittest

from exactus.adapters.async_work import blocking
from exactus.adapters.digest import HashlibHasher
from exactus.adapters.endpoint_adapter import ExactusCoordinator
from exactus.adapters.endpoint_route import attach
from exactus.adapters.in_process import InProcessExternalComputer
from exactus.adapters.tokenizer import ByteTokenizer
from exactus.domain.ports import Segment

try:
    from fastapi import FastAPI
except ImportError:
    FastAPI = None


async def post(app, *, upload_delay=0):
    """Send a request to the local test endpoint and capture its result."""
    body = json.dumps({"prompt": "p", "exact_compute": True}).encode()
    response, pending = [], [body]

    async def receive():
        """Provide the next request or disconnect event to the ASGI application."""
        await asyncio.sleep(upload_delay)
        return {
            "type": "http.request",
            "body": pending.pop() if pending else b"",
            "more_body": False,
        }

    async def send(message):
        """Capture ASGI response events for the transport assertions."""
        response.append(message)

    await app(
        {
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
            "server": ("127.0.0.1", 80),
        },
        receive,
        send,
    )
    return next(
        item["status"] for item in response if item["type"] == "http.response.start"
    )


@unittest.skipUnless(FastAPI, "server extra is not installed")
class EndpointLimits(unittest.TestCase):
    """Exercise Endpoint Limits behavior with isolated fixtures."""

    def app(self, external_computer):
        """Create the local endpoint application for this isolated regression."""

        class Engine:
            """Exercise Engine behavior with isolated fixtures."""

            async def generate(self, *args):
                """Yield the configured generation result for a correlated engine request."""
                text = "<calc-final>2+3</calc-final>"
                return Segment(text, tuple(text.encode()), len(text), "stop")

        app = FastAPI()
        attach(
            app,
            ExactusCoordinator(Engine(), ByteTokenizer(), external_computer),
            ("m",),
            deadline_s=0.03,
            max_concurrent=1,
        )
        return app

    def test_deadline_covers_upload(self):
        """Verify deadline covers upload."""

        async def run():
            """Execute one synchronized workload case."""
            app = self.app(InProcessExternalComputer(HashlibHasher()))
            self.assertEqual(await post(app, upload_delay=0.2), 504)
            await asyncio.sleep(0.01)
            self.assertEqual(await post(app), 200)

        asyncio.run(run())

    def test_timeout_does_not_free_running_thread(self):
        """Verify timeout does not free running thread."""

        class ExternalComputer:
            """Exercise External Computer behavior with isolated fixtures."""

            delay = 0.2

            def evaluate(self, *args, **kwargs):
                """Evaluate or deliberately reject the requested expression in this isolated test double."""
                time.sleep(self.delay)
                return InProcessExternalComputer(HashlibHasher()).evaluate(
                    *args, **kwargs
                )

        async def run():
            """Execute one synchronized workload case."""
            external_computer = ExternalComputer()
            app = self.app(external_computer)
            self.assertEqual(await post(app), 504)
            self.assertEqual(await post(app), 503)
            await asyncio.sleep(0.25)
            external_computer.delay = 0
            self.assertEqual(await post(app), 200)

        asyncio.run(run())

    def test_repeated_cancellation_still_drains_thread(self):
        """Verify repeated cancellation still drains thread."""

        async def run():
            """Execute one synchronized workload case."""
            started, release = threading.Event(), threading.Event()

            def slow():
                """Simulate blocking computation that remains active after transport cancellation."""
                started.set()
                release.wait(2)

            task = asyncio.create_task(blocking(slow))
            await asyncio.to_thread(started.wait, 1)
            try:
                task.cancel()
                await asyncio.sleep(0.01)
                task.cancel()
                await asyncio.sleep(0.01)
                self.assertFalse(task.done())
            finally:
                release.set()
            with self.assertRaises(asyncio.CancelledError):
                await task

        asyncio.run(run())
