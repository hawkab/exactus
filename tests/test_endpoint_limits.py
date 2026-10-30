"""ASGI deadlines retain ownership of non-cancellable CPU work."""

import asyncio
import json
import time
import threading
import unittest

from exactus.adapters.digest import HashlibHasher
from exactus.adapters.in_process import InProcessExternalComputer
from exactus.adapters.endpoint_adapter import ExactusCoordinator
from exactus.adapters.endpoint_route import attach
from exactus.adapters.tokenizer import ByteTokenizer
from exactus.domain.ports import Segment
from exactus.adapters.async_work import blocking

try:
    from fastapi import FastAPI
except ImportError:
    FastAPI = None


async def post(app, *, upload_delay=0):
    body = json.dumps({"prompt": "p", "exact_compute": True}).encode()
    response, pending = [], [body]

    async def receive():
        await asyncio.sleep(upload_delay)
        return {
            "type": "http.request",
            "body": pending.pop() if pending else b"",
            "more_body": False,
        }

    async def send(message):
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
    def app(self, external_computer):
        class Engine:
            async def generate(self, *args):
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
        async def run():
            app = self.app(InProcessExternalComputer(HashlibHasher()))
            self.assertEqual(await post(app, upload_delay=0.2), 504)
            await asyncio.sleep(0.01)
            self.assertEqual(await post(app), 200)

        asyncio.run(run())

    def test_timeout_does_not_free_running_thread(self):
        class ExternalComputer:
            delay = 0.2

            def evaluate(self, *args, **kwargs):
                time.sleep(self.delay)
                return InProcessExternalComputer(HashlibHasher()).evaluate(
                    *args, **kwargs
                )

        async def run():
            external_computer = ExternalComputer()
            app = self.app(external_computer)
            self.assertEqual(await post(app), 504)
            self.assertEqual(await post(app), 503)
            await asyncio.sleep(0.25)
            external_computer.delay = 0
            self.assertEqual(await post(app), 200)

        asyncio.run(run())

    def test_repeated_cancellation_still_drains_thread(self):
        async def run():
            started, release = threading.Event(), threading.Event()

            def slow():
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
