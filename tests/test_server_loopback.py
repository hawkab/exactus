"""Real Uvicorn/TCP C=3 and disconnect tests; no GPU/model required."""

__author__ = "Olshansky Grigory"

import asyncio
import json
import socket
import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor

from exactus.adapters.digest import HashlibHasher
from exactus.adapters.endpoint_adapter import ExactusCoordinator
from exactus.adapters.endpoint_route import attach
from exactus.adapters.in_process import InProcessExternalComputer
from exactus.adapters.tokenizer import ByteTokenizer
from exactus.domain.ports import Segment
from exactus.http_io import request_bytes

try:
    import uvicorn
    from fastapi import FastAPI
except ImportError:
    uvicorn = None


@unittest.skipUnless(uvicorn, "install .[server,test] for actual TCP/ASGI tests")
class ServerLoopback(unittest.TestCase):
    """Exercise Server Loopback behavior with isolated fixtures."""

    def start(self, engine):
        """Start an isolated loopback server and register its cleanup."""
        app = FastAPI()
        attach(
            app,
            ExactusCoordinator(
                engine, ByteTokenizer(), InProcessExternalComputer(HashlibHasher())
            ),
            ("m",),
            max_concurrent=3,
            deadline_s=2.0,
        )
        sock = socket.socket()
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
        server = uvicorn.Server(
            uvicorn.Config(
                app,
                host="127.0.0.1",
                port=port,
                loop="asyncio",
                log_level="critical",
                access_log=False,
                lifespan="off",
            )
        )
        thread = threading.Thread(
            target=server.run, kwargs={"sockets": [sock]}, daemon=True
        )
        thread.start()
        for _ in range(200):
            if server.started:
                break
            time.sleep(0.005)
        self.assertTrue(server.started)

        def stop():
            """Shut down the loopback server and join its serving thread."""
            server.should_exit = True
            thread.join(3)
            sock.close()

        self.addCleanup(stop)
        return port

    def test_three_requests_reach_engine_and_answers_return(self):
        """Verify three requests reach engine and answers return."""

        class Engine:
            """Exercise Engine behavior with isolated fixtures."""

            active = peak = 0

            async def generate(self, *args):
                """Yield the configured generation result for a correlated engine request."""
                self.active += 1
                self.peak = max(self.peak, self.active)
                try:
                    await asyncio.sleep(0.06)
                    return Segment("<calc-final>2+3</calc-final>", (), 1, "stop")
                finally:
                    self.active -= 1

        engine = Engine()
        port = self.start(engine)
        barrier = threading.Barrier(3)

        def request(_):
            """Invoke one local workload while recording concurrency or response evidence."""
            barrier.wait(1)
            return request_bytes(
                f"http://127.0.0.1:{port}/v1/exact/completions",
                b'{"exact_compute":true,"prompt":"p","mode":"numeric-final"}',
                headers={"Content-Type": "application/json"},
                timeout=2,
            )

        with ThreadPoolExecutor(max_workers=3) as pool:
            results = list(pool.map(request, range(3)))
        self.assertEqual(engine.peak, 3)
        self.assertTrue(
            all(
                status == 200 and json.loads(body)["text"] == "5"
                for status, body in results
            )
        )
        self.assertEqual(
            len({json.loads(body)["request_id"] for _, body in results}), 3
        )

    def test_disconnect_cancels_inference(self):
        """Verify disconnect cancels inference."""
        started, aborted = threading.Event(), threading.Event()

        class Engine:
            """Exercise Engine behavior with isolated fixtures."""

            async def generate(self, *args):
                """Yield the configured generation result for a correlated engine request."""
                started.set()
                try:
                    await asyncio.Event().wait()
                except asyncio.CancelledError:
                    aborted.set()
                    raise

        port = self.start(Engine())
        body = b'{"exact_compute":true,"prompt":"p"}'
        with socket.create_connection(("127.0.0.1", port)) as client:
            client.sendall(
                b"POST /v1/exact/completions HTTP/1.1\r\nHost: local\r\nContent-Type: application/json\r\nContent-Length: "
                + str(len(body)).encode()
                + b"\r\n\r\n"
                + body
            )
            self.assertTrue(started.wait(1))
        self.assertTrue(aborted.wait(1))
