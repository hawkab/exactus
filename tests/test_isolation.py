"""Tests for request-data isolation and worker request ownership.

Author: Olshansky Grigory
"""

import hashlib
import sys
import threading
import time
import unittest

from exactus.adapters.digest import HashlibHasher
from exactus.adapters.in_process import InProcessExternalComputer
from exactus.adapters.worker import WorkerExternalComputer
from exactus.domain.errors import DataError, ExactusError
from exactus.domain.expression import EvalContext
from tests._isolation_support import IDLE_CHILD_SCRIPT, PARTIAL_REPLY_SCRIPT
from tests._isolation_support import (
    ScriptedChild,
    ScriptedExternalComputer,
    echo_reply,
    stub_external_computer,
)

__author__ = "Olshansky Grigory"


class TestEvaluatorIsolation(unittest.TestCase):
    """Verifies request-scoped data isolation under concurrent evaluation."""

    def test_concurrent_requests_never_share_data(self):
        """Concurrent threads never observe each other's request data."""
        external_computer = InProcessExternalComputer(HashlibHasher())
        expression = "+".join(["@x"] * 100)
        results: dict[int, list[tuple]] = {name: [] for name in range(4)}

        def run(name: int) -> None:
            """Evaluate 200 private-data expressions and hashes for one thread."""
            payload = (name + 1).to_bytes(8, "big")
            for _ in range(200):
                total = external_computer.evaluate(expression, {"x": payload}).integer
                digest = external_computer.evaluate(
                    "sha256(@m)", {"m": payload}
                ).integer
                results[name].append((total, digest))

        interval = sys.getswitchinterval()
        sys.setswitchinterval(1e-6)
        self.addCleanup(sys.setswitchinterval, interval)
        threads = [threading.Thread(target=run, args=(name,)) for name in range(4)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=60)
        for name in range(4):
            want = int.from_bytes(
                hashlib.sha256((name + 1).to_bytes(8, "big")).digest(), "big"
            )
            self.assertEqual(results[name], [(100 * (name + 1), want)] * 200)
        source = {"x": (7).to_bytes(2, "big")}
        context = EvalContext(source, None)
        source["x"] = b"\x00\x00"
        self.assertEqual(context.data_value("x"), 7)
        self.assertRaises(DataError, context.data_value, "y")


class TestWorkerOwnership(unittest.TestCase):
    """Verifies reply ownership under controlled thread interleaving."""

    def test_controlled_schedule_keeps_replies_owned(self):
        """A second thread cannot cross a first thread's in-flight reply."""
        release = threading.Event()

        def blocking(child, request):
            """Hold the 111 reply until the test releases it."""
            if request["expression"] == "111":
                release.wait(timeout=10)
            return {
                "id": request["id"],
                "ok": True,
                "value": request["expression"],
                "format": "dec",
            }

        child = ScriptedChild(blocking)
        external_computer = ScriptedExternalComputer(lambda: child)
        results, errors = {}, []

        def call(name: str, expression: str) -> None:
            """Evaluate one expression and record the outcome."""
            try:
                results[name] = external_computer.evaluate(expression).integer
            except Exception as exc:
                errors.append(exc)

        first = threading.Thread(target=call, args=("a", "111"))
        first.start()
        self.assertEqual(child.inbox.get(timeout=10)["expression"], "111")
        second = threading.Thread(target=call, args=("b", "222"))
        second.start()
        time.sleep(0.3)
        self.assertTrue(child.inbox.empty())
        release.set()
        first.join(timeout=10)
        second.join(timeout=10)
        self.assertEqual(child.inbox.get(timeout=10)["expression"], "222")
        self.assertEqual(results, {"a": 111, "b": 222})
        self.assertEqual(errors, [])
        external_computer.close()

    def test_mismatched_correlation_id_recycles_worker(self):
        """A reply with a foreign correlation id is rejected and recycled."""
        handlers = [
            lambda c, r: {
                "id": r["id"] + 1000,
                "ok": True,
                "value": r["expression"],
                "format": "dec",
            },
            echo_reply,
        ]
        external_computer = ScriptedExternalComputer(
            lambda: ScriptedChild(handlers.pop(0))
        )
        with self.assertRaises(ExactusError):
            external_computer.evaluate("7")
        self.assertFalse(external_computer.spawned[0].alive)
        self.assertEqual(external_computer.evaluate("7").integer, 7)
        self.assertEqual(len(external_computer.spawned), 2)
        external_computer.close()

    def test_close_serializes_with_inflight_roundtrip(self):
        """Close waits for an in-flight roundtrip before recycling the worker."""
        release = threading.Event()

        def blocking(child, request):
            """Hold every reply until the test releases it."""
            release.wait(timeout=10)
            return {"id": request["id"], "ok": True, "value": "5", "format": "dec"}

        child = ScriptedChild(blocking)
        external_computer = ScriptedExternalComputer(lambda: child)
        results: dict[str, int] = {}
        worker = threading.Thread(
            target=lambda: results.update(v=external_computer.evaluate("5").integer)
        )
        worker.start()
        child.inbox.get(timeout=10)
        closing = threading.Thread(target=external_computer.close)
        closing.start()
        time.sleep(0.3)
        self.assertTrue(closing.is_alive())
        release.set()
        worker.join(timeout=10)
        closing.join(timeout=10)
        self.assertEqual(results["v"], 5)
        self.assertFalse(child.alive)


class TestWorkerTimeouts(unittest.TestCase):
    """Verifies deadline enforcement on partial or stalled replies."""

    def test_partial_reply_times_out_before_child_exit(self):
        """One byte without LF ends by timeout, not at child exit."""
        external_computer, child = stub_external_computer(PARTIAL_REPLY_SCRIPT)
        self.addCleanup(child.wait)
        self.addCleanup(external_computer.close)
        started = time.monotonic()
        with self.assertRaises(ExactusError) as caught:
            external_computer.evaluate("1+1")
        self.assertEqual(str(caught.exception), "worker timed out")
        self.assertLess(time.monotonic() - started, 5.0)

    def test_write_deadline_covers_a_stalled_child(self):
        """A child that never reads cannot stall a write past its deadline."""
        external_computer, child = stub_external_computer(IDLE_CHILD_SCRIPT)
        self.addCleanup(child.wait)
        self.addCleanup(external_computer.close)
        started = time.monotonic()
        with self.assertRaises(ExactusError) as caught:
            external_computer.evaluate("sha256(@payload)", {"payload": b"x" * 200000})
        self.assertEqual(str(caught.exception), "worker timed out")
        self.assertLess(time.monotonic() - started, 5.0)


class TestWorkerLifecycle(unittest.TestCase):
    """Verifies respawn behaviour and strict session cleanup."""

    def test_respawn_after_worker_death(self):
        """A dead worker is replaced transparently on the next call."""
        external_computer = WorkerExternalComputer(timeout=2.0)
        self.addCleanup(external_computer.close)
        self.assertEqual(external_computer.evaluate("2+2").integer, 4)
        first = external_computer._proc
        first.kill()
        first.wait()
        self.assertEqual(external_computer.evaluate("3+3").integer, 6)
        self.assertIsNot(first, external_computer._proc)

    def test_session_ships_data_once_and_resets(self):
        """Session data travels in one init frame and is reset at release."""
        child = ScriptedChild(echo_reply)
        external_computer = ScriptedExternalComputer(lambda: child)
        with external_computer.session({"x": b"\x07"}) as session:
            self.assertEqual(session.evaluate("7").integer, 7)
        frames = [child.inbox.get(timeout=10) for _ in range(3)]
        self.assertEqual(
            [frame.get("op") for frame in frames], ["init", "eval", "reset"]
        )
        self.assertIn("data", frames[0])
        self.assertNotIn("data", frames[1])
        external_computer.close()

    def test_session_data_is_cleared_after_lease(self):
        """Leased request data is invisible to later one-shot calls."""
        external_computer = WorkerExternalComputer(timeout=2.0)
        self.addCleanup(external_computer.close)
        with external_computer.session({"x": (5).to_bytes(1, "big")}) as session:
            self.assertEqual(session.evaluate("@x").integer, 5)
        with self.assertRaises(DataError):
            external_computer.evaluate("@x")


if __name__ == "__main__":
    unittest.main()
