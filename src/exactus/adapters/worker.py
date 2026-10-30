"""Resource-limited subprocess external computer with per-call timeouts.

Author: Olshansky Grigory
"""

from __future__ import annotations

import json
import subprocess
import sys
import threading
import time
from collections.abc import Mapping
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from exactus.adapters.worker_frame import (
    FrameClosed,
    FrameStream,
    FrameTimeout,
    FrameTooLarge,
    MAX_FRAME,
)
from exactus.adapters.worker_session import WorkerSession, encode_data
from exactus.domain.errors import (
    BudgetError,
    DataError,
    GrammarError,
    WorkerError,
    FormatError,
    VerificationError,
)
from exactus.domain.value import ComputeValue

__author__ = "Olshansky Grigory"


class WorkerExternalComputer:
    """Evaluates expressions in one long-lived isolated subprocess.

    Replaces per-call interpreter spawn with a persistent worker while
    keeping isolation: minimal environment, address-space rlimit, per-call
    wall timeout and kill plus respawn on any anomaly. The whole roundtrip,
    decode and recycle run under one exclusive lease, every frame carries
    a correlation id that must be echoed, frames are size-capped and the
    deadline covers writing, flushing and reading up to the frame LF. The
    worker recycles after max_calls evaluations.
    """

    def __init__(self, *, timeout: float = 2.0, max_calls: int = 100) -> None:
        """Store worker lifecycle policy and start with no worker."""
        if timeout <= 0 or type(max_calls) is not int or max_calls <= 0:
            raise ValueError("positive timeout and max_calls required")
        self._timeout = timeout
        self._max_calls = max_calls
        self._lock = threading.RLock()
        self._proc: subprocess.Popen | None = None
        self._in: FrameStream | None = None
        self._out: FrameStream | None = None
        self._calls = 0
        self._next_id = 0
        self._session_owner = None

    def evaluate(
        self,
        expression: str,
        data: Mapping[str, bytes] | None = None,
        *,
        output_format: str = "dec",
    ) -> ComputeValue:
        """Evaluate one expression inside an exclusive worker lease."""
        if not isinstance(expression, str) or len(expression) > 4096:
            raise BudgetError("expression exceeds the worker input budget")
        with self._lock:
            reply = self._exchange(
                {
                    "op": "eval",
                    "expression": expression,
                    "format": output_format,
                    "data": encode_data(data or {}),
                }
            )
            try:
                return self._decode(reply)
            finally:
                self._calls += 1
                if self._calls >= self._max_calls:
                    self._close_locked()

    def session(self, data: Mapping[str, bytes] | None = None) -> WorkerSession:
        """Return an exclusive lease that ships request data to the worker once."""
        return WorkerSession(self, data)

    def request(self, data=None):
        """Cache data once per request, with thread-neutral per-call ownership."""
        from exactus.adapters.worker_request import WorkerRequest

        return WorkerRequest(self, data)

    def evaluate_many(self, expressions, data=None, *, output_format="dec"):
        if not expressions:
            return []
        with self.request(data) as request:
            return request.evaluate_many(expressions, output_format=output_format)

    def run_external_computers(self, tasks):
        """Execute a bounded dependent batch in the resource-limited child."""
        from exactus.external_computers.values import packed

        # Cap and type-check before IPC/child creation as well as in the child.
        if type(tasks) is not list or not 1 <= len(tasks) <= 16:
            raise DataError("external computer batch must contain 1..16 tasks")
        packed(tasks)
        return self._run_external_computer_request(
            {"op": "external_computers", "tasks": tasks}
        )

    def run_plan(self, plan, *, inputs=None, checks=None):
        """Execute/reify a source-bound compute plan under one IPC lease."""
        from exactus.external_computers.values import packed

        request = {"plan": plan, "inputs": inputs, "checks": checks}
        packed(request)
        return self._run_external_computer_request({"op": "plan", **request})

    def _run_external_computer_request(self, request):
        with self._lock:
            try:
                reply = self._exchange(request)
                if not reply.get("ok"):
                    self._decode(reply)  # common typed domain-error mapping
                result = reply.get("result")
                if not isinstance(result, dict) or not isinstance(
                    result.get("results"), list
                ):
                    self._close_locked()
                    raise WorkerError("worker external computer reply is invalid")
                return result
            finally:
                self._calls += 1
                if self._calls >= self._max_calls:
                    self._close_locked()

    def close(self) -> None:
        """Terminate the worker process if it is running."""
        with self._lock:
            self._close_locked()

    def _exchange(self, request: dict) -> dict:
        """Send one correlated request frame and return the validated reply."""
        self._next_id += 1
        request_id = self._next_id
        frame = json.dumps({**request, "id": request_id}, separators=(",", ":")).encode(
            "utf-8"
        )
        if len(frame) > MAX_FRAME:
            raise BudgetError("request frame exceeds the frame limit")
        self._ensure_worker()
        deadline = time.monotonic() + self._timeout
        try:
            self._out.write_frame(frame, deadline)
            raw = self._in.read_frame(deadline)
        except FrameTimeout as exc:
            self._close_locked()
            raise WorkerError("worker timed out") from exc
        except FrameClosed as exc:
            self._close_locked()
            raise WorkerError("worker exited unexpectedly") from exc
        except FrameTooLarge as exc:
            self._close_locked()
            raise WorkerError("worker reply exceeds the frame limit") from exc
        try:
            reply = json.loads(raw)
        except (ValueError, TypeError) as exc:
            self._close_locked()
            raise WorkerError("worker returned invalid data") from exc
        if not isinstance(reply, dict):
            self._close_locked()
            raise WorkerError("worker returned an invalid reply")
        if reply.get("id") != request_id:
            self._close_locked()
            raise WorkerError("worker returned a mismatched reply")
        return reply

    def _decode(self, reply: dict) -> ComputeValue:
        """Convert a validated worker reply into a ComputeValue or raise."""
        if not reply.get("ok"):
            error = reply.get("error", "arithmetic failed")
            for cls in (
                BudgetError,
                DataError,
                GrammarError,
                FormatError,
                VerificationError,
            ):
                if error.startswith(cls.__name__):
                    raise cls(error.split(": ", 1)[-1])
            raise GrammarError(str(error))
        try:
            return ComputeValue(int(reply["value"]), str(reply["format"]))
        except (KeyError, TypeError, ValueError, FormatError) as exc:
            self._close_locked()
            raise WorkerError("worker reply is missing value fields") from exc

    def _ensure_worker(self) -> None:
        """Start the worker subprocess if none is running."""
        if self._proc is not None and self._proc.poll() is None:
            return
        self._close_locked()
        proc = self._spawn()
        self._proc = proc
        self._in = FrameStream(proc.stdout.fileno())
        self._out = FrameStream(proc.stdin.fileno())
        self._calls = 0

    def _spawn(self) -> subprocess.Popen:
        """Launch one fresh isolated worker subprocess."""
        return subprocess.Popen(
            [sys.executable, "-I", str(Path(__file__).resolve()), "--serve"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            env={"PATH": "/usr/bin:/bin", "LANG": "C.UTF-8"},
            bufsize=0,
        )

    def _close_locked(self) -> None:
        """Kill the worker and close its pipes; caller holds the lease."""
        proc, self._proc = self._proc, None
        self._in = None
        self._out = None
        self._calls = 0
        if proc is None:
            return
        try:
            if proc.poll() is None:
                proc.kill()
                proc.wait()
        finally:
            for pipe in (proc.stdin, proc.stdout):
                if pipe is not None:
                    try:
                        pipe.close()
                    except OSError:
                        pass


if __name__ == "__main__":
    if "--serve" in sys.argv:
        from exactus.adapters.worker_serve import serve

        serve()
