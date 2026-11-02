"""Own the isolated subprocess lifecycle and correlated framed transport."""

from __future__ import annotations

__author__ = "Olshansky Grigory"

import json
import subprocess
import sys
import threading
import time
from pathlib import Path

from ..domain.errors import BudgetError, WorkerError
from .worker_frame import (
    MAX_FRAME,
    FrameClosed,
    FrameStream,
    FrameTimeout,
    FrameTooLarge,
)


class WorkerProcess:
    """Maintain one worker process and validate every correlated exchange."""

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
            [
                sys.executable,
                "-I",
                str(Path(__file__).with_name("worker.py").resolve()),
                "--serve",
            ],
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
