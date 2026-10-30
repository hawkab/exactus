"""Shared test support: scripted worker doubles and pipe transports.

Author: Olshansky Grigory
"""

from __future__ import annotations

import json
import os
import queue
import subprocess
import sys
import threading

from exactus.adapters.worker import WorkerExternalComputer
from exactus.adapters.worker_frame import FrameClosed, FrameStream

__author__ = "Olshansky Grigory"

PARTIAL_REPLY_SCRIPT = (
    "import sys,time\nsys.stdout.write('x')\nsys.stdout.flush()\ntime.sleep(30)\n"
)
IDLE_CHILD_SCRIPT = "import time\ntime.sleep(30)\n"


class Pipe:
    """Minimal Popen-pipe double owning one raw file descriptor."""

    def __init__(self, fd: int) -> None:
        """Own the given file descriptor."""
        self._fd = fd

    def fileno(self) -> int:
        """Return the owned file descriptor."""
        return self._fd

    def close(self) -> None:
        """Close the owned file descriptor once."""
        fd, self._fd = self._fd, -1
        if fd >= 0:
            os.close(fd)


class ScriptedChild:
    """Worker subprocess double with test-controlled replies.

    The parent side sees Popen-like stdin and stdout pipes; a responder
    thread reads request frames into an inbox and answers through the
    handler, which may return None so the test drives the reply itself.
    """

    def __init__(self, handler) -> None:
        """Create the pipes, the inbox and the responder thread."""
        req_r, req_w = os.pipe()
        rep_r, rep_w = os.pipe()
        self.stdin = Pipe(req_w)
        self.stdout = Pipe(rep_r)
        self.inbox: queue.Queue = queue.Queue()
        self.alive = True
        self._handler = handler
        self._req_r = req_r
        self._rep_w = rep_w
        threading.Thread(target=self._respond, daemon=True).start()

    def poll(self):
        """Return None while alive and a kill code afterwards."""
        return None if self.alive else -9

    def kill(self) -> None:
        """Mark the scripted child as dead."""
        self.alive = False

    def wait(self):
        """Return the kill code of the scripted child."""
        return -9

    def send(self, reply: dict) -> None:
        """Write one reply frame to the parent side."""
        os.write(self._rep_w, json.dumps(reply).encode("utf-8") + b"\n")

    def _respond(self) -> None:
        """Read request frames and answer them through the handler."""
        stream = FrameStream(self._req_r)
        try:
            while True:
                request = json.loads(stream.read_frame(None))
                self.inbox.put(request)
                reply = self._handler(self, request)
                if reply is not None:
                    self.send(reply)
        except (FrameClosed, OSError, ValueError):
            pass
        finally:
            for fd in (self._req_r, self._rep_w):
                try:
                    os.close(fd)
                except OSError:
                    pass


class ScriptedExternalComputer(WorkerExternalComputer):
    """WorkerExternalComputer whose spawned subprocesses are scripted doubles."""

    def __init__(self, factory, *, timeout: float = 2.0) -> None:
        """Store the child factory and inherit the lifecycle policy."""
        super().__init__(timeout=timeout, max_calls=1000)
        self.spawned: list = []
        self._factory = factory

    def _spawn(self):
        """Create and record one scripted child."""
        child = self._factory()
        self.spawned.append(child)
        return child


def echo_reply(child: ScriptedChild, request: dict) -> dict:
    """Acknowledge init and reset frames and echo eval expressions as values."""
    reply = {"id": request["id"], "ok": True}
    if request.get("op") in ("init", "reset"):
        reply["op"] = request["op"]
    else:
        reply["value"] = request["expression"]
        reply["format"] = request.get("format", "dec")
    return reply


def stub_external_computer(script: str, timeout: float = 0.25):
    """Return a worker external computer and its real child running the given script."""
    child = subprocess.Popen(
        [sys.executable, "-c", script],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        bufsize=0,
    )
    return ScriptedExternalComputer(lambda: child, timeout=timeout), child
