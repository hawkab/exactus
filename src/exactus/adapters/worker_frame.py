"""Deadline-bounded framed byte transport for worker subprocess IPC.

Author: Olshansky Grigory
"""

from __future__ import annotations

import os
import select
import time

__author__ = "Olshansky Grigory"

MAX_FRAME = 1024 * 1024
_CHUNK = 4096


class FrameError(Exception):
    """Base failure of one frame transfer."""


class FrameTimeout(FrameError):
    """Frame transfer did not finish within its deadline."""


class FrameClosed(FrameError):
    """Peer closed the stream before one full frame arrived."""


class FrameTooLarge(FrameError):
    """Frame content exceeds the size cap."""


def _wait(fd: int, write: bool, deadline: float | None) -> None:
    """Block until fd is ready for one direction or the deadline expires."""
    timeout = None if deadline is None else max(0.0, deadline - time.monotonic())
    if write:
        ready = select.select([], [fd], [], timeout)[1]
    else:
        ready = select.select([fd], [], [], timeout)[0]
    if not ready:
        raise FrameTimeout("frame transfer timed out")


class FrameStream:
    """One endpoint of a newline-framed, size-capped byte stream on a raw fd.

    Reads tolerate arbitrary chunking through a pushback buffer and run
    to the frame LF; writes run to completion. Both directions enforce
    the frame size cap and an explicit deadline, and at most one frame
    per direction is in flight, so no framing ambiguity is possible.
    """

    def __init__(self, fd: int) -> None:
        """Take the fd and switch it to non-blocking mode."""
        self._fd = fd
        self._buf = bytearray()
        os.set_blocking(fd, False)

    def read_frame(self, deadline: float | None) -> bytes:
        """Read one LF-terminated frame and return it without the LF."""
        scan_from = 0
        while True:
            newline = self._buf.find(b"\n", scan_from)
            if newline >= 0:
                if newline > MAX_FRAME:
                    raise FrameTooLarge("frame exceeds the size limit")
                frame = bytes(self._buf[:newline])
                del self._buf[: newline + 1]
                return frame
            if len(self._buf) > MAX_FRAME:
                raise FrameTooLarge("frame exceeds the size limit")
            scan_from = len(self._buf)
            self._fill(deadline)

    def write_frame(self, frame: bytes, deadline: float | None) -> None:
        """Write one LF-terminated frame before the deadline."""
        if len(frame) > MAX_FRAME:
            raise FrameTooLarge("frame exceeds the size limit")
        view = memoryview(frame + b"\n")
        while view:
            _wait(self._fd, True, deadline)
            try:
                written = os.write(self._fd, view)
            except BlockingIOError:
                continue
            except OSError as exc:
                raise FrameClosed("frame write failed") from exc
            if written <= 0:
                raise FrameClosed("frame stream stopped accepting bytes")
            view = view[written:]

    def _fill(self, deadline: float | None) -> None:
        """Append one read chunk to the pushback buffer."""
        _wait(self._fd, False, deadline)
        try:
            chunk = os.read(self._fd, _CHUNK)
        except BlockingIOError:
            return
        except OSError as exc:
            raise FrameClosed("frame read failed") from exc
        if not chunk:
            raise FrameClosed("frame stream closed")
        self._buf.extend(chunk)
