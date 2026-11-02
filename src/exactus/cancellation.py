"""Thread-safe transport cancellation; CPU ownership is released only on completion."""

__author__ = "Olshansky Grigory"

import threading
from contextvars import ContextVar


class Cancellation:
    """Coordinate cancellation callbacks without reopening cancelled I/O."""

    __slots__ = ("_lock", "_cancelled", "_callbacks")

    def __init__(self):
        """Initialize Cancellation state and dependencies."""
        self._lock, self._cancelled, self._callbacks = threading.Lock(), False, set()

    def register(self, callback):
        """Register an interruption callback or invoke it if already cancelled."""
        with self._lock:
            cancelled = self._cancelled
            if not cancelled:
                self._callbacks.add(callback)
        if cancelled:
            callback()
            raise OSError("transport cancelled")

    def discard(self, callback):
        """Remove a completed operation interruption callback."""
        with self._lock:
            self._callbacks.discard(callback)

    def check(self):
        """Raise when the shared operation has been cancelled."""
        with self._lock:
            if self._cancelled:
                raise OSError("transport cancelled")

    def cancel(self):
        """Mark the operation cancelled and interrupt registered I/O."""
        with self._lock:
            self._cancelled = True
            callbacks = tuple(self._callbacks)
            self._callbacks.clear()
        for callback in callbacks:
            callback()


current = ContextVar("exactus_cancellation", default=None)
