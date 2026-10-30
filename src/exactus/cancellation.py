"""Thread-safe transport cancellation; CPU ownership is released only on completion."""

from contextvars import ContextVar
import threading


class Cancellation:
    __slots__ = ("_lock", "_cancelled", "_callbacks")

    def __init__(self):
        self._lock, self._cancelled, self._callbacks = threading.Lock(), False, set()

    def register(self, callback):
        with self._lock:
            cancelled = self._cancelled
            if not cancelled:
                self._callbacks.add(callback)
        if cancelled:
            callback()
            raise OSError("transport cancelled")

    def discard(self, callback):
        with self._lock:
            self._callbacks.discard(callback)

    def check(self):
        with self._lock:
            if self._cancelled:
                raise OSError("transport cancelled")

    def cancel(self):
        with self._lock:
            self._cancelled = True
            callbacks = tuple(self._callbacks)
            self._callbacks.clear()
        for callback in callbacks:
            callback()


current = ContextVar("exactus_cancellation", default=None)
