"""Admission leases survive response timeout until underlying work finishes."""

__author__ = "Olshansky Grigory"

import contextvars
import threading
import time
from concurrent.futures import TimeoutError as FutureTimeout

from .cancellation import current as current_cancellation
from .http_io import deadline


class WorkLease:
    """Retain admission until both transport and in-flight work have released ownership."""

    def __init__(self, release):
        """Initialize Work Lease state and dependencies."""
        self._release = release
        self._lock = threading.Lock()
        self._owners = 1

    def hold(self):
        """Add an owner for work that may outlive the request transport."""
        with self._lock:
            self._owners += 1

    def release(self):
        """Release one owner and return capacity exactly when ownership reaches zero."""
        with self._lock:
            self._owners -= 1
            if self._owners == 0:
                self._release()


def invoke_with_deadline(
    fn, timeout, on_complete=lambda: None, *, executor=None, cancellation=None
):
    """Wait for work without releasing its lease while a background thread still runs."""
    box = {}
    context = contextvars.copy_context()
    expires = time.monotonic() + timeout

    def target():
        """Run the operation with copied context and release ownership on completion."""
        try:
            deadline.set(expires)
            if cancellation is not None:
                current_cancellation.set(cancellation)
            box["result"] = fn()
        except Exception as exc:
            box["error"] = exc
        finally:
            on_complete()

    if executor is not None:
        try:
            future = executor.submit(context.run, target)
        except BaseException:
            on_complete()
            raise
        try:
            future.result(timeout=max(0, expires - time.monotonic()))
        except FutureTimeout:
            return None
        if "error" in box:
            raise box["error"]
        return box.get("result")
    worker = threading.Thread(target=lambda: context.run(target), daemon=True)
    try:
        worker.start()
    except BaseException:
        on_complete()
        raise
    worker.join(max(0, timeout))
    if worker.is_alive():
        return None
    if "error" in box:
        raise box["error"]
    return box.get("result")
