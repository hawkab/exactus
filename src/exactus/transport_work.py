"""Admission leases survive response timeout until underlying work finishes."""

import threading
import contextvars
import time
from concurrent.futures import TimeoutError as FutureTimeout
from .http_io import deadline
from .cancellation import current as current_cancellation


class WorkLease:
    def __init__(self, release):
        self._release = release
        self._lock = threading.Lock()
        self._owners = 1

    def hold(self):
        with self._lock:
            self._owners += 1

    def release(self):
        with self._lock:
            self._owners -= 1
            if self._owners == 0:
                self._release()


def invoke_with_deadline(
    fn, timeout, on_complete=lambda: None, *, executor=None, cancellation=None
):
    box = {}
    context = contextvars.copy_context()
    expires = time.monotonic() + timeout

    def target():
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
