"""Thread-neutral request data handles; no lock is held while the LLM runs."""

from uuid import uuid4

from ..domain.errors import ExactusError, WorkerError
from .worker_session import encode_data


class WorkerRequest:
    __slots__ = ("_external_computer", "_data", "_key", "_process", "_closed")

    def __init__(self, external_computer, data):
        self._external_computer, self._data = external_computer, encode_data(data or {})
        self._key, self._process, self._closed = uuid4().hex, None, False

    def evaluate_many(self, expressions, data=None, *, output_format="dec"):
        if isinstance(expressions, (str, bytes)):
            raise ExactusError("expression batch must be a sequence of expressions")
        if data is not None:
            raise ExactusError("request data is immutable; create another handle")
        expressions = tuple(expressions)
        if len(expressions) > 128 or any(
            not isinstance(s, str) or len(s) > 4096 for s in expressions
        ):
            raise ExactusError("invalid expression batch")
        external_computer = self._external_computer
        with external_computer._lock:
            if self._closed:
                raise ExactusError("worker request is closed")
            external_computer._ensure_worker()
            if self._process is not external_computer._proc:
                reply = external_computer._exchange(
                    {"op": "open", "request_key": self._key, "data": self._data}
                )
                if not reply.get("ok"):
                    raise WorkerError("worker rejected request data")
                self._process = external_computer._proc
            reply = external_computer._exchange(
                {
                    "op": "batch",
                    "request_key": self._key,
                    "expressions": expressions,
                    "format": output_format,
                }
            )
            try:
                if not reply.get("ok"):
                    external_computer._decode(
                        reply
                    )  # preserve typed arithmetic failures
                rows = reply.get("values")
                if (
                    not isinstance(rows, list)
                    or len(rows) != len(expressions)
                    or any(not isinstance(row, dict) for row in rows)
                ):
                    external_computer._close_locked()
                    raise WorkerError("worker returned an invalid batch")
                return [external_computer._decode({"ok": True, **row}) for row in rows]
            finally:
                # Failed batches also consume the process-lifetime budget.
                external_computer._calls += len(expressions)
                if external_computer._calls >= external_computer._max_calls:
                    external_computer._close_locked()

    def evaluate(self, expression, data=None, *, output_format="dec"):
        return self.evaluate_many((expression,), data, output_format=output_format)[0]

    def close(self):
        external_computer = self._external_computer
        with external_computer._lock:
            if self._closed:
                return
            self._closed = True
            try:
                # Never start a dead/recycled child just to delete stale state.
                if (
                    external_computer._proc is self._process
                    and self._process is not None
                    and self._process.poll() is None
                ):
                    reply = external_computer._exchange(
                        {"op": "release", "request_key": self._key}
                    )
                    if not reply.get("ok"):
                        external_computer._close_locked()
            except ExactusError:
                external_computer._close_locked()
            finally:
                self._data.clear()

    def __enter__(self):
        if self._closed:
            raise ExactusError("worker request is closed")
        return self

    def __exit__(self, *exc):
        self.close()
