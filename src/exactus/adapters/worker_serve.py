"""Child-side request loop for the sandboxed worker subprocess.

Author: Olshansky Grigory
"""

from __future__ import annotations

import base64
import json
from collections.abc import Mapping

from exactus.adapters.worker_frame import FrameClosed, FrameStream, FrameTooLarge
from exactus.domain.errors import ExactusError, BudgetError, DataError

__author__ = "Olshansky Grigory"

_MEMORY_LIMIT = 128 * 1024 * 1024


def _decode_data(raw: Mapping[str, str]) -> dict[str, bytes]:
    """Decode the base64 data mapping of one request frame."""
    return {name: base64.b64decode(payload) for name, payload in raw.items()}


class RequestLoop:
    """Dispatches frames one at a time over the framed stdio transport.

    Session data enters only through an init frame, serves eval frames
    that omit data, and is dropped on a reset frame, so no request
    payload can outlive its lease.
    """

    def __init__(self, external_computer) -> None:
        """Bind the evaluator and start with no leased session data."""
        self._external_computer = external_computer
        self._session: dict[str, bytes] | None = None
        self._requests = {}

    def run(self, source: FrameStream, sink: FrameStream) -> None:
        """Serve request frames until the parent closes the stream."""
        while True:
            try:
                raw = source.read_frame(None)
            except (FrameClosed, FrameTooLarge):
                return
            reply = self.dispatch(raw)
            try:
                sink.write_frame(
                    json.dumps(reply, separators=(",", ":")).encode("utf-8"), None
                )
            except (FrameClosed, FrameTooLarge):
                return

    def dispatch(self, raw: bytes) -> dict:
        """Handle one request frame and build its correlated reply."""
        try:
            request = json.loads(raw)
            if not isinstance(request, dict) or "id" not in request:
                raise ValueError("request must carry an id")
            reply_id = request["id"]
        except Exception:
            return {"id": None, "ok": False, "error": "GrammarError: invalid request"}
        try:
            return {"id": reply_id, "ok": True, **self._apply(request)}
        except ExactusError as exc:
            return {
                "id": reply_id,
                "ok": False,
                "error": f"{type(exc).__name__}: {exc}",
            }
        except Exception:
            return {
                "id": reply_id,
                "ok": False,
                "error": "GrammarError: invalid request",
            }

    def _apply(self, request: dict) -> dict:
        """Execute one request operation and return its result fields."""
        op = request.get("op", "eval")
        if op == "plan":
            from exactus.external_computers.plan import run_plan

            return {
                "result": run_plan(
                    request.get("plan"),
                    self._external_computer,
                    inputs=request.get("inputs"),
                    checks=request.get("checks"),
                )
            }
        if op == "external_computers":
            from exactus.external_computers.batch import run_external_computers

            return {
                "result": run_external_computers(
                    request.get("tasks"), self._external_computer
                )
            }
        if op in ("open", "release"):
            key = request.get("request_key")
            if not isinstance(key, str) or not 1 <= len(key) <= 64:
                raise DataError("invalid request key")
            if op == "release":
                self._requests.pop(key, None)
            else:
                if key not in self._requests and len(self._requests) >= 32:
                    raise BudgetError("worker request-data capacity exceeded")
                self._requests[key] = _decode_data(request.get("data", {}))
            return {"op": op}
        if op == "init":
            self._session = _decode_data(request.get("data", {}))
            return {"op": "init"}
        if op == "reset":
            self._session = None
            return {"op": "reset"}
        if "request_key" in request:
            key = request["request_key"]
            if key not in self._requests:
                raise DataError("request data handle is missing")
            data = self._requests[key]
        elif "data" in request:
            data = _decode_data(request["data"])
        else:
            data = self._session if self._session is not None else {}
        if op == "batch":
            expressions = request.get("expressions")
            if not isinstance(expressions, list) or len(expressions) > 128:
                raise BudgetError("invalid expression batch")
            return {
                "values": [self._value(expr, data, request) for expr in expressions]
            }
        if op != "eval":
            raise DataError("unknown worker operation")
        return self._value(request["expression"], data, request)

    def _value(self, expression, data, request):
        value = self._external_computer.evaluate(
            expression, data, output_format=request.get("format", "dec")
        )
        return {"value": str(value.integer), "format": value.format}


def serve() -> None:
    """Run the worker request loop over stdio under the memory limit."""
    import resource

    resource.setrlimit(resource.RLIMIT_AS, (_MEMORY_LIMIT, _MEMORY_LIMIT))
    from exactus.adapters.digest import HashlibHasher
    from exactus.adapters.in_process import InProcessExternalComputer

    RequestLoop(InProcessExternalComputer(HashlibHasher())).run(
        FrameStream(0), FrameStream(1)
    )
