"""Resource-limited subprocess external computer with per-call timeouts."""

from __future__ import annotations

import sys
from collections.abc import Mapping
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from exactus.adapters.worker_session import WorkerSession, encode_data
from exactus.domain.errors import (
    BudgetError,
    DataError,
    FormatError,
    GrammarError,
    VerificationError,
    WorkerError,
)
from exactus.domain.value import ComputeValue

__author__ = "Olshansky Grigory"


from exactus.adapters.worker_process import WorkerProcess


class WorkerExternalComputer(WorkerProcess):
    """Evaluates expressions in one long-lived isolated subprocess.

    Replaces per-call interpreter spawn with a persistent worker while
    keeping isolation: minimal environment, address-space rlimit, per-call
    wall timeout and kill plus respawn on any anomaly. The whole roundtrip,
    decode and recycle run under one exclusive lease, every frame carries
    a correlation id that must be echoed, frames are size-capped and the
    deadline covers writing, flushing and reading up to the frame LF. The
    worker recycles after max_calls evaluations.
    """

    def evaluate(
        self,
        expression: str,
        data: Mapping[str, bytes] | None = None,
        *,
        output_format: str = "dec",
    ) -> ComputeValue:
        """Evaluate one expression inside an exclusive worker lease.

        Args:
            expression: Exact integer expression sent to the isolated child.
            data: Request-scoped byte payloads, encoded once for this call.
            output_format: Rendering format returned with the value.

        Returns:
            The exact integer value and its requested rendering format.

        Raises:
            ExactusError: Evaluation fails its expression or data contract.
            WorkerError: The child process or correlated transport fails.
        """
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
        """Evaluate expressions under one request-scoped data binding."""
        if not expressions:
            return []
        with self.request(data) as request:
            return request.evaluate_many(expressions, output_format=output_format)

    def run_external_computers(self, tasks):
        """Execute a bounded dependent batch in the resource-limited child."""
        from exactus.external_computers.values import packed

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
        """Execute a technical request and decode typed failures under an exclusive lease."""
        with self._lock:
            try:
                reply = self._exchange(request)
                if not reply.get("ok"):
                    self._decode(reply)
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


if __name__ == "__main__":
    if "--serve" in sys.argv:
        from exactus.adapters.worker_serve import serve

        serve()
