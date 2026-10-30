"""Exclusive multi-operation lease API for the sandboxed worker.

Author: Olshansky Grigory
"""

from __future__ import annotations

import base64
import threading
from collections.abc import Mapping

from exactus.domain.errors import ExactusError
from exactus.domain.value import ComputeValue

__author__ = "Olshansky Grigory"


def encode_data(data: Mapping[str, bytes]) -> dict[str, str]:
    """Encode one request data mapping as base64 text."""
    return {
        name: base64.b64encode(payload).decode("ascii")
        for name, payload in data.items()
    }


class WorkerSession:
    """Exclusive multi-operation lease over one worker subprocess.

    Ships the request data mapping to the worker once in an init frame,
    evaluates many expressions without reserializing payloads and
    strictly clears child-side data on exit: a reset frame is verified,
    and any anomaly recycles the worker, so no session data can outlive
    the lease. Recycle at max_calls is deferred to lease release. This
    class is a friend of WorkerExternalComputer and runs under its lease lock.
    """

    def __init__(self, external_computer, data: Mapping[str, bytes] | None) -> None:
        """Bind the owning external computer and the request data mapping."""
        self._external_computer = external_computer
        self._data = dict(data or {})
        self._leased = False
        self._owner_thread = None

    def __enter__(self) -> WorkerSession:
        """Acquire the exclusive lease and install the request data."""
        self._external_computer._lock.acquire()
        if self._leased or self._external_computer._session_owner is not None:
            self._external_computer._lock.release()
            raise ExactusError("nested or reused worker session lease")
        try:
            reply = self._external_computer._exchange(
                {"op": "init", "data": encode_data(self._data)}
            )
            if not reply.get("ok"):
                raise ExactusError("worker rejected the session data")
        except BaseException:
            self._external_computer._close_locked()
            self._external_computer._lock.release()
            raise
        self._leased = True
        self._owner_thread = threading.get_ident()
        self._external_computer._session_owner = self
        self._process = self._external_computer._proc
        return self

    def evaluate(self, expression: str, *, output_format: str = "dec") -> ComputeValue:
        """Evaluate one expression against the leased request data."""
        self._check_owner()
        if (
            self._external_computer._proc is not self._process
            or self._process.poll() is not None
        ):
            raise ExactusError("worker session lost its process; enter a new session")
        reply = self._external_computer._exchange(
            {"op": "eval", "expression": expression, "format": output_format}
        )
        value = self._external_computer._decode(reply)
        self._external_computer._calls += 1
        return value

    def __exit__(self, exc_type, exc, tb) -> bool:
        """Clear child-side request data and release the lease."""
        self._check_owner()
        try:
            try:
                if (
                    self._external_computer._proc is self._process
                    and self._process.poll() is None
                ):
                    reply = self._external_computer._exchange({"op": "reset"})
                    if not reply.get("ok"):
                        self._external_computer._close_locked()
            except ExactusError:
                self._external_computer._close_locked()
            if self._external_computer._calls >= self._external_computer._max_calls:
                self._external_computer._close_locked()
        finally:
            self._leased = False
            self._owner_thread = None
            self._external_computer._session_owner = None
            self._external_computer._lock.release()
        return False

    def _check_owner(self) -> None:
        if (
            not self._leased
            or self._owner_thread != threading.get_ident()
            or self._external_computer._session_owner is not self
        ):
            raise ExactusError(
                "worker session is inactive or belongs to another thread"
            )
