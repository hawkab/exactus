"""Per-request forcing state for batch and speculative decoding layouts.

Author: Olshansky Grigory
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from itertools import islice

from ..domain.errors import ExactusError
from .logits_audit import RequestAudit

__author__ = "Olshansky Grigory"

DEFAULT_REQUEST_ID = "__default__"


class LogitsStateError(ExactusError):
    """Forcing state or batch row layout violates the documented contract."""


class RequestSlot:
    """Forcing plan and cursor of a single request."""

    __slots__ = ("plan", "cursor", "audit", "failure")

    def __init__(self) -> None:
        """Open an empty plan with the cursor at the start."""
        self.plan: list[int] = []
        self.cursor = 0
        self.audit = RequestAudit()
        self.failure = None

    def target(self) -> int | None:
        """Return the pending token id, or None when the plan is exhausted."""
        if self.cursor < len(self.plan):
            return self.plan[self.cursor]
        return None

    def advance(self) -> None:
        """Move the cursor one token forward or reject an exhausted plan."""
        if self.target() is None:
            raise LogitsStateError("cannot advance an exhausted forcing plan")
        self.cursor += 1


class ForcingState:
    """Request slots, row layout and the documented failure policy.

    Every request owns one slot keyed by its request id; row blocks are
    mapped by the order given to update_layout. Contract violations raise
    LogitsStateError so an unverified integration fails loudly instead of
    silently dropping the forcing guarantee.
    """

    def __init__(self) -> None:
        """Open the default slot and an empty row layout."""
        self._slots: dict[object, RequestSlot] = {DEFAULT_REQUEST_ID: RequestSlot()}
        self._layout: tuple[object, ...] = ()
        self.guard_failures = 0
        self.last_guard_error: str | None = None

    @property
    def layout(self) -> tuple[object, ...]:
        """Return request ids owning row blocks in batch order."""
        return self._layout

    def plan(self, request_id: object, tokens: Iterable[int]) -> int:
        """Append forced tokens to the request plan and return its length."""
        _check_unique((request_id,))
        tokens = list(islice(tokens, 8193))
        if len(tokens) > 8192 or any(type(t) is not int or t < 0 for t in tokens):
            raise LogitsStateError(
                "forcing plan must contain at most 8192 non-negative token IDs"
            )
        if request_id not in self._slots and len(self._slots) >= 1024:
            raise LogitsStateError("forcing request-state limit exceeded")
        slot = self._slots.get(request_id)
        if slot is None:
            slot = self._slots[request_id] = RequestSlot()
        if slot.failure is not None:
            raise LogitsStateError("reset a failed request before planning more tokens")
        if len(slot.plan) + len(tokens) > 8192:
            raise LogitsStateError("forcing plan token budget exceeded")
        slot.plan.extend(tokens)
        return len(slot.plan)

    def next_target(self, request_id: object) -> int | None:
        """Return the pending token id, or None when forcing is finished."""
        return self._slot(request_id).target()

    def advance(self, request_id: object) -> None:
        """Move the request cursor exactly one forced token forward."""
        self._slot(request_id).advance()

    def reset(self, request_id: object) -> None:
        """Drop the request plan, cursor and audit but keep the slot registered."""
        slot = self._slots.get(request_id)
        if slot is None:
            raise LogitsStateError("no request slot to reset")
        slot.failure = None
        slot.plan.clear()
        slot.cursor = 0
        slot.audit.clear()

    def update_layout(self, batch_update: object) -> None:
        """Bind request ids to row blocks and release finished requests.

        Accepted forms: a sequence of request ids in row-block order, a
        mapping with request_ids and optional finished_ids keys, or an
        object with the same attributes. Finished ids lose their slot so
        the row block is free for reuse; ids absent without being finished
        keep their plan across preemption.
        """
        if batch_update is None:
            return
        request_ids, finished_ids = _parse_batch_update(batch_update)
        _check_unique(request_ids + finished_ids)
        if len(set(request_ids) | (set(self._slots) - set(finished_ids))) > 1024:
            raise LogitsStateError("forcing request-state limit exceeded")
        if set(request_ids) & set(finished_ids):
            raise LogitsStateError("request id is both active and finished")
        for request_id in finished_ids:
            self._slots.pop(request_id, None)
        for request_id in request_ids:
            if request_id not in self._slots:
                self._slots[request_id] = RequestSlot()
        self._layout = request_ids

    def mark_guard_failure(self, exc: Exception, request_id=DEFAULT_REQUEST_ID) -> None:
        """Record an unexpected engine-side fault on the failing request slot."""
        self.guard_failures += 1
        self.last_guard_error = f"{type(exc).__name__}: {exc}"
        slot = self._slots.get(request_id)
        if slot is not None:
            slot.failure = self.last_guard_error

    def record(
        self,
        request_id: object,
        forced: int,
        snapshot: tuple[int, float] | None,
        attempted: bool = True,
    ) -> None:
        """Count every action, including disabled or failed snapshots.

        Position is the plan cursor before advance. Missing snapshots are
        coverage gaps, not evidence that no token was forced.
        """
        slot = self._slots.get(request_id)
        if slot is None:
            return
        slot.audit.record(request_id, slot.cursor, forced, snapshot, attempted)

    def audit(self, request_id: object = DEFAULT_REQUEST_ID) -> list[dict]:
        """Return the forcing-audit records of one request."""
        slot = self._slots.get(request_id)
        return (
            [dict(record) for record in slot.audit.records] if slot is not None else []
        )

    def audit_summary(self) -> dict[str, object]:
        """Return forced/corrected totals and their correction rate."""
        audits = [slot.audit for slot in self._slots.values()]
        forced = sum(audit.total_forced for audit in audits)
        corrected = sum(audit.total_corrected for audit in audits)
        audited = sum(audit.audited for audit in audits)
        return {
            "forced": forced,
            "corrected": corrected,
            "rate": corrected / audited if audited else None,
            "rate_denominator": audited,
            "snapshot_errors": sum(a.snapshot_errors for a in audits),
            "skipped": sum(a.skipped for a in audits),
        }

    def diagnostics(self) -> dict[str, object]:
        """Return an externally checkable forcing-health snapshot."""
        return {
            "guard_failures": self.guard_failures,
            "last_guard_error": self.last_guard_error,
            "active_requests": len(self._layout),
            "tracked_requests": len(self._slots),
            "failed_requests": sum(s.failure is not None for s in self._slots.values()),
        }

    def _slot(self, request_id: object) -> RequestSlot:
        """Return the request slot or reject an untracked request id."""
        slot = self._slots.get(request_id)
        if slot is None:
            raise LogitsStateError(f"no forcing state for request {request_id!r}")
        if slot.failure is not None:
            raise LogitsStateError(
                f"forcing failed for request {request_id!r}: {slot.failure}"
            )
        return slot


def batch_rows(batch):
    """Return the batch as a two-dimensional row sequence."""
    if hasattr(batch, "dim"):
        if batch.dim() == 1:
            return batch.unsqueeze(0)
        if batch.dim() != 2:
            raise LogitsStateError("logits must be a 1D row or a 2D batch")
    return batch


def _parse_batch_update(
    batch_update: object,
) -> tuple[tuple[object, ...], tuple[object, ...]]:
    """Return (request_ids, finished_ids) from the documented update forms."""
    if isinstance(batch_update, Sequence) and not isinstance(
        batch_update, (str, bytes)
    ):
        return tuple(batch_update), ()
    if isinstance(batch_update, Mapping):
        request_ids = batch_update.get("request_ids")
        finished_ids = batch_update.get("finished_ids") or ()
    else:
        request_ids = getattr(batch_update, "request_ids", None)
        finished_ids = getattr(batch_update, "finished_ids", ()) or ()
    if request_ids is None:
        raise LogitsStateError("batch update must expose request_ids")
    return tuple(request_ids), tuple(finished_ids)


def _check_unique(request_ids: tuple[object, ...]) -> None:
    """Reject duplicate or unhashable request ids in one batch update."""
    try:
        if len(set(request_ids)) != len(request_ids):
            raise LogitsStateError("batch update lists duplicate request ids")
    except TypeError as exc:
        raise LogitsStateError("request ids must be hashable") from exc
