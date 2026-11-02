"""Validate request identifiers and plain logits batch layouts."""

from __future__ import annotations

__author__ = "Olshansky Grigory"

from collections.abc import Mapping, Sequence

from ..domain.errors import ExactusError


class LogitsStateError(ExactusError):
    """Forcing state or batch row layout violates the documented contract."""


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
