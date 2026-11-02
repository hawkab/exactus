"""Validate and force score rows without changing request cursors."""

__author__ = "Olshansky Grigory"

import math

from .logits_layout import LogitsStateError

_FLOOR = float("-inf")


def _validate_row(row, target):
    """Reject immutable score rows and targets masked by existing constraints."""
    if not hasattr(row, "__setitem__"):
        raise TypeError("score row is immutable")
    if not math.isfinite(float(row[target])):
        raise LogitsStateError(
            "forced token conflicts with an existing mask or invalid logits"
        )


def _force_row(row, target: int, vocab: int, *, validated=False) -> None:
    """Pin a row; mask validation must never be removed for performance."""
    if not validated:
        _validate_row(row, target)
    if hasattr(row, "fill_"):
        row.fill_(_FLOOR)
    else:
        for index in range(vocab):
            row[index] = _FLOOR
    row[target] = 0.0
