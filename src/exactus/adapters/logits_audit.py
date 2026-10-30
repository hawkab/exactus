"""Bounded per-request audit of forced versus wanted tokens.

Author: Olshansky Grigory
"""

from __future__ import annotations

from collections import deque

__author__ = "Olshansky Grigory"

AUDIT_LIMIT = 256


class RequestAudit:
    """Bounded record buffer and lifetime counters of one request.

    The buffer keeps at most AUDIT_LIMIT records so memory stays bounded,
    while total_forced and total_corrected count every forced token of
    the request regardless of buffer truncation.
    """

    def __init__(self) -> None:
        """Open an empty record buffer with zero totals."""
        self.records: deque[dict[str, object]] = deque(maxlen=AUDIT_LIMIT)
        self.total_forced = 0
        self.total_corrected = 0
        self.audited = 0
        self.snapshot_errors = 0
        self.skipped = 0

    def record(
        self,
        request_id: object,
        position: int,
        forced: int,
        snapshot,
        attempted: bool = True,
    ) -> None:
        """Count every action; retain readable snapshots in the bounded buffer."""
        self.total_forced += 1
        if snapshot is None:
            self.snapshot_errors += int(attempted)
            self.skipped += int(not attempted)
            return
        wanted, logit_gap = snapshot
        self.audited += 1
        self.records.append(
            {
                "request_id": request_id,
                "position": position,
                "forced": forced,
                "wanted": wanted,
                "logit_gap": logit_gap,
            }
        )
        if wanted != forced:
            self.total_corrected += 1

    def clear(self) -> None:
        """Drop every record and zero the lifetime totals."""
        self.records.clear()
        self.total_forced = 0
        self.total_corrected = 0
        self.audited = self.snapshot_errors = self.skipped = 0


def audit_snapshot(row, vocab: int) -> tuple[int, float] | None:
    """Return (wanted_id, logit_gap) of the row before forcing, or None.

    The wanted id is the row argmax and the gap is the wanted logit minus
    the second largest one, i.e. how sure the model was of its own choice.
    Unreadable rows yield None so auditing stays best-effort and can never
    disturb forcing.
    """
    try:
        if callable(getattr(row, "topk", None)):
            # Explicit diagnostic mode only: one vector reduction and compact
            # transfer, never vocab-size Python scalar reads on a CUDA tensor.
            values, indices = row.detach().topk(min(2, vocab))
            values, indices = values.cpu().tolist(), indices.cpu().tolist()
            return indices[0], values[0] - values[1] if len(values) > 1 else 0.0
        wanted, best = 0, float(row[0])
        second: float | None = None
        for index in range(1, vocab):
            value = float(row[index])
            if value > best:
                wanted, best, second = index, value, best
            elif second is None or value > second:
                second = value
        gap = best - second if second is not None else 0.0
        return wanted, gap
    except Exception:
        return None
