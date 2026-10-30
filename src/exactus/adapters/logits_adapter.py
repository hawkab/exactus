"""vLLM logits-processor adapter enforcing computed spans token-exactly.

Author: Olshansky Grigory
"""

from __future__ import annotations

from collections.abc import MutableSequence
import math

from ..domain.ports import Tokenizer
from .logits_audit import audit_snapshot
from .logits_state import DEFAULT_REQUEST_ID, ForcingState, LogitsStateError, batch_rows

__author__ = "Olshansky Grigory"

_FLOOR = float("-inf")


class ExactusLogitsProcessor:
    """Offline/plain batch forcing helper, not the vLLM plugin ABI.

    Invalid masks/state fail closed. Speculative forcing is unsupported;
    the native endpoint uses prefill and keeps engine-owned MTP. Audit is
    off by default; sampled/full are explicit diagnostic modes. Full GPU
    audit performs compact synchronization and is not a zero-cost feature.
    """

    def __init__(
        self,
        tokenizer: Tokenizer | None = None,
        *,
        audit_mode="off",
        audit_every=32,
        **_,
    ) -> None:
        """Store the tokenizer port and open per-request forcing state."""
        self._tokenizer = tokenizer
        self._state = ForcingState()
        if (
            audit_mode not in ("off", "sampled", "full")
            or type(audit_every) is not int
            or audit_every < 1
        ):
            raise LogitsStateError("invalid audit configuration")
        self._audit_mode, self._audit_every = audit_mode, audit_every
        self._steps = 0

    def _snapshot(self, row, vocab):
        attempted = self._audit_mode == "full" or (
            self._audit_mode == "sampled" and self._steps % self._audit_every == 0
        )
        self._steps += 1
        return (audit_snapshot(row, vocab) if attempted else None), attempted

    @staticmethod
    def validate_params(_) -> None:
        """Accept any construction parameters."""

    def update_state(self, batch_update: object) -> None:
        """Map request ids to row blocks and release finished slots.

        The i-th request id owns row block i of the next applied batch;
        see ForcingState.update_layout for the accepted update forms.
        """
        self._state.update_layout(batch_update)

    def is_argmax_invariant(self) -> bool:
        """Report False because forcing changes the argmax on purpose."""
        return False

    def diagnostics(self) -> dict[str, object]:
        """Return the externally checkable forcing-health snapshot."""
        return self._state.diagnostics()

    def plan_text(self, text: str, request_id: object = DEFAULT_REQUEST_ID) -> int:
        """Append the encoded text to the request plan and return its length."""
        tokens = self._tokenizer.encode(text) if self._tokenizer is not None else ()
        return self._state.plan(request_id, tokens)

    def next_target(self, request_id: object = DEFAULT_REQUEST_ID) -> int | None:
        """Return the pending forced token id, or None outside forcing."""
        return self._state.next_target(request_id)

    def advance(self, request_id: object = DEFAULT_REQUEST_ID) -> None:
        """Move the request cursor exactly one forced token forward."""
        self._state.advance(request_id)

    def reset(self, request_id: object = DEFAULT_REQUEST_ID) -> None:
        """Drop the request plan, cursor and audit but keep its slot registered."""
        self._state.reset(request_id)

    def forcing(self, request_id: object = DEFAULT_REQUEST_ID) -> bool:
        """Report whether the request still has tokens left to force."""
        return self._state.next_target(request_id) is not None

    def audit(self, request_id: object = DEFAULT_REQUEST_ID) -> list[dict]:
        """Return the forcing-audit records of one request."""
        return self._state.audit(request_id)

    def audit_summary(self) -> dict[str, object]:
        """Return the forcing-audit totals across tracked requests."""
        return self._state.audit_summary()

    def apply_offline(
        self, scores: MutableSequence[float], request_id: object = DEFAULT_REQUEST_ID
    ) -> MutableSequence[float]:
        """Force one offline score row for one request and advance it by one."""
        try:
            target = self._state.next_target(request_id)
            if target is None:
                return scores
            vocab = len(scores)
            if not 0 <= target < vocab:
                raise LogitsStateError("forced token is outside the row vocabulary")
            snapshot, attempted = self._snapshot(scores, vocab)
            _force_row(scores, target, vocab)
            self._state.record(request_id, target, snapshot, attempted)
            self._state.advance(request_id)
            return scores
        except LogitsStateError:
            raise
        except Exception as exc:
            self._state.mark_guard_failure(exc, request_id)
            raise LogitsStateError("forcing failed; request must not continue") from exc

    def apply(self, logits):
        """Force one target row per active request in a plain batch layout."""
        return self._apply_layout(logits, 1)

    def apply_with_spec_decode(self, logits, num_draft_tokens: int):
        """Fail before mutation: no acceptance/rollback contract is available."""
        raise LogitsStateError(
            "speculative forcing needs an engine acceptance callback and is not supported; "
            "use the coordinator prefill path, which retains engine MTP"
        )

    def _apply_layout(self, batch, block_size: int):
        """Force target rows under the documented row-block contract."""
        rows = batch_rows(batch)
        expected = len(self._state.layout) * block_size
        if len(rows) != expected:
            raise LogitsStateError(
                f"batch has {len(rows)} rows but the layout expects {expected}"
            )
        targets = []
        for index, request_id in enumerate(self._state.layout):
            target = self._state.next_target(request_id)
            if target is None:
                continue
            row = rows[index * block_size]
            if not hasattr(row, "__len__"):
                raise LogitsStateError("batch rows must be score sequences")
            vocab = len(row)
            if not 0 <= target < vocab:
                raise LogitsStateError("forced token is outside the row vocabulary")
            try:
                _validate_row(row, target)
            except LogitsStateError:
                raise
            except Exception as exc:
                self._state.mark_guard_failure(exc, request_id)
                raise LogitsStateError("forcing validation failed for request") from exc
            targets.append((request_id, row, target, vocab))
        # Validate every row before mutation, and commit cursors only after ALL
        # mutations succeed. A device/kernel fault aborts only this affected batch.
        snapshots = [self._snapshot(row, vocab) for _, row, _, vocab in targets]
        try:
            for _, row, target, vocab in targets:
                _force_row(row, target, vocab, validated=True)
        except Exception as exc:
            for request_id, _, _, _ in targets:
                self._state.mark_guard_failure(exc, request_id)
            raise LogitsStateError(
                "forcing mutation failed; affected requests must abort"
            ) from exc
        for (request_id, _, target, _), (snapshot, attempted) in zip(
            targets, snapshots
        ):
            self._state.record(request_id, target, snapshot, attempted)
            self._state.advance(request_id)
        return batch


def _validate_row(row, target):
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


def build_vllm_processor(*args, **kwargs):
    """Reject the old factory ABI instead of silently installing a broken hook."""
    raise LogitsStateError(
        "use exactus.adapters.vllm_logits:LegacyExactusLogitsProcessor "
        "on the legacy non-speculative runner, or the native endpoint plugin"
    )
