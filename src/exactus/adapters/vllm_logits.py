"""Experimental legacy ABI. For MTP use the native endpoint/prefill path."""

from dataclasses import dataclass
import os

from .logits_adapter import _force_row, _validate_row
from .logits_state import LogitsStateError

try:
    from vllm.v1.sample.logits_processor.interface import LogitsProcessor
except ImportError:

    class LogitsProcessor:
        """Importable without the GPU runtime for contract tests."""


@dataclass(slots=True)
class Plan:
    tokens: tuple
    output: list
    checked: int = 0


class LegacyExactusLogitsProcessor(LogitsProcessor):
    """Cursor follows committed tokens, not calls to apply; legacy BatchUpdate ABI."""

    def __init__(self, vllm_config, device, is_pin_memory):
        if os.environ.get("VLLM_USE_V2_MODEL_RUNNER") != "0":
            raise LogitsStateError(
                "this class supports only the explicit legacy runner"
            )
        if getattr(vllm_config, "speculative_config", None) is not None:
            raise LogitsStateError(
                "speculative forcing is unsupported; use the endpoint plugin"
            )
        self.slots, self.batch_size = {}, 0
        self._occupied = set()

    @classmethod
    def validate_params(cls, params):
        tokens = (getattr(params, "extra_args", None) or {}).get(
            "exactus_token_ids", []
        )
        if (
            not isinstance(tokens, list)
            or len(tokens) > 8192
            or any(type(t) is not int or t < 0 for t in tokens)
        ):
            raise ValueError(
                "exactus_token_ids must be a bounded list of non-negative ints"
            )

    def is_argmax_invariant(self):
        return False

    def update_state(self, batch_update):
        if batch_update is None:
            return
        for index in batch_update.removed:
            self.slots.pop(index, None)
            self._occupied.discard(index)
        for index, params, prompt_ids, output_ids in batch_update.added:
            self._occupied.add(index)
            self.validate_params(params)
            tokens = tuple(
                (getattr(params, "extra_args", None) or {}).get("exactus_token_ids", [])
            )
            # extra_args defines a prefix from generation position zero;
            # re-admission after preemption must not restart that prefix.
            if tokens:
                self.slots[index] = Plan(tokens, output_ids)
            else:
                self.slots.pop(index, None)
        for source, target, direction in batch_update.moved:
            if direction.name == "SWAP":
                first, second = (
                    self.slots.pop(source, None),
                    self.slots.pop(target, None),
                )
                if first is not None:
                    self.slots[target] = first
                if second is not None:
                    self.slots[source] = second
            else:
                self._occupied.discard(source)
                self._occupied.add(target)
                plan = self.slots.pop(source, None)
                if plan is None:
                    self.slots.pop(target, None)
                else:
                    self.slots[target] = plan
        self.batch_size = len(self._occupied)
        if self._occupied != set(range(self.batch_size)):
            raise LogitsStateError("legacy batch indices must be dense after update")

    def apply(self, logits):
        if len(logits) != self.batch_size:
            raise LogitsStateError(
                "unexpected layout; speculative rows are unsupported"
            )
        targets = []
        completed = []
        for index, plan in self.slots.items():
            emitted = len(plan.output)
            if emitted < plan.checked:
                raise LogitsStateError(
                    "committed history rolled back on non-speculative runner"
                )
            until = min(emitted, len(plan.tokens))
            for position in range(plan.checked, until):
                if plan.output[position] != plan.tokens[position]:
                    raise LogitsStateError(
                        "committed tokens diverged from the forcing plan"
                    )
            plan.checked = until
            if emitted < len(plan.tokens):
                target = plan.tokens[emitted]
                if not 0 <= index < len(logits) or target >= len(logits[index]):
                    raise LogitsStateError("forced token or slot is out of range")
                targets.append((index, target))
            else:
                completed.append(index)
        for index in completed:
            self.slots.pop(index)
        for index, target in targets:
            _validate_row(logits[index], target)
        for index, target in targets:
            _force_row(logits[index], target, len(logits[index]), validated=True)
        return logits
