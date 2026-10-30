"""Bounded evaluation semantics for exact integer expressions.

Author: Olshansky Grigory
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType

from .errors import BudgetError, DataError, GrammarError
from .grammar import GrammarParser
from .lexer import tokenize
from .ports import Hasher
from .value import ComputeValue

__author__ = "Olshansky Grigory"


@dataclass(frozen=True, slots=True)
class EvalContext:
    """Immutable request-scoped data and hashing for one evaluation.

    The data mapping is snapshotted at construction, so concurrent
    requests sharing one evaluator can never observe each other's
    payloads and callers cannot mutate a running evaluation.
    """

    data: Mapping[str, bytes] = field(default_factory=dict)
    hasher: Hasher | None = None

    def __post_init__(self) -> None:
        """Freeze the data mapping against caller mutation."""
        object.__setattr__(self, "data", MappingProxyType(dict(self.data)))

    def data_payload(self, name: str) -> bytes:
        """Return the raw bytes of one request-scoped data reference."""
        payload = self.data.get(name)
        if payload is None:
            raise DataError(f"unknown data reference: {name}")
        if not isinstance(payload, bytes):
            raise DataError(f"data reference {name} is not bytes")
        return payload

    def data_value(self, name: str) -> int:
        """Return the big-endian unsigned integer of one data payload."""
        payload = self.data_payload(name)
        return int.from_bytes(payload, "big") if payload else 0

    def hash_digest(self, name: str, payload: bytes) -> int:
        """Return the digest of payload for one supported hash function."""
        if self.hasher is None or name not in self.hasher.algorithms:
            raise GrammarError(f"unknown function: {name}")
        return self.hasher.digest(name, payload)


class ExpressionEvaluator:
    """Evaluates bounded integer expressions with immutable budgets.

    The evaluator holds only immutable configuration (hasher port and
    budgets); request data flows through an EvalContext passed down the
    call stack, so one instance is safe for concurrent requests.

    ``/`` and ``%`` truncate toward zero, ``//`` floors, ``xor`` is
    bitwise exclusive or and ``^`` is power. Hash calls consume data
    references only, keeping byte payloads out of numeric ambiguity.
    """

    def __init__(
        self,
        hasher: Hasher | None = None,
        *,
        max_chars: int = 4096,
        max_nodes: int = 512,
        max_depth: int = 64,
        max_bits: int = 4096,
        max_exponent: int = 100000,
    ) -> None:
        """Store the hasher port and the evaluation budgets."""
        self._hasher = hasher
        self._max_chars = max_chars
        self._max_nodes = max_nodes
        self._max_depth = max_depth
        self._max_bits = max_bits
        self._max_exponent = max_exponent

    @property
    def max_bits(self) -> int:
        """Return the integer bit budget."""
        return self._max_bits

    def evaluate(
        self,
        expression: str,
        data: Mapping[str, bytes] | None = None,
        *,
        output_format: str = "dec",
    ) -> ComputeValue:
        """Parse and evaluate one expression with request-scoped data."""
        if not isinstance(expression, str) or not expression.strip():
            raise GrammarError("expression must be a non-empty string")
        if len(expression) > self._max_chars:
            raise BudgetError("expression exceeds the character budget")
        context = EvalContext(data or {}, self._hasher)
        parser = GrammarParser(self, tokenize(expression), context,
                              max_nodes=self._max_nodes, max_depth=self._max_depth)
        return ComputeValue(self.bound(parser.parse()), output_format)

    def bound(self, value: int) -> int:
        """Return value after enforcing the integer bit budget."""
        if value.bit_length() > self._max_bits:
            raise BudgetError("integer exceeds the bit budget")
        return value

    def mul_case(self, op: str, left: int, right: int) -> int:
        """Apply one multiplicative operator with trunc or floor semantics."""
        if op == "*":
            return left * right
        if right == 0:
            raise GrammarError("division by zero")
        if op == "//":
            return left // right
        quotient = abs(left) // abs(right)
        quotient = -quotient if (left < 0) != (right < 0) else quotient
        return left - quotient * right if op in ("%", "mod") else quotient

    def power(self, base: int, exponent: int) -> int:
        """Compute base**exponent under exponent and bit budgets."""
        if exponent < 0:
            raise GrammarError("negative exponent")
        if exponent > self._max_exponent:
            raise BudgetError("exponent outside allowed range")
        if abs(base) > 1 and (abs(base).bit_length() - 1) * exponent >= self._max_bits:
            raise BudgetError("power exceeds the bit budget")
        return pow(base, exponent)
