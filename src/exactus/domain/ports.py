"""Ports of the Exactus hexagonal architecture."""

from __future__ import annotations

__author__ = "Olshansky Grigory"

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from .value import ComputeValue


@dataclass(frozen=True, slots=True)
class Candidate:
    """One speculative draft continuation with its predictor score."""

    tokens: tuple[int, ...]
    score: float


@dataclass(frozen=True, slots=True)
class Segment:
    """One finished engine generation segment inside a request."""

    text: str
    token_ids: tuple[int, ...]
    sampled_tokens: int
    finish_reason: str
    token_count_source: str = "engine_output_ids"


@runtime_checkable
class Hasher(Protocol):
    """Computes integer digests over request-scoped byte payloads."""

    @property
    def algorithms(self) -> tuple[str, ...]:
        """Return the supported algorithm names."""

    def digest(self, algorithm: str, payload: bytes) -> int:
        """Return the digest of payload as an unsigned integer."""


@runtime_checkable
class ExternalComputer(Protocol):
    """Evaluates exact expressions into integer values."""

    def evaluate(
        self,
        expression: str,
        data: Mapping[str, bytes] | None = None,
        *,
        output_format: str = "dec",
    ) -> ComputeValue:
        """Evaluate an expression using only its request-scoped inputs.

        Args:
            expression: Exact integer expression in the supported grammar.
            data: Named byte payloads available to this evaluation only.
            output_format: Rendering format of the resulting integer.

        Returns:
            An exact value with its requested rendering format.

        Raises:
            ExactusError: The expression or its inputs cannot be evaluated.
        """


@runtime_checkable
class Tokenizer(Protocol):
    """Encodes text to token ids and back."""

    def encode(self, text: str) -> list[int]:
        """Return the token ids for text."""

    def decode(self, ids: Sequence[int]) -> str:
        """Return the text for token ids."""


@runtime_checkable
class DraftPredictor(Protocol):
    """Proposes speculative continuations for a token prefix."""

    def predict(self, prefix: Sequence[int], k: int = 3) -> list[Candidate]:
        """Return up to k draft continuations, best first."""


@runtime_checkable
class Engine(Protocol):
    """Generates one text segment per call inside a request."""

    async def generate(
        self, prompt_ids: Sequence[int], max_tokens: int, request_id: str
    ) -> Segment:
        """Generate one segment and return it finished."""
