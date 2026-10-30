"""Exact-span computation over generated transcripts.

Author: Olshansky Grigory
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from ..domain.errors import ExactusError, GrammarError
from ..domain.ports import ExternalComputer
from ..domain.finalization import finalize, validate_final
from ..domain.protocol import SpanMachine

__author__ = "Olshansky Grigory"


class ComputeError(ExactusError):
    """The external computer port failed without raising a typed domain error."""


@dataclass(frozen=True, slots=True)
class Operation:
    """One recorded exact operation of a transcript."""

    expression: str
    value: str
    final: bool
    format: str


@dataclass(frozen=True, slots=True)
class ComputeResult:
    """Outcome of running the exact-span protocol over one transcript."""

    text: str
    operations: tuple[Operation, ...]
    exact_final: bool


class ComputeService:
    """Orchestrates span detection and exact evaluation over generated text.

    Contour one of the design: deterministic spans are computed by an
    external computer and substituted into the transcript, so computed
    digits never pass through sampling. Evaluation failures surface as
    typed exceptions and never as a guessed success value.
    """

    def __init__(self, external_computer: ExternalComputer) -> None:
        """Bind the external computer port."""
        self._external_computer = external_computer

    def run(
        self,
        transcript: str,
        data: Mapping[str, bytes] | None = None,
        *,
        require_final: bool = True,
        output_format: str = "dec",
    ) -> ComputeResult:
        """Compute every span of one transcript and return the completed text.

        The full frame state is validated before any result is returned.
        With require_final set, a transcript without a final span raises
        GrammarError instead of returning free text; otherwise computed
        values are rendered at their span positions.
        """
        machine = SpanMachine()
        spans = machine.feed(transcript)
        parsed = machine.finish()
        validate_final(parsed, "preserve-text")
        exact_final = any(span.final for span in spans)
        if require_final and not exact_final:
            raise GrammarError("transcript has no final exact span")
        operations = [
            self._record(span.expression, span.final, data, output_format)
            for span in spans
        ]
        text, _, exact_final = finalize(
            parsed, operations, "numeric-final" if require_final else "preserve-text"
        )
        return ComputeResult(text, tuple(operations), exact_final)

    def _record(
        self,
        expression: str,
        final: bool,
        data: Mapping[str, bytes] | None,
        output_format: str,
    ) -> Operation:
        """Evaluate one expression and wrap it as a recorded operation."""
        try:
            value = self._external_computer.evaluate(
                expression, data, output_format=output_format
            )
        except ExactusError:
            raise
        except Exception as exc:
            raise ComputeError(f"evaluation failed: {exc}") from exc
        return Operation(expression, value.render(), final, value.format)
