"""Audited proofs of computed spans: what was computed and how it grounds.

Author: Olshansky Grigory
"""

from __future__ import annotations

from dataclasses import dataclass

__author__ = "Olshansky Grigory"

DERIVED = "\u22a2"
QED = "\u220e"


@dataclass(frozen=True, slots=True)
class SpanProof:
    """One computed span: expression, substituted value and role."""

    expression: str
    value: str
    final: bool = False
    origin: str = "model_marker"
    source_start: int | None = None
    source_end: int | None = None


@dataclass(frozen=True, slots=True)
class AnswerCheck:
    """External computer verdict for the model's final answer when the prompt is solvable."""

    expected: str
    given: str | None
    verdict: str


class Proof:
    """Audited proof of one response with inline and detailed renderings."""

    __slots__ = ("spans", "check", "_parsed")

    def __init__(
        self,
        spans: tuple[SpanProof, ...],
        check: AnswerCheck | None = None,
        *,
        parsed=None,
    ):
        self.spans = spans
        self.check = check
        self._parsed = parsed

    def inline(self, content: str, values: list[str] | None = None) -> str:
        """Return content with every span replaced by a marked computed value."""
        from .errors import GrammarError
        from .protocol import parse

        parsed = (
            self._parsed
            if self._parsed is not None and self._parsed.content == content
            else parse(content)
        )
        spans = parsed.spans
        if len(spans) != len(self.spans) or any(
            s.expression != r.expression or s.final != r.final
            for s, r in zip(spans, self.spans)
        ):
            raise GrammarError("proof is not bound to this transcript")
        if values is not None and values != [s.value for s in self.spans]:
            raise GrammarError("proof values cannot be overridden")
        return parsed.render(
            values if values is not None else [s.value for s in self.spans], marks=True
        )

    def detail(self) -> str:
        """Return the one-line audit summary in the derived-equals style."""
        parts = [f"{DERIVED} {s.expression}={s.value} {QED}" for s in self.spans]
        if self.check is not None:
            parts.append(f"check {self.check.verdict}")
        return "  ".join(parts) if parts else "no computed spans"

    def verification(self) -> dict:
        """Separate expression evaluation from verifying the user's task."""
        checked = self.check
        fallback = any(s.origin == "prompt_dispatch" for s in self.spans)
        return {
            "task_verified": (fallback or checked.verdict == "match")
            if checked
            else None,
            "verification_scope": "calculator_prompt"
            if checked
            else "computed_expressions_only",
        }

    def as_dict(self, *, detail: bool = True) -> dict:
        """Return the machine-readable audit record for response payloads."""
        record: dict = {
            **self.verification(),
            "spans": [
                {
                    "expression": s.expression,
                    "value": s.value,
                    "final": s.final,
                    "origin": s.origin,
                    "source_start": s.source_start,
                    "source_end": s.source_end,
                }
                for s in self.spans
            ],
        }
        if detail:
            record["detail"] = self.detail()
        if self.check is not None:
            record["answer_check"] = {
                "expected": self.check.expected,
                "given": self.check.given,
                "verdict": self.check.verdict,
            }
        return record


def check_answer(expected: str, given: str | None) -> AnswerCheck:
    """Compare the external computer value with the model's own final value."""
    if given is None:
        return AnswerCheck(expected, None, "no_answer")
    return AnswerCheck(expected, given, "match" if given == expected else "mismatch")
