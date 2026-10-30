"""One finalization policy shared by library, HTTP and native adapters."""

from __future__ import annotations

from collections.abc import Sequence
import re

from .audit import Proof, SpanProof, check_answer
from .errors import GrammarError
from .protocol import ParsedTranscript, parse


class FinalizationError(GrammarError):
    """Protocol failure with a stable adapter-independent error code."""

    def __init__(self, message: str, code: str) -> None:
        super().__init__(message)
        self.error_code = code


def validate_final(parsed: ParsedTranscript, mode: str) -> list[int]:
    """Validate structure before paying for any expression evaluation."""
    if mode not in ("preserve-text", "audit", "strict", "numeric-final"):
        raise FinalizationError("unknown finalization mode", "invalid_mode")
    finals = [i for i, span in enumerate(parsed.spans) if span.final]
    if len(finals) > 1:
        raise FinalizationError(
            "response has several final exact spans", "multiple_final"
        )
    if finals and finals[0] != len(parsed.spans) - 1:
        raise FinalizationError(
            "operation after final exact span", "operation_after_final"
        )
    if mode in ("strict", "numeric-final") and not finals:
        raise FinalizationError("response has no final exact span", "missing_final")
    return finals


def finalize(
    content: str | ParsedTranscript,
    records: Sequence,
    mode: str,
    *,
    external_computer_result: tuple[str, str] | None = None,
    prompt: str = "",
) -> tuple[str, Proof, bool]:
    """Validate frames and bind evaluated records to their original spans.

    A final span is unique and terminal among operations in every mode;
    narrative after it is preserved. Strict/numeric-final require a final.
    No expression is evaluated again here.
    """
    parsed = content if isinstance(content, ParsedTranscript) else parse(content)
    spans = parsed.spans
    if len(spans) != len(records) or any(
        s.expression != r.expression or s.final != r.final
        for s, r in zip(spans, records)
    ):
        raise FinalizationError(
            "audit records do not match transcript", "invalid_proof"
        )
    fallback = False
    try:
        finals = validate_final(parsed, mode)
    except FinalizationError as exc:
        if (
            exc.error_code != "missing_final"
            or mode != "numeric-final"
            or external_computer_result is None
        ):
            raise
        finals, fallback = [], True
    proof = Proof(
        tuple(
            r
            if isinstance(r, SpanProof)
            else SpanProof(
                r.expression, r.value, r.final, source_start=s.start, source_end=s.end
            )
            for s, r in zip(spans, records)
        ),
        parsed=parsed,
    )
    if external_computer_result is not None:
        found = re.findall(r"\\answer\{(-?\d+)\}", parsed.content)
        given = records[finals[0]].value if finals else found[-1] if found else None
        proof.check = check_answer(external_computer_result[1], given)
        if (
            mode in ("strict", "numeric-final")
            and finals
            and proof.check.verdict != "match"
        ):
            raise FinalizationError("answer check failed", "answer_mismatch")
    if fallback:
        expression, value = external_computer_result
        start = prompt.find(expression)
        proof.spans += (
            SpanProof(
                expression,
                value,
                True,
                "prompt_dispatch",
                start if start >= 0 else None,
                start + len(expression) if start >= 0 else None,
            ),
        )
        return value, proof, True
    if mode == "numeric-final":
        text = records[finals[0]].value
    elif mode == "audit":
        text = parsed.render([r.value for r in records], marks=True)
    else:
        text = parsed.render([r.value for r in records])
    return text, proof, bool(finals)
