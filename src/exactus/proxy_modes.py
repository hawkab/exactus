"""Request modes, control fields and exact-answer finalization."""

from __future__ import annotations

__author__ = "Olshansky Grigory"

import re
from collections.abc import Mapping
from dataclasses import dataclass

from .domain.audit import Proof, SpanProof
from .domain.errors import ExactusError
from .domain.finalization import finalize, validate_final
from .domain.protocol import SpanMachine
from .proxy_data import CONTROL_FIELDS as CONTROL_FIELDS
from .proxy_data import decode_data as decode_data
from .proxy_data import split_payload as split_payload
from .request_errors import (
    RequestError as RequestError,
)
from .request_errors import (
    as_request_error as as_request_error,
)

MODES = ("numeric-final", "preserve-text", "strict", "audit")
DEFAULT_MODE = "preserve-text"


@dataclass(frozen=True, slots=True)
class Answer:
    """Finalized text of one processed response choice."""

    text: str
    computed: tuple[str, ...]
    exact_final: bool
    proof: Proof | None = None
    model_final_emitted: bool = False
    generation_complete: bool = False


def resolve_mode(raw, *, legacy_require_final=None) -> str:
    """Resolve the request mode from the explicit and the legacy field."""
    if raw is None:
        return "numeric-final" if legacy_require_final else DEFAULT_MODE
    if raw not in MODES:
        raise RequestError(
            "mode must be one of numeric-final, preserve-text, strict, audit",
            "invalid_mode",
        )
    return raw


def model_answer(text: str, answer) -> str | None:
    """Return the model's own final value: final span value or answer{} tag."""
    if answer.proof is not None:
        finals = [
            s.value
            for s in answer.proof.spans
            if s.final and s.origin == "model_marker"
        ]
        if finals:
            return finals[-1]
    elif answer.exact_final and answer.computed:
        return answer.computed[-1]
    found = re.findall(r"\\answer\{(-?\d+)\}", text)
    return found[-1] if found else None


def compute_answer(
    content: str,
    external_computer,
    data: Mapping[str, bytes] | None,
    *,
    mode: str,
    output_format: str = "dec",
    external_computer_result=None,
    prompt="",
) -> Answer:
    """Compute every span of one text and finalize it under the mode policy.

    ``preserve-text`` substitutes values in place and tolerates a missing
    final span; ``audit`` additionally marks every substituted value with
    derived/QED proof marks; ``strict`` demands exactly one final span;
    ``numeric-final`` answers with the final value only. Any protocol or
    evaluation violation raises instead of returning a soft success.
    """
    machine = SpanMachine()
    try:
        spans = machine.feed(content)
        parsed = machine.finish()
        validate_final(parsed, "preserve-text")
    except ExactusError as exc:
        raise as_request_error(exc) from exc
    records: list[SpanProof] = []
    from .application.evaluate import evaluate_spans

    try:
        evaluated = (
            evaluate_spans(external_computer, spans, data, output_format=output_format)
            if spans
            else []
        )
        for span, operation in zip(spans, evaluated):
            records.append(
                SpanProof(
                    span.expression,
                    operation.value,
                    span.final,
                    source_start=span.start,
                    source_end=span.end,
                )
            )
    except ExactusError as exc:
        raise as_request_error(exc) from exc
    try:
        text, proof, exact = finalize(
            parsed,
            records,
            mode,
            external_computer_result=external_computer_result,
            prompt=prompt,
        )
    except ExactusError as exc:
        raise as_request_error(exc) from exc
    return Answer(
        text,
        tuple(s.value for s in proof.spans),
        exact,
        proof,
        any(s.final for s in spans),
        True,
    )
