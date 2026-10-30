"""Request modes, control fields and exact-answer finalization.

Author: Olshansky Grigory
"""

from __future__ import annotations

import base64
import re
from collections.abc import Mapping
from dataclasses import dataclass

from .domain.audit import Proof, SpanProof
from .domain.errors import ExactusError
from .domain.finalization import finalize, validate_final
from .domain.protocol import SpanMachine

__author__ = "Olshansky Grigory"

MODES = ("numeric-final", "preserve-text", "strict", "audit")
DEFAULT_MODE = "preserve-text"
CONTROL_FIELDS = frozenset({"data", "mode"})
_MAX_PAYLOADS = 64
_MAX_PAYLOAD_BYTES = 256 * 1024
_NAME_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]{0,63}\Z")
_ERROR_CODES = {
    "GrammarError": "grammar_error",
    "BudgetError": "budget_error",
    "DataError": "data_error",
    "FormatError": "format_error",
}


class RequestError(ExactusError):
    """Request-level failure carrying a machine-readable code and status."""

    def __init__(self, message: str, error_code: str, status: int = 422) -> None:
        """Store the message, the machine-readable code and the HTTP status."""
        super().__init__(message)
        self.error_code = error_code
        self.status = status

    def body(self) -> dict:
        """Return the machine-readable JSON body of the failure."""
        return {"error": str(self), "error_code": self.error_code}


@dataclass(frozen=True, slots=True)
class Answer:
    """Finalized text of one processed response choice."""

    text: str
    computed: tuple[str, ...]
    exact_final: bool
    proof: Proof | None = None
    model_final_emitted: bool = False
    generation_complete: bool = False


def as_request_error(exc: ExactusError) -> RequestError:
    """Wrap one domain error into a request error with a stable code."""
    code = getattr(
        exc, "error_code", _ERROR_CODES.get(type(exc).__name__, "compute_error")
    )
    return RequestError(str(exc), code, 422)


def decode_data(raw) -> dict[str, bytes]:
    """Decode and strictly validate the request-scoped data map.

    Every payload is base64-decoded exactly once with strict alphabet
    checking; names, payload sizes and payload counts are bounded.
    """
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise RequestError("data must be an object of base64 payloads", "invalid_data")
    if len(raw) > _MAX_PAYLOADS:
        raise RequestError("too many data payloads", "invalid_data")
    decoded: dict[str, bytes] = {}
    for name, payload in raw.items():
        if not isinstance(name, str) or not _NAME_RE.match(name):
            raise RequestError(
                "data names must match [A-Za-z_][A-Za-z0-9_]{0,63}", "invalid_data"
            )
        if not isinstance(payload, str):
            raise RequestError(
                "data entries must map names to base64 strings", "invalid_data"
            )
        try:
            value = base64.b64decode(payload, validate=True)
        except Exception as exc:
            raise RequestError(
                "data payload is not valid base64", "invalid_data"
            ) from exc
        if len(value) > _MAX_PAYLOAD_BYTES:
            raise RequestError("data payload exceeds 256 KiB", "invalid_data")
        decoded[name] = value
    return decoded


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


def split_payload(payload: Mapping) -> tuple[dict, dict[str, bytes], object]:
    """Split one request into upstream fields, decoded data and the raw mode."""
    if not isinstance(payload, Mapping):
        raise RequestError("request body must be a JSON object", "invalid_request")
    upstream = {
        key: value for key, value in payload.items() if key not in CONTROL_FIELDS
    }
    data = decode_data(payload.get("data"))
    return upstream, data, payload.get("mode")


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
