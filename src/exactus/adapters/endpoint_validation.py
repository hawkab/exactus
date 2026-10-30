"""Strict request admission before invoking an inference engine."""

from collections.abc import Mapping, Sequence
from ..domain.errors import BudgetError, ExactusError, FormatError
from ..domain.value import FORMATS


class RequestTypeError(ExactusError, TypeError):
    """Request argument violates the strict type contract."""


class RequestValueError(ExactusError, ValueError):
    """Request argument is outside its supported range."""


def validate_request(
    prompt_ids,
    data,
    max_tokens,
    require_final,
    output_format,
    request_id,
    parent_id,
    planner,
    context_limit,
):
    if isinstance(prompt_ids, (str, bytes)) or not isinstance(prompt_ids, Sequence):
        raise RequestTypeError("prompt_ids must be a sequence of ints")
    if any(type(token) is not int for token in prompt_ids):
        raise RequestTypeError("prompt_ids must contain plain ints")
    if not prompt_ids or any(token < 0 for token in prompt_ids):
        raise RequestValueError("prompt_ids must be non-empty and non-negative")
    if type(max_tokens) is not int or max_tokens <= 0:
        raise RequestTypeError("max_tokens must be a positive int")
    if type(require_final) is not bool:
        raise RequestTypeError("require_final must be a bool")
    if type(output_format) is not str or output_format not in FORMATS:
        raise FormatError(f"unknown format: {output_format}")
    if data is not None and (
        not isinstance(data, Mapping)
        or any(
            not isinstance(key, str) or type(val) is not bytes
            for key, val in data.items()
        )
    ):
        raise RequestTypeError("data must map names to bytes payloads")
    for name, value in (("request_id", request_id), ("parent_id", parent_id)):
        if value is not None and type(value) is not str:
            raise RequestTypeError(f"{name} must be a str or None")
        if value is not None and not 0 < len(value) <= 256:
            raise RequestValueError(f"{name} must contain 1..256 characters")
    if planner is not None:
        raise RequestValueError(
            "prefill insertion cannot also use a logits forcing planner"
        )
    if context_limit is not None and len(prompt_ids) + max_tokens > context_limit:
        raise BudgetError("prompt plus max_tokens exceeds the context limit")
    return list(prompt_ids)
