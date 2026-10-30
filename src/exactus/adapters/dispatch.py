"""Prompt dispatch: detect self-contained computable tasks and solve them.

Adopted from the Turnstyle dispatch idea: a prompt that carries a
computable expression is grounded by an external computer, so correctness does not
depend on the model emitting protocol markers. Extraction is
conservative: a candidate must look like an expression and evaluate
exactly under the bounded grammar, otherwise dispatch abstains.

Author: Olshansky Grigory
"""

from __future__ import annotations

import re
from collections.abc import Mapping

from ..domain.errors import ExactusError

__author__ = "Olshansky Grigory"

_CUE_RE = re.compile(
    r"\s*(?:вычисли(?:ть)?|посчитай|сколько будет|найди|compute|calculate|"
    r"evaluate|what\s+is|solve|result\s+of)\s*:?\s+(.+?)\s*[?.!]?\s*",
    re.IGNORECASE,
)
_BARE_RE = re.compile(r"^\s*(.+?)\s*=\s*\?\s*$")
_EXPR_CHARS = set("0123456789abcdefxABCDEFX@()_+-*/%<>&|^~ ")
_OPERATOR_CHARS = set("+-*/%<>&|^~(")
_MAX_CANDIDATE = 4096


def _clean(candidate: str) -> str:
    """Trim whitespace, never truncate decimals or trailing instructions."""
    return candidate.strip()


def _looks_like_expression(candidate: str) -> bool:
    """Return whether a candidate has expression shape worth evaluating."""
    if not candidate or len(candidate) > _MAX_CANDIDATE:
        return False
    if not set(candidate) <= _EXPR_CHARS:
        return False
    if not any(ch.isdigit() for ch in candidate):
        return False
    return any(ch in _OPERATOR_CHARS for ch in candidate) or "@" in candidate


def find_expression(prompt: str) -> str | None:
    """Accept only an entire single-line calculator request; otherwise abstain.

    A successful evaluation of a substring is not evidence that it represents
    the user's task. Quotes, decimals and additional instructions are not
    dispatched, even when an initial arithmetic substring would be valid.
    """
    if not isinstance(prompt, str) or len(prompt) > _MAX_CANDIDATE + 64:
        return None
    for pattern in (_CUE_RE, _BARE_RE):
        match = pattern.fullmatch(prompt)
        if match is None:
            continue
        candidate = _clean(match.group(1))
        if _looks_like_expression(candidate):
            return candidate
    return None


def external_computer_answer(
    prompt: str,
    external_computer,
    data: Mapping[str, bytes] | None = None,
    output_format: str = "dec",
) -> tuple[str, str] | None:
    """Return (expression, exact value) for a prompt, or None when not solvable."""
    expression = find_expression(prompt)
    if expression is None:
        return None
    try:
        value = external_computer.evaluate(
            expression, data, output_format=output_format
        )
    except ExactusError:
        return None
    return expression, value.render()


def last_user_text(payload: Mapping) -> str:
    """Return the last user message text of one chat request payload."""
    messages = payload.get("messages")
    if not isinstance(messages, list):
        return ""
    for message in reversed(messages):
        if isinstance(message, Mapping) and message.get("role") == "user":
            content = message.get("content")
            if isinstance(content, str):
                return content
    return ""


def calculator_prompt(payload: Mapping) -> str:
    """Abstain if any conversation context can change the calculator task.

    Call on the original client payload, before injecting our own system hint.
    Even apparently harmless user-provided system/developer messages are not
    silently whitelisted. Multimodal and tool conversations also abstain.
    """
    messages = payload.get("messages")
    if not isinstance(messages, list) or len(messages) != 1:
        return ""
    message = messages[0]
    if not isinstance(message, Mapping) or message.get("role") != "user":
        return ""
    content = message.get("content")
    return content if isinstance(content, str) else ""
