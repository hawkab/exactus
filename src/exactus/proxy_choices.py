"""Validate the upstream choice and tool-call protocol."""

from __future__ import annotations

__author__ = "Olshansky Grigory"

from .request_errors import RequestError


def validate_choices(choices):
    """Validate completion and tool-call structure before any computation."""
    tool_stages = []
    for choice in choices:
        if not isinstance(choice, dict) or choice.get("finish_reason") not in (
            "stop",
            "tool_calls",
        ):
            raise RequestError(
                "upstream generation did not complete", "incomplete_generation"
            )
        message = choice.get("message")
        if not isinstance(message, dict) or not isinstance(
            message.get("content"), (str, type(None))
        ):
            raise RequestError(
                "upstream message must have text or null content",
                "upstream_error",
                502,
            )
        calls = message.get("tool_calls")
        intermediate = bool(calls) or choice["finish_reason"] == "tool_calls"
        if (
            (calls is not None and not isinstance(calls, list))
            or intermediate
            and (
                not isinstance(calls, list)
                or not 1 <= len(calls) <= 128
                or any(
                    not isinstance(c, dict)
                    or not isinstance(c.get("id"), str)
                    or not isinstance(c.get("function"), dict)
                    or not isinstance(c["function"].get("name"), str)
                    or not isinstance(c["function"].get("arguments"), str)
                    for c in calls
                )
            )
        ):
            raise RequestError("invalid upstream tool calls", "upstream_error", 502)
        tool_stages.append(intermediate)
    return tool_stages
