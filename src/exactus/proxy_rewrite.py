"""Pure request and response rewriting helpers for the proxy tier.

Author: Olshansky Grigory
"""

from __future__ import annotations

__author__ = "Olshansky Grigory"


def prepare_request(payload: dict, hint: str) -> dict:
    """Return the payload with the marker hint injected into the system role."""
    if not hint:
        return payload
    messages = payload.get("messages")
    if not isinstance(messages, list):
        return payload
    if any(
        isinstance(m, dict)
        and m.get("role") == "system"
        and "<calc-final>" in str(m.get("content", ""))
        for m in messages
    ):
        return payload
    augmented = [{"role": "system", "content": hint}] + messages
    return {**payload, "messages": augmented}


def extract_content(response: dict) -> str:
    """Return the assistant text of the first choice."""
    choices = response.get("choices") or []
    if not choices:
        return ""
    message = choices[0].get("message") or {}
    return str(message.get("content") or "")


def replace_content(response: dict, text: str) -> dict:
    """Return the response with the first choice text replaced."""
    choices = response.get("choices") or []
    if not choices:
        return response
    message = dict(choices[0].get("message") or {})
    message["content"] = text
    updated = dict(choices[0])
    updated["message"] = message
    return {**response, "choices": [updated, *choices[1:]]}


def map_choices(response: dict, transform) -> dict:
    """Return the response with every choice content transformed."""
    updated = []
    for choice in response.get("choices") or []:
        entry = dict(choice) if isinstance(choice, dict) else {}
        message = dict(entry.get("message") or {})
        message["content"] = transform(str(message.get("content") or ""))
        entry["message"] = message
        updated.append(entry)
    return {**response, "choices": updated}
