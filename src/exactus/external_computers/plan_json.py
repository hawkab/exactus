"""Decode model-authored plans without accepting duplicate JSON keys."""

__author__ = "Olshansky Grigory"

import json

from ..domain.errors import DataError
from .values import packed


def decode_plan(text):
    """Decode UTF-8 plan JSON while rejecting duplicate keys and trailing content."""
    if type(text) is not str:
        raise DataError("model plan must be text under 32 KiB")
    try:
        size = len(text.encode("utf-8"))
    except UnicodeError as exc:
        raise DataError("model plan must be valid UTF-8") from exc
    if size > 32768:
        raise DataError("model plan must be text under 32 KiB")
    text = text.strip()
    if text.startswith("```json\n") and text.endswith("\n```"):
        text = text[8:-4]

    def unique(pairs):
        """Build an object only when each JSON key occurs once."""
        value = {}
        for key, item in pairs:
            if key in value:
                raise DataError("duplicate model plan JSON key")
            value[key] = item
        return value

    try:
        plan = json.loads(text, object_pairs_hook=unique)
        packed(plan)
    except (ValueError, RecursionError) as exc:
        raise DataError("model did not return a valid JSON plan") from exc
    return plan
