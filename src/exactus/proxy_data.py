"""Decode request-scoped payloads without forwarding local control fields."""

__author__ = "Olshansky Grigory"

import base64
import re
from collections.abc import Mapping

from .request_errors import RequestError

CONTROL_FIELDS = frozenset({"data", "mode"})
_MAX_PAYLOADS = 64
_MAX_PAYLOAD_BYTES = 256 * 1024
_NAME_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]{0,63}\Z")


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


def split_payload(payload: Mapping) -> tuple[dict, dict[str, bytes], object]:
    """Split one request into upstream fields, decoded data and the raw mode."""
    if not isinstance(payload, Mapping):
        raise RequestError("request body must be a JSON object", "invalid_request")
    upstream = {
        key: value for key, value in payload.items() if key not in CONTROL_FIELDS
    }
    data = decode_data(payload.get("data"))
    return upstream, data, payload.get("mode")
