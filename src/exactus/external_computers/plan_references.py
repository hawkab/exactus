"""Resolve source-bound references and answer templates."""

__author__ = "Olshansky Grigory"

from ..domain.errors import DataError
from .values import packed


def select(value, path):
    """Select JSON keys or indices without exposing Python attributes."""
    if type(path) is not list or len(path) > 16:
        raise DataError("reference path must be a list of at most 16 keys/indices")
    for key in path:
        if type(value) is dict and type(key) is str and key in value:
            value = value[key]
        elif type(value) is list and type(key) is int and 0 <= key < len(value):
            value = value[key]
        else:
            raise DataError("reference path does not exist or has the wrong type")
    return value


def _reference(node, key, scope):
    """Validate and resolve a reference within its declared input or output scope."""
    if (
        set(node) - {key, "path"}
        or type(node[key]) is not str
        or node[key] not in scope
    ):
        raise DataError(f"invalid or unknown {key} reference")
    return select(scope[node[key]], node.get("path", []))


def resolve(value, outputs, inputs, *, template=False):
    """Interpret only the plan syntax, never syntax inside referenced data."""
    if type(value) is list:
        return [resolve(v, outputs, inputs, template=template) for v in value]
    if type(value) is not dict:
        return value
    if "$ref" in value:
        return _reference(value, "$ref", outputs)
    if "$input" in value:
        return _reference(value, "$input", inputs)
    if "$if" in value and template:
        if set(value) != {"$if", "then", "else"}:
            raise DataError("answer conditional requires $if/then/else")
        test = resolve(value["$if"], outputs, inputs, template=True)
        if type(test) is not bool:
            raise DataError("answer conditional requires a computed boolean")
        return resolve(
            value["then"] if test else value["else"], outputs, inputs, template=True
        )
    if "$text" in value and template:
        if set(value) != {"$text"} or type(value["$text"]) is not list:
            raise DataError("$text requires a list of literal/reference parts")
        parts = []
        size = 0
        for part in value["$text"]:
            item = resolve(part, outputs, inputs, template=True)
            text = item if type(item) is str else packed(item)
            size += len(text.encode("utf-8"))
            if size > 65536:
                raise DataError("answer text exceeds 64 KiB")
            parts.append(text)
        return "".join(parts)
    if any(k.startswith("$") for k in value):
        raise DataError("unknown plan/template directive")
    return {k: resolve(v, outputs, inputs, template=template) for k, v in value.items()}


def references(value, inputs, *, template=False):
    """Validate static reference syntax before any task is executed."""
    found = set()
    if template and type(value) in (int, bool):
        raise DataError("numeric/boolean answer literals must be computed references")
    if type(value) is list:
        for item in value:
            found.update(references(item, inputs, template=template))
    elif type(value) is dict:
        if "$input" in value:
            _reference(value, "$input", inputs)
        elif "$ref" in value:
            if set(value) - {"$ref", "path"} or type(value["$ref"]) is not str:
                raise DataError("invalid result reference")
            path = value.get("path", [])
            if (
                type(path) is not list
                or len(path) > 16
                or any(
                    type(k) not in (int, str) or (type(k) is int and k < 0)
                    for k in path
                )
            ):
                raise DataError("invalid result reference path")
            found.add(value["$ref"])
        elif template and "$if" in value:
            if set(value) != {"$if", "then", "else"}:
                raise DataError("invalid answer conditional")
            for item in value.values():
                found.update(references(item, inputs, template=True))
        elif template and "$text" in value:
            if set(value) != {"$text"} or type(value["$text"]) is not list:
                raise DataError("invalid answer text")
            found.update(references(value["$text"], inputs, template=True))
        else:
            if any(k.startswith("$") for k in value):
                raise DataError("unknown plan directive")
            for item in value.values():
                found.update(references(item, inputs, template=template))
    return found
