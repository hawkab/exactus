"""Validate caller-supplied contracts against computed answers."""

__author__ = "Olshansky Grigory"

import operator

from ..domain.errors import DataError, VerificationError
from .plan_references import select
from .values import packed

_RELATIONS = {
    "ge": operator.ge,
    "le": operator.le,
    "gt": operator.gt,
    "lt": operator.lt,
}
_TYPES = {
    "int": int,
    "bool": bool,
    "str": str,
    "list": list,
    "dict": dict,
    "null": type(None),
}


def validate_checks(checks):
    """Validate every caller predicate before any computation starts."""
    if type(checks) is not list or len(checks) > 32:
        raise DataError("checks must contain at most 32 predicates")
    for check in checks:
        if type(check) is not dict or set(check) != {"path", "op", "value"}:
            raise DataError("check requires path/op/value")
        path, op = check["path"], check["op"]
        if (
            type(path) is not list
            or len(path) > 16
            or any(
                type(k) not in (int, str) or (type(k) is int and k < 0) for k in path
            )
        ):
            raise DataError("invalid check path")
        if type(op) is not str or op not in {"type", "eq", "ne", *_RELATIONS}:
            raise DataError("invalid check operator")
        if op == "type" and (
            type(check["value"]) is not str or check["value"] not in _TYPES
        ):
            raise DataError("invalid check type")
        if op in _RELATIONS and type(check["value"]) is not int:
            raise DataError("ordered check requires integer value")


def check_answer(answer, checks):
    """Apply typed equality, ordering, and type predicates to the computed answer."""
    validate_checks(checks)
    for index, check in enumerate(checks):
        actual = select(answer, check["path"])
        expected, op = check["value"], check["op"]
        if op == "type":
            if type(expected) is not str or expected not in _TYPES:
                raise DataError("invalid check type")
            ok = type(actual) is _TYPES[expected]
        elif op in ("eq", "ne"):
            ok = packed(actual) == packed(expected)
            ok = ok if op == "eq" else not ok
        elif type(op) is str and op in _RELATIONS:
            if type(actual) is not int or type(expected) is not int:
                raise DataError("ordered checks require integer operands")
            ok = _RELATIONS[op](actual, expected)
        else:
            raise DataError("unknown check operator")
        if not ok:
            raise VerificationError(f"answer contract failed at check {index}")
