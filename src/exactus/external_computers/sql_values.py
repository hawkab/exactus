"""Validate SQL identifiers, scalar cells, and checked integer arithmetic."""

__author__ = "Olshansky Grigory"

import re

from ..domain.errors import DataError

_NAME = re.compile(r"[A-Za-z][A-Za-z0-9_]{0,47}\Z")
_FUNCTIONS = frozenset(
    {
        "abs",
        "count",
        "sum",
        "min",
        "max",
        "length",
        "lower",
        "upper",
        "coalesce",
        "nullif",
        "ifnull",
        "int_add",
        "int_sub",
        "int_mul",
        "int_div",
        "int_mod",
        "substr",
        "substring",
        "trim",
        "ltrim",
        "rtrim",
    }
)


def _integer(value):
    """Validate an exact signed 64-bit SQL integer without accepting booleans."""
    if type(value) is not int or not -(2**63) <= value < 2**63:
        raise ValueError("SQL checked arithmetic requires signed int64")
    return value


def _operation(op):
    """Wrap a binary operator with checked integer input and output validation."""

    def call(a, b):
        """Apply the operator only to validated signed 64-bit operands."""
        return _integer(op(_integer(a), _integer(b)))

    return call


class _IntegerSum:
    """Aggregate SQL integers without implicit floating-point conversion."""

    def __init__(self):
        """Initialize Integer Sum state and dependencies."""
        self.total, self.seen = 0, False

    def step(self, value):
        """Accumulate one non-null checked integer."""
        if value is not None:
            self.total += _integer(value)
            self.seen = True

    def finalize(self):
        """Return the checked total or SQL null when no value was seen."""
        return _integer(self.total) if self.seen else None


def identifier(name):
    """Validate and quote a public SQL identifier."""
    if (
        type(name) is not str
        or not _NAME.fullmatch(name)
        or name.lower().startswith("sqlite_")
    ):
        raise DataError(
            "SQL identifiers must be public ASCII names under 49 characters"
        )
    return '"' + name + '"'


def _cell(cell):
    """Reject unsupported SQL cells and integers outside the signed 64-bit range."""
    if cell is not None and type(cell) not in (str, int, bool):
        raise DataError("SQL cells must be null/string/bool/int64")
    if type(cell) is int and not -(2**63) <= cell < 2**63:
        raise DataError("SQL integers must fit signed int64")
