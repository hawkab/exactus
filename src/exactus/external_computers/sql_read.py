"""External computer using read-only SQLite over explicit ephemeral tables.

Capabilities are denied at SQLite authorization time, not by a SELECT regex.
Run through WorkerExternalComputer for address-space, frame and wall-clock isolation.
"""

__author__ = "Olshansky Grigory"

import sqlite3
import time

from ..domain.errors import BudgetError, DataError, GrammarError
from .sql_connection import _prepare as _prepare
from .sql_values import _NAME, _cell
from .sql_values import identifier as identifier
from .values import packed

_ARITHMETIC = frozenset(
    {
        "Add",
        "Subtract",
        "Multiply",
        "Divide",
        "Remainder",
        "Real",
        "RealAffinity",
        "Cast",
    }
)


class SQLSession:
    """Private request-local DB reuse; never shared across requests/users."""

    def __init__(self, *, _owned=False):
        """Initialize SQLSession state and dependencies."""
        self._connections = {}
        self._owned, self._last = _owned, None

    def reuses(self, tables):
        """Allow identity reuse only for private snapshots owned by the current request."""
        return self._owned and self._last is not None and self._last[0] is tables

    def connection(self, tables):
        """Reuse a canonical table snapshot or prepare a fresh read-only database."""
        if self.reuses(tables):
            return self._last[1]
        key = packed(tables)
        if key not in self._connections:
            if len(self._connections) == 4:
                self._connections.pop(next(iter(self._connections))).close()
            self._connections[key] = _prepare(tables)
        conn = self._connections[key]
        self._last = (tables, conn) if self._owned else None
        return conn

    def close(self):
        """Close all request-owned databases and discard identity reuse state."""
        for connection in self._connections.values():
            connection.close()
        self._connections.clear()
        self._last = None

    def __enter__(self):
        """Enter this managed context and return its active resource."""
        return self

    def __exit__(self, *exc):
        """Release owned resources when leaving the managed context."""
        self.close()


def evaluate(query, tables, parameters=None, *, session=None):
    """Execute a read-only query over explicit, ephemeral tables.

    SQLite authorization controls capabilities; opcode inspection independently
    rejects implicit floating-point arithmetic and casts.

    Args:
        query: Read-only SQL returning a result set.
        tables: Named tables containing explicit columns and scalar rows.
        parameters: Positional or named scalar bindings.
        session: Optional request-local connection owner.

    Returns:
        A JSON object containing result column names and rows.

    Raises:
        GrammarError: SQLite rejects a query or unsupported arithmetic.
        DataError: Tables, identifiers, or parameter values are invalid.
        BudgetError: Query work or output exceeds its execution contract.
    """
    if type(query) is not str or not 0 < len(query) <= 8192:
        raise BudgetError("SQL query must contain 1..8192 characters")
    if type(tables) is not dict or not 1 <= len(tables) <= 8:
        raise DataError("SQL requires 1..8 explicit tables")
    parameters = [] if parameters is None else parameters
    if type(parameters) not in (list, dict) or len(parameters) > 128:
        raise DataError(
            "SQL parameters must be a list or named map of at most 128 cells"
        )
    if type(parameters) is dict and any(
        type(k) is not str or not _NAME.fullmatch(k) for k in parameters
    ):
        raise DataError(
            "SQL parameter names must be public ASCII names under 49 characters"
        )
    if session is not None and session.reuses(tables):
        packed(parameters)
    else:
        packed({"tables": tables, "parameters": parameters})
    conn = cursor = None
    try:
        conn = session.connection(tables) if session is not None else _prepare(tables)
        for cell in parameters.values() if type(parameters) is dict else parameters:
            _cell(cell)
        expires, steps = time.monotonic() + 0.25, 0

        def progress():
            """Interrupt a SQL virtual machine when its configured work contract is exceeded."""
            nonlocal steps
            steps += 100
            return int(steps > 100000 or time.monotonic() >= expires)

        conn.set_progress_handler(progress, 100)

        for instruction in conn.execute("EXPLAIN " + query, parameters):
            if instruction[1] in _ARITHMETIC:
                raise GrammarError(
                    "SQL REAL/CAST and unchecked arithmetic are unsupported; "
                    "use int_add/int_sub/int_mul/int_div/int_mod or python_expr"
                )
        cursor = conn.execute(query, parameters)
        if cursor.description is None:
            raise GrammarError("SQL must return a result set")
        rows = [list(row) for row in cursor.fetchmany(513)]
        if len(rows) > 512:
            raise BudgetError("SQL output exceeds 512 rows; add a selective LIMIT")
        value = {"columns": [col[0] for col in cursor.description], "rows": rows}
        packed(value)
        return value
    except (sqlite3.Error, sqlite3.Warning) as exc:
        if "interrupted" in str(exc):
            raise BudgetError("SQL VM work/deadline budget exceeded") from exc
        raise GrammarError(f"SQL rejected: {exc}") from exc
    except MemoryError as exc:
        raise BudgetError("SQL memory/opcode budget exceeded") from exc
    finally:
        if cursor is not None:
            cursor.close()
        if conn is not None and session is None:
            conn.close()
