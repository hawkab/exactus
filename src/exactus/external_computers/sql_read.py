"""External computer using read-only SQLite over explicit ephemeral tables.

Capabilities are denied at SQLite authorization time, not by a SELECT regex.
Run through WorkerExternalComputer for address-space, frame and wall-clock isolation.
"""

import re
import sqlite3
import time

from ..domain.errors import BudgetError, DataError, GrammarError
from .values import packed

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


def _integer(value):
    if type(value) is not int or not -(2**63) <= value < 2**63:
        raise ValueError("SQL checked arithmetic requires signed int64")
    return value


def _operation(op):
    def call(a, b):
        return _integer(op(_integer(a), _integer(b)))

    return call


class _IntegerSum:
    def __init__(self):
        self.total, self.seen = 0, False

    def step(self, value):
        if value is not None:
            self.total += _integer(value)
            self.seen = True

    def finalize(self):
        return _integer(self.total) if self.seen else None


def identifier(name):
    if (
        type(name) is not str
        or not _NAME.fullmatch(name)
        or name.lower().startswith("sqlite_")
    ):
        raise DataError(
            "SQL identifiers must be public ASCII names under 49 characters"
        )
    return '"' + name + '"'


class SQLSession:
    """Private request-local DB reuse; never shared across requests/users."""

    def __init__(self, *, _owned=False):
        self._connections = {}
        self._owned, self._last = _owned, None

    def reuses(self, tables):
        # Only batch/plan callers with private, sealed JSON snapshots enable
        # identity reuse. Ordinary callers may mutate their tables: re-key them.
        return self._owned and self._last is not None and self._last[0] is tables

    def connection(self, tables):
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
        for connection in self._connections.values():
            connection.close()
        self._connections.clear()
        self._last = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def _prepare(tables):
    import operator

    conn = sqlite3.connect(":memory:", cached_statements=0)
    try:
        for name, op in (
            ("int_add", operator.add),
            ("int_sub", operator.sub),
            ("int_mul", operator.mul),
            ("int_div", operator.floordiv),
            ("int_mod", operator.mod),
        ):
            conn.create_function(name, 2, _operation(op), deterministic=True)
        conn.create_function(
            "abs", 1, lambda value: _integer(abs(_integer(value))), deterministic=True
        )
        conn.create_aggregate("sum", 1, _IntegerSum)
        conn.execute("PRAGMA temp_store=MEMORY")
        conn.execute("PRAGMA trusted_schema=OFF")
        if hasattr(conn, "setlimit"):
            for category, limit in (
                (sqlite3.SQLITE_LIMIT_LENGTH, 65536),
                (sqlite3.SQLITE_LIMIT_SQL_LENGTH, 8192),
                (sqlite3.SQLITE_LIMIT_COLUMN, 64),
                (sqlite3.SQLITE_LIMIT_EXPR_DEPTH, 32),
                (sqlite3.SQLITE_LIMIT_COMPOUND_SELECT, 16),
                (sqlite3.SQLITE_LIMIT_VDBE_OP, 20000),
                (sqlite3.SQLITE_LIMIT_VARIABLE_NUMBER, 128),
                (sqlite3.SQLITE_LIMIT_ATTACHED, 0),
            ):
                conn.setlimit(category, limit)
        total = 0
        for name, table in tables.items():
            qname = identifier(name)
            if type(table) is not dict or set(table) != {"columns", "rows"}:
                raise DataError("SQL table requires columns and rows only")
            columns, rows = table["columns"], table["rows"]
            if (
                type(columns) is not list
                or not 1 <= len(columns) <= 32
                or type(rows) is not list
            ):
                raise DataError("invalid SQL table shape")
            quoted = [identifier(c) for c in columns]
            if len({c.lower() for c in columns}) != len(columns):
                raise DataError("duplicate SQL columns")
            total += len(rows)
            if total > 2048:
                raise BudgetError("SQL input exceeds 2048 rows")
            conn.execute(f"CREATE TABLE {qname} ({','.join(quoted)})")
            for row in rows:
                if type(row) is not list or len(row) != len(columns):
                    raise DataError("SQL row width does not match columns")
                for cell in row:
                    _cell(cell)
            conn.executemany(
                f"INSERT INTO {qname} VALUES ({','.join('?' for _ in columns)})", rows
            )
        conn.commit()
        conn.execute("PRAGMA query_only=ON")
        names = frozenset(tables)

        def authorize(action, a, b, db, source):
            if action in (sqlite3.SQLITE_SELECT, sqlite3.SQLITE_RECURSIVE):
                return sqlite3.SQLITE_OK
            if (
                action == sqlite3.SQLITE_READ
                and a in names
                and (db == "main" or (db is None and b == ""))
            ):
                return sqlite3.SQLITE_OK
            if action == sqlite3.SQLITE_FUNCTION and (b or "").lower() in _FUNCTIONS:
                return sqlite3.SQLITE_OK
            return sqlite3.SQLITE_DENY

        conn.set_authorizer(authorize)
        return conn
    except BaseException:
        conn.close()
        raise


def evaluate(query, tables, parameters=None, *, session=None):
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
            nonlocal steps
            steps += 100
            return int(steps > 100000 or time.monotonic() >= expires)

        conn.set_progress_handler(progress, 100)
        # Authorizer controls capabilities. Opcode guard independently refuses
        # implicit REAL arithmetic/CAST even when the final output is an int.
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


def _cell(cell):
    if cell is not None and type(cell) not in (str, int, bool):
        raise DataError("SQL cells must be null/string/bool/int64")
    if type(cell) is int and not -(2**63) <= cell < 2**63:
        raise DataError("SQL integers must fit signed int64")
