"""Validate typed operation specifications and dispatch CPU evaluators."""

__author__ = "Olshansky Grigory"

from dataclasses import dataclass

from ..domain.errors import DataError


@dataclass(frozen=True, slots=True)
class PythonTask:
    """Describe a pure Python expression and its explicit JSON inputs."""

    expression: str
    inputs: dict


@dataclass(frozen=True, slots=True)
class SQLTask:
    """Describe a read-only SQL query, explicit tables, and bound parameters."""

    query: str
    tables: dict
    parameters: list | dict


@dataclass(frozen=True, slots=True)
class CalcTask:
    """Describe one exact integer expression."""

    expression: str


def parse(spec):
    """Validate task fields and construct the corresponding typed operation."""
    kind = spec.get("kind")
    if (
        kind == "python_expr"
        and set(spec) <= {"kind", "expression", "inputs"}
        and "expression" in spec
    ):
        return PythonTask(spec["expression"], spec.get("inputs", {}))
    if (
        kind == "sql_read"
        and set(spec) <= {"kind", "query", "tables", "parameters"}
        and {"query", "tables"} <= set(spec)
    ):
        return SQLTask(spec["query"], spec["tables"], spec.get("parameters", []))
    if kind == "calc" and set(spec) == {"kind", "expression"}:
        return CalcTask(spec["expression"])
    raise DataError("unsupported external computer kind or task fields")


def execute(task, external_computer, sql_session=None):
    """Dispatch a typed task to its concrete CPU evaluator."""
    if isinstance(task, PythonTask):
        from .python_expr import evaluate

        return evaluate(task.expression, task.inputs)
    if isinstance(task, SQLTask):
        from .sql_read import evaluate

        return evaluate(task.query, task.tables, task.parameters, session=sql_session)
    if isinstance(task, CalcTask):
        return external_computer.evaluate(task.expression).integer
    raise DataError("unsupported external computer task")
