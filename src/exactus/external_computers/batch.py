"""Typed external computer tasks and one bounded, request-local dependent batch.

This is execution evidence for the supplied program/data, not a proof that a
model chose the correct program for a natural-language problem. There is no
automatic substring extraction, remote DB connection, shared result cache or
retry with a guessed answer. Unsupported operations fail closed.
"""

import hashlib
import json
import re
import time
from dataclasses import dataclass

from ..domain.errors import DataError
from .values import OwnedJSON, packed

_ID = re.compile(r"[a-zA-Z][a-zA-Z0-9_]{0,47}\Z")


@dataclass(frozen=True, slots=True)
class PythonTask:
    expression: str
    inputs: dict


@dataclass(frozen=True, slots=True)
class SQLTask:
    query: str
    tables: dict
    parameters: list | dict


@dataclass(frozen=True, slots=True)
class CalcTask:
    expression: str


def parse(spec):
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
    if isinstance(task, PythonTask):
        from .python_expr import evaluate

        return evaluate(task.expression, task.inputs)
    if isinstance(task, SQLTask):
        from .sql_read import evaluate

        return evaluate(task.query, task.tables, task.parameters, session=sql_session)
    if isinstance(task, CalcTask):
        return external_computer.evaluate(task.expression).integer
    raise DataError("unsupported external computer task")


def _resolve(value, outputs):
    if type(value) is list:
        return [_resolve(item, outputs) for item in value]
    if type(value) is dict:
        if "$ref" in value:
            if (
                set(value) != {"$ref"}
                or type(value["$ref"]) is not str
                or value["$ref"] not in outputs
            ):
                raise DataError(
                    "external computer reference must name a previous task; cycles/forward refs forbidden"
                )
            return outputs[value["$ref"]]
        return {k: _resolve(v, outputs) for k, v in value.items()}
    return value


def run_external_computers(tasks, external_computer):
    if type(tasks) is not list or not 1 <= len(tasks) <= 16:
        raise DataError("external computer batch must contain 1..16 tasks")
    tasks = json.loads(packed(tasks))
    outputs, cache, rows = {}, {}, []
    encoding = OwnedJSON()
    aggregate = encoding.sequence()
    start = time.monotonic()
    from .sql_read import SQLSession

    with SQLSession(_owned=True) as sql_session:
        for task in tasks:
            if (
                type(task) is not dict
                or type(task.get("id")) is not str
                or not _ID.fullmatch(task["id"])
            ):
                raise DataError("external computer task requires a public id")
            ident = task["id"]
            if ident in outputs:
                raise DataError("duplicate external computer task id")
            spec = _resolve({k: v for k, v in task.items() if k != "id"}, outputs)
            canonical = encoding.encode(spec)
            parsed = parse(spec)
            digest = hashlib.sha256(
                ("exactus-external-computer-v1\0" + canonical).encode()
            ).hexdigest()
            cached = digest in cache
            value = (
                cache[digest]
                if cached
                else execute(parsed, external_computer, sql_session)
            )
            outputs[ident] = cache[digest] = value
            rows.append(
                {
                    "id": ident,
                    "kind": spec["kind"],
                    "value": value,
                    "input_sha256": digest,
                    "cached": cached,
                }
            )
            aggregate.append(rows[-1])
    return {
        "results": rows,
        "guarantee": "execution_of_supplied_program_and_data_only",
        "model_calls": 0,
        "sampled_tokens": 0,
        "engine_bypassed": True,
        "elapsed_s": time.monotonic() - start,
    }
