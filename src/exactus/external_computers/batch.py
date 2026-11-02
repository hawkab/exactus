"""Typed external computer tasks and one bounded, request-local dependent batch.

This is execution evidence for the supplied program/data, not a proof that a
model chose the correct program for a natural-language problem. There is no
automatic substring extraction, remote DB connection, shared result cache or
retry with a guessed answer. Unsupported operations fail closed.
"""

__author__ = "Olshansky Grigory"

import json
import re
import time

from ..domain.errors import DataError
from .task_execution import TaskExecution
from .task_specs import CalcTask as CalcTask
from .task_specs import PythonTask as PythonTask
from .task_specs import SQLTask as SQLTask
from .task_specs import execute as execute
from .task_specs import parse as parse
from .values import OwnedJSON, packed

_ID = re.compile(r"[a-zA-Z][a-zA-Z0-9_]{0,47}\Z")


def _resolve(value, outputs):
    """Resolve backward-only task references without accepting cycles or forward edges."""
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
    """Execute a dependent batch using a private snapshot of its inputs.

    Args:
        tasks: Typed operations with unique IDs and backward-only references.
        external_computer: Exact integer evaluator used by calculator tasks.

    Returns:
        Result rows, canonical input digests, and execution accounting.

    Raises:
        DataError: A task or dependency does not satisfy the batch contract.
        ExactusError: A requested computation cannot be completed.
    """
    if type(tasks) is not list or not 1 <= len(tasks) <= 16:
        raise DataError("external computer batch must contain 1..16 tasks")
    tasks = json.loads(packed(tasks))
    encoding = OwnedJSON()
    aggregate = encoding.sequence()
    start = time.monotonic()
    from .sql_read import SQLSession

    with SQLSession(_owned=True) as sql_session:
        execution = TaskExecution(
            external_computer, sql_session, encoding.encode, aggregate
        )
        for task in tasks:
            if (
                type(task) is not dict
                or type(task.get("id")) is not str
                or not _ID.fullmatch(task["id"])
            ):
                raise DataError("external computer task requires a public id")
            ident = task["id"]
            if ident in execution.outputs:
                raise DataError("duplicate external computer task id")
            spec = _resolve(
                {k: v for k, v in task.items() if k != "id"}, execution.outputs
            )
            execution.run(ident, spec)
    return {
        "results": execution.rows,
        "guarantee": "execution_of_supplied_program_and_data_only",
        "model_calls": 0,
        "sampled_tokens": 0,
        "engine_bypassed": True,
        "elapsed_s": time.monotonic() - start,
    }
