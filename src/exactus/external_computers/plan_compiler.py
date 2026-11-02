"""Validate, order, and prune a source-bound computation graph."""

__author__ = "Olshansky Grigory"

import re
from collections import deque
from dataclasses import dataclass

from ..domain.errors import DataError
from .batch import parse
from .plan_references import references

_ID = re.compile(r"[a-zA-Z][a-zA-Z0-9_]{0,47}\Z")


@dataclass(frozen=True, slots=True)
class CompiledPlan:
    """Store ordered live tasks, pruned task identifiers, and the answer template."""

    ordered: tuple
    pruned: tuple
    answer: object


def compile_plan(plan, inputs):
    """Validate all tasks, detect cycles, and retain the answer dependency closure."""
    if type(plan) is not dict or set(plan) != {"tasks", "answer"}:
        raise DataError("plan requires tasks and answer only")
    tasks = plan["tasks"]
    if type(tasks) is not list or not 1 <= len(tasks) <= 16:
        raise DataError("plan requires 1..16 tasks")
    specs, deps = {}, {}
    for row in tasks:
        if (
            type(row) is not dict
            or type(row.get("id")) is not str
            or not _ID.fullmatch(row["id"])
        ):
            raise DataError("invalid semantic task id")
        ident = row["id"]
        if ident in specs:
            raise DataError("duplicate plan task id")
        spec = {k: v for k, v in row.items() if k != "id"}
        parse(spec)
        deps[ident] = references(spec, inputs)
        specs[ident] = spec
    roots = references(plan["answer"], inputs, template=True)
    if (roots | set().union(*deps.values())) - specs.keys():
        raise DataError("plan has unknown task references")
    incoming = {name: len(d) for name, d in deps.items()}
    consumers = {name: [] for name in specs}
    for name, parents in deps.items():
        for parent in parents:
            consumers[parent].append(name)
    ready = deque(name for name in specs if not incoming[name])
    order = []
    while ready:
        name = ready.popleft()
        order.append(name)
        for child in consumers[name]:
            incoming[child] -= 1
            if not incoming[child]:
                ready.append(child)
    if len(order) != len(specs):
        raise DataError("cyclic plan dependencies")
    live = set(roots)
    pending = list(roots)
    while pending:
        for dep in deps[pending.pop()]:
            if dep not in live:
                live.add(dep)
                pending.append(dep)
    return CompiledPlan(
        tuple((name, specs[name]) for name in order if name in live),
        tuple(name for name in order if name not in live),
        plan["answer"],
    )
