"""Compile a bounded external computer DAG and render its answer without an LLM.

PoT: semantic task names and source-bound inputs. CoA: abstract references in
the answer. Execution is event-triggered, never a per-decode-token callback.
"""

from collections import deque
from dataclasses import dataclass
import hashlib
import json
import operator
import re
import time

from ..domain.errors import DataError, VerificationError
from .batch import execute, parse
from .values import OwnedJSON, packed

_ID = re.compile(r"[a-zA-Z][a-zA-Z0-9_]{0,47}\Z")
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


def select(value, path):
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


@dataclass(frozen=True, slots=True)
class CompiledPlan:
    ordered: tuple
    pruned: tuple
    answer: object


def compile_plan(plan, inputs):
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
        parse(spec)  # enforce the task kind/field schema, including dead code
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


def validate_checks(checks):
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


def run_plan(plan, external_computer, *, inputs=None, checks=None):
    start = time.monotonic()
    inputs, checks = (
        ({} if inputs is None else inputs),
        ([] if checks is None else checks),
    )
    if type(inputs) is not dict:
        raise DataError("plan inputs must be an object")
    canonical = packed({"plan": plan, "inputs": inputs, "checks": checks})
    snapshot = json.loads(canonical)
    plan, inputs, checks = snapshot["plan"], snapshot["inputs"], snapshot["checks"]
    compiled = compile_plan(plan, inputs)
    # Validate all contract syntax before executing a potentially expensive task.
    validate_checks(checks)
    outputs, cache, rows = {}, {}, []
    # Small scalar plans are cheaper with the C encoder/boundary validator;
    # larger DAGs profit from sealed-child measurements and incremental rows.
    encoding = OwnedJSON() if len(compiled.ordered) > 1 else None
    encode = encoding.encode if encoding is not None else packed
    aggregate = encoding.sequence() if encoding is not None else None
    from .sql_read import SQLSession

    with SQLSession(_owned=True) as sql_session:
        for name, abstract in compiled.ordered:
            spec = resolve(abstract, outputs, inputs)
            digest = hashlib.sha256(
                ("exactus-external-computer-v1\0" + encode(spec)).encode()
            ).hexdigest()
            cached = digest in cache
            value = (
                cache[digest]
                if cached
                else execute(parse(spec), external_computer, sql_session)
            )
            outputs[name] = cache[digest] = value
            rows.append(
                {
                    "id": name,
                    "kind": spec["kind"],
                    "value": value,
                    "input_sha256": digest,
                    "cached": cached,
                }
            )
            if aggregate is not None:
                aggregate.append(rows[-1])
            else:
                packed(value)
    answer = resolve(compiled.answer, outputs, inputs, template=True)
    encoded_answer = encode(answer)
    check_answer(answer, checks)
    result = {
        "answer": answer,
        "results": rows,
        "pruned_ids": list(compiled.pruned),
        "verified_checks": len(checks),
        "model_calls": 0,
        "sampled_tokens": 0,
        "plan_sha256": hashlib.sha256(
            ("exactus-plan-v1\0" + canonical).encode()
        ).hexdigest(),
        "answer_sha256": hashlib.sha256(encoded_answer.encode()).hexdigest(),
        "verification_scope": "supplied_program_and_caller_contracts",
    }
    encode(result)
    result["elapsed_s"] = time.monotonic() - start
    return result


def decode_plan(text):
    if type(text) is not str:
        raise DataError("model plan must be text under 32 KiB")
    try:
        size = len(text.encode("utf-8"))
    except UnicodeError as exc:
        raise DataError("model plan must be valid UTF-8") from exc
    if size > 32768:
        raise DataError("model plan must be text under 32 KiB")
    text = text.strip()
    if text.startswith("```json\n") and text.endswith("\n```"):
        text = text[8:-4]

    def unique(pairs):
        value = {}
        for key, item in pairs:
            if key in value:
                raise DataError("duplicate model plan JSON key")
            value[key] = item
        return value

    try:
        plan = json.loads(text, object_pairs_hook=unique)
        packed(plan)
    except (ValueError, RecursionError) as exc:
        raise DataError("model did not return a valid JSON plan") from exc
    return plan
