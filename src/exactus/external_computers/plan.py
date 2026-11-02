"""Compile a bounded external computer DAG and render its answer without an LLM.

PoT: semantic task names and source-bound inputs. CoA: abstract references in
the answer. Execution is event-triggered, never a per-decode-token callback.
"""

__author__ = "Olshansky Grigory"

import hashlib
import json
import time

from ..domain.errors import DataError
from .plan_checks import (
    check_answer as check_answer,
)
from .plan_checks import (
    validate_checks as validate_checks,
)
from .plan_compiler import CompiledPlan as CompiledPlan
from .plan_compiler import compile_plan as compile_plan
from .plan_json import decode_plan as decode_plan
from .plan_references import (
    references as references,
)
from .plan_references import (
    resolve as resolve,
)
from .plan_references import (
    select as select,
)
from .task_execution import TaskExecution
from .values import OwnedJSON, packed


def run_plan(plan, external_computer, *, inputs=None, checks=None):
    """Execute a graph and verify its rendered answer against caller contracts.

    All task schemas and check syntax are validated before execution, including
    tasks pruned from the answer dependency closure. Result reuse is confined
    to this request. Single-task plans retain the direct canonical encoder.

    Args:
        plan: Task graph and an answer template referencing computed values.
        external_computer: Exact integer evaluator for calculator operations.
        inputs: Named JSON values snapshotted before execution.
        checks: Typed predicates over paths in the rendered answer.

    Returns:
        The answer, result evidence, pruned IDs, and canonical digests.

    Raises:
        DataError: Graph, inputs, references, or check syntax are invalid.
        VerificationError: A computed answer fails a caller-supplied predicate.
    """
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

    validate_checks(checks)

    encoding = OwnedJSON() if len(compiled.ordered) > 1 else None
    encode = encoding.encode if encoding is not None else packed
    aggregate = encoding.sequence() if encoding is not None else None
    from .sql_read import SQLSession

    with SQLSession(_owned=True) as sql_session:
        execution = TaskExecution(external_computer, sql_session, encode, aggregate)
        for name, abstract in compiled.ordered:
            spec = resolve(abstract, execution.outputs, inputs)
            execution.run(name, spec)
    answer = resolve(compiled.answer, execution.outputs, inputs, template=True)
    encoded_answer = encode(answer)
    check_answer(answer, checks)
    result = {
        "answer": answer,
        "results": execution.rows,
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
