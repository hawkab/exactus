"""External computer for a small Python-expression subset, not eval/exec/sandbox.

Only JSON values, local names, integer operations, comprehensions and named
pure builtins below are implemented. Imports, attributes, functions, arbitrary
objects, statements, floating point and host capabilities do not exist here.
"""

__author__ = "Olshansky Grigory"

import ast

from ..domain.errors import BudgetError, DataError, GrammarError
from .python_builtins import _LazyItems as _LazyItems
from .python_interpreter import Interpreter as Interpreter
from .python_syntax import _FUNCTIONS, _NODES
from .values import packed


def evaluate(expression, inputs):
    """Validate and interpret the supported Python-expression subset.

    Args:
        expression: One expression containing only supported syntax.
        inputs: Public names mapped to explicit JSON values.

    Returns:
        The computed JSON value, with integer arithmetic preserved exactly.

    Raises:
        GrammarError: Syntax or operand types are unsupported.
        DataError: Input names or values violate the JSON contract.
        BudgetError: Evaluation exceeds its configured execution contract.
    """
    if type(expression) is not str or not 0 < len(expression) <= 8192:
        raise BudgetError("Python expression must contain 1..8192 characters")
    if type(inputs) is not dict or any(
        type(k) is not str
        or not k.isidentifier()
        or k.startswith("_")
        or k in _FUNCTIONS
        for k in inputs
    ):
        raise DataError("Python inputs require public identifier names")
    packed(inputs)
    try:
        tree = ast.parse(expression, mode="eval")
    except (SyntaxError, RecursionError, ValueError) as exc:
        raise GrammarError("invalid Python expression") from exc
    for count, node in enumerate(ast.walk(tree), 1):
        if count > 512:
            raise BudgetError("Python expression exceeds AST node budget")
        if not isinstance(node, _NODES):
            raise GrammarError(f"unsupported Python syntax: {type(node).__name__}")
    try:
        value = Interpreter().visit(tree.body, inputs)
        packed(value)
        return value
    except (
        KeyError,
        IndexError,
        TypeError,
        ValueError,
        ZeroDivisionError,
        OverflowError,
        RecursionError,
    ) as exc:
        raise GrammarError(f"Python expression failed: {type(exc).__name__}") from exc
