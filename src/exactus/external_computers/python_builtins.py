"""Execute pure builtins and lazy comprehensions with shared work accounting."""

__author__ = "Olshansky Grigory"

import ast

from ..domain.errors import BudgetError, GrammarError
from .python_syntax import _BUILTINS, _FUNCTIONS
from .values import MAX_ITEMS, bounded


class _LazyItems:
    """Private interpreter iterator: never accepted as input or JSON output."""

    __slots__ = ("_items",)

    def __init__(self, items):
        """Initialize Lazy Items state and dependencies."""
        self._items = items

    def __iter__(self):
        """Return the iterator without eagerly consuming its values."""
        return self._items


def comprehension(interpreter, node, env, depth):
    """Evaluate one bounded comprehension and retain lazy generator semantics."""
    child_depth = depth + 1
    if len(node.generators) != 1:
        raise GrammarError("only single-generator comprehensions are supported")
    gen = node.generators[0]
    if (
        gen.is_async
        or not isinstance(gen.target, ast.Name)
        or gen.target.id.startswith("_")
        or gen.target.id in _FUNCTIONS
    ):
        raise GrammarError("invalid comprehension target")
    source = interpreter.visit(gen.iter, env, child_depth)
    if type(source) not in (list, str, dict):
        raise GrammarError("bounded JSON iterable required")
    local = dict(env)

    def items():
        """Yield filtered elements while charging interpreter work per source item."""
        for item in source:
            interpreter.tick(depth)
            local[gen.target.id] = item
            if all(interpreter.visit(test, local, depth + 1) for test in gen.ifs):
                yield interpreter.visit(node.elt, local, depth + 1)

    iterator = _LazyItems(items())
    return iterator if isinstance(node, ast.GeneratorExp) else bounded(list(iterator))


def call(interpreter, node, env, depth):
    """Execute an allowed pure builtin while accounting for work and short-circuiting."""
    child_depth = depth + 1
    if (
        not isinstance(node.func, ast.Name)
        or node.func.id not in _FUNCTIONS
        or node.keywords
    ):
        raise GrammarError("only allowlisted pure positional calls are supported")
    args = [interpreter.visit(a, env, child_depth) for a in node.args]
    name = node.func.id
    if name == "range":
        if not 1 <= len(args) <= 3 or any(type(a) is not int for a in args):
            raise GrammarError("range requires 1..3 integers")
        value = range(*args)
        if len(value) > MAX_ITEMS:
            raise BudgetError("Python range exceeds item budget")
        interpreter.spend(len(value))
        return list(value)
    if name == "sum":
        if len(args) != 1 or type(args[0]) not in (list, _LazyItems):
            raise GrammarError("sum requires bounded integers")
        values = args[0]
        if type(values) is list:
            if any(type(v) is not int for v in values):
                raise GrammarError("sum requires bounded integers")
            interpreter.spend(len(values))
            return bounded(sum(values))

        def integers():
            """Yield validated integer operands and charge their consumption."""
            for value in values:
                interpreter.spend(1)
                if type(value) is not int:
                    raise GrammarError("sum requires bounded integers")
                yield value

        return bounded(sum(integers()))
    if len(args) == 1 and type(args[0]) is _LazyItems:
        if name not in ("min", "max", "sorted", "all", "any"):
            raise GrammarError("generator requires an iterable consumer")
        if name == "sorted":
            values = list(args[0])
            interpreter.spend(len(values) * max(1, len(values).bit_length()))
            return bounded(sorted(values))

        return bounded(_BUILTINS[name](args[0]))
    if name in ("min", "max", "sorted", "all", "any"):
        size = (
            len(args[0])
            if len(args) == 1 and type(args[0]) in (list, str, dict)
            else len(args)
        )
        interpreter.spend(
            size * max(1, size.bit_length()) if name == "sorted" else size
        )
    return bounded(_BUILTINS[name](*args))
