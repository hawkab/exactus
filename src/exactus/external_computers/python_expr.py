"""External computer for a small Python-expression subset, not eval/exec/sandbox.

Only JSON values, local names, integer operations, comprehensions and named
pure builtins below are implemented. Imports, attributes, functions, arbitrary
objects, statements, floating point and host capabilities do not exist here.
"""

import ast
import operator

from ..domain.errors import BudgetError, DataError, GrammarError
from .values import MAX_BITS, MAX_BYTES, MAX_ITEMS, bounded, packed

_BINARY = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.BitAnd: operator.and_,
    ast.BitOr: operator.or_,
    ast.BitXor: operator.xor,
}
_COMPARE = {
    ast.Eq: operator.eq,
    ast.NotEq: operator.ne,
    ast.Lt: operator.lt,
    ast.LtE: operator.le,
    ast.Gt: operator.gt,
    ast.GtE: operator.ge,
    ast.In: lambda a, b: a in b,
    ast.NotIn: lambda a, b: a not in b,
}
_FUNCTIONS = frozenset(
    {"sum", "len", "min", "max", "sorted", "range", "abs", "all", "any"}
)
_UNARY = {ast.UAdd: operator.pos, ast.USub: operator.neg, ast.Invert: operator.invert}
_BUILTINS = {
    "len": len,
    "min": min,
    "max": max,
    "sorted": sorted,
    "abs": abs,
    "all": all,
    "any": any,
}
_NODES = (
    ast.Expression,
    ast.Constant,
    ast.Name,
    ast.Load,
    ast.Store,
    ast.List,
    ast.Dict,
    ast.Subscript,
    ast.BinOp,
    ast.UnaryOp,
    ast.UAdd,
    ast.USub,
    ast.Not,
    ast.Invert,
    ast.BoolOp,
    ast.And,
    ast.Or,
    ast.Compare,
    ast.IfExp,
    ast.Call,
    ast.ListComp,
    ast.GeneratorExp,
    ast.comprehension,
    *_BINARY,
    *_COMPARE,
)


def evaluate(expression, inputs):
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


class Interpreter:
    def __init__(self):
        self.remaining = 20000

    def tick(self, depth):
        self.remaining -= 1
        if self.remaining < 0 or depth > 32:
            raise BudgetError("Python work/depth budget exceeded")

    def spend(self, units):
        self.remaining -= units
        if self.remaining < 0:
            raise BudgetError("Python builtin work budget exceeded")

    def visit(self, node, env, depth=0):
        self.tick(depth)
        child_depth = depth + 1
        if isinstance(node, ast.Constant):
            return bounded(node.value)
        if isinstance(node, ast.Name):
            if node.id.startswith("_"):
                raise GrammarError("private names are forbidden")
            return env[node.id]
        if isinstance(node, ast.List):
            return bounded([self.visit(item, env, child_depth) for item in node.elts])
        if isinstance(node, ast.Dict):
            return bounded(
                {
                    self.visit(k, env, child_depth): self.visit(v, env, child_depth)
                    for k, v in zip(node.keys, node.values)
                }
            )
        if isinstance(node, ast.Subscript):
            value = self.visit(node.value, env, child_depth)
            key = self.visit(node.slice, env, child_depth)
            if type(value) not in (list, dict, str):
                raise GrammarError("only JSON subscripts are supported")
            return value[key]
        if isinstance(node, ast.UnaryOp):
            value = self.visit(node.operand, env, child_depth)
            if isinstance(node.op, ast.Not):
                return not value
            if type(value) is not int:
                raise GrammarError("integer unary operand required")
            return bounded(_UNARY[type(node.op)](value))
        if isinstance(node, ast.BinOp):
            a = self.visit(node.left, env, child_depth)
            b = self.visit(node.right, env, child_depth)
            if type(a) is int and type(b) is int:
                if isinstance(node.op, ast.Pow) and (
                    b < 0
                    or b > MAX_BITS
                    or (abs(a) > 1 and a.bit_length() * b > MAX_BITS)
                ):
                    raise BudgetError("Python power exceeds integer budget")
            elif (
                isinstance(node.op, ast.Add)
                and type(a) is type(b)
                and type(a) in (list, str)
            ):
                if len(a) + len(b) > (MAX_ITEMS if type(a) is list else MAX_BYTES):
                    raise BudgetError("Python concatenation exceeds budget")
            elif isinstance(node.op, ast.Mult) and (
                (type(a) in (list, str) and type(b) is int)
                or (type(b) in (list, str) and type(a) is int)
            ):
                seq, n = (a, b) if type(b) is int else (b, a)
                if len(seq) * max(0, n) > (
                    MAX_ITEMS if type(seq) is list else MAX_BYTES
                ):
                    raise BudgetError("Python repetition exceeds budget")
            else:
                raise GrammarError("unsupported Python operand types")
            return bounded(_BINARY[type(node.op)](a, b))
        if isinstance(node, ast.BoolOp):
            result = self.visit(node.values[0], env, child_depth)
            for item in node.values[1:]:
                if (isinstance(node.op, ast.And) and not result) or (
                    isinstance(node.op, ast.Or) and result
                ):
                    break
                result = self.visit(item, env, child_depth)
            return result
        if isinstance(node, ast.Compare):
            a = self.visit(node.left, env, child_depth)
            for op, item in zip(node.ops, node.comparators):
                b = self.visit(item, env, child_depth)
                if isinstance(op, (ast.In, ast.NotIn)) and type(b) in (str, list, dict):
                    self.spend(len(b) * (len(a) if type(a) is str else 1))
                if not _COMPARE[type(op)](a, b):
                    return False
                a = b
            return True
        if isinstance(node, ast.IfExp):
            branch = (
                node.body if self.visit(node.test, env, child_depth) else node.orelse
            )
            return self.visit(branch, env, child_depth)
        if isinstance(node, (ast.ListComp, ast.GeneratorExp)):
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
            source = self.visit(gen.iter, env, child_depth)
            if type(source) not in (list, str, dict):
                raise GrammarError("bounded JSON iterable required")
            local = dict(env)

            def items():
                for item in source:
                    self.tick(depth)
                    local[gen.target.id] = item
                    if all(self.visit(test, local, depth + 1) for test in gen.ifs):
                        yield self.visit(node.elt, local, depth + 1)

            iterator = _LazyItems(items())
            return (
                iterator
                if isinstance(node, ast.GeneratorExp)
                else bounded(list(iterator))
            )
        if isinstance(node, ast.Call):
            if (
                not isinstance(node.func, ast.Name)
                or node.func.id not in _FUNCTIONS
                or node.keywords
            ):
                raise GrammarError(
                    "only allowlisted pure positional calls are supported"
                )
            args = [self.visit(a, env, child_depth) for a in node.args]
            name = node.func.id
            if name == "range":
                if not 1 <= len(args) <= 3 or any(type(a) is not int for a in args):
                    raise GrammarError("range requires 1..3 integers")
                value = range(*args)
                if len(value) > MAX_ITEMS:
                    raise BudgetError("Python range exceeds item budget")
                self.spend(len(value))
                return list(value)
            if name == "sum":
                if len(args) != 1 or type(args[0]) not in (list, _LazyItems):
                    raise GrammarError("sum requires bounded integers")
                values = args[0]
                if type(values) is list:
                    if any(type(v) is not int for v in values):
                        raise GrammarError("sum requires bounded integers")
                    self.spend(len(values))
                    return bounded(sum(values))

                def integers():
                    for value in values:
                        self.spend(1)
                        if type(value) is not int:
                            raise GrammarError("sum requires bounded integers")
                        yield value

                return bounded(sum(integers()))
            if len(args) == 1 and type(args[0]) is _LazyItems:
                if name not in ("min", "max", "sorted", "all", "any"):
                    raise GrammarError("generator requires an iterable consumer")
                if name == "sorted":
                    values = list(args[0])
                    self.spend(len(values) * max(1, len(values).bit_length()))
                    return bounded(sorted(values))
                # Iteration itself charges the interpreter work budget. any/all
                # must not visit later elements after their answer is known.
                return bounded(_BUILTINS[name](args[0]))
            if name in ("min", "max", "sorted", "all", "any"):
                size = (
                    len(args[0])
                    if len(args) == 1 and type(args[0]) in (list, str, dict)
                    else len(args)
                )
                self.spend(
                    size * max(1, size.bit_length()) if name == "sorted" else size
                )
            return bounded(_BUILTINS[name](*args))
        raise GrammarError("unsupported Python expression")


class _LazyItems:
    """Private interpreter iterator: never accepted as input or JSON output."""

    __slots__ = ("_items",)

    def __init__(self, items):
        self._items = items

    def __iter__(self):
        return self._items
