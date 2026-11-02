"""Interpret validated expression nodes without eval, imports, or host objects."""

__author__ = "Olshansky Grigory"

import ast

from ..domain.errors import BudgetError, GrammarError
from .python_builtins import call, comprehension
from .python_operators import binary
from .python_syntax import _COMPARE, _UNARY
from .values import bounded


class Interpreter:
    """Evaluate supported AST nodes with shared work and depth accounting."""

    def __init__(self):
        """Initialize Interpreter state and dependencies."""
        self.remaining = 20000

    def tick(self, depth):
        """Charge a visited node and validate the current evaluation depth."""
        self.remaining -= 1
        if self.remaining < 0 or depth > 32:
            raise BudgetError("Python work/depth budget exceeded")

    def spend(self, units):
        """Charge builtin work against this expression evaluation."""
        self.remaining -= units
        if self.remaining < 0:
            raise BudgetError("Python builtin work budget exceeded")

    def visit(self, node, env, depth=0):
        """Evaluate one validated node using only the supplied JSON environment."""
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
            return binary(self, node, env, depth)
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
            return comprehension(self, node, env, depth)
        if isinstance(node, ast.Call):
            return call(self, node, env, depth)
        raise GrammarError("unsupported Python expression")
