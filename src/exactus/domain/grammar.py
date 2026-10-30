"""Recursive-descent grammar parser over the expression token stream.

Author: Olshansky Grigory
"""
from __future__ import annotations

from .errors import BudgetError, GrammarError
from .lexer import Token

__author__ = "Olshansky Grigory"

_UNARY = ("-", "+", "~")
_MUL = ("*", "/", "//", "%", "mod")
_SHIFT = ("<<", ">>")
_POWER = ("^", "**")


class GrammarParser:
    """Parses one token stream with precedence and complexity accounting.

    Precedence, low to high: xor, bor, band, shifts, add, mul, unary,
    right-associative power. Semantic work is delegated to the host
    evaluator and the request-scoped context, so parsing stays free of
    numeric policy and holds no request data of its own.
    """

    def __init__(self, host, tokens: list[Token], context, *, max_nodes: int, max_depth: int) -> None:
        """Store the semantic host, the token stream, the request context and the complexity budgets."""
        self._host = host
        self._tokens = tokens
        self._context = context
        self._max_nodes = max_nodes
        self._max_depth = max_depth
        self._pos = 0
        self._nodes = 0
        self._depth = 0

    def parse(self) -> int:
        """Parse a full expression and return its exact integer value."""
        try:
            value = self._parse_xor()
        except RecursionError as exc:
            raise BudgetError("expression exceeds the complexity budget") from exc
        if self._peek().kind != "END":
            raise GrammarError("trailing input after expression")
        return value

    def _peek(self) -> Token:
        """Return the current token without consuming it."""
        return self._tokens[self._pos]

    def _take(self) -> Token:
        """Consume and return the current token."""
        token = self._tokens[self._pos]
        self._pos += 1
        return token

    def _accept(self, text: str) -> bool:
        """Consume the current token if it matches text."""
        if self._peek().text == text:
            self._pos += 1
            return True
        return False

    def _count(self) -> None:
        """Charge one node of complexity budget."""
        self._nodes += 1
        if self._nodes > self._max_nodes:
            raise BudgetError("expression exceeds the node budget")

    def _parse_xor(self) -> int:
        """Parse a left-associative chain of xor operators."""
        value = self._parse_bor()
        while self._peek().kind == "NAME" and self._peek().text == "xor":
            self._take()
            self._count()
            value = self._host.bound(value ^ self._parse_bor())
        return value

    def _parse_bor(self) -> int:
        """Parse a left-associative chain of bitwise or operators."""
        value = self._parse_band()
        while self._accept("|"):
            self._count()
            value = self._host.bound(value | self._parse_band())
        return value

    def _parse_band(self) -> int:
        """Parse a left-associative chain of bitwise and operators."""
        value = self._parse_shift()
        while self._accept("&"):
            self._count()
            value = self._host.bound(value & self._parse_shift())
        return value

    def _parse_shift(self) -> int:
        """Parse a left-associative chain of shift operators."""
        value = self._parse_add()
        while self._peek().text in _SHIFT:
            op = self._take().text
            self._count()
            amount = self._parse_add()
            if amount < 0 or amount > self._host.max_bits:
                raise BudgetError("shift amount outside allowed range")
            value = self._host.bound(value << amount if op == "<<" else value >> amount)
        return value

    def _parse_add(self) -> int:
        """Parse a left-associative chain of addition and subtraction."""
        value = self._parse_mul()
        while self._peek().text in ("+", "-"):
            op = self._take().text
            self._count()
            right = self._parse_mul()
            value = self._host.bound(value + right if op == "+" else value - right)
        return value

    def _parse_mul(self) -> int:
        """Parse a left-associative chain of multiplication and division."""
        value = self._parse_unary()
        while self._peek().text in _MUL:
            op = self._take().text
            self._count()
            value = self._host.bound(self._host.mul_case(op, value, self._parse_unary()))
        return value

    def _parse_unary(self) -> int:
        """Parse unary operators and fall through to power."""
        if self._peek().text in _UNARY:
            op = self._take().text
            self._count()
            value = self._parse_unary()
            return self._host.bound(-value if op == "-" else ~value if op == "~" else value)
        return self._parse_power()

    def _parse_power(self) -> int:
        """Parse right-associative power with the exponent budget."""
        base = self._parse_atom()
        if self._peek().text in _POWER:
            self._take()
            self._count()
            return self._host.bound(self._host.power(base, self._parse_unary()))
        return base

    def _parse_atom(self) -> int:
        """Parse a literal, data reference, hash call or parenthesised expression."""
        self._count()
        token = self._peek()
        if token.kind == "INT":
            return self._host.bound(self._take().value)
        if token.kind == "REF":
            return self._host.bound(self._context.data_value(self._take().text))
        if token.kind == "NAME":
            return self._host.bound(self._parse_call())
        if token.kind == "LPAREN":
            self._take()
            self._depth += 1
            if self._depth > self._max_depth:
                raise BudgetError("parentheses exceed the nesting budget")
            value = self._parse_xor()
            if not self._accept(")"):
                raise GrammarError("missing closing parenthesis")
            self._depth -= 1
            return value
        raise GrammarError(f"unexpected token: {token.text!r}")

    def _parse_call(self) -> int:
        """Parse a hash call over one data reference and return its digest."""
        name = self._take().text
        if not self._accept("("):
            raise GrammarError("function call needs parentheses")
        if self._peek().kind != "REF":
            raise GrammarError("hash functions accept only data references")
        payload = self._context.data_payload(self._take().text)
        if not self._accept(")"):
            raise GrammarError("missing closing parenthesis in function call")
        return self._context.hash_digest(name, payload)
