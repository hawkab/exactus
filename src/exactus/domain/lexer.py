"""Tokenizer for exact integer expression source code.

Author: Olshansky Grigory
"""
from __future__ import annotations

from dataclasses import dataclass

from .errors import GrammarError

__author__ = "Olshansky Grigory"

_TWO_CHAR = ("**", "//", "<<", ">>")
_ONE_CHAR = "+-*/%^&|~(),"
_NAME_START = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ_"
_NAME_BODY = _NAME_START + "0123456789"
_DIGITS = "0123456789"


@dataclass(frozen=True, slots=True)
class Token:
    """One lexical unit of an expression."""

    kind: str
    text: str
    value: int | None = None


def tokenize(source: str) -> list[Token]:
    """Split expression source into a flat token list ending with an END token."""
    tokens: list[Token] = []
    i, n = 0, len(source)
    while i < n:
        char = source[i]
        if char in " \t\r\n":
            i += 1
            continue
        if char in _DIGITS:
            i = _scan_number(source, i, tokens)
            continue
        if char == "@":
            i = _scan_ref(source, i, tokens)
            continue
        if char in _NAME_START:
            i = _scan_name(source, i, tokens)
            continue
        pair = source[i : i + 2]
        if pair in _TWO_CHAR:
            tokens.append(Token("OP", pair))
            i += 2
            continue
        if char in _ONE_CHAR:
            kind = {"(": "LPAREN", ")": "RPAREN", ",": "COMMA"}.get(char, "OP")
            tokens.append(Token(kind, char))
            i += 1
            continue
        raise GrammarError(f"unexpected character: {char!r}")
    tokens.append(Token("END", ""))
    return tokens


def _scan_number(source: str, start: int, tokens: list[Token]) -> int:
    """Scan a decimal, hexadecimal or binary integer literal starting at start."""
    if source.startswith(("0x", "0X"), start):
        return _scan_prefixed(source, start + 2, "0123456789abcdefABCDEF", 16, tokens)
    if source.startswith(("0b", "0B"), start):
        return _scan_prefixed(source, start + 2, "01", 2, tokens)
    i, plain = start, False
    while i < len(source) and (source[i] in _DIGITS or source[i] == "_"):
        plain = plain or source[i] in _DIGITS
        if source[i] == "_" and (i + 1 >= len(source) or source[i + 1] not in _DIGITS):
            raise GrammarError("misplaced separator in number literal")
        i += 1
    if not plain:
        raise GrammarError("number literal has no digits")
    tokens.append(Token("INT", source[start:i], int(source[start:i].replace("_", ""))))
    return i


def _scan_prefixed(source: str, start: int, alphabet: str, base: int, tokens: list[Token]) -> int:
    """Scan the body of a 0x/0b literal and append its INT token."""
    i, plain = start, False
    while i < len(source) and (source[i] in alphabet or source[i] == "_"):
        plain = plain or source[i] in alphabet
        if source[i] == "_" and (i + 1 >= len(source) or source[i + 1] not in alphabet):
            raise GrammarError("misplaced separator in number literal")
        i += 1
    if not plain:
        raise GrammarError("number literal has no digits")
    tokens.append(Token("INT", source[start - 2 : i], int(source[start:i].replace("_", ""), base)))
    return i


def _scan_name(source: str, start: int, tokens: list[Token]) -> int:
    """Scan an identifier used as an operator keyword or hash algorithm name."""
    i = start + 1
    while i < len(source) and source[i] in _NAME_BODY:
        i += 1
    tokens.append(Token("NAME", source[start:i]))
    return i


def _scan_ref(source: str, start: int, tokens: list[Token]) -> int:
    """Scan a request-scoped data reference of the form @name."""
    i = start + 1
    if i >= len(source) or source[i] not in _NAME_START:
        raise GrammarError("data reference needs a name after @")
    i += 1
    while i < len(source) and source[i] in _NAME_BODY:
        i += 1
    tokens.append(Token("REF", source[start + 1 : i]))
    return i
