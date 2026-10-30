"""Shared JSON and work budgets for untrusted external computer requests."""

import json
from dataclasses import dataclass

from ..domain.errors import BudgetError, DataError

MAX_ITEMS = 2048
MAX_BYTES = 65536
MAX_BITS = 4096


def bounded(value, *, depth=0, counter=None):
    """Validate a JSON tree without accepting arbitrary objects or cycles."""
    if counter is None:
        counter = [0]
    counter[0] += 1
    if counter[0] > 16384 or depth > 32:
        raise BudgetError("external computer value tree exceeds budget")
    kind = type(value)
    if value is None or kind is bool:
        return value
    if kind is int:
        if value.bit_length() > MAX_BITS:
            raise BudgetError("external computer integer exceeds 4096 bits")
        return value
    if kind is str:
        if len(value) > MAX_BYTES:
            raise BudgetError("external computer string exceeds 64 KiB")
        try:
            size = len(value.encode("utf-8"))
        except UnicodeError as exc:
            raise DataError("external computer strings must be valid UTF-8") from exc
        if size > MAX_BYTES:
            raise BudgetError("external computer string exceeds 64 KiB")
        return value
    if kind not in (list, dict) or len(value) > MAX_ITEMS:
        raise DataError(
            "external computer supports bounded JSON null/bool/int/string/list/object only"
        )
    if kind is dict:
        for key, item in value.items():
            if type(key) is not str:
                raise DataError("external computer object keys must be strings")
            bounded(key, depth=depth + 1, counter=counter)
            bounded(item, depth=depth + 1, counter=counter)
    else:
        for item in value:
            bounded(item, depth=depth + 1, counter=counter)
    return value


def packed(value):
    bounded(value)
    result = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    if len(result.encode("utf-8")) > MAX_BYTES:
        raise BudgetError("external computer JSON exceeds 64 KiB")
    return result


@dataclass(frozen=True, slots=True)
class _Encoding:
    size: int
    nodes: int
    depth: int
    owner: object = None


class OwnedJSON:
    """Request-local measurements for PRIVATE, no-longer-mutated trees.

    Not a validator for shared/mutable caller objects. Callers first snapshot
    at the trust boundary using packed/json.loads. Aliases still count once
    per occurrence against byte/node/depth limits. Cache holds strong refs
    (no id reuse), is bounded, and never survives its request.
    """

    __slots__ = ("_memo",)

    def __init__(self):
        self._memo = {}

    def encode(self, value):
        self._measure(value, 0)
        # Keep the C JSON encoder for small/fresh objects; Python fragment
        # concatenation was slower on ordinary plans. Reuse measurements of
        # sealed children, not repeated whole-tree validation.
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )

    def _measure(self, value, nesting):
        if nesting > 32:
            raise BudgetError("external computer value tree exceeds budget")
        kind = type(value)
        if kind not in (list, dict):
            bounded(value)
            if kind is str:
                size = len(json.dumps(value, ensure_ascii=False).encode("utf-8"))
            elif value is None:
                size = 4
            elif kind is bool:
                size = 4 if value else 5
            else:
                size = len(str(value))
            return size, 1, 0
        cached = self._memo.get(id(value))
        if cached is not None:
            self._check(cached, nesting)
            return cached.size, cached.nodes, cached.depth
        if len(value) > MAX_ITEMS:
            raise DataError("external computer container exceeds item budget")
        size, nodes, depth = 2, 1, 0
        for index, (key, item) in enumerate(
            value.items() if kind is dict else enumerate(value)
        ):
            if kind is dict:
                if type(key) is not str:
                    raise DataError("external computer object keys must be strings")
                key_size, _, _ = self._measure(key, nesting + 1)
                size += key_size + 1
                nodes += 1
            child_size, child_nodes, child_depth = self._measure(item, nesting + 1)
            size += child_size + bool(index)
            nodes += child_nodes
            depth = max(depth, child_depth + 1)
            if size > MAX_BYTES or nodes > 16384 or depth + nesting > 32:
                raise BudgetError("external computer JSON/tree exceeds budget")
        if len(self._memo) < 16384:
            self._memo[id(value)] = _Encoding(size, nodes, depth, value)
        return size, nodes, depth

    @staticmethod
    def _check(value, nesting=0):
        if value.size > MAX_BYTES:
            raise BudgetError("external computer JSON exceeds 64 KiB")
        if value.nodes > 16384 or value.depth + nesting > 32:
            raise BudgetError("external computer value tree exceeds budget")

    def sequence(self):
        return _SequenceBudget(self)


class _SequenceBudget:
    """Charge each completed row once; never cache a growing list."""

    __slots__ = ("owner", "size", "nodes", "depth", "count")

    def __init__(self, owner):
        self.owner, self.size, self.nodes, self.depth, self.count = owner, 2, 1, 0, 0

    def append(self, value):
        if self.count >= MAX_ITEMS:
            raise DataError("external computer aggregate exceeds item budget")
        size, nodes, depth = self.owner._measure(value, 1)
        self.size += size + bool(self.count)
        self.nodes += nodes
        self.depth = max(self.depth, depth + 1)
        self.count += 1
        self.owner._check(_Encoding(self.size, self.nodes, self.depth))
