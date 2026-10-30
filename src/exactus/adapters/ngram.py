"""Experimental tuple-indexed n-gram proposals; no target-model verifier.

Author: Olshansky Grigory
"""

from __future__ import annotations

from collections import Counter, defaultdict, deque
from collections.abc import Iterable, Sequence

from ..domain.ports import Candidate

__author__ = "Olshansky Grigory"


class NgramDraftPredictor:
    """Predicts continuations using token tuples as lookup keys.

    This CPU experiment is not connected to the production MTP loop.
    Drafts are unverified proposals: they
    require a verifier in the calling loop, and without one the predictor
    gives no accuracy guarantee for the produced answer.
    """

    def __init__(self, order: int = 4) -> None:
        """Set the maximum n-gram order and start with an empty index."""
        if type(order) is not int or not 1 <= order <= 32:
            raise ValueError("order must be in 1..32")
        self._order = order
        self._index: dict[tuple[int, ...], Counter] = defaultdict(Counter)

    def fit(self, sequences: Iterable[Sequence[int]]) -> None:
        """Index every continuation of every sequence."""
        for sequence in sequences:
            ids = tuple(sequence)
            for position in range(len(ids)):
                for size in range(1, self._order + 1):
                    if position < size:
                        break
                    key = ids[position - size : position]
                    self._index[key][ids[position]] += 1

    def predict(self, prefix: Iterable[int], k: int = 3) -> list[Candidate]:
        """Return up to k continuations from the longest matching n-gram."""
        ids = _tail(prefix, self._order)
        for size in range(len(ids), 0, -1):
            counter = self._index.get(ids[-size:])
            if counter:
                total = sum(counter.values())
                best = counter.most_common(k)
                return [Candidate((token,), count / total) for token, count in best]
        return []


def _tail(prefix: Iterable[int], order: int) -> tuple[int, ...]:
    """Copy at most the last order tokens of prefix, never the whole prefix."""
    if isinstance(prefix, Sequence):
        return tuple(prefix[max(0, len(prefix) - order) :])
    return tuple(deque(prefix, maxlen=order))
