"""Speculative draft suggestions from an external predictor.

Author: Olshansky Grigory
"""

from __future__ import annotations

from collections.abc import Sequence

from ..domain.ports import Candidate, DraftPredictor

__author__ = "Olshansky Grigory"


class DraftService:
    """Wraps a predictor into the draft-suggestion use case.

    Proposals only: this service contains no target-model verifier.
    A caller must implement speculative acceptance/rollback before using
    these tokens in generated output; unverified drafts are not lossless.
    """

    def __init__(self, predictor: DraftPredictor) -> None:
        """Bind the predictor port."""
        self._predictor = predictor

    def suggest(self, prefix: Sequence[int], k: int = 3) -> list[Candidate]:
        """Return up to k draft continuations for the prefix, best first."""
        return self._predictor.predict(prefix, k)
