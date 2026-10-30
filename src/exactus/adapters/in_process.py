"""In-process external computer adapter built on the domain evaluator.

Author: Olshansky Grigory
"""

from __future__ import annotations

from collections.abc import Mapping

from ..domain.expression import ExpressionEvaluator
from ..domain.ports import Hasher
from ..domain.value import ComputeValue

__author__ = "Olshansky Grigory"


class InProcessExternalComputer:
    """Evaluates expressions in the current process with the domain evaluator.

    Fastest adapter; best for trusted expressions and test suites. Use the
    worker adapter when the isolation of a disposable process is required.
    The evaluator keeps no request state, so one instance shared by
    concurrent threads cannot mix request data between evaluations.
    """

    def __init__(self, hasher: Hasher | None = None) -> None:
        """Build the evaluator with an optional hasher port."""
        self._evaluator = ExpressionEvaluator(hasher)

    def evaluate(
        self,
        expression: str,
        data: Mapping[str, bytes] | None = None,
        *,
        output_format: str = "dec",
    ) -> ComputeValue:
        """Evaluate one expression in-process."""
        return self._evaluator.evaluate(expression, data, output_format=output_format)
