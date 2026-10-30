"""Exactus: bounded external computers for LLMs.

Author: Olshansky Grigory
"""

from __future__ import annotations

__author__ = "Olshansky Grigory"
__version__ = "0.7.0"

from .application.compute import ComputeResult, ComputeService, Operation
from .application.predict import DraftService
from .domain.errors import (
    BudgetError,
    DataError,
    ExactusError,
    FormatError,
    GrammarError,
)
from .domain.expression import ExpressionEvaluator
from .domain.ports import (
    Candidate,
    ExternalComputer,
    DraftPredictor,
    Engine,
    Hasher,
    Segment,
    Tokenizer,
)
from .domain.protocol import Span, SpanMachine
from .domain.value import ComputeValue

__all__ = [
    "BudgetError",
    "Candidate",
    "ExternalComputer",
    "ComputeResult",
    "ComputeService",
    "ComputeValue",
    "DataError",
    "DraftPredictor",
    "DraftService",
    "Engine",
    "ExactusError",
    "ExpressionEvaluator",
    "FormatError",
    "GrammarError",
    "Hasher",
    "Operation",
    "Segment",
    "Span",
    "SpanMachine",
    "Tokenizer",
]
