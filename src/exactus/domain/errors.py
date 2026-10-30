"""Exception hierarchy for the Exactus domain.

Author: Olshansky Grigory
"""

from __future__ import annotations

__author__ = "Olshansky Grigory"


class ExactusError(Exception):
    """Base error for every domain failure."""


class GrammarError(ExactusError):
    """Expression violates syntax or operator semantics."""


class BudgetError(ExactusError):
    """Expression exceeds a size, depth, node or bit budget."""


class DataError(ExactusError):
    """Request-scoped data reference is missing or unusable."""


class FormatError(ExactusError):
    """Requested output format is unknown."""


class VerificationError(ExactusError):
    """The computed answer did not satisfy the caller's contract."""

    error_code = "answer_contract_failed"


class WorkerError(ExactusError):
    """Transient IPC/process failure, not invalid client arithmetic."""
