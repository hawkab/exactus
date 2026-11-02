"""Translate domain failures into stable HTTP error codes."""

__author__ = "Olshansky Grigory"

from .domain.errors import ExactusError

_ERROR_CODES = {
    "GrammarError": "grammar_error",
    "BudgetError": "budget_error",
    "DataError": "data_error",
    "FormatError": "format_error",
}


class RequestError(ExactusError):
    """Request-level failure carrying a machine-readable code and status."""

    def __init__(self, message: str, error_code: str, status: int = 422) -> None:
        """Store the message, the machine-readable code and the HTTP status."""
        super().__init__(message)
        self.error_code = error_code
        self.status = status

    def body(self) -> dict:
        """Return the machine-readable JSON body of the failure."""
        return {"error": str(self), "error_code": self.error_code}


def as_request_error(exc: ExactusError) -> RequestError:
    """Wrap one domain error into a request error with a stable code."""
    code = getattr(
        exc, "error_code", _ERROR_CODES.get(type(exc).__name__, "compute_error")
    )
    return RequestError(str(exc), code, 422)
