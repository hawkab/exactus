"""Preserve upstream evidence when a local computation fails."""

from __future__ import annotations

__author__ = "Olshansky Grigory"

from .domain.errors import WorkerError
from .request_errors import RequestError, as_request_error


def computation_failure(exc, response):
    """Retain spent upstream usage when local computation fails."""
    if isinstance(exc, WorkerError):
        failure_status = 503
        body = {
            "error": "compute worker unavailable",
            "error_code": "worker_unavailable",
            "retryable": True,
        }
    else:
        failure = exc if isinstance(exc, RequestError) else as_request_error(exc)
        failure_status, body = failure.status, failure.body()
    if type(response) is dict:
        if "usage" in response:
            body["upstream_usage"] = response["usage"]
        body["upstream_choices"] = response.get("choices")
    body["exactus"] = {"upstream_requests": 1, "answer_rendered": False}
    return failure_status, body, {}
