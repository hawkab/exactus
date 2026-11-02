"""Zero-change OpenAI-compatible proxy activating exact computation."""

from __future__ import annotations

__author__ = "Olshansky Grigory"

import json
from typing import Any

from .adapters.dispatch import external_computer_answer
from .domain.errors import ExactusError, WorkerError
from .http_io import request_bytes
from .proxy_failures import computation_failure
from .proxy_modes import (
    RequestError,
    as_request_error,
)
from .proxy_requests import prepare_chat, technical_result
from .proxy_response import ResponseProcessor
from .proxy_rewrite import (
    extract_content as extract_content,
)
from .proxy_rewrite import (
    prepare_request as prepare_request,
)
from .proxy_rewrite import (
    replace_content as replace_content,
)
from .proxy_server import serve as serve
from .proxy_transport import decode_reply, select_headers

_PATHS = (
    "/v1/chat/completions",
    "/v1/completions",
    "/v1/models",
    "/v1/exact/external-computers",
    "/v1/exact/plans",
)


class ExactusProxy:
    """Sits between any OpenAI client and any OpenAI backend.

    One command activates exact computation without touching the model
    stack: the proxy injects the marker hint, forwards the client method
    and headers upstream and replaces computed spans with exact values in
    every response choice. Local control fields ``data`` and ``mode`` are
    validated and consumed here and never reach the upstream. The honest
    limited subset is non-streaming chat completions plus passthrough GET.
    """

    def __init__(
        self,
        upstream: str,
        external_computer,
        *,
        hint: str | None = None,
        require_final: bool = False,
        deadline_s: float = 600.0,
    ) -> None:
        """Store the upstream URL, the external computer port and the policy."""
        self._upstream = upstream.rstrip("/")
        self._external_computer = external_computer
        self._hint = hint
        self._require_final = require_final
        self._deadline_s = deadline_s
        from .adapters.worker import WorkerExternalComputer

        self._external_computer_worker = WorkerExternalComputer(
            timeout=min(deadline_s, 2.0)
        )

    def close(self):
        """Release the lazily spawned external computer worker."""
        self._external_computer_worker.close()

    @property
    def deadline_s(self) -> float:
        """Return the end-to-end deadline of one request in seconds."""
        return self._deadline_s

    def handle(
        self, path: str, payload: dict, *, method: str = "POST", headers=None
    ) -> tuple[int, Any]:
        """Process one request and return the status and the response body."""
        status, body, _ = self.handle_ex(path, payload, method=method, headers=headers)
        return status, body

    def handle_ex(
        self, path: str, payload: dict, *, method: str = "POST", headers=None
    ) -> tuple[int, Any, dict]:
        """Process one request and return status, body and extra headers."""
        if path not in _PATHS:
            return 404, {"error": "unknown path", "error_code": "unknown_path"}, {}
        if (
            path in ("/v1/exact/external-computers", "/v1/exact/plans")
            and method != "POST"
        ):
            return (
                405,
                {"error": "POST required", "error_code": "method_not_allowed"},
                {},
            )
        if method == "GET":
            status, body = self._forward(path, None, method="GET", headers=headers)
            return status, body, {}
        try:
            return self._post(path, payload, headers)
        except RequestError as exc:
            return exc.status, exc.body(), {}
        except WorkerError:
            return (
                503,
                {
                    "error": "external computer worker unavailable",
                    "error_code": "worker_unavailable",
                    "retryable": True,
                },
                {},
            )
        except ExactusError as exc:
            error = as_request_error(exc)
            return error.status, error.body(), {}

    def _post(self, path: str, payload: dict, headers) -> tuple[int, Any, dict]:
        """Dispatch technical requests or forward a prepared chat completion."""
        if path in ("/v1/exact/plans", "/v1/exact/external-computers"):
            result = technical_result(path, payload, self._external_computer_worker)
            return 200, result, {"X-Exactus-Model-Calls": "0"}
        if any(
            k in payload
            for k in ("compute_plan", "plan_inputs", "plan_checks", "plan_input_view")
        ):
            if path != "/v1/chat/completions":
                raise RequestError(
                    "plan generation requires chat completions", "invalid_request"
                )
            from .adapters.plan_chat import complete

            return complete(self, payload, headers)
        prepared = prepare_chat(
            path,
            payload,
            self._external_computer_worker,
            self._hint,
            self._require_final,
        )
        status, response = self._forward(
            path, prepared.payload, method="POST", headers=headers
        )
        if status != 200 or path != "/v1/chat/completions":
            return status, response, {}
        try:
            updated = self._compute(
                payload, response, data=prepared.data, mode=prepared.mode
            )
        except ExactusError as exc:
            return computation_failure(exc, response)
        if prepared.facts is not None:
            updated["exactus"].update(
                external_computers=prepared.facts,
                upstream_requests=1,
                engine_bypassed=False,
            )
        return status, updated, {"X-Exactus-Ops": str(updated["exactus"]["operations"])}

    def _forward(self, path: str, payload, *, method: str, headers) -> tuple[int, Any]:
        """Send one request upstream and decode the JSON reply."""
        body = json.dumps(payload).encode("utf-8") if payload is not None else None
        try:
            status, raw = request_bytes(
                self._upstream + path,
                body,
                method=method,
                headers=select_headers(headers, has_body=body is not None),
                timeout=self._deadline_s,
            )
            return decode_reply(status, raw)
        except TimeoutError:
            return 504, {
                "error": "upstream deadline exceeded",
                "error_code": "deadline_exceeded",
                "retryable": True,
            }
        except Exception as exc:
            return 502, {
                "error": f"upstream failure: {exc}",
                "error_code": "upstream_error",
            }

    def _compute(self, payload: dict, response, *, data=None, mode=None) -> dict:
        """Process every response choice through the external-computer port."""
        processor = ResponseProcessor(
            self._external_computer, self._require_final, external_computer_answer
        )
        return processor.process(payload, response, data=data, mode=mode)
