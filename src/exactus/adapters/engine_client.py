"""OpenAI-compatible HTTP client implementing the engine port.

Author: Olshansky Grigory
"""

from __future__ import annotations

import json
from collections.abc import Sequence

from ..domain.errors import ExactusError
from ..domain.ports import Segment, Tokenizer
from ..domain.protocol import SpanMachine
from .async_work import blocking
from ..http_io import request_bytes

__author__ = "Olshansky Grigory"


class EngineError(ExactusError):
    """Engine transport, schema or response failure with a typed origin."""


class EngineClient:
    """Engine port over one OpenAI-compatible chat completions endpoint.

    HTTP fallback contract: POST {base_url}/v1/chat/completions with request
    fields model, messages, max_tokens, temperature and stop; response
    fields choices[0].message.content, choices[0].finish_reason and the
    optional usage.completion_tokens. Tokenization counts come from the
    provided Tokenizer port (or usage.completion_tokens), never from
    characters. With stop holding closing tags, the client restores the
    closing tag of a trailing open frame, because OpenAI-style servers
    omit stop strings from the returned text (stop-boundary handling).
    """

    def __init__(
        self,
        tokenizer: Tokenizer,
        *,
        base_url: str,
        model: str,
        api_key: str | None = None,
        timeout_s: float = 60.0,
        stop: Sequence[str] | None = None,
        context_limit: int | None = None,
        system: str | None = None,
        require_usage: bool = True,
    ) -> None:
        """Store the endpoint configuration and the tokenizer port."""
        self._tokenizer = tokenizer
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._api_key = api_key
        self._timeout_s = timeout_s
        self._stop = list(stop) if stop else None
        self._context_limit = context_limit
        self._system = system
        self._require_usage = require_usage

    @property
    def context_limit(self) -> int | None:
        """Return the model context limit when one is configured."""
        return self._context_limit

    async def generate(
        self, prompt_ids: Sequence[int], max_tokens: int, request_id: str
    ) -> Segment:
        """Generate one segment from the accumulated prompt over HTTP."""
        return await blocking(self._generate, prompt_ids, max_tokens, request_id)

    def _generate(self, prompt_ids, max_tokens, request_id):
        payload = self._payload(self._decode(prompt_ids), max_tokens)
        return self._segment(self._post(payload, request_id))

    def _decode(self, prompt_ids: Sequence[int]) -> str:
        """Decode the accumulated prompt ids with the tokenizer port."""
        try:
            return self._tokenizer.decode(prompt_ids)
        except Exception as exc:
            raise EngineError(f"prompt decode failed: {exc}") from exc

    def _payload(self, text: str, max_tokens: int) -> dict:
        """Build one chat completions request for the accumulated prompt."""
        messages = []
        if self._system:
            messages.append({"role": "system", "content": self._system})
        messages.append({"role": "user", "content": text})
        if self._context_limit is not None:
            encode = getattr(self._tokenizer, "encode_messages", None)
            if not callable(encode):
                raise EngineError(
                    "HTTP context limits require tokenizer.encode_messages(messages)"
                )
            if len(encode(messages)) + max_tokens > self._context_limit:
                raise EngineError("chat template plus output exceeds the context limit")
        payload = {
            "model": self._model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": 0.0,
        }
        if self._stop:
            payload["stop"] = list(self._stop)
        return payload

    def _post(self, payload: dict, request_id: str) -> dict:
        """Send one JSON request and decode the JSON reply."""
        body = json.dumps(payload).encode("utf-8")
        headers = {"Content-Type": "application/json", "X-Request-Id": request_id}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        try:
            status, raw = request_bytes(
                self._base_url + "/v1/chat/completions",
                body,
                headers=headers,
                timeout=self._timeout_s,
            )
        except Exception as exc:
            raise EngineError(f"engine request failed: {exc}") from exc
        if status != 200:
            raise EngineError(f"engine returned HTTP {status}: {raw[:200]!r}")
        try:
            decoded = json.loads(raw)
        except ValueError as exc:
            raise EngineError("engine reply is not valid JSON") from exc
        if not isinstance(decoded, dict):
            raise EngineError("engine reply is not a JSON object")
        return decoded

    def _segment(self, reply: dict) -> Segment:
        """Extract one generation segment from a chat completions reply."""
        choices = reply.get("choices")
        if not choices or not isinstance(choices[0], dict):
            raise EngineError("engine reply carries no choices")
        message = choices[0].get("message") or {}
        text = str(message.get("content") or "")
        finish_reason = str(choices[0].get("finish_reason") or "unknown")
        text = self._restore_stop_boundary(
            text, finish_reason, choices[0].get("stop_reason")
        )
        try:
            token_ids = tuple(self._tokenizer.encode(text))
        except Exception as exc:
            raise EngineError(f"segment encode failed: {exc}") from exc
        usage = reply.get("usage") or {}
        completion = usage.get("completion_tokens")
        valid = type(completion) is int and completion >= 0
        if not valid and (self._require_usage or completion is not None):
            raise EngineError(
                "valid completion usage is required for output budget accounting"
            )
        return Segment(
            text,
            token_ids,
            completion if valid else len(token_ids),
            finish_reason,
            "usage" if valid else "tokenizer_estimate",
        )

    def _restore_stop_boundary(
        self, text: str, finish_reason: str, stop_reason=None
    ) -> str:
        """Re-append the closing tag removed as a stop string, when needed."""
        if not self._stop or finish_reason != "stop":
            return text
        machine = SpanMachine()
        try:
            machine.feed(text)
        except ExactusError:
            return text
        closing = machine.pending_close()
        if closing is not None and closing in self._stop and stop_reason == closing:
            return text + closing
        return text
