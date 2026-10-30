"""FastAPI route tests for the exact-compute endpoint registration.

Author: Olshansky Grigory
"""

import asyncio
import hashlib
import json
import unittest

from exactus.adapters.digest import HashlibHasher
from exactus.adapters.endpoint_adapter import ExactusCoordinator
from exactus.adapters.endpoint_route import attach
from exactus.adapters.in_process import InProcessExternalComputer
from exactus.domain.ports import Segment

try:
    import fastapi
except ImportError:
    fastapi = None

__author__ = "Olshansky Grigory"

_ABC_SHA256 = str(int.from_bytes(hashlib.sha256(b"abc").digest(), "big"))


class ScriptedEngine:
    """Engine stub returning one scripted segment per call."""

    def __init__(self, texts) -> None:
        """Store the scripted segment texts and an empty call log."""
        self._texts = list(texts)
        self.calls: list[str] = []

    async def generate(self, prompt_ids, max_tokens, request_id) -> Segment:
        """Return the next scripted segment and record the request id."""
        self.calls.append(request_id)
        text = self._texts.pop(0) if self._texts else ""
        return Segment(text, tuple(), 1, "stop")


class WordTokenizer:
    """Tokenizer stub encoding one token per whitespace-separated word."""

    def encode(self, text: str) -> list[int]:
        """Return one placeholder token id per word."""
        return [1] * max(1, len(text.split()))

    def decode(self, ids) -> str:
        """Return the empty text for any token ids."""
        return ""


def asgi_call(app, method: str, path: str, body: bytes = b"", headers=None):
    """Invoke one ASGI request and return the status and decoded JSON body."""

    async def run():
        """Run the app once and collect the raw response."""
        state = {"status": None}
        chunks = []
        pending = [body] if body else []

        async def receive():
            """Deliver the request body exactly once."""
            if pending:
                return {
                    "type": "http.request",
                    "body": pending.pop(0),
                    "more_body": False,
                }
            # A fully uploaded request is NOT a disconnected client. Real ASGI
            # receive waits here until the peer actually disconnects.
            await asyncio.Event().wait()

        async def send(message):
            """Collect the response status and body chunks."""
            if message["type"] == "http.response.start":
                state["status"] = message["status"]
            elif message["type"] == "http.response.body":
                chunks.append(message.get("body", b""))

        pairs = [
            (name.lower().encode("utf-8"), value.encode("utf-8"))
            for name, value in (headers or {}).items()
        ]
        scope = {
            "type": "http",
            "asgi": {"version": "3.0"},
            "http_version": "1.1",
            "method": method,
            "scheme": "http",
            "path": path,
            "raw_path": path.encode("utf-8"),
            "query_string": b"",
            "root_path": "",
            "headers": pairs,
            "client": ("127.0.0.1", 12345),
            "server": ("127.0.0.1", 80),
        }
        await app(scope, receive, send)
        return state["status"], b"".join(chunks)

    status, raw = asyncio.run(run())
    return status, json.loads(raw) if raw else {}


@unittest.skipUnless(fastapi is not None, "fastapi is not installed")
class TestExactRoute(unittest.TestCase):
    """Verifies the route registration, the modes and the request limits."""

    def _app(self, texts):
        """Build one FastAPI app with a scripted engine behind the route."""
        engine = ScriptedEngine(texts)
        coordinator = ExactusCoordinator(
            engine, WordTokenizer(), InProcessExternalComputer(HashlibHasher())
        )
        app = fastapi.FastAPI()
        attach(app, coordinator, ("model-1",))
        return app, engine

    def _post(self, app, payload):
        """POST one JSON request to the exact-compute route."""
        return asgi_call(
            app, "POST", "/v1/exact/completions", json.dumps(payload).encode("utf-8")
        )

    def test_plain_post_runs_engine(self):
        """A normal POST returns 200 and invokes the engine without extra query."""
        app, engine = self._app(["value <calc>2+3</calc> done"])
        status, body = self._post(app, {"exact_compute": True, "prompt": "compute 2+3"})
        self.assertEqual(status, 200)
        self.assertEqual(body["text"], "value 5 done")
        self.assertGreaterEqual(len(engine.calls), 1)

    def test_default_mode_preserves_report(self):
        """The default mode inserts values in place instead of eating the report."""
        app, _ = self._app(
            ["Summary. Budget <calc-final>6*7</calc-final>. Recommendations follow."]
        )
        status, body = self._post(app, {"exact_compute": True, "prompt": "report"})
        self.assertEqual(status, 200)
        self.assertEqual(body["text"], "Summary. Budget 42. Recommendations follow.")
        self.assertEqual(body["mode"], "preserve-text")

    def test_numeric_final_returns_number_only(self):
        """numeric-final answers with the computed number only."""
        app, _ = self._app(["Answer <calc-final>6*7</calc-final>"])
        status, body = self._post(
            app, {"exact_compute": True, "prompt": "calc", "mode": "numeric-final"}
        )
        self.assertEqual(status, 200)
        self.assertEqual(body["text"], "42")
        self.assertTrue(body["exact_final"])

    def test_numeric_final_free_text_is_coded_error(self):
        """Free text without a final span is a coded failure, not a soft success."""
        app, _ = self._app(["guessed 42"])
        status, body = self._post(
            app, {"exact_compute": True, "prompt": "calc", "mode": "numeric-final"}
        )
        self.assertEqual(status, 422)
        self.assertEqual(body["error_code"], "missing_final")

    def test_strict_mode_violation_is_coded_error(self):
        """strict mode rejects any protocol violation with a machine-readable code."""
        app, _ = self._app(["guessed 42"])
        status, body = self._post(
            app, {"exact_compute": True, "prompt": "calc", "mode": "strict"}
        )
        self.assertEqual(status, 422)
        self.assertEqual(body["error_code"], "missing_final")

    def test_data_hash_end_to_end(self):
        """Request data is decoded once and hashes to the exact value."""
        app, _ = self._app(["<calc-final>sha256(@m)</calc-final>"])
        status, body = self._post(
            app, {"exact_compute": True, "prompt": "hash", "data": {"m": "YWJj"}}
        )
        self.assertEqual(status, 200)
        self.assertEqual(body["text"], _ABC_SHA256)

    def test_invalid_data_is_rejected(self):
        """Invalid base64 in data is refused with a coded 422."""
        app, engine = self._app(["ok"])
        status, body = self._post(
            app, {"exact_compute": True, "prompt": "x", "data": {"m": "!!!"}}
        )
        self.assertEqual(status, 422)
        self.assertEqual(body["error_code"], "invalid_data")
        self.assertEqual(len(engine.calls), 0)

    def test_oversized_body_is_413(self):
        """A body above the limit is refused with 413 before execution."""
        app, engine = self._app(["ok"])
        status, body = self._post(app, {"exact_compute": True, "prompt": "x" * 40000})
        self.assertEqual(status, 413)
        self.assertEqual(body["error_code"], "body_too_large")
        self.assertEqual(len(engine.calls), 0)

    def test_invalid_json_is_400(self):
        """A malformed JSON body yields 400 with a machine-readable code."""
        app, _ = self._app(["ok"])
        status, body = asgi_call(app, "POST", "/v1/exact/completions", b"{not json")
        self.assertEqual(status, 400)
        self.assertEqual(body["error_code"], "invalid_json")


if __name__ == "__main__":
    unittest.main()
