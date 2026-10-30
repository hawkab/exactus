"""HTTP engine client, stop-boundary and plan_text wiring tests.

Author: Olshansky Grigory
"""

import asyncio
import json
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from exactus.adapters.digest import HashlibHasher
from exactus.adapters.endpoint_adapter import ExactusCoordinator
from exactus.adapters.endpoint_validation import RequestValueError
from exactus.adapters.engine_client import EngineClient, EngineError
from exactus.adapters.in_process import InProcessExternalComputer
from exactus.adapters.tokenizer import ByteTokenizer

__author__ = "Olshansky Grigory"


class UnitTokenizer:
    """Tokenizer stub mapping each whole string to exactly one token."""

    def encode(self, text):
        """Return one placeholder token id per non-empty string."""
        return [1] if text else []

    def decode(self, ids):
        """Return the empty text for any token ids."""
        return ""


class RecordingPlanner:
    """Planner stub recording every plan_text call with its request id."""

    def __init__(self):
        """Start with an empty call log."""
        self.calls = []

    def plan_text(self, text, request_id):
        """Record one planned injection and return a unit length."""
        self.calls.append((text, request_id))
        return 1


class _State:
    """Scripted replies and recorded requests of the fake upstream."""

    def __init__(self):
        """Start with empty reply and request logs."""
        self.replies = []
        self.requests = []


class _Handler(BaseHTTPRequestHandler):
    """Records one chat request and replies with the next scripted body."""

    def do_POST(self):
        """Serve the next scripted chat completions reply."""
        length = int(self.headers.get("Content-Length", "0"))
        payload = json.loads(self.rfile.read(length))
        self.server.state.requests.append((self.path, dict(self.headers), payload))
        code, body = (
            self.server.state.replies.pop(0)
            if self.server.state.replies
            else (200, _chat(""))
        )
        raw = json.dumps(body).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def log_message(self, *args):
        """Keep the fake upstream silent."""


def _chat(text, *, usage=None, finish_reason="stop"):
    """Build one OpenAI-style chat completions reply body."""
    reply = {
        "choices": [{"message": {"content": text}, "finish_reason": finish_reason}]
    }
    reply["usage"] = {
        "completion_tokens": usage if usage is not None else len(text.encode())
    }
    return reply


class TestEngineClient(unittest.TestCase):
    """Verifies the HTTP engine client against a stdlib fake upstream."""

    @classmethod
    def setUpClass(cls):
        """Start one fake OpenAI-compatible upstream on a local port."""
        cls.state = _State()
        server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
        server.state = cls.state
        server.daemon_threads = True
        cls.server = server
        cls.thread = threading.Thread(target=server.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        """Shut the fake upstream down."""
        cls.server.shutdown()
        cls.server.server_close()

    def setUp(self):
        """Reset the reply and request logs before each test."""
        self.state.replies.clear()
        self.state.requests.clear()

    def _client(self, tokenizer=None, **kwargs):
        """Build one client bound to the fake upstream."""
        base = f"http://127.0.0.1:{self.server.server_address[1]}"
        return EngineClient(
            tokenizer or ByteTokenizer(), base_url=base, model="m", **kwargs
        )

    def test_generate_returns_segment_and_forwards_request(self):
        """Usage counts sample tokens and the prompt travels as one message."""
        self.state.replies.append((200, _chat("hi", usage=7)))
        segment = asyncio.run(self._client().generate([65, 66], 32, "r-1"))
        self.assertEqual(segment.text, "hi")
        self.assertEqual(segment.sampled_tokens, 7)
        path, headers, payload = self.state.requests[0]
        self.assertEqual(path, "/v1/chat/completions")
        self.assertEqual(headers.get("X-Request-Id"), "r-1")
        self.assertEqual(payload["max_tokens"], 32)
        self.assertEqual(payload["messages"][-1]["content"], "AB")

    def test_sampled_counts_tokenizer_ids_without_usage(self):
        """Without usage the sampled count is tokenizer ids, not characters."""
        reply = _chat("hello")
        reply.pop("usage")
        self.state.replies.append((200, reply))
        segment = asyncio.run(
            self._client(UnitTokenizer(), require_usage=False).generate([1], 8, "r")
        )
        self.assertEqual(segment.sampled_tokens, 1)
        self.assertEqual(segment.token_count_source, "tokenizer_estimate")
        self.assertEqual(segment.token_ids, (1,))

    def test_missing_or_invalid_usage_fails_closed_by_default(self):
        for usage in (None, -1, True):
            reply = _chat("hello")
            if usage is None:
                reply.pop("usage")
            else:
                reply["usage"]["completion_tokens"] = usage
            self.state.replies.append((200, reply))
            with self.assertRaises(EngineError):
                asyncio.run(self._client().generate([65], 32, "r"))

    def test_eos_does_not_invent_close_tag(self):
        self.state.replies.append((200, _chat("<calc-final>2+3")))
        segment = asyncio.run(
            self._client(stop=["</calc-final>"]).generate([65], 32, "r")
        )
        self.assertEqual(segment.text, "<calc-final>2+3")

    def test_chat_template_context_checked_before_http(self):
        client = self._client(context_limit=16)
        with self.assertRaises(EngineError):
            asyncio.run(client.generate([65], 8, "r"))
        self.assertEqual(self.state.requests, [])

    def test_http_error_is_typed(self):
        """Upstream failures surface as typed engine errors."""
        self.state.replies.append((500, {"error": "boom"}))
        with self.assertRaises(EngineError):
            asyncio.run(self._client().generate([65], 8, "r"))

    def test_stop_boundary_restores_close_tag(self):
        """A close tag removed as a stop string is restored on the segment."""
        reply = _chat("<calc>2+3")
        reply["choices"][0]["stop_reason"] = "</calc>"
        self.state.replies.append((200, reply))
        client = self._client(stop=["</calc>"])
        segment = asyncio.run(client.generate([65], 8, "r"))
        self.assertEqual(segment.text, "<calc>2+3</calc>")

    def test_prefill_does_not_also_plan_forcing(self):
        """Computed values are inserted once, not repeated by logits forcing."""
        planner = RecordingPlanner()
        coordinator = ExactusCoordinator(
            self._client(), ByteTokenizer(), InProcessExternalComputer(HashlibHasher())
        )
        before = len(self.state.requests)
        with self.assertRaises(RequestValueError):
            asyncio.run(coordinator.run([65], require_final=False, planner=planner))
        self.assertEqual(planner.calls, [])
        self.assertEqual(len(self.state.requests), before)

    def test_known_tokens_use_tokenizer_and_cover_final(self):
        """Known tokens are tokenizer counts and include the final value."""
        self.state.replies.append((200, _chat("<calc-final>2^200</calc-final>")))
        coordinator = ExactusCoordinator(
            self._client(UnitTokenizer()),
            UnitTokenizer(),
            InProcessExternalComputer(HashlibHasher()),
        )
        result = asyncio.run(coordinator.run([1]))
        self.assertEqual(result.known_tokens, 1)
        self.assertTrue(result.exact_final)
        self.state.replies.append((200, _chat("<calc>2^200</calc>")))
        self.state.replies.append((200, _chat("")))
        result = asyncio.run(coordinator.run([1], require_final=False))
        self.assertEqual(result.known_tokens, 1)


if __name__ == "__main__":
    unittest.main()
