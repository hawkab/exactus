"""Extended protocol framing and coordinator behaviour tests.

Author: Olshansky Grigory
"""

import asyncio
import random
import unittest

from exactus.adapters.digest import HashlibHasher
from exactus.adapters.endpoint_adapter import (
    ExactusCoordinator,
    RequestTypeError,
    RequestValueError,
)
from exactus.adapters.in_process import InProcessExternalComputer
from exactus.adapters.tokenizer import ByteTokenizer
from exactus.application.compute import ComputeService
from exactus.domain.errors import GrammarError
from exactus.domain.ports import Segment
from exactus.domain.protocol import SpanMachine

__author__ = "Olshansky Grigory"

_CORPUS = (
    "plain <calc>1<<8</calc> tail",
    "a <calc>2+3</calc> b <calc-final>256>>2</calc-final> c",
    "<calc-final>1<<8</calc-final>",
    "x <calc>a<b>1</calc> y",
    "<calc>1</calc><calc>2</calc>",
    "<calc>1</calc><calc><calc-final>2</calc-final>",
    "<calc>1</calc-final>",
    "mixed </calc> noise",
    "end <c",
)
_SUFFIXES = ("", "<", "</ca", ">>", " <c", "</calc>")


def _run(chunks):
    """Feed chunks into one machine; return spans and the error name."""
    machine = SpanMachine()
    spans = []
    try:
        for chunk in chunks:
            spans.extend(machine.feed(chunk))
        machine.finish()
    except GrammarError as exc:
        return spans, type(exc).__name__
    return spans, None


def _split(text, rng):
    """Split text at random cut points into non-empty chunks."""
    points = sorted({rng.randrange(1, len(text)) for _ in range(rng.randrange(4))})
    pieces, last = [], 0
    for point in points:
        pieces.append(text[last:point])
        last = point
    pieces.append(text[last:])
    return [piece for piece in pieces if piece]


class ScriptedEngine:
    """Engine stub returning scripted segments and recording calls."""

    def __init__(self, texts):
        """Store the scripted segment texts and empty call logs."""
        self._texts = list(texts)
        self.prompts = []
        self.ids = []

    async def generate(self, prompt_ids, max_tokens, request_id):
        """Return the next scripted segment and record the call."""
        self.prompts.append(tuple(prompt_ids))
        self.ids.append(request_id)
        text = self._texts.pop(0) if self._texts else ""
        return Segment(text, tuple(), 1, "stop")


class TestFrameScanner(unittest.TestCase):
    """Verifies strict framing, shift operators and chunk invariance."""

    def test_shift_operators_in_spans(self):
        """Expressions may contain << and >> inside a span."""
        spans = SpanMachine().feed(
            "x <calc>1<<8</calc> y <calc-final>256>>2</calc-final>"
        )
        self.assertEqual([s.expression for s in spans], ["1<<8", "256>>2"])
        service = ComputeService(InProcessExternalComputer(HashlibHasher()))
        self.assertEqual(service.run("<calc-final>1<<8</calc-final>").text, "256")
        self.assertEqual(service.run("<calc-final>256>>2</calc-final>").text, "64")

    def test_nested_frame_after_span_rejected(self):
        """A frame opened after a completed span cannot nest another frame."""
        with self.assertRaises(GrammarError):
            SpanMachine().feed("<calc>1</calc><calc><calc-final>2</calc-final>")

    def test_stray_and_mismatched_close_rejected(self):
        """Stray or mismatched closing tags are rejected as violations."""
        for text in ("a</calc>b", "<calc>1</calc-final>", "<calc-final>1</calc>"):
            with self.subTest(text=text):
                with self.assertRaises(GrammarError):
                    SpanMachine().feed(text)

    def test_chunk_boundary_fuzz(self):
        """1000 random chunk splits give the whole-text outcome exactly."""
        rng = random.Random(20261029)
        for _ in range(1000):
            text = rng.choice(_CORPUS) + rng.choice(_SUFFIXES)
            whole_spans, whole_err = _run([text])
            split_spans, split_err = _run(_split(text, rng))
            self.assertEqual(whole_err, split_err, repr(text))
            if whole_err is None:
                self.assertEqual(whole_spans, split_spans, repr(text))


class TestCoordinatorExt(unittest.TestCase):
    """Verifies coordinator injection, rendering, identity and budgets."""

    def test_two_intermediate_spans_both_injected(self):
        """Every intermediate span of one segment gets its own result frame."""
        engine = ScriptedEngine(["<calc>1</calc><calc>2</calc>", "done"])
        coordinator = ExactusCoordinator(
            engine, ByteTokenizer(), InProcessExternalComputer(HashlibHasher())
        )
        result = asyncio.run(coordinator.run([65], require_final=False))
        self.assertEqual(result.text, "12done")
        echoed = ByteTokenizer().decode(engine.prompts[1])
        self.assertIn("<calc-result>1</calc-result>", echoed)
        self.assertIn("<calc-result>2</calc-result>", echoed)
        self.assertLess(
            echoed.index("1</calc-result>"), echoed.index("2</calc-result>")
        )

    def test_plain_end_renders_computed_values(self):
        """A plain end renders computed values instead of raw frames."""
        engine = ScriptedEngine(["<calc>2+3</calc> ok", ""])
        coordinator = ExactusCoordinator(
            engine, ByteTokenizer(), InProcessExternalComputer(HashlibHasher())
        )
        result = asyncio.run(coordinator.run([65], require_final=False))
        self.assertEqual(result.text, "5 ok")
        self.assertFalse(result.exact_final)

    def test_require_final_without_final_raises(self):
        """require_final with no final span raises instead of free text."""
        engine = ScriptedEngine(["guessed 42", ""])
        coordinator = ExactusCoordinator(
            engine, ByteTokenizer(), InProcessExternalComputer(HashlibHasher())
        )
        with self.assertRaises(GrammarError):
            asyncio.run(coordinator.run([65], require_final=True))
        self.assertEqual(len(engine.ids), 1)

    def test_request_ids_unique_under_concurrency(self):
        """Parallel runs carry unique uuid request and segment ids."""
        frame = "<calc>1</calc><calc-final>2</calc-final>"
        engine = ScriptedEngine([frame, frame, frame])

        async def one():
            """Run one anonymous request and return its result."""
            coordinator = ExactusCoordinator(
                engine, ByteTokenizer(), InProcessExternalComputer(HashlibHasher())
            )
            await asyncio.sleep(0)
            return await coordinator.run([65])

        async def three():
            """Run three requests concurrently and collect results."""
            return await asyncio.gather(one(), one(), one())

        results = asyncio.run(three())
        rids = [item.request_id for item in results]
        self.assertEqual(len(set(rids)), 3)
        self.assertEqual(len(set(engine.ids)), len(engine.ids))
        for segment_id in engine.ids:
            self.assertIn(segment_id.split(":")[0], rids)

    def test_strict_argument_types_rejected(self):
        """Bad request argument types fail before the first engine call."""
        engine = ScriptedEngine(["x"])
        coordinator = ExactusCoordinator(
            engine, ByteTokenizer(), InProcessExternalComputer(HashlibHasher())
        )
        cases = [
            {"prompt_ids": [65], "max_tokens": "large"},
            {"prompt_ids": [65], "max_tokens": 0},
            {"prompt_ids": [65], "require_final": 1},
            {"prompt_ids": ["a"]},
            {"prompt_ids": [True]},
            {"prompt_ids": []},
            {"prompt_ids": [65], "request_id": 7},
        ]
        for kwargs in cases:
            with self.subTest(kwargs=kwargs):
                with self.assertRaises((RequestTypeError, RequestValueError)):
                    asyncio.run(coordinator.run(**kwargs))
        self.assertEqual(engine.ids, [])


if __name__ == "__main__":
    unittest.main()
