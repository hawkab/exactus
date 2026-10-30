"""Tests for the marker span protocol.

Author: Olshansky Grigory
"""
import unittest

from exactus.domain.errors import GrammarError
from exactus.domain.protocol import SPAN_RE, SpanMachine

__author__ = "Olshansky Grigory"


class TestSpanMachine(unittest.TestCase):
    """Verifies streamed span extraction and frame validation."""

    def test_single_span(self):
        """A complete span is emitted once."""
        machine = SpanMachine()
        spans = machine.feed("a <calc>1+1</calc> b")
        self.assertEqual(len(spans), 1)
        self.assertEqual(spans[0].expression, "1+1")
        self.assertFalse(spans[0].final)

    def test_final_span(self):
        """The final marker sets the final flag."""
        machine = SpanMachine()
        spans = machine.feed("<calc-final>2*3</calc-final>")
        self.assertTrue(spans[0].final)

    def test_split_across_chunks(self):
        """Spans split across chunks are emitted when complete."""
        machine = SpanMachine()
        self.assertEqual(machine.feed("<cal"), [])
        self.assertEqual(machine.feed("c>1</calc>")[0].expression, "1")

    def test_emitted_once(self):
        """Already emitted spans are not re-emitted."""
        machine = SpanMachine()
        machine.feed("<calc>1</calc>")
        self.assertEqual(machine.feed(" more"), [])

    def test_open_frame_detected(self):
        """An unclosed marker frame is reported."""
        machine = SpanMachine()
        machine.feed("start <calc>1+1")
        self.assertTrue(machine.has_open_frame())

    def test_nested_frames_rejected(self):
        """Marker text inside a span expression raises GrammarError."""
        machine = SpanMachine()
        with self.assertRaises(GrammarError):
            machine.feed("<calc><calc>1</calc>")

    def test_substitution_pattern(self):
        """The public pattern replaces spans with computed text."""
        self.assertEqual(SPAN_RE.sub("X", "a <calc>1</calc> b"), "a X b")


if __name__ == "__main__":
    unittest.main()
