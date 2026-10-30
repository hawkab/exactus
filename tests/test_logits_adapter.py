"""Tests for the logits forcing adapter.

Author: Olshansky Grigory
"""
import unittest

from exactus.adapters.logits_adapter import ExactusLogitsProcessor
from exactus.adapters.tokenizer import ByteTokenizer

__author__ = "Olshansky Grigory"


class TestExactusLogitsProcessor(unittest.TestCase):
    """Verifies forcing-plan state transitions and offline row forcing."""

    def setUp(self) -> None:
        """Create one processor with the byte tokenizer."""
        self.processor = ExactusLogitsProcessor(ByteTokenizer())

    def test_plan_and_target(self):
        """Planning text exposes its first token as the target."""
        self.processor.plan_text("42")
        self.assertEqual(self.processor.next_target(), ord("4"))

    def test_offline_forcing(self):
        """The offline row forces exactly the target token."""
        self.processor.plan_text("A")
        scores = [1.0] * 256
        self.processor.apply_offline(scores)
        self.assertEqual(scores[ord("A")], 0.0)
        self.assertLess(scores[ord("B")], -100)

    def test_plan_advances(self):
        """Consecutive calls consume the plan in order."""
        self.processor.plan_text("AB")
        self.assertEqual(self.processor.next_target(), ord("A"))
        self.processor.apply_offline([0.0] * 256)
        self.assertEqual(self.processor.next_target(), ord("B"))

    def test_no_plan_is_noop(self):
        """Without a plan the row passes through unchanged."""
        scores = [1.0] * 8
        self.assertIs(self.processor.apply_offline(scores), scores)
        self.assertEqual(scores, [1.0] * 8)

    def test_out_of_vocab_target_raises(self):
        """Out-of-vocabulary targets raise instead of silently resetting."""
        from exactus.adapters.logits_state import LogitsStateError
        self.processor.plan_text("A")
        with self.assertRaises(LogitsStateError):
            self.processor.apply_offline([1.0, 1.0])

    def test_interface_stubs(self):
        """Params stub never raises; malformed batch updates are rejected."""
        from exactus.adapters.logits_state import LogitsStateError
        self.processor.validate_params(object())
        with self.assertRaises(LogitsStateError):
            self.processor.update_state(object())
        self.assertFalse(self.processor.is_argmax_invariant())


if __name__ == "__main__":
    unittest.main()
