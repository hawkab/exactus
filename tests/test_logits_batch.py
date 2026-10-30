"""Shape-only batch and speculative layout tests for logits forcing.

Author: Olshansky Grigory
"""

import unittest
from collections.abc import Sequence

from exactus.adapters.logits_adapter import ExactusLogitsProcessor
from exactus.adapters.logits_state import LogitsStateError
from exactus.adapters.ngram import NgramDraftPredictor
from exactus.adapters.tokenizer import ByteTokenizer

__author__ = "Olshansky Grigory"

VOCAB = 256


def make_rows(count: int, width: int = VOCAB) -> list[list[float]]:
    """Return count untouched score rows of the given vocabulary width."""
    return [[1.0] * width for _ in range(count)]


class CountingSequence(Sequence):
    """Sequence double counting how many items callers copy out."""

    def __init__(self, items) -> None:
        """Store the wrapped items with a zero copy counter."""
        self._items, self.copied = list(items), 0

    def __len__(self) -> int:
        """Return the wrapped length."""
        return len(self._items)

    def __getitem__(self, index):
        """Return indexed items and count every element exposed."""
        value = self._items[index]
        self.copied += len(value) if isinstance(index, slice) else 1
        return value


class ProcessorCase(unittest.TestCase):
    """Shared base case holding one byte-tokenizer forcing processor."""

    def setUp(self) -> None:
        """Create one processor with the byte tokenizer."""
        self.processor = ExactusLogitsProcessor(ByteTokenizer())


class TestPerRequestPlans(ProcessorCase):
    """Independent plans, cursors and resets for concurrent requests."""

    def test_state_is_request_local(self):
        """Plans, cursors and resets never bleed between two requests."""
        self.processor.plan_text("AB", "A")
        self.processor.plan_text("XY", "B")
        self.assertEqual(self.processor.next_target("A"), ord("A"))
        self.assertEqual(self.processor.next_target("B"), ord("X"))
        self.processor.advance("A")
        self.assertEqual(self.processor.next_target("A"), ord("B"))
        self.assertEqual(self.processor.next_target("B"), ord("X"))
        self.processor.reset("A")
        self.assertIsNone(self.processor.next_target("A"))
        self.assertEqual(self.processor.next_target("B"), ord("X"))


class TestBatchLayout(ProcessorCase):
    """Documented row contracts for plain and verify layouts."""

    def setUp(self) -> None:
        """Register two requests with short byte plans."""
        super().setUp()
        self.processor.update_state(["A", "B"])
        self.processor.plan_text("ABC", "A")
        self.processor.plan_text("XY", "B")

    def test_plain_batch_forces_one_row_per_request(self):
        """Plain apply forces row i for the i-th active request."""
        rows = make_rows(2)
        self.processor.apply(rows)
        self.assertEqual(rows[0][ord("A")], 0.0)
        self.assertEqual(rows[1][ord("X")], 0.0)
        self.assertLess(rows[0][ord("B")], -100)
        self.assertEqual(self.processor.next_target("A"), ord("B"))
        self.assertEqual(self.processor.next_target("B"), ord("Y"))

    def test_spec_layout_forces_only_target_rows(self):
        """Unsupported speculative forcing fails before changing any row/cursor."""
        rows = make_rows(6)
        with self.assertRaises(LogitsStateError):
            self.processor.apply_with_spec_decode(rows, 2)
        self.assertEqual(rows, make_rows(6))
        self.assertEqual(self.processor.next_target("A"), ord("A"))


class TestFailurePolicy(ProcessorCase):
    """Locked policy: contract violations raise, faults flag disabling."""

    def test_missing_state_raises(self):
        """Operations on an untracked request raise LogitsStateError."""
        with self.assertRaises(LogitsStateError):
            self.processor.next_target("ghost")
        with self.assertRaises(LogitsStateError):
            self.processor.advance("ghost")

    def test_layout_violations_raise(self):
        """Row counts outside the documented contract raise."""
        self.processor.update_state(["A", "B"])
        with self.assertRaises(LogitsStateError):
            self.processor.apply(make_rows(3))
        with self.assertRaises(LogitsStateError):
            self.processor.apply_with_spec_decode(make_rows(6), 1)
        with self.assertRaises(LogitsStateError):
            ExactusLogitsProcessor(ByteTokenizer()).apply(make_rows(1))

    def test_out_of_vocabulary_target_raises(self):
        """Targets outside the row vocabulary raise instead of resetting."""
        self.processor.update_state(["A"])
        self.processor.plan_text("A", "A")
        with self.assertRaises(LogitsStateError):
            self.processor.apply(make_rows(1, 8))

    def test_engine_guard_fails_closed(self):
        """Unexpected faults fail closed, never continuing free generation."""
        self.processor.update_state(["A"])
        self.processor.plan_text("A", "A")
        batch = [(1.0,) * VOCAB]
        with self.assertRaises(LogitsStateError):
            self.processor.apply(batch)
        self.assertEqual(self.processor.diagnostics()["guard_failures"], 1)
        self.assertEqual(self.processor.diagnostics()["failed_requests"], 1)
        self.processor.plan_text("B", "unaffected")
        self.processor.apply_offline([0.0] * VOCAB, "unaffected")


class TestSlotLifecycle(ProcessorCase):
    """Slot release on finish and plan survival across preemption."""

    def test_finished_slot_release_allows_reuse(self):
        """A finished request frees its slot for the next occupant."""
        self.processor.update_state(["A"])
        self.processor.plan_text("AB", "A")
        self.processor.update_state({"request_ids": ["B"], "finished_ids": ["A"]})
        with self.assertRaises(LogitsStateError):
            self.processor.next_target("A")
        self.processor.plan_text("X", "B")
        rows = make_rows(1)
        self.processor.apply(rows)
        self.assertEqual(rows[0][ord("X")], 0.0)
        self.assertIsNone(self.processor.next_target("B"))

    def test_preempted_request_keeps_its_plan(self):
        """A temporarily absent request resumes its own cursor later."""
        self.processor.update_state(["A", "B"])
        self.processor.plan_text("AB", "A")
        self.processor.plan_text("XY", "B")
        self.processor.apply(make_rows(2))
        self.processor.update_state(["A"])
        self.processor.update_state(["A", "B"])
        rows = make_rows(2)
        self.processor.apply(rows)
        self.assertEqual(rows[0][ord("B")], 0.0)
        self.assertEqual(rows[1][ord("Y")], 0.0)


class TestNgramPrefixSizing(unittest.TestCase):
    """Predict copies only the n-gram suffix of a prefix, never all of it."""

    def setUp(self) -> None:
        """Fit one predictor on a small corpus."""
        self.predictor = NgramDraftPredictor(order=3)
        self.predictor.fit([[1, 2, 3, 4], [1, 2, 3, 5], [1, 2, 3, 5]])

    def test_prefix_is_not_copied_whole(self):
        """Lookup reads at most order tokens out of a long prefix."""
        prefix = CountingSequence([7] * 5000 + [1, 2, 3])
        self.predictor.predict(prefix)
        self.assertLessEqual(prefix.copied, 3)

    def test_long_prefix_and_iterator_match_list(self):
        """Long and streamed prefixes return the same drafts as the list."""
        listed = [(c.tokens, c.score) for c in self.predictor.predict([1, 2, 3])]
        long = [
            (c.tokens, c.score) for c in self.predictor.predict([7] * 5000 + [1, 2, 3])
        ]
        streamed = [
            (c.tokens, c.score) for c in self.predictor.predict(iter([1, 2, 3]))
        ]
        self.assertEqual(listed, long)
        self.assertEqual(listed, streamed)

    def test_docstring_has_no_verify_guarantee(self):
        """The docstring demands a verifier instead of promising protection."""
        doc = NgramDraftPredictor.__doc__
        self.assertIn("verifier", doc)
        self.assertIn("no accuracy guarantee", doc)
        self.assertNotIn("never change", doc)
