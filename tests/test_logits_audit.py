"""Audit coverage and hot-path regressions."""

__author__ = "Olshansky Grigory"

import unittest

from exactus.adapters.logits_adapter import ExactusLogitsProcessor
from exactus.adapters.logits_audit import audit_snapshot
from exactus.adapters.logits_state import LogitsStateError
from exactus.adapters.tokenizer import ByteTokenizer


class TestAudit(unittest.TestCase):
    """Exercise Audit behavior with isolated fixtures."""

    def full(self):
        """Create a logits processor with explicit full diagnostic auditing."""
        return ExactusLogitsProcessor(ByteTokenizer(), audit_mode="full")

    def test_wanted_and_corrected(self):
        """Verify wanted and corrected."""
        p = self.full()
        p.plan_text("AB")
        row = [0.0] * 256
        row[65], row[66] = 5.0, 2.0
        p.apply_offline(row)
        self.assertEqual(p.audit()[0]["wanted"], 65)
        self.assertEqual(p.audit()[0]["logit_gap"], 3.0)
        row = [0.0] * 256
        row[3], row[66] = 9.0, 4.0
        p.apply_offline(row)
        self.assertEqual(p.audit()[1]["wanted"], 3)
        self.assertEqual(
            p.audit_summary(),
            {
                "forced": 2,
                "corrected": 1,
                "rate": 0.5,
                "rate_denominator": 2,
                "snapshot_errors": 0,
                "skipped": 0,
            },
        )

    def test_default_off_has_no_vocab_scan(self):
        """Verify default off has no vocab scan."""

        class Row(list):
            """Exercise Row behavior with isolated fixtures."""

            reads = 0

            def __getitem__(self, i):
                """Return the item selected by the supplied key or index."""
                self.reads += 1
                return super().__getitem__(i)

        p = ExactusLogitsProcessor(ByteTokenizer())
        p.plan_text("A")
        row = Row([0.0] * 131072)
        p.apply_offline(row)
        self.assertEqual(row.reads, 1)
        self.assertEqual(p.audit(), [])
        self.assertEqual(p.audit_summary()["forced"], 1)
        self.assertEqual(p.audit_summary()["skipped"], 1)
        self.assertIsNone(p.audit_summary()["rate"])

    def test_missing_snapshot_counts_real_forcing(self):
        """Verify missing snapshot counts real forcing."""

        class Opaque:
            """Exercise Opaque behavior with isolated fixtures."""

            def __float__(self):
                """Simulate an unavailable scalar snapshot conversion."""
                raise TypeError("unreadable")

        p = self.full()
        p.plan_text("A")
        row = [Opaque()] * 256
        row[65] = 0.0
        p.apply_offline(row)
        summary = p.audit_summary()
        self.assertEqual(summary["forced"], 1)
        self.assertEqual(summary["snapshot_errors"], 1)
        self.assertEqual(summary["rate_denominator"], 0)
        self.assertIsNone(summary["rate"])
        self.assertEqual(row[65], 0.0)

    def test_bounded_buffer_and_totals(self):
        """Verify bounded buffer and totals."""
        p = self.full()
        p.plan_text("x" * 300)
        for _ in range(300):
            p.apply_offline([0.0] * 256)
        self.assertEqual(len(p.audit()), 256)
        self.assertEqual(p.audit()[0]["position"], 44)
        self.assertEqual(p.audit_summary()["forced"], 300)
        self.assertEqual(p.audit_summary()["corrected"], 300)

    def test_sampled_coverage(self):
        """Verify sampled coverage."""
        p = ExactusLogitsProcessor(ByteTokenizer(), audit_mode="sampled", audit_every=3)
        p.plan_text("A" * 7)
        for _ in range(7):
            p.apply_offline([0.0] * 256)
        self.assertEqual(p.audit_summary()["forced"], 7)
        self.assertEqual(p.audit_summary()["rate_denominator"], 3)
        self.assertEqual(p.audit_summary()["skipped"], 4)

    def test_reset_and_finished_clear(self):
        """Verify reset and finished clear."""
        p = self.full()
        p.update_state(["a"])
        p.plan_text("A", "a")
        p.apply([[0.0] * 256])
        self.assertEqual(len(p.audit("a")), 1)
        p.reset("a")
        self.assertEqual(p.audit_summary()["forced"], 0)
        p.update_state({"request_ids": [], "finished_ids": ["a"]})
        self.assertEqual(p.audit("a"), [])

    def test_masks_never_reenabled(self):
        """Verify masks never reenabled."""
        p = self.full()
        p.plan_text("A")
        row = [0.0] * 256
        row[65] = float("-inf")
        before = row[:]
        with self.assertRaises(LogitsStateError):
            p.apply_offline(row)
        self.assertEqual(row, before)
        self.assertEqual(p.next_target(), 65)

    def test_full_tensor_snapshot_uses_vector_reduction(self):
        """Verify full tensor snapshot uses vector reduction."""

        class Values:
            """Exercise Values behavior with isolated fixtures."""

            def __init__(self, values):
                """Initialize Values state and dependencies."""
                self.values = values

            def cpu(self):
                """Return the compact tensor fixture as a host-side value."""
                return self

            def tolist(self):
                """Convert the compact tensor fixture to a plain value list."""
                return self.values

        class Tensor:
            """Exercise Tensor behavior with isolated fixtures."""

            def detach(self):
                """Return the detached tensor fixture without scalar reads."""
                return self

            def topk(self, n):
                """Return the fixture top-value reduction used by the audit snapshot."""
                self.n = n
                return Values([9.0, 7.0]), Values([3, 65])

            def __getitem__(self, _):
                """Return the item selected by the supplied key or index."""
                raise AssertionError("no scalar extraction")

        row = Tensor()
        self.assertEqual(audit_snapshot(row, 131072), (3, 2.0))
        self.assertEqual(row.n, 2)

    def test_invalid_config(self):
        """Verify invalid config."""
        for kwargs in ({"audit_mode": "bad"}, {"audit_every": 0}):
            with self.assertRaises(LogitsStateError):
                ExactusLogitsProcessor(ByteTokenizer(), **kwargs)
