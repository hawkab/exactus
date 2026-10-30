"""Tests for the exact-span compute use case.

Author: Olshansky Grigory
"""

import unittest

from exactus.adapters.digest import HashlibHasher
from exactus.adapters.in_process import InProcessExternalComputer
from exactus.application.compute import ComputeService
from exactus.domain.errors import GrammarError

__author__ = "Olshansky Grigory"


class TestComputeService(unittest.TestCase):
    """Verifies contour one end to end with the in-process external computer."""

    def setUp(self) -> None:
        """Create one compute service over the in-process adapter."""
        self.service = ComputeService(InProcessExternalComputer(HashlibHasher()))

    def test_intermediate_substitution(self):
        """Computed values replace spans in the transcript."""
        result = self.service.run("total <calc>2+3</calc> ok", require_final=False)
        self.assertEqual(result.text, "total 5 ok")
        self.assertEqual(len(result.operations), 1)

    def test_final_answer_wins(self):
        """A final span sets the answer to the computed value only."""
        result = self.service.run(
            "guess 7 <calc-final>37*184500</calc-final>", require_final=True
        )
        self.assertEqual(result.text, "6826500")
        self.assertTrue(result.exact_final)

    def test_hash_span_with_data(self):
        """Hash calls resolve request-scoped data references."""
        result = self.service.run(
            "<calc-final>sha256(@m) % 1000</calc-final>",
            {"m": b"abc"},
            require_final=True,
        )
        self.assertEqual(len(result.operations), 1)
        self.assertTrue(result.operations[0].value.isdigit())

    def test_output_format(self):
        """The output format applies to rendered values."""
        result = self.service.run(
            "<calc-final>255</calc-final>", require_final=True, output_format="hex"
        )
        self.assertEqual(result.text, "0xff")

    def test_missing_final_rejected(self):
        """Require-final raises when no final span exists."""
        with self.assertRaises(GrammarError):
            self.service.run("<calc>1+1</calc>", require_final=True)

    def test_open_frame_rejected(self):
        """Incomplete frames raise GrammarError."""
        with self.assertRaises(GrammarError):
            self.service.run("start <calc>1+1", require_final=False)

    def test_multiple_spans_ordered(self):
        """Spans substitute in order across the transcript."""
        result = self.service.run(
            "<calc>1</calc>-<calc>2</calc>-<calc-final>3</calc-final>",
            require_final=True,
        )
        self.assertEqual(result.text, "3")
        self.assertEqual(len(result.operations), 3)


if __name__ == "__main__":
    unittest.main()
