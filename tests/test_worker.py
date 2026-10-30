"""Tests for the persistent sandboxed worker adapter.

Author: Olshansky Grigory
"""

import unittest

from exactus.adapters.worker import WorkerExternalComputer
from exactus.domain.errors import GrammarError

__author__ = "Olshansky Grigory"


class TestWorkerExternalComputer(unittest.TestCase):
    """Verifies worker parity, recycling and error mapping."""

    def setUp(self) -> None:
        """Create one persistent worker external computer."""
        self.worker = WorkerExternalComputer(max_calls=4)

    def tearDown(self) -> None:
        """Terminate the worker process."""
        self.worker.close()

    def test_parity_with_reference(self):
        """Worker results match known exact values."""
        self.assertEqual(self.worker.evaluate("2^10").integer, 1024)
        self.assertEqual(self.worker.evaluate("-7/2").integer, -3)

    def test_data_reference_roundtrip(self):
        """Request-scoped data survives the JSON transport."""
        value = self.worker.evaluate("sha256(@m) % 1000", {"m": b"abc"})
        self.assertTrue(value.integer >= 0)

    def test_recycling_after_max_calls(self):
        """The worker recycles transparently after max_calls evaluations."""
        for i in range(7):
            self.assertEqual(self.worker.evaluate(f"{i}+1").integer, i + 1)

    def test_error_mapping(self):
        """Domain failures map to domain exceptions, not crashes."""
        with self.assertRaises(GrammarError):
            self.worker.evaluate("1/0")

    def test_hex_output_format(self):
        """Output format passes through the transport."""
        self.assertEqual(
            self.worker.evaluate("255", output_format="hex").render(), "0xff"
        )


if __name__ == "__main__":
    unittest.main()
