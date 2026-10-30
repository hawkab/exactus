"""Tests for computed value rendering formats.

Author: Olshansky Grigory
"""
import unittest

from exactus.domain.errors import FormatError
from exactus.domain.value import ComputeValue

__author__ = "Olshansky Grigory"


class TestComputeValue(unittest.TestCase):
    """Verifies format validation and rendering."""

    def test_decimal(self):
        """Decimal rendering is the default."""
        self.assertEqual(ComputeValue(255).render(), "255")

    def test_hex(self):
        """Hex rendering is prefixed like Python hex."""
        self.assertEqual(ComputeValue(255, "hex").render(), "0xff")

    def test_bin(self):
        """Binary rendering is prefixed like Python bin."""
        self.assertEqual(ComputeValue(5, "bin").render(), "0b101")

    def test_negative(self):
        """Negative values render with a sign in every format."""
        self.assertEqual(ComputeValue(-1, "hex").render(), "-0x1")

    def test_unknown_format(self):
        """Unknown formats raise FormatError."""
        with self.assertRaises(FormatError):
            ComputeValue(1, "oct")

    def test_non_int_rejected(self):
        """Non-int payloads raise FormatError."""
        with self.assertRaises(FormatError):
            ComputeValue(1.5)


if __name__ == "__main__":
    unittest.main()
