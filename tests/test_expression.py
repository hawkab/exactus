"""Tests for the bounded expression evaluator.

Author: Olshansky Grigory
"""
import unittest

from exactus.domain.errors import BudgetError, DataError, GrammarError
from exactus.domain.expression import ExpressionEvaluator

__author__ = "Olshansky Grigory"


class TestExpressionEvaluator(unittest.TestCase):
    """Verifies grammar, semantics, budgets and data references."""

    def setUp(self) -> None:
        """Create one evaluator with default budgets."""
        self.ev = ExpressionEvaluator()

    def value(self, source: str, data=None) -> int:
        """Evaluate source and return its integer value."""
        return self.ev.evaluate(source, data).integer

    def test_arithmetic_precedence(self):
        """Addition binds looser than multiplication."""
        self.assertEqual(self.value("2+3*4"), 14)

    def test_power_right_associative(self):
        """Power chains to the right and beats unary minus."""
        self.assertEqual(self.value("2^3^2"), 512)
        self.assertEqual(self.value("-2^2"), -4)

    def test_power_synonym(self):
        """Double-star is an exact synonym of caret power."""
        self.assertEqual(self.value("2**10"), 1024)

    def test_trunc_division_and_remainder(self):
        """Slash and percent truncate toward zero."""
        self.assertEqual(self.value("-7/2"), -3)
        self.assertEqual(self.value("-7%2"), -1)

    def test_floor_division(self):
        """Double-slash floors like Python."""
        self.assertEqual(self.value("-7//2"), -4)

    def test_mod_keyword(self):
        """The mod keyword matches percent semantics."""
        self.assertEqual(self.value("5 mod 2"), 1)

    def test_bitwise_operators(self):
        """And, or, not, xor and shifts follow Python semantics."""
        self.assertEqual(self.value("6&3"), 2)
        self.assertEqual(self.value("6|3"), 7)
        self.assertEqual(self.value("~0"), -1)
        self.assertEqual(self.value("2 xor 3"), 1)
        self.assertEqual(self.value("1<<10"), 1024)
        self.assertEqual(self.value("(-8)>>1"), -4)

    def test_hex_and_binary_literals(self):
        """Prefixed literals and underscores parse correctly."""
        self.assertEqual(self.value("0xff + 0b1"), 256)
        self.assertEqual(self.value("1_000_000"), 1000000)

    def test_data_reference_big_endian(self):
        """A data reference yields the big-endian integer of its bytes."""
        self.assertEqual(self.value("@n", {"n": bytes([1, 0])}), 256)

    def test_unknown_data_reference(self):
        """Missing data references raise DataError."""
        with self.assertRaises(DataError):
            self.value("@missing")

    def test_division_by_zero(self):
        """Division by zero is a grammar-level failure."""
        with self.assertRaises(GrammarError):
            self.value("1/0")

    def test_negative_exponent(self):
        """Negative exponents are rejected."""
        with self.assertRaises(GrammarError):
            self.value("2^-1")

    def test_trailing_input(self):
        """Trailing tokens after the expression are rejected."""
        with self.assertRaises(GrammarError):
            self.value("1 2")

    def test_bit_budget(self):
        """Results above the bit budget raise BudgetError."""
        with self.assertRaises(BudgetError):
            self.value("2^4096")

    def test_shift_budget(self):
        """Shift amounts outside range raise BudgetError."""
        with self.assertRaises(BudgetError):
            self.value("1<<100000")

    def test_flat_chain_long(self):
        """Flat operator chains pass up to the node budget."""
        self.assertEqual(self.value("+".join(["1"] * 66)), 66)

    def test_nesting_budget(self):
        """Parenthesis nesting above the depth budget raises BudgetError."""
        with self.assertRaises(BudgetError):
            self.value("(" * 65 + "1" + ")" * 65)

    def test_node_budget(self):
        """Expressions above the node budget raise BudgetError."""
        with self.assertRaises(BudgetError):
            ExpressionEvaluator(max_nodes=10).evaluate("+".join(["1"] * 20))


if __name__ == "__main__":
    unittest.main()
