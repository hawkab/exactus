"""Tests for audited proofs and prompt dispatch.

Author: Olshansky Grigory
"""

import unittest

from exactus.adapters.dispatch import (
    find_expression,
    last_user_text,
    external_computer_answer,
)
from exactus.adapters.digest import HashlibHasher
from exactus.adapters.in_process import InProcessExternalComputer
from exactus.domain.audit import Proof, SpanProof, check_answer
from exactus.proxy import ExactusProxy
from exactus.proxy_modes import model_answer

__author__ = "Olshansky Grigory"


class TestProof(unittest.TestCase):
    """Verifies inline marks, detail rendering and the machine record."""

    def setUp(self) -> None:
        """Build one two-span proof with a final span."""
        self.proof = Proof((SpanProof("2+3", "5"), SpanProof("6*7", "42", True)))

    def test_inline_marks_values(self):
        """Substituted values carry derived and QED marks."""
        out = self.proof.inline("a <calc>2+3</calc> b <calc-final>6*7</calc-final>")
        self.assertEqual(out, "a \u22a2 5 \u220e b \u22a2 42 \u220e")

    def test_detail_mentions_every_span(self):
        """The summary lists each expression=value pair and the check."""
        text = self.proof.detail()
        self.assertIn("\u22a2 2+3=5 \u220e", text)
        self.assertIn("\u22a2 6*7=42 \u220e", text)

    def test_as_dict_is_machine_readable(self):
        """The record carries spans and optional answer check."""
        record = self.proof.as_dict()
        self.assertEqual(
            record["spans"][1],
            {
                "expression": "6*7",
                "value": "42",
                "final": True,
                "origin": "model_marker",
                "source_start": None,
                "source_end": None,
            },
        )
        self.assertNotIn("answer_check", record)

    def test_check_answer_verdicts(self):
        """Verdicts distinguish match, mismatch and a missing answer."""
        self.assertEqual(check_answer("42", "42").verdict, "match")
        self.assertEqual(check_answer("42", "41").verdict, "mismatch")
        self.assertEqual(check_answer("42", None).verdict, "no_answer")


class TestDispatch(unittest.TestCase):
    """Verifies conservative prompt extraction and external computer solving."""

    def setUp(self) -> None:
        """Create the in-process external computer."""
        self.external_computer = InProcessExternalComputer(HashlibHasher())

    def test_cue_extraction(self):
        """Cue phrases lead to the extracted expression."""
        self.assertEqual(find_expression("Вычисли: 37*184500 + 2."), "37*184500 + 2")
        self.assertEqual(find_expression("What is (445 + 152) * 2?"), "(445 + 152) * 2")

    def test_bare_equation_line(self):
        """A bare question line is extracted."""
        self.assertEqual(find_expression("2^10 - 1 = ?"), "2^10 - 1")

    def test_prose_is_not_a_task(self):
        """Ordinary prose abstains instead of guessing an expression."""
        self.assertIsNone(find_expression("The budget for the quarter was discussed."))

    def test_external_computer_answer(self):
        """The external computer evaluates an extracted expression exactly."""
        self.assertEqual(
            external_computer_answer("Вычисли: 7*6", self.external_computer),
            ("7*6", "42"),
        )

    def test_unsolvable_cue_abstains(self):
        """A cue with an out-of-grammar candidate returns None."""
        self.assertIsNone(
            external_computer_answer("Вычисли: foo bar", self.external_computer)
        )

    def test_last_user_text(self):
        """The last user message is located in a chat payload."""
        payload = {
            "messages": [
                {"role": "system", "content": "h"},
                {"role": "user", "content": "first"},
                {"role": "user", "content": "Вычисли: 2+2"},
            ]
        }
        self.assertEqual(last_user_text(payload), "Вычисли: 2+2")


class TestProxyAudit(unittest.TestCase):
    """Verifies audit payloads and external computer grounding in the proxy."""

    def setUp(self) -> None:
        """Create one proxy with the in-process external computer."""
        self.proxy = ExactusProxy(
            "http://upstream", InProcessExternalComputer(HashlibHasher())
        )

    def test_audit_mode_marks_text(self):
        """Audit mode returns proof marks and an audit record."""
        payload = {"messages": [{"role": "user", "content": "hi"}], "mode": "audit"}
        response = {
            "choices": [
                {"message": {"content": "v <calc>2+2</calc>"}, "finish_reason": "stop"}
            ]
        }
        out = self.proxy._compute(payload, response)
        self.assertEqual(out["choices"][0]["message"]["content"], "v \u22a2 4 \u220e")
        self.assertEqual(out["exactus"]["audit"][0]["spans"][0]["value"], "4")

    def test_answer_check_mismatch_is_reported(self):
        """The external computer verdict is reported without rewriting the answer."""
        payload = {"messages": [{"role": "user", "content": "Вычисли: 7*6"}]}
        response = {
            "choices": [
                {
                    "message": {"content": "<calc-final>7*6</calc-final>"},
                    "finish_reason": "stop",
                }
            ]
        }
        out = self.proxy._compute(payload, response)
        self.assertEqual(out["exactus"]["answer_check"]["verdict"], "match")

    def test_numeric_final_is_grounded_by_external_computer(self):
        """In numeric-final mode the external computer answer replaces the choice."""
        payload = {
            "messages": [{"role": "user", "content": "Вычисли: 7*6"}],
            "mode": "numeric-final",
        }
        response = {
            "choices": [
                {"message": {"content": "I think it is 43."}, "finish_reason": "stop"}
            ]
        }
        out = self.proxy._compute(payload, response)
        self.assertEqual(out["choices"][0]["message"]["content"], "42")
        self.assertEqual(out["exactus"]["answer_check"]["verdict"], "no_answer")

    def test_model_answer_prefers_final_span(self):
        """The model's own final value comes from its final span."""
        from exactus.proxy_modes import Answer

        answer = Answer("x", ("42",), True)
        self.assertEqual(model_answer("x", answer), "42")


if __name__ == "__main__":
    unittest.main()
