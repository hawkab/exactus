"""Tests for the drop-in proxy request and response rewriting.

Author: Olshansky Grigory
"""

import unittest

from exactus.adapters.digest import HashlibHasher
from exactus.adapters.in_process import InProcessExternalComputer
from exactus.proxy import (
    ExactusProxy,
    extract_content,
    prepare_request,
    replace_content,
)
from exactus.proxy_modes import RequestError
from exactus.prompts import system_hint

__author__ = "Olshansky Grigory"


class TestProxyRewriting(unittest.TestCase):
    """Verifies hint injection and content substitution without a network."""

    def setUp(self) -> None:
        """Create one proxy with the in-process external computer."""
        self.proxy = ExactusProxy(
            "http://upstream", InProcessExternalComputer(HashlibHasher())
        )

    def test_hint_injected(self):
        """The marker hint becomes the leading system message."""
        out = prepare_request({"messages": [{"role": "user", "content": "q"}]}, "HINT")
        self.assertEqual(out["messages"][0], {"role": "system", "content": "HINT"})

    def test_hint_not_duplicated(self):
        """Requests already carrying the protocol are forwarded unchanged."""
        payload = {
            "messages": [{"role": "system", "content": "<calc-final>1</calc-final>"}]
        }
        self.assertIs(prepare_request(payload, "HINT"), payload)

    def test_content_extraction(self):
        """Assistant text is extracted from an OpenAI-style response."""
        response = {"choices": [{"message": {"content": "hi"}}]}
        self.assertEqual(extract_content(response), "hi")

    def test_content_replacement(self):
        """Assistant text is replaced without touching other choices."""
        response = {
            "choices": [{"message": {"content": "old"}, "finish_reason": "stop"}]
        }
        out = replace_content(response, "new")
        self.assertEqual(out["choices"][0]["message"]["content"], "new")
        self.assertEqual(out["choices"][0]["finish_reason"], "stop")

    def test_compute_substitution(self):
        """Computed spans are substituted into the response content."""
        response = {
            "choices": [
                {
                    "message": {"content": "a <calc>2+3</calc> b"},
                    "finish_reason": "stop",
                }
            ]
        }
        out = self.proxy._compute({"messages": []}, response)
        self.assertEqual(extract_content(out), "a 5 b")

    def test_compute_final_answer(self):
        """A final span yields only the computed value."""
        response = {
            "choices": [
                {
                    "message": {"content": "<calc-final>6*7</calc-final>"},
                    "finish_reason": "stop",
                }
            ]
        }
        out = self.proxy._compute({"messages": []}, response)
        self.assertEqual(extract_content(out), "42")

    def test_compute_error_is_strict(self):
        """Compute failures raise a machine-readable request error."""
        response = {
            "choices": [
                {"message": {"content": "<calc>1/0</calc>"}, "finish_reason": "stop"}
            ]
        }
        with self.assertRaises(RequestError) as caught:
            self.proxy._compute({"messages": []}, response)
        body = caught.exception.body()
        self.assertIn("error_code", body)

    def test_unknown_path(self):
        """Unknown paths return 404."""
        status, body = self.proxy.handle("/v1/audio", {})
        self.assertEqual(status, 404)
        self.assertIn("error", body)

    def test_hint_preset_model(self):
        """The user model name maps to a marker hint."""
        self.assertIn("<calc-final>", system_hint("nvidia/qwen3.8-flash-next-nvfp4"))

    def test_unknown_model_gets_default(self):
        """Unknown models receive the generic hint."""
        self.assertIn("<calc-final>", system_hint(None))


if __name__ == "__main__":
    unittest.main()
