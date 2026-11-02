"""Actual loopback transport caps/deadlines and ASGI lifecycle regressions."""

__author__ = "Olshansky Grigory"


import unittest

from exactus.adapters.digest import HashlibHasher
from exactus.adapters.in_process import InProcessExternalComputer


class DeepHTTPFixture(unittest.TestCase):
    """Provide shared setup and helpers for DeepHTTP regressions."""

    def external_computer(self):
        """Create an in-process external computer for the current test."""
        return InProcessExternalComputer(HashlibHasher())
