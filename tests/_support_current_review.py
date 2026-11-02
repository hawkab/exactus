"""Executable counterexamples from the e511bf4 review."""

__author__ = "Olshansky Grigory"


import unittest

from exactus.adapters.digest import HashlibHasher
from exactus.adapters.in_process import InProcessExternalComputer
from exactus.proxy import ExactusProxy


class CurrentReviewFixture(unittest.TestCase):
    """Provide shared setup and helpers for CurrentReview regressions."""

    def setUp(self):
        """Prepare isolated dependencies for each test."""
        self.proxy = ExactusProxy("unused", InProcessExternalComputer(HashlibHasher()))
        self.addCleanup(self.proxy.close)
