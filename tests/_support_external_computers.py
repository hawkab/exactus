"""External computer correctness, capability boundaries, IPC and HTTP contracts."""

__author__ = "Olshansky Grigory"


import unittest

from exactus.adapters.digest import HashlibHasher
from exactus.adapters.in_process import InProcessExternalComputer

TABLES = {
    "events": {
        "columns": ["root", "attempt", "delivered", "tokens"],
        "rows": [
            ["a", 1, True, 101],
            ["b", 1, False, 20],
            ["b", 2, True, 73],
            ["c", 1, False, 32],
        ],
    }
}


class BatchExternalComputerTestsFixture(unittest.TestCase):
    """Provide shared setup and helpers for BatchExternalComputerTests regressions."""

    def setUp(self):
        """Prepare isolated dependencies for each test."""
        self.external_computer = InProcessExternalComputer(HashlibHasher())
