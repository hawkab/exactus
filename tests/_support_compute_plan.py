"""PoT/CoA plan compilation, faithful reification and transport boundaries."""

__author__ = "Olshansky Grigory"


import unittest

from exactus.adapters.digest import HashlibHasher
from exactus.adapters.in_process import InProcessExternalComputer
from exactus.proxy import ExactusProxy


def fixture():
    """Build a source-bound plan fixture with independently specified expected results."""
    return {
        "tasks": [
            {
                "id": "safe",
                "kind": "python_expr",
                "expression": "available >= required",
                "inputs": {
                    "available": {"$ref": "capacity"},
                    "required": {"$input": "demand"},
                },
            },
            {
                "id": "capacity",
                "kind": "python_expr",
                "expression": "min(sum(n) for n in nodes)",
                "inputs": {"nodes": {"$input": "nodes"}},
            },
            {"id": "dead", "kind": "calc", "expression": "2**9999999"},
        ],
        "answer": {
            "capacity": {"$ref": "capacity"},
            "decision": {"$if": {"$ref": "safe"}, "then": "go", "else": "no-go"},
            "explanation": {"$text": ["N+1: ", {"$ref": "capacity"}, " tokens/s"]},
        },
    }


class PlanTestsFixture(unittest.TestCase):
    """Provide shared setup and helpers for PlanTests regressions."""

    def setUp(self):
        """Prepare isolated dependencies for each test."""
        self.external_computer = InProcessExternalComputer(HashlibHasher())
        self.inputs = {"nodes": [[51, 48], [62, 55]], "demand": 110}


class PlanTransportTestsFixture(unittest.TestCase):
    """Provide shared setup and helpers for PlanTransportTests regressions."""

    def proxy(self):
        """Create an isolated proxy with deterministic computation dependencies."""
        proxy = ExactusProxy("unused", InProcessExternalComputer(HashlibHasher()))
        self.addCleanup(proxy.close)
        return proxy
