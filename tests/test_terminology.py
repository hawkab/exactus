"""Canonical external computer names across library, worker and CLI."""

import hashlib
import json
from pathlib import Path
import subprocess
import sys
import unittest

from exactus import ExternalComputer
from exactus.adapters.digest import HashlibHasher
from exactus.adapters.in_process import InProcessExternalComputer
from exactus.adapters.worker import WorkerExternalComputer
from exactus.external_computers import run_external_computers, run_plan
from exactus.external_computers.values import packed


class ExternalComputerTerminology(unittest.TestCase):
    def test_public_port_and_batch_exports(self):
        external_computer = InProcessExternalComputer(HashlibHasher())
        self.assertIsInstance(external_computer, ExternalComputer)
        spec = {"kind": "calc", "expression": "6*7"}
        result = run_external_computers([{"id": "answer", **spec}], external_computer)
        self.assertEqual(result["results"][0]["value"], 42)
        self.assertEqual(
            result["results"][0]["input_sha256"],
            hashlib.sha256(
                ("exactus-external-computer-v1\0" + packed(spec)).encode()
            ).hexdigest(),
        )
        plan = run_plan(
            {"tasks": [{"id": "answer", **spec}], "answer": {"$ref": "answer"}},
            external_computer,
        )
        self.assertEqual(plan["answer"], 42)
        self.assertEqual(
            plan["results"][0]["input_sha256"], result["results"][0]["input_sha256"]
        )

    def test_worker_uses_external_computer_task_protocol(self):
        external_computer = WorkerExternalComputer()
        self.addCleanup(external_computer.close)
        result = external_computer.run_external_computers(
            [
                {
                    "id": "answer",
                    "kind": "python_expr",
                    "expression": "sum(xs)",
                    "inputs": {"xs": [1, 2, 3]},
                }
            ]
        )
        self.assertEqual(result["results"][0]["value"], 6)
        self.assertEqual(result["model_calls"], 0)

    def test_external_computers_cli(self):
        result = subprocess.run(
            [sys.executable, "-m", "exactus.adapters.cli", "external-computers"],
            cwd=Path(__file__).resolve().parents[1],
            input=json.dumps(
                {"tasks": [{"id": "answer", "kind": "calc", "expression": "7*8"}]}
            ),
            capture_output=True,
            text=True,
            timeout=10,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout)["results"][0]["value"], 56)


if __name__ == "__main__":
    unittest.main()
