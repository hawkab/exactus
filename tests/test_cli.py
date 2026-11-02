"""Exercise CLI data decoding through the real command-line entry point."""

__author__ = "Olshansky Grigory"

import base64
import hashlib
import subprocess
import sys
import unittest


class CLIDataTests(unittest.TestCase):
    """Keep malformed inputs from silently changing an exact computation."""

    def _evaluate(self, assignment, *, worker=False, expression="sha256(@blob)"):
        """Run an evaluation and capture its exit status and output streams."""
        command = [
            sys.executable,
            "-m",
            "exactus.adapters.cli",
            "eval",
            expression,
            "--data",
            assignment,
        ]
        if worker:
            command.append("--worker")
        return subprocess.run(command, capture_output=True, text=True)

    def test_invalid_base64_is_rejected_without_a_result_or_traceback(self):
        """Reject invalid alphabet, padding and non-ASCII data in either mode."""
        for payload in ("%%%", "Zg", "Zg===", "данные"):
            for worker in (False, True):
                with self.subTest(payload=payload, worker=worker):
                    result = self._evaluate(f"blob={payload}", worker=worker)
                    self.assertEqual(result.returncode, 1)
                    self.assertEqual(result.stdout, "")
                    self.assertIn("base64", result.stderr)
                    self.assertNotIn("Traceback", result.stderr)
                    self.assertNotIn(payload, result.stderr)

    def test_data_assignment_requires_a_name_and_separator(self):
        """Reject malformed assignments before evaluating even an unused input."""
        for assignment in ("blob", "=Zg=="):
            with self.subTest(assignment=assignment):
                result = self._evaluate(assignment, expression="6*7")
                self.assertEqual(result.returncode, 1)
                self.assertEqual(result.stdout, "")
                self.assertIn("name=base64", result.stderr)
                self.assertNotIn("Traceback", result.stderr)

    def test_binary_and_empty_payloads_keep_their_exact_values(self):
        """Accept valid padded binary data and intentionally empty payloads."""
        for value in (b"", b"\x00\xff\x01\x80", b"exactus"):
            payload = base64.b64encode(value).decode("ascii")
            expected = str(int.from_bytes(hashlib.sha256(value).digest(), "big"))
            for worker in (False, True):
                with self.subTest(value=value, worker=worker):
                    result = self._evaluate(f"blob={payload}", worker=worker)
                    self.assertEqual(result.returncode, 0, result.stderr)
                    self.assertEqual(result.stdout.strip(), expected)
                    self.assertEqual(result.stderr, "")
