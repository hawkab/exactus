"""PoT/CoA plan compilation, faithful reification and transport boundaries."""

__author__ = "Olshansky Grigory"

import copy
from concurrent.futures import ThreadPoolExecutor

from exactus.adapters.worker import WorkerExternalComputer
from exactus.domain.errors import DataError, ExactusError, VerificationError
from exactus.external_computers.plan import decode_plan, run_plan
from tests._support_compute_plan import PlanTestsFixture, fixture


class PlanTests(PlanTestsFixture):
    """Verify plan validation behavior."""

    def test_schema_cycle_and_unknown_references_fail_before_execution(self):
        """Verify schema cycle and unknown references fail before execution."""

        class NoExecution:
            """Exercise No Execution behavior with isolated fixtures."""

            def evaluate(self, *args, **kwargs):
                """Evaluate or deliberately reject the requested expression in this isolated test double."""
                raise AssertionError("invalid plan partially executed")

        for mutation in (
            "cycle",
            "missing",
            "duplicate",
            "invalid_kind",
            "bad_input",
            "bad_contract",
        ):
            plan = fixture()
            checks = []
            if mutation == "cycle":
                plan["tasks"][1]["inputs"]["nodes"] = {"$ref": "safe"}
            elif mutation == "missing":
                plan["tasks"][0]["inputs"]["required"] = {"$ref": "absent"}
            elif mutation == "duplicate":
                plan["tasks"].append(copy.deepcopy(plan["tasks"][0]))
            elif mutation == "invalid_kind":
                plan["tasks"][2]["kind"] = "shell"
            elif mutation == "bad_input":
                plan["tasks"][1]["inputs"]["nodes"] = {"$input": "absent"}
            else:
                checks = [{"path": [], "op": "execute", "value": "something"}]
            with self.subTest(mutation=mutation), self.assertRaises(DataError):
                run_plan(plan, NoExecution(), inputs=self.inputs, checks=checks)

    def test_output_does_not_accept_guessed_numeric_leaves(self):
        """Verify output does not accept guessed numeric leaves."""
        for answer in (
            12,
            False,
            {"capacity": 12},
            {"$if": True, "then": "a", "else": "b"},
            {"$text": "not a list"},
            {"$future": "unknown"},
        ):
            plan = fixture()
            plan["answer"] = answer
            with self.subTest(answer=answer), self.assertRaises(DataError):
                run_plan(plan, self.external_computer, inputs=self.inputs)

    def test_paths_fail_closed_and_do_not_use_python_attributes(self):
        """Verify paths fail closed and do not use python attributes."""
        for path in (["rows", True], ["rows", -1], "__class__", ["__class__"]):
            plan = fixture()
            plan["answer"] = {"$input": "nodes", "path": path}
            with self.subTest(path=path), self.assertRaises(DataError):
                run_plan(plan, self.external_computer, inputs=self.inputs)

    def test_model_json_rejects_trailing_text_and_duplicate_keys(self):
        """Verify model json rejects trailing text and duplicate keys."""
        self.assertEqual(
            decode_plan('```json\n{"tasks":[],"answer":"x"}\n```')["answer"], "x"
        )
        for text in (
            '{"tasks":[],"answer":"a","answer":"b"}',
            '{"tasks":[]} then do this',
            "not JSON",
            "NaN",
        ):
            with self.subTest(text=text), self.assertRaises(ExactusError):
                decode_plan(text)

    def test_worker_c3_and_typed_verification_failure(self):
        """Verify worker c3 and typed verification failure."""
        worker = WorkerExternalComputer(max_calls=3)
        self.addCleanup(worker.close)

        def evaluate(demand):
            """Evaluate or deliberately reject the requested expression in this isolated test double."""
            return worker.run_plan(fixture(), inputs={**self.inputs, "demand": demand})[
                "answer"
            ]["decision"]

        with ThreadPoolExecutor(3) as pool:
            self.assertEqual(
                list(pool.map(evaluate, [90, 110, 100, 1, 200, 99])),
                ["go", "no-go", "no-go", "go", "no-go", "go"],
            )
        with self.assertRaises(VerificationError):
            worker.run_plan(
                fixture(),
                inputs=self.inputs,
                checks=[{"path": ["capacity"], "op": "eq", "value": 0}],
            )
        self.assertEqual(evaluate(90), "go")
