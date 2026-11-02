"""PoT/CoA plan compilation, faithful reification and transport boundaries."""

__author__ = "Olshansky Grigory"


from unittest.mock import patch

from exactus.adapters.digest import HashlibHasher
from exactus.adapters.in_process import InProcessExternalComputer
from exactus.domain.errors import DataError
from exactus.external_computers.plan import decode_plan, run_plan
from exactus.proxy import ExactusProxy
from tests._support_compute_plan import PlanTestsFixture


class PlanTests(PlanTestsFixture):
    """Verify plan prompting behavior."""

    def test_sql_hint_is_gated_by_tables_not_unrelated_numeric_inputs(self):
        """Verify sql hint is gated by tables not unrelated numeric inputs."""
        from exactus.adapters.plan_chat import PLAN_HINT, SQL_EXAMPLE_INPUTS, SQL_HINT
        from exactus.adapters.plan_schema import has_tables

        for inputs, extra in (
            ({"documents": [[4, 5, 1, 2]]}, ""),
            (SQL_EXAMPLE_INPUTS, SQL_HINT),
        ):
            self.assertEqual(has_tables(inputs), bool(extra))
            proxy = ExactusProxy(
                "http://unused", InProcessExternalComputer(HashlibHasher())
            )
            self.addCleanup(proxy.close)
            with patch.object(proxy, "_forward", return_value=(503, {})) as forward:
                proxy.handle(
                    "/v1/chat/completions",
                    {
                        "compute_plan": True,
                        "plan_inputs": inputs,
                        "messages": [{"role": "user", "content": "compute"}],
                    },
                )
            self.assertEqual(
                forward.call_args.args[1]["messages"][0]["content"], PLAN_HINT + extra
            )

    def test_marker_failure_retains_already_spent_inference_usage(self):
        """Verify marker failure retains already spent inference usage."""
        response = {
            "usage": {"prompt_tokens": 8, "completion_tokens": 8192},
            "choices": [{"finish_reason": "length", "message": {"content": ""}}],
        }
        proxy = ExactusProxy(
            "http://unused", InProcessExternalComputer(HashlibHasher())
        )
        self.addCleanup(proxy.close)
        with patch.object(proxy, "_forward", return_value=(200, response)):
            status, body = proxy.handle(
                "/v1/chat/completions",
                {
                    "messages": [{"role": "user", "content": "calculate"}],
                },
            )
        self.assertEqual(status, 422)
        self.assertEqual(body["error_code"], "incomplete_generation")
        self.assertEqual(body["upstream_usage"], response["usage"])
        self.assertEqual(body["upstream_choices"], response["choices"])
        self.assertFalse(body["exactus"]["answer_rendered"])

    def test_sql_few_shot_is_an_executable_verified_example(self):
        """Verify sql few shot is an executable verified example."""
        from exactus.adapters.plan_chat import SQL_EXAMPLE_INPUTS, SQL_EXAMPLE_PLAN

        result = run_plan(
            SQL_EXAMPLE_PLAN,
            InProcessExternalComputer(HashlibHasher()),
            inputs=SQL_EXAMPLE_INPUTS,
            checks=[{"path": ["unique_items"], "op": "eq", "value": 2}],
        )
        self.assertEqual(result["answer"], {"unique_items": 2})

    def test_benchmark_transport_timeout_is_a_recorded_failure(self):
        """Verify benchmark transport timeout is a recorded failure."""
        from benchmarks.bench_external_computers import post

        records = []
        with patch("urllib.request.urlopen", side_effect=TimeoutError("read timeout")):
            result = post(
                "http://unused/v1/chat/completions", {}, capture=records.append
            )
        self.assertEqual(result["status"], 0)
        self.assertEqual(result["error"], "read timeout")
        self.assertEqual(
            [r["record_type"] for r in records],
            ["request_started", "request_completed"],
        )
        self.assertEqual(records[0]["request_id"], records[1]["request_id"])

    def test_model_plan_rejects_invalid_utf8_with_typed_error(self):
        """Verify model plan rejects invalid utf8 with typed error."""
        with self.assertRaises(DataError):
            decode_plan("\ud800")

    def test_independent_operational_goldens(self):
        """Verify independent operational goldens."""
        from benchmarks.bench_external_computers import same_json
        from benchmarks.plan_cases import cases, gold_plan

        for seed in range(60):
            for item in cases(seed):
                result = run_plan(
                    gold_plan(item),
                    InProcessExternalComputer(HashlibHasher()),
                    inputs=item["inputs"],
                    checks=item["checks"],
                )
                self.assertTrue(
                    same_json(result["answer"], item["expected"]), item["id"]
                )
