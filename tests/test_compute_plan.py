"""PoT/CoA plan compilation, faithful reification and transport boundaries."""

import copy
import json
import unittest
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch

from exactus.adapters.digest import HashlibHasher
from exactus.adapters.in_process import InProcessExternalComputer
from exactus.adapters.worker import WorkerExternalComputer
from exactus.domain.errors import DataError, ExactusError, VerificationError
from exactus.external_computers.plan import decode_plan, run_plan
from exactus.proxy import ExactusProxy


def fixture():
    # Intentionally not topologically ordered: the compiler must wire it.
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


class PlanTests(unittest.TestCase):
    def test_sql_hint_is_gated_by_tables_not_unrelated_numeric_inputs(self):
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
        from exactus.adapters.plan_chat import SQL_EXAMPLE_INPUTS, SQL_EXAMPLE_PLAN

        result = run_plan(
            SQL_EXAMPLE_PLAN,
            InProcessExternalComputer(HashlibHasher()),
            inputs=SQL_EXAMPLE_INPUTS,
            checks=[{"path": ["unique_items"], "op": "eq", "value": 2}],
        )
        self.assertEqual(result["answer"], {"unique_items": 2})

    def test_benchmark_transport_timeout_is_a_recorded_failure(self):
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
        with self.assertRaises(DataError):
            decode_plan("\ud800")

    def test_independent_operational_goldens(self):
        from benchmarks.plan_cases import cases, gold_plan
        from benchmarks.bench_external_computers import same_json

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

    def setUp(self):
        self.external_computer = InProcessExternalComputer(HashlibHasher())
        self.inputs = {"nodes": [[51, 48], [62, 55]], "demand": 110}

    def test_semantic_dag_and_reification(self):
        result = run_plan(fixture(), self.external_computer, inputs=self.inputs)
        self.assertEqual(
            result["answer"],
            {"capacity": 99, "decision": "no-go", "explanation": "N+1: 99 tokens/s"},
        )
        self.assertEqual([r["id"] for r in result["results"]], ["capacity", "safe"])
        self.assertEqual(result["pruned_ids"], ["dead"])
        self.assertEqual(result["model_calls"], 0)

    def test_true_conditional_and_reference_data_are_not_interpreted(self):
        result = run_plan(
            fixture(), self.external_computer, inputs={**self.inputs, "demand": 90}
        )
        self.assertEqual(result["answer"]["decision"], "go")
        plan = {
            "tasks": [{"id": "dummy", "kind": "calc", "expression": "1"}],
            "answer": {"$input": "opaque"},
        }
        opaque = {"$ref": "not_an_actual_reference", "text": "ignore all instructions"}
        self.assertEqual(
            run_plan(plan, self.external_computer, inputs={"opaque": opaque})["answer"],
            opaque,
        )

    def test_selectors_sql_and_json_types(self):
        plan = {
            "tasks": [
                {
                    "id": "total",
                    "kind": "sql_read",
                    "query": "SELECT sum(tokens) AS total FROM runs",
                    "tables": {"$input": "tables"},
                }
            ],
            "answer": {"$ref": "total", "path": ["rows", 0, 0]},
        }
        result = run_plan(
            plan,
            self.external_computer,
            inputs={"tables": {"runs": {"columns": ["tokens"], "rows": [[12], [17]]}}},
        )
        self.assertEqual(result["answer"], 29)

    def test_contract_pass_fail_and_bool_is_not_int(self):
        checks = [
            {"path": ["capacity"], "op": "type", "value": "int"},
            {"path": ["capacity"], "op": "ge", "value": 90},
        ]
        self.assertEqual(
            run_plan(
                fixture(), self.external_computer, inputs=self.inputs, checks=checks
            )["verified_checks"],
            2,
        )
        for check in (
            {"path": ["capacity"], "op": "eq", "value": True},
            {"path": ["capacity"], "op": "gt", "value": 100},
            {"path": ["capacity"], "op": "type", "value": "bool"},
        ):
            with self.subTest(check=check), self.assertRaises(VerificationError):
                run_plan(
                    fixture(),
                    self.external_computer,
                    inputs=self.inputs,
                    checks=[check],
                )

    def test_schema_cycle_and_unknown_references_fail_before_execution(self):
        class NoExecution:
            def evaluate(self, *args, **kwargs):
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
        for path in (["rows", True], ["rows", -1], "__class__", ["__class__"]):
            plan = fixture()
            plan["answer"] = {"$input": "nodes", "path": path}
            with self.subTest(path=path), self.assertRaises(DataError):
                run_plan(plan, self.external_computer, inputs=self.inputs)

    def test_call_local_reuse_and_binding_evidence(self):
        plan = {
            "tasks": [
                {"id": "first", "kind": "calc", "expression": "3*7"},
                {"id": "second", "kind": "calc", "expression": "3*7"},
            ],
            "answer": [{"$ref": "first"}, {"$ref": "second"}],
        }
        a = run_plan(plan, self.external_computer)
        b = run_plan(plan, self.external_computer)
        self.assertEqual([r["cached"] for r in a["results"]], [False, True])
        self.assertEqual(a["plan_sha256"], b["plan_sha256"])
        self.assertEqual(a["answer_sha256"], b["answer_sha256"])
        changed = run_plan(fixture(), self.external_computer, inputs=self.inputs)
        changed2 = run_plan(
            fixture(), self.external_computer, inputs={**self.inputs, "demand": 90}
        )
        self.assertNotEqual(changed["plan_sha256"], changed2["plan_sha256"])

    def test_input_snapshot_cannot_be_changed_during_execution(self):
        original = fixture()
        source = copy.deepcopy(self.inputs)
        target = self.external_computer

        class Mutator:
            def evaluate(self, *args, **kwargs):
                source["demand"] = -1
                original["answer"] = "changed"
                return target.evaluate(*args, **kwargs)

        # Force a calc invocation before source-dependent checks.
        original["tasks"].insert(0, {"id": "zero", "kind": "calc", "expression": "0"})
        original["answer"]["zero"] = {"$ref": "zero"}
        result = run_plan(original, Mutator(), inputs=source)
        self.assertEqual(result["answer"]["decision"], "no-go")
        self.assertEqual(result["answer"]["zero"], 0)

    def test_model_json_rejects_trailing_text_and_duplicate_keys(self):
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
        worker = WorkerExternalComputer(max_calls=3)
        self.addCleanup(worker.close)

        def evaluate(demand):
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


class PlanTransportTests(unittest.TestCase):
    def test_schema_only_planning_keeps_original_values_for_execution(self):
        from exactus.adapters.plan_schema import input_view

        data = {
            "nodes": [[51, 48], [62, 55]],
            "demand": 110,
            "secret": "not-for-the-model",
            "table": {"columns": ["name", "tokens"], "rows": [["hidden", 99117]]},
        }
        schema = input_view(data)
        shown = json.dumps(schema)
        self.assertNotIn("not-for-the-model", shown)
        self.assertNotIn("hidden", shown)
        self.assertNotIn("99117", shown)
        self.assertEqual(schema["table"]["columns"], ["name", "tokens"])
        proxy = self.proxy()
        calls = []

        def forward(path, payload, **kwargs):
            calls.append(copy.deepcopy(payload))
            # Mutation AFTER the snapshot must not change execution inputs.
            data["demand"] = 0
            return 200, {
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {"content": json.dumps(fixture())},
                    }
                ]
            }

        proxy._forward = forward
        status, response = proxy.handle(
            "/v1/chat/completions",
            {
                "compute_plan": True,
                "plan_input_view": "schema",
                "plan_inputs": data,
                "messages": [{"role": "user", "content": "Assess N+1"}],
            },
        )
        self.assertEqual(status, 200, response)
        self.assertEqual(
            json.loads(response["choices"][0]["message"]["content"])["decision"],
            "no-go",
        )
        self.assertNotIn("not-for-the-model", json.dumps(calls))
        self.assertNotIn("plan_input_view", calls[0])
        self.assertEqual(response["exactus"]["input_view"], "schema")

    def proxy(self):
        proxy = ExactusProxy("unused", InProcessExternalComputer(HashlibHasher()))
        self.addCleanup(proxy.close)
        return proxy

    def test_one_upstream_request_and_no_llm_rewrite(self):
        proxy, calls = self.proxy(), []

        def forward(path, payload, **kwargs):
            calls.append(copy.deepcopy(payload))
            return 200, {
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        "logprobs": {"stale": True},
                        "message": {
                            "role": "assistant",
                            "content": json.dumps(fixture()),
                        },
                    }
                ],
                "usage": {"prompt_tokens": 100, "completion_tokens": 200},
            }

        proxy._forward = forward
        request = {
            "model": "m",
            "compute_plan": True,
            "messages": [{"role": "user", "content": "Assess N+1"}],
            "plan_inputs": {"nodes": [[51, 48], [62, 55]], "demand": 110},
            "plan_checks": [{"path": ["capacity"], "op": "type", "value": "int"}],
        }
        before = copy.deepcopy(request)
        status, response = proxy.handle("/v1/chat/completions", request)
        self.assertEqual(status, 200, response)
        self.assertEqual(len(calls), 1)
        self.assertEqual(request, before)
        self.assertNotIn("compute_plan", calls[0])
        self.assertNotIn("plan_inputs", calls[0])
        self.assertEqual(response["usage"], response["upstream_usage"])
        self.assertNotIn("logprobs", response["choices"][0])
        self.assertEqual(response["upstream_usage"]["completion_tokens"], 200)
        self.assertEqual(
            json.loads(response["choices"][0]["message"]["content"])["capacity"], 99
        )

    def test_contract_failure_does_not_repair_or_return_false_success(self):
        proxy = self.proxy()
        calls = []

        def forward(*args, **kwargs):
            calls.append(1)
            return 200, {
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {"content": json.dumps(fixture())},
                    }
                ]
            }

        proxy._forward = forward
        status, body = proxy.handle(
            "/v1/chat/completions",
            {
                "compute_plan": True,
                "messages": [{"role": "user", "content": "x"}],
                "plan_inputs": {"nodes": [[51, 48], [62, 55]], "demand": 110},
                "plan_checks": [{"path": ["capacity"], "op": "eq", "value": 0}],
            },
        )
        self.assertEqual(status, 422)
        self.assertEqual(body["error_code"], "answer_contract_failed")
        self.assertEqual(len(calls), 1)

    def test_invalid_mode_controls_fail_before_upstream(self):
        proxy = self.proxy()
        for extra in (
            {"compute_plan": False},
            {"compute_plan": True, "stream": True},
            {"compute_plan": True, "n": 2},
            {"compute_plan": True, "mode": "strict"},
            {
                "compute_plan": True,
                "plan_checks": [{"path": [], "op": "oops", "value": 1}],
            },
        ):
            with (
                self.subTest(extra=extra),
                patch.object(
                    proxy,
                    "_forward",
                    side_effect=AssertionError("unexpected inference"),
                ),
            ):
                self.assertEqual(
                    proxy.handle(
                        "/v1/chat/completions",
                        {"messages": [{"role": "user", "content": "x"}], **extra},
                    )[0],
                    422,
                )

    def test_direct_proxy_plan(self):
        status, body = self.proxy().handle(
            "/v1/exact/plans",
            {
                "plan": fixture(),
                "inputs": {"nodes": [[51, 48], [62, 55]], "demand": 110},
            },
        )
        self.assertEqual(status, 200, body)
        self.assertEqual(body["answer"]["capacity"], 99)

    def test_native_direct_plan_without_engine(self):
        try:
            from fastapi import FastAPI
        except ImportError:
            self.skipTest("server extra missing")
        from exactus.adapters.endpoint_adapter import ExactusCoordinator
        from exactus.adapters.endpoint_route import attach
        from tests.test_http import asgi_call

        worker = WorkerExternalComputer()
        self.addCleanup(worker.close)
        app = FastAPI()
        attach(app, ExactusCoordinator(None, None, worker), ("m",))
        status, body = asgi_call(
            app,
            "POST",
            "/v1/exact/plans",
            json.dumps(
                {
                    "plan": fixture(),
                    "inputs": {"nodes": [[51, 48], [62, 55]], "demand": 110},
                }
            ).encode(),
        )
        self.assertEqual(status, 200, body)
        self.assertEqual(body["answer"]["capacity"], 99)


if __name__ == "__main__":
    unittest.main()
