"""PoT/CoA plan compilation, faithful reification and transport boundaries."""

__author__ = "Olshansky Grigory"

import copy
import json

from tests._support_compute_plan import PlanTransportTestsFixture, fixture


class PlanTransportTests(PlanTransportTestsFixture):
    """Verify plan generation behavior."""

    def test_schema_only_planning_keeps_original_values_for_execution(self):
        """Verify schema only planning keeps original values for execution."""
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
            """Return the configured upstream fixture and capture the forwarded request."""
            calls.append(copy.deepcopy(payload))

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

    def test_one_upstream_request_and_no_llm_rewrite(self):
        """Verify one upstream request and no llm rewrite."""
        proxy, calls = self.proxy(), []

        def forward(path, payload, **kwargs):
            """Return the configured upstream fixture and capture the forwarded request."""
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
