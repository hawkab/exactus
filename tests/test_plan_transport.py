"""PoT/CoA plan compilation, faithful reification and transport boundaries."""

__author__ = "Olshansky Grigory"


import json
from unittest.mock import patch

from exactus.adapters.worker import WorkerExternalComputer
from tests._support_compute_plan import PlanTransportTestsFixture, fixture


class PlanTransportTests(PlanTransportTestsFixture):
    """Verify plan transport behavior."""

    def test_contract_failure_does_not_repair_or_return_false_success(self):
        """Verify contract failure does not repair or return false success."""
        proxy = self.proxy()
        calls = []

        def forward(*args, **kwargs):
            """Return the configured upstream fixture and capture the forwarded request."""
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
        """Verify invalid mode controls fail before upstream."""
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
        """Verify direct proxy plan."""
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
        """Verify native direct plan without engine."""
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
