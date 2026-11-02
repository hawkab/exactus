"""Executable counterexamples from the e511bf4 review."""

__author__ = "Olshansky Grigory"


from unittest.mock import patch

from tests._support_current_review import CurrentReviewFixture


class CurrentReview(CurrentReviewFixture):
    """Verify graph replay behavior."""

    def test_graph_replay_rejects_tampered_verdict(self):
        """Verify graph replay rejects tampered verdict."""
        import copy
        import json

        from benchmarks import bench_external_computers as module

        tasks, expected = module.case(7403)

        def chat(url, prompt, model, capture=None):
            """Send a model request and retain its actual usage and completion status."""
            value = (
                {"checks": ["verify source"] * 3, "rollback": ["retain work"] * 3}
                if "Plan a canary" in prompt
                else {
                    "reason": "A sufficiently detailed explanation of the facts.",
                    "actions": [
                        {"kind": k}
                        for k in ("retain_pending", "maintain_reserve", "hold_rollout")
                    ],
                }
            )
            return {
                "status": 200,
                "elapsed_s": 0,
                "response": {
                    "choices": [
                        {
                            "finish_reason": "stop",
                            "message": {"content": json.dumps(value)},
                        }
                    ],
                    "upstream_usage": {"prompt_tokens": 5, "completion_tokens": 7},
                },
            }

        def post(url, payload, capture=None):
            """Send a request to the local test endpoint and capture its result."""
            ident = payload["tasks"][0]["id"]
            return {
                "status": 200,
                "elapsed_s": 0,
                "response": {"results": [{"value": expected[ident]}]},
            }

        with (
            patch.object(module, "chat", side_effect=chat),
            patch.object(module, "post", side_effect=post),
        ):
            row = module.graph("unused", "after", 7403, "fake")
        module.verify_graph(row)
        damaged = copy.deepcopy(row)
        damaged["completion_tokens"] = 0
        with self.assertRaises(ValueError):
            module.verify_graph(damaged)
        damaged = copy.deepcopy(row)
        damaged["merge"]["response"]["choices"][0]["message"]["content"] = json.dumps(
            {
                "reason": "A sufficiently detailed explanation of the facts.",
                "actions": [None] * 3,
            }
        )
        with self.assertRaises(ValueError):
            module.verify_graph(damaged)
