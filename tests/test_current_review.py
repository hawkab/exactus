"""Executable counterexamples from the e511bf4 review."""

__author__ = "Olshansky Grigory"


from unittest.mock import patch

from tests._support_current_review import CurrentReviewFixture


class CurrentReview(CurrentReviewFixture):
    """Verify current review behavior."""

    def test_tool_only_and_mixed_responses_preserve_calls_usage_and_null(self):
        """Verify tool only and mixed responses preserve calls usage and null."""
        calls = [
            {
                "id": "c",
                "type": "function",
                "function": {"name": "read_file", "arguments": "{}"},
            }
        ]
        for content, expected in ((None, None), ("value <calc>2+3</calc>", "value 5")):
            upstream = {
                "choices": [
                    {
                        "finish_reason": "tool_calls",
                        "message": {
                            "role": "assistant",
                            "content": content,
                            "tool_calls": calls,
                        },
                    }
                ],
                "usage": {"completion_tokens": 12, "prompt_tokens": 8},
            }
            with patch.object(self.proxy, "_forward", return_value=(200, upstream)):
                status, response = self.proxy.handle(
                    "/v1/chat/completions",
                    {
                        "messages": [{"role": "user", "content": "read source"}],
                        "tools": [],
                    },
                )
            self.assertEqual(status, 200)
            self.assertEqual(response["choices"][0]["message"]["content"], expected)
            self.assertEqual(response["choices"][0]["message"]["tool_calls"], calls)
            self.assertEqual(response["usage"], upstream["usage"])
            self.assertFalse(response["exactus"]["generation_complete"])
            self.assertIsNone(response["exactus"]["task_verified"])

    def test_no_hint_preserves_history_and_unchanged_logits_metadata(self):
        """Verify no hint preserves history and unchanged logits metadata."""
        from exactus.proxy_rewrite import prepare_request

        payload = {"messages": [{"role": "user", "content": "hello"}]}
        self.assertIs(prepare_request(payload, ""), payload)
        result = self.proxy._compute(
            payload,
            {
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {"content": "hello"},
                        "logprobs": {"content": []},
                    }
                ]
            },
        )
        self.assertIn("logprobs", result["choices"][0])

    def test_forced_tool_with_stop_is_not_a_final_answer(self):
        """Verify forced tool with stop is not a final answer."""
        reply = {
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {
                        "content": "",
                        "tool_calls": [
                            {
                                "id": "c",
                                "type": "function",
                                "function": {"name": "read_file", "arguments": "{}"},
                            }
                        ],
                    },
                }
            ],
        }
        with (
            patch.object(self.proxy, "_forward", return_value=(200, reply)),
            patch(
                "exactus.proxy.external_computer_answer",
                side_effect=AssertionError("tool is not final"),
            ),
        ):
            code, body = self.proxy.handle(
                "/v1/chat/completions",
                {
                    "messages": [{"role": "user", "content": "Compute: 6*7"}],
                    "mode": "strict",
                },
            )
        self.assertEqual(code, 200)
        self.assertEqual(body["choices"][0]["finish_reason"], "tool_calls")
        self.assertEqual(body["choices"][0]["upstream_finish_reason"], "stop")
        self.assertFalse(body["exactus"]["generation_complete"])
        self.assertFalse(body["exactus"]["exact_final"])
        self.assertEqual(reply["choices"][0]["finish_reason"], "stop")
        for bad in ({}, "invalid", [{"function": {}}]):
            reply["choices"][0]["message"]["tool_calls"] = bad
            with patch.object(self.proxy, "_forward", return_value=(200, reply)):
                self.assertEqual(
                    self.proxy.handle(
                        "/v1/chat/completions",
                        {
                            "messages": [{"role": "user", "content": "hello"}],
                        },
                    )[0],
                    502,
                )

    def test_numeric_mismatch_fails_and_unknown_task_is_not_verified(self):
        """Verify numeric mismatch fails and unknown task is not verified."""

        def invoke(text, mode="numeric-final", prompt="Compute: 6*7"):
            """Invoke the local fixture and return its status and decoded response."""
            reply = {
                "choices": [{"finish_reason": "stop", "message": {"content": text}}]
            }
            with patch.object(self.proxy, "_forward", return_value=(200, reply)):
                return self.proxy.handle(
                    "/v1/chat/completions",
                    {"messages": [{"role": "user", "content": prompt}], "mode": mode},
                )

        self.assertEqual(invoke("<calc-final>43</calc-final>")[0], 422)
        code, body = invoke("<calc-final>6*7</calc-final>")
        self.assertEqual(code, 200)
        self.assertTrue(body["exactus"]["task_verified"])
        code, body = invoke(
            "<calc-final>43</calc-final>", prompt="An ambiguous business question"
        )
        self.assertEqual(code, 200)
        self.assertIsNone(body["exactus"]["task_verified"])
        code, body = invoke("\\answer{43}")
        self.assertEqual(body["choices"][0]["message"]["content"], "42")
        self.assertTrue(body["exactus"]["task_verified"])
