"""Executable counterexamples from the e511bf4 review."""

__author__ = "Olshansky Grigory"

import subprocess
import tempfile
from pathlib import Path
from unittest.mock import patch

from tests._support_current_review import CurrentReviewFixture


class CurrentReview(CurrentReviewFixture):
    """Verify current benchmark contracts behavior."""

    def test_action_contract_rejects_null_unsafe_and_extra_fields(self):
        """Verify action contract rejects null unsafe and extra fields."""
        from benchmarks.agent_quality import action_contract

        for actions in (
            [None] * 3,
            ["delete pending"],
            [{"kind": "drop_pending"}],
            [{"kind": "retain_pending", "detail": "delete everything"}],
            [{"kind": "hold_rollout"}, {"kind": "proceed_rollout"}],
        ):
            self.assertFalse(action_contract(actions))
        self.assertTrue(
            action_contract([{"kind": "retain_pending"}], {"retain_pending"})
        )

    def test_streamed_tool_with_stop_is_intermediate_in_benchmark(self):
        """Verify streamed tool with stop is intermediate in benchmark."""
        import json

        from benchmarks.bench_live_support import chat

        chunks = [
            {
                "choices": [
                    {
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": "c",
                                    "function": {"name": "calc", "arguments": "{}"},
                                }
                            ]
                        },
                        "finish_reason": "stop",
                    }
                ]
            }
        ]
        with patch(
            "benchmarks.bench_live_support.stream_lines",
            return_value=iter(
                ("data:" + json.dumps(chunk)).encode() for chunk in chunks
            ),
        ):
            row = chat(1, "unused", [{"role": "user", "content": "hello"}])
        self.assertEqual(row["finish"], "tool_calls")
        self.assertEqual(row["upstream_finish"], "stop")

    def test_archive_provenance_cannot_inherit_parent_git(self):
        """Verify archive provenance cannot inherit parent git."""
        from benchmarks.bench_manifest import source_provenance

        with tempfile.TemporaryDirectory() as root:
            subprocess.run(["git", "init", "-q", root], check=True)
            nested = Path(root) / "archive"
            nested.mkdir()
            result = source_provenance(nested)
        self.assertIsNone(result["source_sha"])
        self.assertIsNone(result["dirty"])

    def test_accounting_retains_censored_cost_and_unknown_usage(self):
        """Verify accounting retains censored cost and unknown usage."""
        from benchmarks.bench_external_computers import RunAccounting

        ledger = RunAccounting()
        ledger.record({"record_type": "graph_started", "graph_id": "aborted"})
        for ident in ("a", "b"):
            ledger.record(
                {
                    "record_type": "request_started",
                    "request_id": ident,
                    "url": "http://x/v1/chat/completions",
                }
            )
        ledger.record(
            {
                "record_type": "request_completed",
                "request_id": "a",
                "status": 200,
                "response": {"usage": {"completion_tokens": 11847}},
            }
        )
        result = ledger.summary()
        self.assertEqual(result["known_completion_tokens_partial"], 11847)
        self.assertIsNone(result["completion_tokens"])
        self.assertEqual(result["censored_graph_ids"], ["aborted"])

    def test_fixture_has_both_go_and_no_go(self):
        """Verify fixture has both go and no go."""
        from benchmarks.bench_external_computers import case, decide

        self.assertEqual(
            {decide(case(seed)[1]) for seed in range(7403, 7407)}, {"go", "no-go"}
        )

    def test_tool_benchmark_keeps_tools_through_multiple_rounds(self):
        """Verify tool benchmark keeps tools through multiple rounds."""
        from benchmarks import bench_live as live

        usage = {"completion_tokens": 5, "prompt_tokens": 7}
        replies = [
            {
                "text": "",
                "finish": "tool_calls",
                "calls": [
                    {"id": "a", "name": "calc", "arguments": '{"expression":"2+3"}'}
                ],
            },
            {
                "text": "",
                "finish": "tool_calls",
                "calls": [
                    {"id": "b", "name": "calc", "arguments": '{"expression":"5*4"}'}
                ],
            },
            {"text": "\\answer{20}", "finish": "stop", "calls": []},
        ]
        replies = [{**r, "usage": usage, "wall": 0.1, "ttft": None} for r in replies]
        with patch.object(live, "chat", side_effect=replies) as call:
            result = live.run_toolcall(
                1, "m", {"id": "a", "q": "(2+3)*4", "truth": 20}, max_tokens=20
            )
        self.assertTrue(result["correct"])
        self.assertEqual(result["ctok"], 15)
        self.assertEqual(
            [c.kwargs["max_tokens"] for c in call.call_args_list], [20, 15, 10]
        )
        self.assertTrue(all(c.kwargs.get("tools") for c in call.call_args_list))
