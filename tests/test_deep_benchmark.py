"""Offline correctness, truthful request artifacts and replay regression corpus."""

__author__ = "Olshansky Grigory"

import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stderr
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks"))
import bench_live as live
import bench_live_support as support
from bench_report import aggregate, replay, stats

from exactus.adapters.digest import HashlibHasher
from exactus.adapters.in_process import InProcessExternalComputer


class DeepBenchmark(unittest.TestCase):
    """Exercise Deep Benchmark behavior with isolated fixtures."""

    def task(self):
        """Return an independent arithmetic task fixture."""
        return {"id": "fixture", "q": "2+3?", "truth": 5, "expr": "2+3"}

    def test_all_intermediates_must_evaluate(self):
        """Verify all intermediates must evaluate."""
        result = support.score_marker(
            "<calc>1/0</calc><calc-final>2+3</calc-final>",
            InProcessExternalComputer(HashlibHasher()),
        )
        self.assertEqual(result, (False, None, False))

    def test_negative_usage_not_aggregated_as_tokens_or_zero_errors(self):
        """Verify negative usage not aggregated as tokens or zero errors."""
        row = support.pack(
            self.task(),
            5,
            {
                "usage": {"completion_tokens": -7, "prompt_tokens": 3},
                "finish": "stop",
                "text": r"\answer{5}",
                "wall": 0.1,
                "ttft": None,
            },
            False,
            True,
        )
        row["arm"] = "alone"
        summary = aggregate([row])["alone"]
        self.assertEqual(summary["errors"], 1)
        self.assertEqual(summary["failure_reasons"], {"invalid_usage": 1})
        self.assertIsNone(summary["tokens"]["median"])
        self.assertIsNone(stats([float("nan"), -1, None])["mean"])

    def test_wire_snapshot_is_immutable(self):
        """Verify wire snapshot is immutable."""

        class SSE:
            """Exercise SSE behavior with isolated fixtures."""

            def __enter__(self):
                """Enter this managed context and return its active resource."""
                return self

            def __exit__(self, *args):
                """Release owned resources when leaving the managed context."""
                pass

            def __iter__(self):
                """Return the iterator without eagerly consuming its values."""
                return iter(
                    [
                        b'data: {"choices":[],"usage":{"completion_tokens":1}}\n',
                        b"data: [DONE]\n",
                    ]
                )

        messages = [{"role": "user", "content": "before"}]
        with patch.object(support, "stream_lines", return_value=iter(SSE())):
            result = support.chat(1, "m", messages)
        messages[0]["content"] = "after"
        messages.append({"role": "tool", "content": "future"})
        self.assertEqual(
            result["request"]["messages"], [{"role": "user", "content": "before"}]
        )

    def test_raw_replay_reproduces_summary(self):
        """Verify raw replay reproduces summary."""
        row = support.pack(
            self.task(),
            5,
            {
                "usage": {"completion_tokens": 7, "prompt_tokens": 3},
                "finish": "stop",
                "text": r"\answer{5}",
                "wall": 0.1,
                "ttft": None,
            },
            False,
            True,
        )
        row["arm"] = "alone"
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "raw.jsonl"
            path.write_text(
                json.dumps({"record_type": "manifest", "manifest": {"test": True}})
                + "\n"
                + json.dumps(row)
                + "\n"
            )
            self.assertEqual(replay(path)["arms"], aggregate([row]))
            row["correct"] = False
            path.write_text(
                json.dumps({"record_type": "manifest", "manifest": {"test": True}})
                + "\n"
                + json.dumps(row)
                + "\n"
            )
            with self.assertRaises(ValueError):
                replay(path)

    def test_out_jsonl_collision_rejected_before_network(self):
        """Verify out jsonl collision rejected before network."""
        with tempfile.TemporaryDirectory() as folder:
            runtime = Path(folder) / "runtime.json"
            runtime.write_text(
                json.dumps(
                    {
                        "model_revision": "a",
                        "tokenizer_revision": "b",
                        "vllm_revision": "c",
                        "container_digest": "d",
                        "profile": {"test": True},
                    }
                )
            )
            with (
                patch.object(
                    sys,
                    "argv",
                    [
                        "bench_live.py",
                        "--out",
                        str(Path(folder) / "collision.jsonl"),
                        "--runtime-manifest",
                        str(runtime),
                    ],
                ),
                patch.object(live, "chat") as chat,
            ):
                stderr = io.StringIO()
                with redirect_stderr(stderr), self.assertRaises(SystemExit):
                    live.main()
                self.assertIn("--out must end", stderr.getvalue())
                self.assertEqual(chat.call_count, 0)
