"""Offline all-arm smoke and independent scorer checks, without model traffic."""

from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks"))
import bench_live as live
import bench_live_support as support
from exactus.adapters.digest import HashlibHasher
from exactus.adapters.in_process import InProcessExternalComputer


def reply(text, finish="stop", calls=None, ttft=0.1):
    return {
        "text": text,
        "finish": finish,
        "calls": calls or [],
        "wall": 1.0,
        "ttft": ttft,
        "usage": {"completion_tokens": 5, "prompt_tokens": 7},
    }


class BenchLiveTests(unittest.TestCase):
    def setUp(self):
        self.task = {"id": "test", "q": "2+3?", "truth": 5, "expr": "2+3"}
        self.external_computer = InProcessExternalComputer(HashlibHasher())

    def test_alone_and_marker_imports_and_budget(self):
        for fn, text, extra in (
            (live.run_alone, "\\answer{5}", []),
            (live.run_marker, "<calc-final>2+3</calc-final>", [self.external_computer]),
        ):
            with patch.object(live, "chat", return_value=reply(text)) as chat:
                row = fn(1, "m", self.task, *extra, max_tokens=77)
                self.assertTrue(row["correct"])
                self.assertEqual(chat.call_args.kwargs["max_tokens"], 77)

    def test_marker_loop_injects_all_results(self):
        replies = [
            reply("<calc>1+1</calc><calc>2+2</calc>"),
            reply("<calc-final>2+3</calc-final>"),
        ]
        with patch.object(live, "chat", side_effect=replies) as chat:
            row = live.run_marker_loop(
                1, "m", self.task, self.external_computer, max_tokens=77
            )
            self.assertTrue(row["correct"])
            prompt = chat.call_args.args[2][-1]["content"]
            self.assertIn("<calc-result>2</calc-result>", prompt)
            self.assertIn("<calc-result>4</calc-result>", prompt)
            self.assertEqual(chat.call_args.kwargs["max_tokens"], 72)

    def test_toolcall_result_and_visible_ttft(self):
        clock = [0.0]
        messages = []

        def chat(port, model, msgs, **kwargs):
            messages.append(msgs[:])
            clock[0] += 1
            if len(messages) == 1:
                return reply(
                    "",
                    "tool_calls",
                    [{"id": "c", "arguments": '{"expression":"2+3"}'}],
                    None,
                )
            return reply("\\answer{5}")

        with (
            patch.object(live, "chat", side_effect=chat),
            patch.object(live.time, "perf_counter", side_effect=lambda: clock[0]),
        ):
            row = live.run_toolcall(1, "m", self.task)
        self.assertTrue(row["correct"])
        self.assertEqual(messages[1][-1]["content"], "5")
        self.assertEqual(row["ttft"], 1.1)

    def test_truncation_and_missing_usage_are_failures(self):
        for r in (
            reply("\\answer{5}", "length"),
            {**reply("\\answer{5}"), "usage": {}},
        ):
            row = support.pack(self.task, 5, r, False, True)
            self.assertFalse(row["correct"])
            self.assertTrue(row["answer_match"])

    def test_independent_evaluator_is_bounded(self):
        self.assertEqual(support.independent_eval("(7-2)*3"), 15)
        self.assertEqual(support.independent_eval("-7/2"), -3)
        for expr in ("2**1000000", "'x'*1000000000", "__import__('os')"):
            with self.assertRaises(ValueError):
                support.independent_eval(expr)

    def test_unique_tasks_and_manifest_dry_run(self):
        tasks = support.tasks(60)
        self.assertEqual(len({t["expr"] for t in tasks}), 60)
        self.assertEqual(len({t["id"] for t in tasks}), 60)
