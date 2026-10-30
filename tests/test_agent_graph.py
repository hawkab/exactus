"""DAG scheduling and rubric smoke, not a model quality/performance result."""

import json
from pathlib import Path
import sys
import threading
import time
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks"))
from agent_cases import case, judge
from bench_agent_graph import run_graph


class AgentGraph(unittest.TestCase):
    def test_real_dependencies_and_three_concurrent_requests(self):
        task = case(0)
        lock = threading.Lock()
        active = peak = 0
        calls = []

        def request(prompt):
            nonlocal active, peak
            with lock:
                active += 1
                peak = max(peak, active)
                calls.append(prompt)
            try:
                time.sleep(0.02)
                if "Both auditors finished" in prompt:
                    self.assertIn('"capacity":', prompt)
                    self.assertIn('"audit":', prompt)
                    expected = task["expected"]["lead"]
                elif "You are the capacity auditor" in prompt:
                    expected = task["expected"]["capacity"]
                elif "You are the response-delivery auditor" in prompt:
                    expected = task["expected"]["audit"]
                else:
                    expected = {"parallel_audits": 2}
                return {
                    "text": json.dumps(
                        {
                            "facts": expected,
                            "analysis": "A sufficiently detailed operational analysis.",
                            "actions": [
                                {"kind": k}
                                for k in (
                                    "retain_pending",
                                    "maintain_reserve",
                                    "admission_control",
                                    "replay_stored_answers",
                                    "deduplicate_by_root",
                                )
                            ],
                        }
                    ),
                    "usage": {"completion_tokens": 25, "prompt_tokens": 100},
                    "finish": "stop",
                    "wall": 0.02,
                }
            finally:
                with lock:
                    active -= 1

        result = run_graph(task, request)
        self.assertTrue(result["correct"])
        self.assertEqual(peak, 3)
        self.assertEqual(len(calls), 4)
        self.assertEqual(result["ctok"], 100)
        self.assertEqual(
            set(result["stages"]), {"planner", "capacity", "audit", "lead"}
        )

    def test_no_synthesis_of_failed_audits(self):
        result = run_graph(
            case(1), lambda prompt: {"text": "{}", "usage": {}, "wall": 0}
        )
        self.assertFalse(result["correct"])
        self.assertNotIn("lead", result["stages"])
        self.assertIn("lead:not_run", result["failures"])

    def test_bool_not_accepted_as_integer_and_wrong_decisions_fail(self):
        report = {
            "facts": {"value": True},
            "analysis": "A sufficiently detailed operational analysis.",
            "actions": ["x"],
        }
        self.assertFalse(judge(json.dumps(report), {"value": 1})["correct"])
