"""Executable counterexamples from the e511bf4 review."""

__author__ = "Olshansky Grigory"

import tempfile
from pathlib import Path
from unittest.mock import patch

from tests._support_current_review import CurrentReviewFixture


class CurrentReview(CurrentReviewFixture):
    """Verify benchmark admission behavior."""

    def test_busy_micro_gate_aborts_before_inference_not_as_model_failure(self):
        """Verify busy micro gate aborts before inference not as model failure."""
        import json

        from benchmarks import bench_live as live

        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            runtime = root / "runtime.json"
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
            import sys

            exc = live.QualificationBusy

            class BusyGate:
                """Exercise Busy Gate behavior with isolated fixtures."""

                elapsed_s = 0

                def __call__(self):
                    """Invoke this deterministic test double under the simulated failure conditions."""
                    raise exc("busy")

            with (
                patch.object(
                    sys,
                    "argv",
                    [
                        "bench_live.py",
                        "--n",
                        "2",
                        "--runs",
                        "1",
                        "--warmup",
                        "0",
                        "--arms",
                        "alone",
                        "--out",
                        str(root / "run.json"),
                        "--runtime-manifest",
                        str(runtime),
                        "--capacity-url",
                        "unused",
                        "--engine-metrics-url",
                        "unused",
                    ],
                ),
                patch.object(live, "QualificationGate", return_value=BusyGate()),
                patch.dict(
                    live.chat.__globals__,
                    {
                        "stream_lines": lambda *a, **kw: self.fail(
                            "inference was called"
                        )
                    },
                ),
            ):
                command = list(sys.argv)
                with self.assertRaises(exc):
                    live.main()
            rows = [
                json.loads(x) for x in (root / "run.jsonl").read_text().splitlines()
            ]
            self.assertEqual(
                [r["record_type"] for r in rows], ["manifest", "qualification_refused"]
            )

            def failed_warmup(*args, **kwargs):
                """Record spent inference evidence before raising the simulated warmup failure."""
                live.CAPTURED_SEGMENTS.get().append(
                    {"usage": {"completion_tokens": 17}}
                )
                raise RuntimeError("warmup transport failure")

            command[command.index("--warmup") + 1] = "1"
            command[command.index("--out") + 1] = str(root / "warm.json")
            with (
                patch.object(sys, "argv", command),
                patch.object(live, "QualificationGate", return_value=BusyGate()),
                patch.object(live, "run_alone", side_effect=failed_warmup),
            ):
                with self.assertRaisesRegex(RuntimeError, "warmup transport failure"):
                    live.main()
            rows = [
                json.loads(x) for x in (root / "warm.jsonl").read_text().splitlines()
            ]
            self.assertEqual(rows[-1]["record_type"], "warmup")
            self.assertEqual(rows[-1]["segments"][0]["usage"]["completion_tokens"], 17)
