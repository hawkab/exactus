"""External computer correctness, capability boundaries, IPC and HTTP contracts."""

__author__ = "Olshansky Grigory"

import copy
from concurrent.futures import ThreadPoolExecutor

from exactus.adapters.worker import WorkerExternalComputer
from exactus.domain.errors import ExactusError
from exactus.external_computers.batch import run_external_computers

TABLES = {
    "events": {
        "columns": ["root", "attempt", "delivered", "tokens"],
        "rows": [
            ["a", 1, True, 101],
            ["b", 1, False, 20],
            ["b", 2, True, 73],
            ["c", 1, False, 32],
        ],
    }
}


from tests._support_external_computers import BatchExternalComputerTestsFixture


class BatchExternalComputerTests(BatchExternalComputerTestsFixture):
    """Verify batch computer behavior."""

    def test_benchmark_guard_checks_backend_not_just_ingress(self):
        """Verify benchmark guard checks backend not just ingress."""
        from benchmarks.bench_external_computers import idle

        cap = dict(
            backend_ready=True,
            admission_enabled=True,
            active_weight=0,
            waiting=0,
            reserved=0,
            admitted=0,
        )
        metrics = 'vllm:num_requests_running{engine="0"} 0\nvllm:num_requests_waiting{engine="0"} 0\n'
        self.assertTrue(idle(cap, metrics))
        self.assertFalse(idle(cap, metrics.replace("} 0", "} 3", 1)))
        self.assertFalse(idle(cap, metrics.replace("} 0", "} NaN", 1)))
        self.assertFalse(idle(cap, ""))
        self.assertFalse(idle({**cap, "admitted": 1}, metrics))

    def test_dependent_batch_and_evidence_binding(self):
        """Verify dependent batch and evidence binding."""
        tasks = [
            {"id": "a", "kind": "calc", "expression": "6*7"},
            {
                "id": "b",
                "kind": "python_expr",
                "expression": "n+1",
                "inputs": {"n": {"$ref": "a"}},
            },
            {"id": "c", "kind": "calc", "expression": "6*7"},
        ]
        result = run_external_computers(tasks, self.external_computer)
        self.assertEqual([r["value"] for r in result["results"]], [42, 43, 42])
        self.assertEqual([r["cached"] for r in result["results"]], [False, False, True])
        self.assertEqual(result["model_calls"], 0)
        changed = copy.deepcopy(tasks)
        changed[0]["expression"] = "6*8"
        self.assertNotEqual(
            result["results"][1]["input_sha256"],
            run_external_computers(changed, self.external_computer)["results"][1][
                "input_sha256"
            ],
        )

    def test_invalid_graph_and_unknown_fields(self):
        """Verify invalid graph and unknown fields."""
        for tasks in (
            [],
            [{"id": "a", "kind": "go", "source": "func main(){}"}],
            [
                {
                    "id": "a",
                    "kind": "python_expr",
                    "expression": "x",
                    "inputs": {"x": {"$ref": "a"}},
                }
            ],
            [{"id": "a", "kind": "calc", "expression": "1", "surprise": True}],
            [{"id": "a", "kind": "calc", "expression": "1"}] * 2,
        ):
            with self.subTest(tasks=tasks), self.assertRaises(ExactusError):
                run_external_computers(tasks, self.external_computer)

    def test_request_local_cache(self):
        """Verify request local cache."""
        tasks = [{"id": "a", "kind": "calc", "expression": "123+5"}]
        self.assertFalse(
            run_external_computers(tasks, self.external_computer)["results"][0][
                "cached"
            ]
        )

    def test_operational_goldens_are_independent_of_external_computers(self):
        """Verify operational goldens are independent of external computers."""
        from benchmarks.bench_external_computers import case, same_json

        for seed in range(40):
            tasks, expected = case(seed)
            result = run_external_computers(tasks, self.external_computer)
            self.assertTrue(
                same_json({r["id"]: r["value"] for r in result["results"]}, expected)
            )
        self.assertFalse(
            run_external_computers(tasks, self.external_computer)["results"][0][
                "cached"
            ]
        )

    def test_worker_c3_and_recycle(self):
        """Verify worker c3 and recycle."""
        worker = WorkerExternalComputer(max_calls=2)
        self.addCleanup(worker.close)

        def run(n):
            """Execute one synchronized workload case."""
            return worker.run_external_computers(
                [
                    {
                        "id": "x",
                        "kind": "python_expr",
                        "expression": "sum(values)",
                        "inputs": {"values": [n, n + 1]},
                    }
                ]
            )["results"][0]["value"]

        with ThreadPoolExecutor(3) as pool:
            self.assertEqual(
                list(pool.map(run, range(12))), [2 * n + 1 for n in range(12)]
            )
        with self.assertRaises(ExactusError):
            worker.run_external_computers(
                [{"id": "x", "kind": "python_expr", "expression": "open('x')"}]
            )
        self.assertEqual(run(100), 201)
