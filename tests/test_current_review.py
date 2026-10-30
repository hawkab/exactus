"""Executable counterexamples from the e511bf4 review."""

from pathlib import Path
import subprocess
import tempfile
import unittest
import random
from unittest.mock import patch

from exactus.adapters.digest import HashlibHasher
from exactus.adapters.in_process import InProcessExternalComputer
from exactus.domain.errors import ExactusError
from exactus.external_computers.python_expr import evaluate
from exactus.external_computers.sql_read import evaluate as sql
from exactus.external_computers.values import OwnedJSON, packed
from exactus.external_computers.plan import run_plan
from exactus.proxy import ExactusProxy


class CurrentReview(unittest.TestCase):
    def setUp(self):
        self.proxy = ExactusProxy("unused", InProcessExternalComputer(HashlibHasher()))
        self.addCleanup(self.proxy.close)

    def test_generators_short_circuit_and_keep_list_comprehensions_eager(self):
        self.assertTrue(evaluate("any(1//x for x in xs)", {"xs": [1, 0]}))
        self.assertFalse(evaluate("all(1//x for x in xs)", {"xs": [2, 0]}))
        with self.assertRaises(ExactusError):
            evaluate("any([1//x for x in xs])", {"xs": [1, 0]})
        self.assertEqual(evaluate("sum(x*x for x in range(20) if x%2)", {}), 1330)
        self.assertEqual(
            evaluate("sorted(x for x in xs)", {"xs": [3, 1, 2]}), [1, 2, 3]
        )
        for expression in (
            "(x for x in xs)",
            "len(x for x in xs)",
            "sum(x for x in range(1000000))",
        ):
            with self.subTest(expression=expression), self.assertRaises(ExactusError):
                evaluate(expression, {"xs": [1, 2]})

    def test_tool_only_and_mixed_responses_preserve_calls_usage_and_null(self):
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
        def invoke(text, mode="numeric-final", prompt="Compute: 6*7"):
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

    def test_sql_forbids_hidden_real_and_uses_checked_integer_math(self):
        tables = {"items": {"columns": ["x"], "rows": [[1], [2]]}}
        for query in (
            "SELECT CAST(9007199254740993*1.0 AS INTEGER)",
            "SELECT CAST((9223372036854775807+1)-9223372036854775807 AS INTEGER)",
            "SELECT 9223372036854775807+1",
            "SELECT int_add(9223372036854775807,1)",
            "SELECT int_div(1,0)",
            "SELECT avg(x)>0 FROM items",
            "SELECT sum('0.5')>0 FROM items",
        ):
            with self.subTest(query=query), self.assertRaises(ExactusError):
                sql(query, tables)
        self.assertEqual(
            sql("SELECT int_sub(sum(x),count(*)) FROM items", tables)["rows"], [[1]]
        )
        self.assertEqual(
            sql("SELECT int_div(-7,2),int_mod(-7,2)", tables)["rows"], [[-4, 1]]
        )

    def test_owned_encoding_matches_boundary_encoding_and_charges_aliases(self):
        values = [
            None,
            True,
            -43,
            "кириллица\n",
            [],
            {},
            {"z": [1, {"x": 'quote"'}], "a": False},
        ]
        for value in values:
            self.assertEqual(OwnedJSON().encode(value), packed(value))
        owner = OwnedJSON()
        repeated = list(range(100))
        owner.encode(repeated)
        self.assertEqual(owner.encode([repeated] * 10), packed([repeated] * 10))
        for bad in ([repeated] * 180, ["x" * 4000] * 20, {"x": 1.5}, "\ud800"):
            with self.subTest(kind=type(bad)), self.assertRaises(ExactusError):
                owner.encode(bad)
        aggregate = owner.sequence()
        with self.assertRaises(ExactusError):
            for _ in range(30):
                aggregate.append("x" * 4000)

    def test_plan_no_repeated_packed_rows_and_same_canonical_digests(self):
        plan = {
            "tasks": [
                {
                    "id": f"t{i}",
                    "kind": "python_expr",
                    "expression": "xs",
                    "inputs": {"xs": {"$input": "xs"}},
                }
                for i in range(16)
            ],
            "answer": [{"$ref": f"t{i}"} for i in range(16)],
        }
        from exactus.external_computers import plan as module

        with patch.object(module, "packed", wraps=packed) as calls:
            result = run_plan(
                plan,
                InProcessExternalComputer(HashlibHasher()),
                inputs={"xs": list(range(256))},
            )
        self.assertEqual(sum(r["cached"] for r in result["results"]), 15)
        self.assertEqual(calls.call_count, 1)  # one caller-owned snapshot boundary

    def test_action_contract_rejects_null_unsafe_and_extra_fields(self):
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
        from benchmarks.bench_manifest import source_provenance

        with tempfile.TemporaryDirectory() as root:
            subprocess.run(["git", "init", "-q", root], check=True)
            nested = Path(root) / "archive"
            nested.mkdir()
            result = source_provenance(nested)
        self.assertIsNone(result["source_sha"])
        self.assertIsNone(result["dirty"])

    def test_accounting_retains_censored_cost_and_unknown_usage(self):
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
        from benchmarks.bench_external_computers import case, decide

        self.assertEqual(
            {decide(case(seed)[1]) for seed in range(7403, 7407)}, {"go", "no-go"}
        )

    def test_sql_catalog_reused_only_inside_one_request(self):
        from exactus.external_computers.batch import run_external_computers
        from exactus.external_computers import sql_read

        mutable = {"items": {"columns": ["x"], "rows": [[1]]}}
        with sql_read.SQLSession() as session:
            self.assertEqual(
                sql("SELECT sum(x) FROM items", mutable, session=session)["rows"], [[1]]
            )
            mutable["items"]["rows"].append([2])
            self.assertEqual(
                sql("SELECT sum(x) FROM items", mutable, session=session)["rows"], [[3]]
            )

        tables = {"items": {"columns": ["x"], "rows": [[1], [2]]}}
        tasks = [
            {
                "id": "a",
                "kind": "sql_read",
                "query": "SELECT count(*) FROM items",
                "tables": tables,
            },
            {
                "id": "b",
                "kind": "sql_read",
                "query": "SELECT sum(x) FROM items",
                "tables": tables,
            },
        ]
        with patch.object(sql_read, "_prepare", wraps=sql_read._prepare) as prepare:
            run_external_computers(tasks, InProcessExternalComputer(HashlibHasher()))
            self.assertEqual(prepare.call_count, 1)
            run_external_computers(tasks, InProcessExternalComputer(HashlibHasher()))
            self.assertEqual(prepare.call_count, 2)
        session = sql_read.SQLSession()
        connection = session.connection(tables)
        session.close()
        import sqlite3

        with self.assertRaises(sqlite3.ProgrammingError):
            connection.execute("SELECT 1")

    def test_tool_benchmark_keeps_tools_through_multiple_rounds(self):
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

    def test_owned_json_generated_trees_match_canonical_encoding(self):
        rng = random.Random(62030)

        def tree(depth):
            if depth == 0 or rng.randrange(3) == 0:
                return rng.choice([None, False, True, -12345, 2**90, '\n\t"\\Ё界'])
            if rng.randrange(2):
                return [tree(depth - 1) for _ in range(rng.randrange(5))]
            return {str(i): tree(depth - 1) for i in range(rng.randrange(5))}

        for _ in range(500):
            value = tree(5)
            self.assertEqual(OwnedJSON().encode(value), packed(value))

    def test_busy_micro_gate_aborts_before_inference_not_as_model_failure(self):
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

            # bench_live supports script and namespace imports. Patch the exact
            # globals bound to the imported helper, not an independent module.
            exc = live.QualificationBusy

            class BusyGate:
                elapsed_s = 0

                def __call__(self):
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

    def test_graph_replay_rejects_tampered_verdict(self):
        import copy
        import json
        from benchmarks import bench_external_computers as module

        tasks, expected = module.case(7403)

        def chat(url, prompt, model, capture=None):
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
