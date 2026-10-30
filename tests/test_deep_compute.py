"""Complexity, context, native boundary and request-scoped IPC regressions."""

import asyncio
from concurrent.futures import ThreadPoolExecutor
import os
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

from exactus.adapters.digest import HashlibHasher
from exactus.adapters.endpoint_adapter import ExactusCoordinator
from exactus.adapters.in_process import InProcessExternalComputer
from exactus.adapters.tokenizer import ByteTokenizer
from exactus.adapters.worker import WorkerExternalComputer
from exactus.adapters.vllm_logits import LegacyExactusLogitsProcessor
from exactus.domain.errors import ExactusError
from exactus.domain.ports import Segment
from exactus.domain.protocol import SpanMachine, parse
from exactus.proxy import ExactusProxy
from exactus.proxy_modes import RequestError


class DeepCompute(unittest.TestCase):
    def setUp(self):
        self.external_computer = InProcessExternalComputer(HashlibHasher())

    def test_oversized_expression_rejected_before_worker_spawn(self):
        worker = WorkerExternalComputer()
        with self.assertRaises(ExactusError):
            worker.evaluate("1" * 200000)
        self.assertIsNone(worker._proc)

    def test_context_is_not_dispatched(self):
        proxy = ExactusProxy("unused", self.external_computer)
        payload = {
            "messages": [
                {"role": "system", "content": "Multiply the result by 10"},
                {"role": "user", "content": "Вычисли: 2+3"},
            ]
        }
        result = {
            "choices": [
                {
                    "message": {"content": "<calc-final>(2+3)*10</calc-final>"},
                    "finish_reason": "stop",
                }
            ]
        }
        out = proxy._compute(payload, result, mode="strict", data={})
        self.assertEqual(out["choices"][0]["message"]["content"], "50")
        self.assertNotIn("answer_check", out["exactus"])
        result["choices"][0]["message"]["content"] = r"\answer{50}"
        with self.assertRaises(RequestError) as caught:
            proxy._compute(payload, result, mode="numeric-final", data={})
        self.assertEqual(caught.exception.error_code, "missing_final")

    def test_parser_does_not_rescan_and_offsets_survive_unicode_chunks(self):
        import exactus.domain.protocol as protocol

        text = "я" + "<" * 10000 + "x<calc>1<<8</calc> <calc-final>2+3</calc-final>"
        with patch.object(protocol, "_tag_at", wraps=protocol._tag_at) as tag:
            machine = SpanMachine()
            for ch in text:
                machine.feed(ch)
            parsed = machine.finish()
            count = tag.call_count
            for _ in range(100):
                self.assertIs(machine.finish(), parsed)
                machine.has_open_frame()
                machine.has_pending_tag()
                machine.render(["256", "5"])
            self.assertEqual(tag.call_count, count)
        self.assertLess(count, len(text) * 14)
        self.assertEqual(parsed, parse(text))
        for span in parsed.spans:
            self.assertTrue(text[span.start : span.end].startswith("<calc"))
        self.assertEqual(parsed.spans[0].expression, "1<<8")

    def test_segment_boundary_before_work(self):
        class Engine:
            calls = 0

            async def generate(self, *args):
                self.calls += 1
                return Segment("<calc>1</calc><calc>2", (), 1, "stop")

        engine = Engine()
        with patch.object(
            self.external_computer, "evaluate", wraps=self.external_computer.evaluate
        ) as evaluate:
            with self.assertRaises(ExactusError):
                asyncio.run(
                    ExactusCoordinator(
                        engine, ByteTokenizer(), self.external_computer
                    ).run([1])
                )
        self.assertEqual(evaluate.call_count, 0)
        self.assertEqual(engine.calls, 1)

    def test_one_cpu_job_per_segment(self):
        import exactus.adapters.endpoint_adapter as adapter

        class Engine:
            async def generate(self, *args):
                return Segment(
                    "<calc>1+1</calc><calc-final>2+3</calc-final>", (), 1, "stop"
                )

        with patch.object(adapter, "blocking", wraps=adapter.blocking) as jobs:
            result = asyncio.run(
                ExactusCoordinator(
                    Engine(), ByteTokenizer(), self.external_computer
                ).run([1])
            )
        self.assertEqual(result.text, "5")
        self.assertEqual(jobs.call_count, 1)

    def test_request_scoped_data_concurrent_and_recycle(self):
        worker = WorkerExternalComputer(max_calls=3)
        try:
            handles = [worker.request({"x": bytes([value])}) for value in (1, 7, 99)]

            def run(item):
                value, handle = item
                for _ in range(3):
                    self.assertEqual(
                        [v.integer for v in handle.evaluate_many(["@x", "@x+1"])],
                        [value, value + 1],
                    )
                handle.close()
                with self.assertRaises(ExactusError):
                    handle.evaluate("@x")

            with ThreadPoolExecutor(max_workers=3) as pool:
                list(pool.map(run, zip((1, 7, 99), handles)))
        finally:
            worker.close()

    def test_request_data_encoded_once_and_failure_does_not_leak(self):
        import exactus.adapters.worker_request as module

        worker = WorkerExternalComputer(max_calls=100)
        try:
            with patch.object(
                module, "encode_data", wraps=module.encode_data
            ) as encode:
                with worker.request({"x": b"\x07"}) as request:
                    for _ in range(3):
                        self.assertEqual(request.evaluate("@x").integer, 7)
                    with self.assertRaises(ExactusError):
                        request.evaluate_many(["@x", "1/0"])
                self.assertEqual(encode.call_count, 1)
            with worker.request({"x": b"\x63"}) as request:
                self.assertEqual(request.evaluate("@x").integer, 99)
        finally:
            worker.close()

    def test_empty_legacy_plans_never_copy_history(self):
        class History(list):
            reads = 0

            def __getitem__(self, key):
                self.reads += 1
                return super().__getitem__(key)

        with patch.dict(os.environ, {"VLLM_USE_V2_MODEL_RUNNER": "0"}):
            proc = LegacyExactusLogitsProcessor(
                NS(speculative_config=None), "cpu", False
            )
        history = History()
        proc.update_state(
            NS(
                batch_size=1,
                removed=[],
                moved=[],
                added=[(0, NS(extra_args={}), [], history)],
            )
        )
        for _ in range(4096):
            proc.apply([[0.0, 0.0]])
            history.append(1)
        self.assertEqual(history.reads, 0)
        self.assertEqual(proc.slots, {})

    def test_slot_defaults_not_allocated_on_existing_keys(self):
        import exactus.adapters.logits_state as state

        value = state.ForcingState()
        value.update_layout(["r"])
        with patch.object(state, "RequestSlot", wraps=state.RequestSlot) as constructor:
            for _ in range(1000):
                value.update_layout(["r"])
            self.assertEqual(constructor.call_count, 0)
