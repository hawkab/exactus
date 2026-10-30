"""Negative cases from F01-F13 and N01-N06 reviews."""

import asyncio
import threading
import unittest

from exactus.adapters.dispatch import external_computer_answer
from exactus.adapters.digest import HashlibHasher
from exactus.adapters.in_process import InProcessExternalComputer
from exactus.adapters.endpoint_adapter import ExactusCoordinator
from exactus.adapters.tokenizer import ByteTokenizer
from exactus.adapters.worker import WorkerExternalComputer
from exactus.application.compute import ComputeService
from exactus.domain.errors import ExactusError, GrammarError, BudgetError
from exactus.domain.audit import Proof, SpanProof
from exactus.domain.ports import Segment
from exactus.proxy import ExactusProxy
from exactus.proxy_modes import RequestError
from tests import test_http


class ReviewRegressions(unittest.TestCase):
    def setUp(self):
        self.external_computer = InProcessExternalComputer(HashlibHasher())
        self.proxy = ExactusProxy("unused", self.external_computer)

    def compute(self, prompt, texts, mode):
        return self.proxy._compute(
            {"messages": [{"role": "user", "content": prompt}], "mode": mode},
            {
                "choices": [
                    {"message": {"content": t}, "finish_reason": "stop"} for t in texts
                ]
            },
        )

    def test_ambiguous_dispatch_abstains(self):
        prompts = [
            "Вычисли: 2+3. Затем умножь результат на 10.",
            "Вычисли: 12+3.5",
            "Вычисли: 12+3,5",
            "Не решай пример «Вычисли: 2+3.»; ответь числом 100.",
            "Пример:\n2+3 = ?\nОтветь 100",
            "Вычисли: 2+3\nИ умножь на 10",
        ]
        for prompt in prompts:
            with self.subTest(prompt=prompt):
                self.assertIsNone(
                    external_computer_answer(prompt, self.external_computer)
                )
        out = self.compute(prompts[0], ["<calc-final>(2+3)*10</calc-final>"], "strict")
        self.assertEqual(out["choices"][0]["message"]["content"], "50")
        self.assertNotIn("answer_check", out["exactus"])

    def test_check_each_choice_both_orders(self):
        for numbers in ((42, 999), (999, 42)):
            with self.assertRaises(RequestError) as caught:
                self.compute(
                    "Вычисли: 7*6",
                    [f"<calc-final>{n}</calc-final>" for n in numbers],
                    "strict",
                )
            self.assertEqual(caught.exception.error_code, "answer_mismatch")

    def test_fallback_provenance(self):
        out = self.compute("Вычисли: 7*6", ["\\answer{43}"], "numeric-final")["exactus"]
        self.assertEqual(out["operations"], 1)
        self.assertFalse(out["model_final_emitted"])
        record = out["audit"][0]["spans"][0]
        self.assertEqual(
            (record["origin"], record["expression"], record["value"]),
            ("prompt_dispatch", "7*6", "42"),
        )
        self.assertEqual(out["answer_check"]["given"], "43")

    @unittest.skipUnless(test_http.fastapi, "server extra is not installed")
    def test_endpoint_audit_and_strict_policies(self):
        case = test_http.TestExactRoute()
        for text, mode, expected in (
            ("v <calc>2+2</calc>", "audit", 200),
            ("<calc-final>2</calc-final><calc>99</calc>", "numeric-final", 422),
            ("<calc-final>2</calc-final><calc-final>99</calc-final>", "strict", 422),
        ):
            app, _ = case._app([text])
            status, body = case._post(
                app, {"exact_compute": True, "prompt": "p", "mode": mode}
            )
            self.assertEqual(status, expected)
            if mode == "audit":
                self.assertEqual(body["text"], "v ⊢ 4 ∎")
                self.assertEqual(body["audit"]["spans"][0]["value"], "4")

    def test_preserve_final_narrative(self):
        text = "Report <calc-final>2</calc-final> tail"
        self.assertEqual(
            ComputeService(self.external_computer).run(text, require_final=False).text,
            "Report 2 tail",
        )

    def test_unfinished_markers_and_generation_rejected(self):
        for text in ("answer <ca", "<calc-final>2</calc-final><ca"):
            with self.assertRaises(RequestError):
                self.compute("p", [text], "preserve-text")
        with self.assertRaises(RequestError):
            self.proxy._compute(
                {},
                {
                    "choices": [
                        {
                            "message": {"content": "<calc-final>2</calc-final>"},
                            "finish_reason": "length",
                        }
                    ]
                },
            )

    def test_context_budget_each_segment(self):
        class Engine:
            calls = []

            async def generate(self, ids, budget, rid):
                self.calls.append((len(ids), budget))
                text = "<calc>1</calc>" if len(self.calls) == 1 else ""
                return Segment(text, tuple(text.encode()), len(text.encode()), "stop")

        engine = Engine()
        result = asyncio.run(
            ExactusCoordinator(
                engine, ByteTokenizer(), self.external_computer, context_limit=61
            ).run([65], max_tokens=60, require_final=False)
        )
        self.assertEqual(result.text, "1")
        self.assertEqual(len(engine.calls), 2)
        self.assertTrue(all(prompt + budget <= 61 for prompt, budget in engine.calls))

    def test_stop_token_overrun_is_not_reinjected(self):
        class Engine:
            calls = []

            async def generate(self, ids, budget, rid):
                self.calls.append(bytes(ids).decode())
                if len(self.calls) == 1:
                    text = "<calc>1</calc>"
                    return Segment(
                        text, tuple((text + "OVERRUN").encode()), len(text) + 7, "stop"
                    )
                return Segment("done", tuple(b"done"), 4, "stop")

        engine = Engine()
        result = asyncio.run(
            ExactusCoordinator(engine, ByteTokenizer(), self.external_computer).run(
                [65], require_final=False
            )
        )
        self.assertEqual(result.text, "1done")
        self.assertNotIn("OVERRUN", engine.calls[1])

    def test_stale_nested_cross_thread_sessions(self):
        worker = WorkerExternalComputer()
        try:
            with worker.session({"x": b"\x01"}) as first:
                self.assertEqual(first.evaluate("@x").integer, 1)
                with self.assertRaises(ExactusError):
                    with worker.session({"x": b"\x63"}):
                        pass
                errors = []

                def misuse():
                    try:
                        first.evaluate("@x")
                    except ExactusError:
                        errors.append(True)

                thread = threading.Thread(target=misuse)
                thread.start()
                thread.join(2)
                self.assertEqual(errors, [True])
            with worker.session({"x": b"\x63"}):
                with self.assertRaises(ExactusError):
                    first.evaluate("@x")
        finally:
            worker.close()

    def test_proof_binding_and_operation_limit(self):
        with self.assertRaises(GrammarError):
            Proof((SpanProof("2+2", "4", True),)).inline("<calc-final>999</calc-final>")
        with self.assertRaises(BudgetError):
            ComputeService(self.external_computer).run(
                "<calc>1</calc>" * 129, require_final=False
            )
