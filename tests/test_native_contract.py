"""Real-shaped async stream and legacy ABI tests; not GPU qualification."""

import asyncio
import os
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

from exactus.adapters.native_engine import NativeEngine
from exactus.adapters.vllm_logits import LegacyExactusLogitsProcessor
from exactus.adapters.logits_state import LogitsStateError
from exactus.adapters.default_plugin import ExactusEndpointPlugin


class NativeContract(unittest.TestCase):
    def test_native_stream_token_ids_and_sampling(self):
        class Client:
            aborted = []

            async def generate(self, prompt, params, rid):
                self.received = (prompt, params, rid)
                yield NS(finished=False, outputs=[])
                yield NS(
                    finished=True,
                    outputs=[
                        NS(
                            text="<calc>2+3</calc>",
                            token_ids=[4, 5],
                            finish_reason="stop",
                        )
                    ],
                )

            async def abort(self, rid):
                self.aborted.append(rid)

        client = Client()
        segment = asyncio.run(
            NativeEngine(client, sampling_factory=lambda **kw: kw).generate(
                [1, 2], 32, "r"
            )
        )
        self.assertEqual(segment.token_ids, (4, 5))
        self.assertEqual(client.received[0], {"prompt_token_ids": [1, 2]})
        self.assertTrue(client.received[1]["include_stop_str_in_output"])
        self.assertEqual(client.aborted, [])

    def test_cancel_aborts_native_request(self):
        class Client:
            aborted = []

            async def generate(self, *args):
                await asyncio.sleep(60)
                yield None

            async def abort(self, rid):
                self.aborted.append(rid)

        async def run():
            client = Client()
            task = asyncio.create_task(
                NativeEngine(client, sampling_factory=lambda **kw: kw).generate(
                    [1], 8, "r"
                )
            )
            await asyncio.sleep(0)
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task
            self.assertEqual(client.aborted, ["r"])

        asyncio.run(run())

    def legacy(self, speculative=None):
        with patch.dict(os.environ, {"VLLM_USE_V2_MODEL_RUNNER": "0"}):
            return LegacyExactusLogitsProcessor(
                NS(speculative_config=speculative), "cpu", False
            )

    def update(self, proc, *, size=1, added=(), removed=(), moved=()):
        # Fields are the actual vLLM legacy BatchUpdate contract.
        proc.update_state(NS(added=added, removed=removed, moved=moved))

    def test_legacy_committed_cursor_and_none_update(self):
        proc, output = self.legacy(), []
        params = NS(extra_args={"exactus_token_ids": [65, 66]})
        self.update(proc, added=[(0, params, [1], output)])
        for _ in range(2):  # repeat apply before sampler commits must not advance
            row = [[0.0] * 256]
            proc.apply(row)
            self.assertEqual(row[0][65], 0.0)
            self.assertEqual(row[0][66], float("-inf"))
        output.append(65)
        proc.update_state(None)
        row = [[0.0] * 256]
        proc.apply(row)
        self.assertEqual(row[0][66], 0.0)
        self.update(proc, removed=[0], added=[(0, params, [1], output)])
        row = [[0.0] * 256]
        proc.apply(row)
        self.assertEqual(row[0][66], 0.0)  # resumed, not restarted at A

    def test_legacy_move_and_release(self):
        proc = self.legacy()
        self.update(
            proc,
            size=2,
            added=[
                (0, NS(extra_args={"exactus_token_ids": [65]}), [], []),
                (1, NS(extra_args={"exactus_token_ids": [66]}), [], []),
            ],
        )
        self.update(proc, size=2, moved=[(0, 1, NS(name="SWAP"))])
        rows = [[0.0] * 256 for _ in range(2)]
        proc.apply(rows)
        self.assertEqual(rows[0][66], 0.0)
        self.update(
            proc, size=1, removed=[0], moved=[(1, 0, NS(name="UNIDIRECTIONAL"))]
        )
        self.assertEqual(len(proc.slots), 1)

    def test_unsafe_runner_rejected(self):
        with self.assertRaises(LogitsStateError):
            self.legacy(speculative=NS(method="mtp"))
        with patch.dict(os.environ, {"VLLM_USE_V2_MODEL_RUNNER": "1"}):
            with self.assertRaises(LogitsStateError):
                LegacyExactusLogitsProcessor(NS(speculative_config=None), "cpu", False)

    def test_zero_arg_plugin(self):
        plugin = ExactusEndpointPlugin()
        self.assertEqual(plugin.name, "exactus")
        self.assertTrue(callable(plugin.attach_router))
