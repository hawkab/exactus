"""Two-phase installed plugin wiring, including route and worker cleanup."""

import asyncio
from types import SimpleNamespace as NS
import unittest

from exactus.adapters.default_plugin import ExactusEndpointPlugin
from tests.test_endpoint_limits import post, FastAPI


@unittest.skipUnless(FastAPI, "server extra is not installed")
class PluginLifecycle(unittest.TestCase):
    def test_late_route_collision_is_not_silently_shadowed(self):
        from fastapi import FastAPI
        from exactus.adapters.endpoint_plugin import PluginStateError

        app = FastAPI()
        plugin = ExactusEndpointPlugin()
        plugin.attach_router(app)
        app.add_api_route("/v1/exact/completions", lambda: {}, methods=["POST"])
        with self.assertRaises(PluginStateError):
            asyncio.run(plugin.init_state(None, app.state, NS()))

    def test_attach_init_generate_and_cleanup(self):
        class Tokenizer:
            def encode(self, text, **kwargs):
                return list(text.encode())

            def decode(self, ids, **kwargs):
                return bytes(ids).decode()

            def apply_chat_template(self, messages, **kwargs):
                self.messages = messages
                # Recent Transformers defaults to BatchEncoding unless the
                # native adapter explicitly requests plain token IDs.
                if kwargs.get("return_dict", True):
                    return {"input_ids": [65]}
                if kwargs.get("return_tensors") is not None:
                    raise AssertionError("native prompt must not be a tensor")
                return [65]

        class Client:
            model_config = NS(max_model_len=4096)

            async def get_tokenizer(self):
                return Tokenizer()

            async def generate(self, prompt, params, rid):
                yield NS(
                    finished=True,
                    outputs=[
                        NS(
                            text="<calc-final>2+3</calc-final>",
                            token_ids=[10, 11],
                            finish_reason="stop",
                        )
                    ],
                )

            async def abort(self, rid):
                raise AssertionError("no abort on successful completion")

        async def run():
            app, plugin = FastAPI(), ExactusEndpointPlugin()
            plugin.attach_router(app)
            await plugin.init_state(
                Client(),
                app.state,
                NS(model="m", served_model_name=["alias"], max_model_len="auto"),
            )
            engine = app.state.exactus_coordinator._engine
            engine._sampling_factory = lambda **kwargs: kwargs
            async with app.router.lifespan_context(app):
                self.assertEqual(await post(app), 200)
                worker = app.state.exactus_worker
                self.assertIsNotNone(worker._proc)
                external_computer_worker = app.state.exactus_external_computer_worker
                self.assertIsNot(worker, external_computer_worker)
                self.assertIsNone(external_computer_worker._proc)
                result = await app.state.exactus_coordinator.run_external_computers(
                    [{"id": "x", "kind": "python_expr", "expression": "sum(range(10))"}]
                )
                self.assertEqual(result["results"][0]["value"], 45)
                self.assertIsNotNone(external_computer_worker._proc)
                self.assertIn(
                    "Markers",
                    app.state.exactus_coordinator.tokenizer.inner.messages[0][
                        "content"
                    ],
                )
            self.assertIsNone(worker._proc)
            self.assertIsNone(external_computer_worker._proc)

        asyncio.run(run())
