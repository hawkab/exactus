"""Two-phase installed plugin wiring, including route and worker cleanup."""

__author__ = "Olshansky Grigory"

import asyncio
import unittest
from types import SimpleNamespace as NS

from exactus.adapters.default_plugin import ExactusEndpointPlugin
from tests.test_endpoint_limits import FastAPI, post


@unittest.skipUnless(FastAPI, "server extra is not installed")
class PluginLifecycle(unittest.TestCase):
    """Exercise Plugin Lifecycle behavior with isolated fixtures."""

    def test_late_route_collision_is_not_silently_shadowed(self):
        """Verify late route collision is not silently shadowed."""
        from fastapi import FastAPI

        from exactus.adapters.endpoint_plugin import PluginStateError

        app = FastAPI()
        plugin = ExactusEndpointPlugin()
        plugin.attach_router(app)
        app.add_api_route("/v1/exact/completions", lambda: {}, methods=["POST"])
        with self.assertRaises(PluginStateError):
            asyncio.run(plugin.init_state(None, app.state, NS()))

    def test_attach_init_generate_and_cleanup(self):
        """Verify attach init generate and cleanup."""

        class Tokenizer:
            """Exercise Tokenizer behavior with isolated fixtures."""

            def encode(self, text, **kwargs):
                """Encode fixture text into deterministic byte token identifiers."""
                return list(text.encode())

            def decode(self, ids, **kwargs):
                """Decode fixture token identifiers into text."""
                return bytes(ids).decode()

            def apply_chat_template(self, messages, **kwargs):
                """Render fixture messages using the test chat template."""
                self.messages = messages

                if kwargs.get("return_dict", True):
                    return {"input_ids": [65]}
                if kwargs.get("return_tensors") is not None:
                    raise AssertionError("native prompt must not be a tensor")
                return [65]

        class Client:
            """Exercise Client behavior with isolated fixtures."""

            model_config = NS(max_model_len=4096)

            async def get_tokenizer(self):
                """Return the tokenizer owned by the qualification client double."""
                return Tokenizer()

            async def generate(self, prompt, params, rid):
                """Yield the configured generation result for a correlated engine request."""
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
                """Abort the correlated request in the qualification engine double."""
                raise AssertionError("no abort on successful completion")

        async def run():
            """Execute one synchronized workload case."""
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
