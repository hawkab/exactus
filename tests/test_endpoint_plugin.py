"""Endpoint plugin factory and generate-wrapper surface tests.

Author: Olshansky Grigory
"""

import asyncio
import inspect
import unittest
from types import SimpleNamespace

from exactus.adapters.digest import HashlibHasher
from exactus.adapters.endpoint_adapter import ExactusCoordinator
from exactus.adapters.endpoint_plugin import (
    ExactusGenerateWrapper,
    PluginStateError,
    _StateCoordinator,
    build_endpoint_plugin,
)
from exactus.adapters.endpoint_validation import RequestValueError
from exactus.adapters.in_process import InProcessExternalComputer
from exactus.adapters.tokenizer import ByteTokenizer
from exactus.domain.ports import Segment

__author__ = "Olshansky Grigory"


class ScriptedEngine:
    """Engine stub returning scripted segments and recording calls."""

    def __init__(self, texts):
        """Store the scripted segment texts and empty call logs."""
        self._texts = list(texts)
        self.ids = []

    async def generate(self, prompt_ids, max_tokens, request_id):
        """Return the next scripted segment and record the call."""
        self.ids.append(request_id)
        text = self._texts.pop(0) if self._texts else ""
        return Segment(text, tuple(), 1, "stop")


class RecordingPlanner:
    """Planner stub recording every plan_text call."""

    def __init__(self):
        """Start with an empty call log."""
        self.calls = []

    def plan_text(self, text, request_id):
        """Record one planned injection and return a unit length."""
        self.calls.append((text, request_id))
        return 1


class FakeLLM:
    """Wrapped engine stub reporting delegated generate calls."""

    async def generate(self, prompt, sampling_params, request_id, **kwargs):
        """Return the delegated call identity."""
        yield ("delegated", request_id)


def build(models=("m",)):
    """Build one plugin over a scripted engine through the factory."""
    engine = ScriptedEngine(["<calc>1</calc><calc-final>2</calc-final>"])
    factory = build_endpoint_plugin(
        tokenizer=ByteTokenizer(),
        external_computer=InProcessExternalComputer(HashlibHasher()),
        engine_factory=lambda engine_client, args: engine,
        models=models,
    )
    return factory(), engine


class TestEndpointPluginSurface(unittest.TestCase):
    """Verifies the vLLM plugin ABI shape and the generate wrapper."""

    def test_factory_yields_plugin_protocol_shape(self):
        """The zero-argument factory yields the endpoint plugin surface."""
        plugin, _ = build()
        self.assertTrue(plugin.name)
        self.assertEqual(plugin.required_tasks, ("generate",))
        self.assertTrue(callable(plugin.attach_router))
        self.assertTrue(inspect.iscoroutinefunction(plugin.init_state))

    def test_init_state_builds_coordinator_on_state(self):
        """init_state places a wired coordinator onto the shared state."""
        plugin, engine = build()

        class State:
            """Plain shared-state stand-in."""

        state = State()
        asyncio.run(plugin.init_state(object(), state, object()))
        self.assertIsInstance(state.exactus_coordinator, ExactusCoordinator)
        result = asyncio.run(state.exactus_coordinator.run([65]))
        self.assertEqual(result.text, "2")
        self.assertEqual(engine.ids, [f"{result.request_id}:0"])

    def test_wrapper_delegates_plain_calls_unchanged(self):
        """Calls without the exactus option reach the wrapped engine as-is."""
        coordinator = ExactusCoordinator(
            ScriptedEngine([]),
            ByteTokenizer(),
            InProcessExternalComputer(HashlibHasher()),
        )
        wrapper = ExactusGenerateWrapper(FakeLLM(), coordinator)

        async def consume():
            return [item async for item in wrapper.generate("p", None, "r-9", extra=1)]

        self.assertEqual(asyncio.run(consume()), [("delegated", "r-9")])

    def test_wrapper_runs_explicit_exact_requests(self):
        """Exact results use an API separate from native output streaming."""
        coordinator = ExactusCoordinator(
            ScriptedEngine(["<calc>2+3</calc>", ""]),
            ByteTokenizer(),
            InProcessExternalComputer(HashlibHasher()),
        )
        wrapper = ExactusGenerateWrapper(FakeLLM(), coordinator)
        result = asyncio.run(
            wrapper.generate_exact([65], "r-1", max_tokens=64, require_final=False)
        )
        self.assertEqual(result.text, "5")

    def test_wrapper_rejects_forcing_planner(self):
        """A duplicate logits forcing planner is refused before any engine call."""
        planner = RecordingPlanner()
        coordinator = ExactusCoordinator(
            ScriptedEngine(["<calc>2+3</calc>", ""]),
            ByteTokenizer(),
            InProcessExternalComputer(HashlibHasher()),
        )
        wrapper = ExactusGenerateWrapper(FakeLLM(), coordinator)
        with self.assertRaises(RequestValueError):
            asyncio.run(
                wrapper.generate_exact(
                    [65], "r-2", max_tokens=64, require_final=False, planner=planner
                )
            )
        self.assertEqual(planner.calls, [])

    def test_wrapper_rejects_non_mapping_options(self):
        """A non-mapping exactus option is refused as a typed state error."""
        coordinator = ExactusCoordinator(
            ScriptedEngine([]),
            ByteTokenizer(),
            InProcessExternalComputer(HashlibHasher()),
        )
        wrapper = ExactusGenerateWrapper(FakeLLM(), coordinator)
        with self.assertRaises(PluginStateError):
            wrapper.generate([65], None, "r", exactus="yes")


class _StubCoordinator:
    """Recording coordinator stand-in for deferred-state delegation."""

    def __init__(self):
        """Start with an empty call log."""
        self.calls = []

    async def run(self, *args, **kwargs):
        """Record one generate call and answer it."""
        self.calls.append("run")
        return "ran"

    async def run_external_computers(self, tasks):
        """Record one external computer call and answer it."""
        self.calls.append("run_external_computers")
        return "external_computers"

    async def run_plan(self, plan, **kwargs):
        """Record one plan call and answer it."""
        self.calls.append("run_plan")
        return "plan"


class TestStateCoordinatorDelegation(unittest.TestCase):
    """Every route entry must survive the deferred plugin state."""

    def test_state_coordinator_forwards_all_route_entries(self):
        """run, run_external_computers and run_plan all reach the phase-B coordinator."""
        stub = _StubCoordinator()
        view = _StateCoordinator(
            ByteTokenizer(), SimpleNamespace(exactus_coordinator=stub)
        )
        self.assertEqual(asyncio.run(view.run([65])), "ran")
        self.assertEqual(
            asyncio.run(view.run_external_computers([])), "external_computers"
        )
        self.assertEqual(asyncio.run(view.run_plan({})), "plan")
        self.assertEqual(stub.calls, ["run", "run_external_computers", "run_plan"])

    def test_state_coordinator_refuses_uninitialized_state(self):
        """Each route entry fails typed before phase B has run."""
        view = _StateCoordinator(
            ByteTokenizer(), SimpleNamespace(exactus_coordinator=None)
        )
        for entry in (
            view.run([65]),
            view.run_external_computers([]),
            view.run_plan({}),
        ):
            with self.assertRaises(PluginStateError):
                asyncio.run(entry)


if __name__ == "__main__":
    unittest.main()
