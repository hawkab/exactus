"""vLLM endpoint plugin factory and application wiring for Exactus.

Integration points:
- vllm.endpoint_plugins entry point: build_endpoint_plugin(...) returns a
  zero-argument factory whose object satisfies the vLLM EndpointPlugin
  protocol (name, required_tasks, attach_router(app) and
  await init_state(engine_client, state, args)); endpoint plugins load
  only under the VLLM_PLUGINS allowlist and the route intentionally
  extends /v1 (shadowing caveat applies per the vLLM endpoint plugin ABI).
- AsyncLLM.generate wrapper preserves the wrapped stream unchanged;
  generate_exact is a separate explicit EndpointResult API.
- HTTP surface: attach_to_app registers the strict exact-compute route on
  a FastAPI app through adapters.endpoint_route.

The coordinator evaluates spans and re-feeds computed values once as
prefill; a duplicate logits planner is rejected. The draft predictor path
is experimental and requires a real verifier for speculative acceptance;
it is not lossless and no such guarantee is made.

Author: Olshansky Grigory
"""

from __future__ import annotations

from collections.abc import Mapping

from ..domain.errors import ExactusError

__author__ = "Olshansky Grigory"


class PluginStateError(ExactusError):
    """Plugin lifecycle state is missing or misconfigured."""


class ExactusGenerateWrapper:
    """Wraps an AsyncLLM-like generate entry with the exact-compute loop.

    generate is native streaming; generate_exact explicitly returns an exact
    result. The draft predictor is unrelated and still requires a verifier.
    """

    def __init__(self, llm, coordinator) -> None:
        """Store the wrapped engine object and the exact coordinator."""
        self._llm = llm
        self._coordinator = coordinator

    def generate(self, prompt, sampling_params=None, request_id=None, **kwargs):
        """Preserve native streaming types; exact results have a separate API."""
        options = kwargs.pop("exactus", None)
        if options is not None:
            raise PluginStateError(
                "use generate_exact; generate returns the native stream"
            )
        return self._llm.generate(prompt, sampling_params, request_id, **kwargs)

    async def generate_exact(self, prompt, request_id=None, **options):
        """Run an explicit exact-result request, not an AsyncLLM stream."""
        if not isinstance(options, Mapping):
            raise PluginStateError("exactus option must be a request mapping")
        if isinstance(prompt, Mapping):
            prompt = prompt.get("prompt_token_ids", prompt)
        allowed = {
            "data",
            "max_tokens",
            "require_final",
            "output_format",
            "parent_id",
            "planner",
            "mode",
        }
        if set(options) - allowed:
            raise PluginStateError("unsupported exact request options")
        ids = prompt
        if isinstance(prompt, str):
            from .async_work import blocking

            tokenizer = self._coordinator.tokenizer
            ids = await blocking(
                getattr(tokenizer, "encode_prompt", tokenizer.encode), prompt
            )
        return await self._coordinator.run(
            ids,
            options.get("data"),
            max_tokens=options.get("max_tokens", 2048),
            require_final=options.get("require_final", True),
            output_format=options.get("output_format", "dec"),
            request_id=request_id,
            parent_id=options.get("parent_id"),
            planner=options.get("planner"),
            mode=options.get("mode"),
            task_prompt=prompt if isinstance(prompt, str) else None,
        )


class _StateCoordinator:
    """Deferred coordinator view resolved through the plugin app state.

    Routes are registered before the engine exists (plugin phase A);
    each request resolves the coordinator built in phase B.
    """

    def __init__(self, tokenizer, state) -> None:
        """Store the tokenizer and the shared plugin state object."""
        self._tokenizer = tokenizer
        self._state = state

    def _bound(self):
        """Return the phase-B coordinator or fail the request."""
        coordinator = getattr(self._state, "exactus_coordinator", None)
        if coordinator is None:
            raise PluginStateError("exactus plugin state is not initialized")
        return coordinator

    async def run(self, *args, **kwargs):
        """Forward one request to the coordinator created in init_state."""
        return await self._bound().run(*args, **kwargs)

    async def run_external_computers(self, tasks):
        """Forward one external computer batch to the coordinator created in init_state."""
        return await self._bound().run_external_computers(tasks)

    async def run_plan(self, plan, **kwargs):
        """Forward one compute plan to the coordinator created in init_state."""
        return await self._bound().run_plan(plan, **kwargs)


class _ExactusEndpointPlugin:
    """Configured plugin satisfying the vLLM EndpointPlugin protocol."""

    required_tasks = ("generate",)

    def __init__(
        self, name: str, tokenizer, external_computer, engine_factory, planner, models
    ) -> None:
        """Store the build-time dependencies of the two-phase lifecycle."""
        self.name = name
        self._tokenizer = tokenizer
        self._external_computer = external_computer
        self._engine_factory = engine_factory
        self._planner = planner
        self._models = tuple(models)

    def attach_router(self, app) -> None:
        """Register the exact-compute route and expose the shared state."""
        from .endpoint_route import attach

        app.state.exactus_planner = self._planner
        attach(app, _StateCoordinator(self._tokenizer, app.state), self._models)

    async def init_state(self, engine_client, state, args) -> None:
        """Build the engine port and coordinator once the engine exists."""
        from .endpoint_adapter import ExactusCoordinator

        engine = self._engine_factory(engine_client, args)
        state.exactus_coordinator = ExactusCoordinator(
            engine,
            self._tokenizer,
            self._external_computer,
            context_limit=getattr(engine, "context_limit", None),
        )
        state.exactus_planner = self._planner


def build_endpoint_plugin(
    *,
    tokenizer,
    external_computer,
    engine_factory,
    models,
    planner=None,
    name: str = "exactus",
):
    """Return a zero-argument factory of one configured plugin object.

    The factory is a valid vllm.endpoint_plugins entry point target (vLLM
    endpoint plugin ABI, two-phase lifecycle). engine_factory receives
    (engine_client, args) and must return the Engine port; the optional
    planner argument is retained only to reject unsafe duplicate insertion.
    """
    if planner is not None:
        raise PluginStateError("native prefill must not also install a forcing planner")

    def factory():
        """Instantiate one configured Exactus endpoint plugin."""
        return _ExactusEndpointPlugin(
            name, tokenizer, external_computer, engine_factory, planner, models
        )

    return factory


def attach_to_app(app, coordinator, models, **kwargs) -> None:
    """Register the strict exact-compute HTTP route on one FastAPI app.

    Thin wrapper over adapters.endpoint_route.attach for callers wiring
    their own coordinator; on a vLLM server prefer build_endpoint_plugin.
    """
    from .endpoint_route import attach

    attach(app, coordinator, models, **kwargs)
