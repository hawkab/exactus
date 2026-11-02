"""Resolve shared plugin dependencies after the engine has initialized."""

__author__ = "Olshansky Grigory"


class DeferredCoordinator:
    """Forward requests through state without capturing an uninitialized engine."""

    def __init__(self, state, tokenizer=None):
        """Bind shared state and an optional tokenizer supplied during wiring."""
        self.state = state
        self._initial_tokenizer = tokenizer

    @property
    def tokenizer(self):
        """Return the configured tokenizer or resolve it from the live coordinator."""
        if self._initial_tokenizer is not None:
            return self._initial_tokenizer
        return self._bound().tokenizer

    def _bound(self):
        """Resolve the initialized coordinator or report a lifecycle error."""
        from .endpoint_plugin import PluginStateError

        coordinator = getattr(self.state, "exactus_coordinator", None)
        if coordinator is None:
            raise PluginStateError("native plugin is not initialized")
        return coordinator

    async def run(self, *args, **kwargs):
        """Forward one generation request to the initialized coordinator."""
        return await self._bound().run(*args, **kwargs)

    async def run_external_computers(self, tasks):
        """Forward one isolated technical batch to the initialized coordinator."""
        return await self._bound().run_external_computers(tasks)

    async def run_plan(self, plan, **kwargs):
        """Forward one source-bound plan to the initialized coordinator."""
        return await self._bound().run_plan(plan, **kwargs)
