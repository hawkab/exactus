"""Installable zero-argument endpoint plugin, opt-in through VLLM_PLUGINS."""

__author__ = "Olshansky Grigory"

import asyncio
import inspect
import weakref

from .async_work import blocking
from .endpoint_adapter import ExactusCoordinator
from .endpoint_plugin import PluginStateError
from .native_engine import NativeEngine, NativeTokenizer
from .plugin_state import DeferredCoordinator
from .worker import WorkerExternalComputer


class _Models:
    """Read served model names from shared plugin state."""

    def __init__(self, state):
        """Initialize Models state and dependencies."""
        self.state = state

    def __getitem__(self, index):
        """Return the item selected by the supplied key or index."""
        return self.state.exactus_models[index]

    def __contains__(self, model):
        """Report whether the supplied value belongs to this collection."""
        return model in self.state.exactus_models


class ExactusEndpointPlugin:
    """Wire the native generation engine into the Exactus endpoint lifecycle."""

    name = "exactus"
    required_tasks = ("generate",)

    def attach_router(self, app):
        """Register routes before initialization and close owned workers at shutdown."""
        from .endpoint_route import attach

        app.state.exactus_models = ("initializing",)
        attach(app, DeferredCoordinator(app.state), _Models(app.state))
        self._app = weakref.ref(app)
        from contextlib import asynccontextmanager

        previous_lifespan = app.router.lifespan_context

        @asynccontextmanager
        async def lifespan(application):
            """Preserve the host lifespan and release both lazy worker processes."""
            try:
                async with previous_lifespan(application) as value:
                    yield value
            finally:
                workers = [
                    getattr(app.state, name, None)
                    for name in ("exactus_worker", "exactus_external_computer_worker")
                ]
                await asyncio.gather(
                    *(blocking(w.close) for w in workers if w is not None)
                )

        app.router.lifespan_context = lifespan

    async def init_state(self, engine_client, state, args):
        """Resolve engine configuration and create separate span and technical workers."""
        app = self._app() if hasattr(self, "_app") else None
        if app is not None and any(
            sum(
                getattr(r, "path", None) == path
                and "POST" in (getattr(r, "methods", None) or ())
                for r in app.routes
            )
            != 1
            for path in (
                "/v1/exact/completions",
                "/v1/exact/external-computers",
                "/v1/exact/plans",
            )
        ):
            raise PluginStateError("Exactus route collision after plugin registration")
        if engine_client is None:
            raise PluginStateError(
                "Exactus requires a generation engine; render is unsupported"
            )
        getter = getattr(engine_client, "get_tokenizer", None)
        if callable(getter):
            raw = getter()
            if inspect.isawaitable(raw):
                raw = await raw
        else:
            from transformers import AutoTokenizer

            raw = await blocking(
                AutoTokenizer.from_pretrained,
                getattr(args, "tokenizer", None) or args.model,
                revision=getattr(args, "tokenizer_revision", None)
                or getattr(args, "revision", None),
                local_files_only=True,
                trust_remote_code=bool(getattr(args, "trust_remote_code", False)),
            )
        template = getattr(args, "chat_template", None)
        if template is not None:
            from vllm.entrypoints.chat_utils import load_chat_template

            template = await blocking(load_chat_template, template)
        tokenizer = NativeTokenizer(
            raw,
            chat_template=template,
            template_kwargs=getattr(args, "default_chat_template_kwargs", None),
        )
        config = getattr(engine_client, "model_config", None)
        config_getter = getattr(engine_client, "get_model_config", None)
        if config is None and callable(config_getter):
            config = config_getter()
            if inspect.isawaitable(config):
                config = await config
        limit = getattr(config, "max_model_len", None)
        if type(limit) is not int:
            limit = getattr(args, "max_model_len", None)
        if type(limit) is not int or limit <= 0:
            raise PluginStateError("cannot resolve the native model context limit")
        worker = WorkerExternalComputer()
        state.exactus_worker = worker

        state.exactus_external_computer_worker = WorkerExternalComputer()
        names = getattr(args, "served_model_name", None) or [args.model]
        state.exactus_models = (names,) if isinstance(names, str) else tuple(names)
        state.exactus_coordinator = ExactusCoordinator(
            NativeEngine(engine_client, context_limit=limit),
            tokenizer,
            worker,
            context_limit=limit,
            task_external_computer=state.exactus_external_computer_worker,
        )
