"""Bounded native segment coordinator; known values enter only via normal prefill."""

from __future__ import annotations
import time
import uuid
from dataclasses import dataclass

from ..domain.audit import Proof
from ..domain.errors import BudgetError, GrammarError
from ..domain.finalization import FinalizationError
from ..domain.protocol import SpanMachine
from .async_work import blocking
from .endpoint_compute import compute_segment
from .endpoint_validation import (
    RequestTypeError as RequestTypeError,
    RequestValueError,
    validate_request,
)


@dataclass(frozen=True, slots=True)
class EndpointResult:
    """Output-token accounting is not a count of GPU forwards or FLOPs."""

    text: str
    operations: tuple
    engine_segments: int
    sampled_tokens: int
    known_tokens: int
    prefill_tokens: int
    exact_final: bool
    elapsed_s: float
    request_id: str
    parent_id: str | None = None
    transcript: str = ""
    finish_reason: str = "stop"
    proof: Proof | None = None
    mode: str = "preserve-text"
    model_final_emitted: bool = False
    engine_wait_s: float = 0.0
    cpu_stage_s: float = 0.0
    prompt_tokens: int = 0
    sampled_tokens_exact: bool = True


class ExactusCoordinator:
    def __init__(
        self,
        engine,
        tokenizer,
        external_computer,
        *,
        max_segments=9,
        context_limit=None,
        task_external_computer=None,
    ):
        self._engine, self._tokenizer, self._external_computer = (
            engine,
            tokenizer,
            external_computer,
        )
        self._max_segments, self._context_limit = max_segments, context_limit
        self._task_external_computer = (
            external_computer
            if task_external_computer is None
            else task_external_computer
        )

    @property
    def tokenizer(self):
        return self._tokenizer

    async def run_external_computers(self, tasks):
        """Run a technical batch without touching the engine or its caches."""
        from ..domain.errors import DataError

        run = getattr(self._task_external_computer, "run_external_computers", None)
        if run is None:
            raise DataError(
                "technical external computers require WorkerExternalComputer isolation"
            )
        return await blocking(run, tasks)

    async def run_plan(self, plan, *, inputs=None, checks=None):
        from ..domain.errors import DataError

        run = getattr(self._task_external_computer, "run_plan", None)
        if run is None:
            raise DataError("compute plans require WorkerExternalComputer isolation")
        return await blocking(run, plan, inputs=inputs, checks=checks)

    async def run(
        self,
        prompt_ids,
        data=None,
        *,
        max_tokens=2048,
        require_final=True,
        output_format="dec",
        request_id=None,
        parent_id=None,
        planner=None,
        mode=None,
        task_prompt=None,
    ):
        """Sampled tokens (including generated tags) plus known values consume budget.
        Re-fed prefix/result framing also consumes context, not output budget.
        Exactly one bounded CPU job is scheduled per model segment.
        """
        tokens = validate_request(
            prompt_ids,
            data,
            max_tokens,
            require_final,
            output_format,
            request_id,
            parent_id,
            planner,
            self._context_limit,
        )
        mode = mode or ("numeric-final" if require_final else "preserve-text")
        if mode not in ("preserve-text", "strict", "numeric-final", "audit"):
            raise RequestValueError("unknown mode")
        rid = request_id or str(uuid.uuid4())
        machine, operations = SpanMachine(), []
        sampled = known = prefill = 0
        engine_wait = cpu_stage = 0.0
        prompt_tokens = 0
        sampled_exact = True
        started = time.monotonic()
        request = None
        external_computer, request_data = (
            self._external_computer,
            dict(data) if data is not None else None,
        )
        factory = getattr(external_computer, "request", None)
        if callable(factory):
            request = await blocking(factory, request_data)
            external_computer, request_data = request, None
        try:
            for step in range(self._max_segments):
                budget = max_tokens - sampled - known
                if self._context_limit is not None:
                    budget = min(budget, self._context_limit - len(tokens))
                if budget <= 0:
                    raise BudgetError("output or reinjected context budget exhausted")
                tick = time.monotonic()
                prompt_tokens += len(tokens)
                segment = await self._engine.generate(tokens, budget, f"{rid}:{step}")
                engine_wait += time.monotonic() - tick
                if segment.finish_reason != "stop":
                    raise FinalizationError(
                        f"generation did not complete: {segment.finish_reason}",
                        "incomplete_generation",
                    )
                if (
                    type(segment.sampled_tokens) is not int
                    or not 0 <= segment.sampled_tokens <= budget
                ):
                    raise BudgetError("engine violated the sampled token budget")
                sampled += segment.sampled_tokens
                sampled_exact = (
                    sampled_exact and segment.token_count_source != "tokenizer_estimate"
                )
                tick = time.monotonic()
                stage = await blocking(
                    compute_segment,
                    machine,
                    segment,
                    tokens,
                    operations,
                    external_computer,
                    request_data,
                    self._tokenizer,
                    output_format,
                    mode,
                    task_prompt,
                )
                cpu_stage += time.monotonic() - tick
                known += stage.known
                prefill += stage.prefill
                if sampled + known > max_tokens:
                    raise BudgetError("computed result exceeds the token budget")
                if stage.answer is not None:
                    text, proof, exact = stage.answer
                    return EndpointResult(
                        text,
                        tuple(operations),
                        step + 1,
                        sampled,
                        known,
                        prefill,
                        exact,
                        time.monotonic() - started,
                        rid,
                        parent_id,
                        machine.transcript(),
                        "stop",
                        proof,
                        mode,
                        stage.final,
                        engine_wait,
                        cpu_stage,
                        prompt_tokens,
                        sampled_exact,
                    )
            raise GrammarError("segment limit exhausted")
        finally:
            if request is not None:
                await blocking(request.close)
