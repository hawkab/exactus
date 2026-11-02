"""Describe a completed exact-generation run and its token accounting."""

from __future__ import annotations

__author__ = "Olshansky Grigory"

from dataclasses import dataclass

from .audit import Proof


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
