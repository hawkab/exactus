"""Compute response choices through the external-computer port."""

from __future__ import annotations

__author__ = "Olshansky Grigory"

from .adapters.dispatch import calculator_prompt
from .proxy_choices import validate_choices
from .proxy_data import split_payload
from .proxy_modes import compute_answer, resolve_mode
from .proxy_rewrite import map_choices
from .proxy_stats import response_stats
from .request_errors import RequestError


class ResponseProcessor:
    """Apply exact-span policies while preserving tool turns and usage."""

    def __init__(self, external_computer, require_final, answer_finder):
        """Bind computation and prompt-answer dependencies."""
        self._external_computer = external_computer
        self._require_final = require_final
        self._answer_finder = answer_finder

    def process(self, payload: dict, response, *, data=None, mode=None) -> dict:
        """Replace spans in every choice while preserving upstream accounting.

        Nonempty tool calls are intermediate turns even if the upstream reports
        ``stop``. They do not trigger final-answer policy or prompt dispatch.
        Usage always describes upstream generation, never rewritten text length.

        Args:
            payload: Original completion request used to bind the task audit.
            response: Completed upstream response with one or more choices.
            data: Optional predecoded request-scoped byte payloads.
            mode: Optional already validated finalization policy.

        Returns:
            Rewritten choices and per-choice computation evidence.

        Raises:
            RequestError: The response structure or finalization is invalid.
        """
        if data is None or mode is None:
            _, fallback_data, raw_mode = split_payload(payload)
            data = fallback_data if data is None else data
            if mode is None:
                mode = resolve_mode(raw_mode, legacy_require_final=self._require_final)
        if not isinstance(response, dict):
            raise RequestError(
                "upstream response is not a JSON object", "upstream_error", 502
            )
        prompt = calculator_prompt(payload)
        answers = []

        choices = response.get("choices")
        if not isinstance(choices, list) or not choices:
            raise RequestError(
                "upstream response carries no choices", "upstream_error", 502
            )
        tool_stages = []
        checks = []

        def transform(text: str) -> str:
            """Compute one choice text and record its answer."""
            intermediate = tool_stages[len(answers)]
            answer = compute_answer(
                text,
                self._external_computer,
                data,
                mode="preserve-text" if intermediate else mode,
                external_computer_result=None
                if intermediate
                else external_computer_result,
                prompt=prompt,
            )
            if answer.proof.check is not None:
                check = answer.proof.check
                choice_index = choices[len(answers)].get("index", len(answers))
                checks.append(
                    {
                        "index": choice_index,
                        "expected": check.expected,
                        "given": check.given,
                        "verdict": check.verdict,
                    }
                )
            answers.append(answer)
            return answer.text

        tool_stages = validate_choices(choices)

        external_computer_result = (
            self._answer_finder(prompt, self._external_computer, data)
            if prompt and not all(tool_stages)
            else None
        )
        updated = map_choices(response, transform)
        for original, rewritten, intermediate in zip(
            choices, updated["choices"], tool_stages
        ):
            if intermediate and rewritten["finish_reason"] != "tool_calls":
                rewritten["upstream_finish_reason"] = rewritten["finish_reason"]
                rewritten["finish_reason"] = "tool_calls"
            if original["message"].get("content") is None:
                rewritten["message"]["content"] = None
            if original["message"].get("content") != rewritten["message"]["content"]:
                rewritten.pop("logprobs", None)
        if "usage" in updated:
            updated["upstream_usage"] = updated["usage"]
        return {
            **updated,
            "exactus": response_stats(answers, mode, tool_stages, checks),
        }
