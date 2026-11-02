"""One bounded CPU stage per segment: parse, batch compute, tokenize and finalize."""

__author__ = "Olshansky Grigory"

from dataclasses import dataclass

from ..application.evaluate import evaluate_spans
from ..domain.finalization import finalize, validate_final
from .dispatch import external_computer_answer, find_expression


@dataclass(slots=True)
class Stage:
    """Describe computed tokens and an optional finalized segment answer."""

    known: int
    prefill: int
    final: bool
    answer: tuple | None


def compute_segment(
    machine,
    segment,
    tokens,
    operations,
    external_computer,
    data,
    tokenizer,
    output_format,
    mode,
    task_prompt,
):
    """Validate a complete segment before evaluating spans or injecting results.

    One CPU stage handles a model segment. A single encoding of all result
    frames preserves BPE boundaries. Prompt-derived fallback values count toward
    returned known tokens even when the model emitted no final marker.
    """
    fresh_spans = machine.feed(segment.text)
    parsed = machine.finish()
    validate_final(parsed, "preserve-text")
    fresh = (
        evaluate_spans(
            external_computer, fresh_spans, data, output_format=output_format
        )
        if fresh_spans
        else []
    )
    operations.extend(fresh)
    known = sum(len(tokenizer.encode(op.value)) for op in fresh)
    final = any(op.final for op in fresh)
    if final or not fresh:
        external_computer_result = None
        if task_prompt:
            expression = find_expression(task_prompt)
            previous = next(
                (op.value for op in operations if op.expression == expression), None
            )
            external_computer_result = (
                (expression, previous)
                if previous is not None
                else external_computer_answer(
                    task_prompt, external_computer, data, output_format
                )
            )
        answer = finalize(
            parsed,
            operations,
            mode,
            external_computer_result=external_computer_result,
            prompt=task_prompt or "",
        )

        if answer[2] and not final:
            known += len(tokenizer.encode(answer[0]))
        return Stage(known, 0, final, answer)
    ids = segment.token_ids
    if not ids or tokenizer.decode(ids) != segment.text:
        ids = tokenizer.encode(segment.text)
    tokens.extend(ids)
    count = len(ids)

    injection = "".join(f"<calc-result>{op.value}</calc-result>\n" for op in fresh)
    ids = tokenizer.encode(injection)
    tokens.extend(ids)
    return Stage(known, count + len(ids), False, None)
