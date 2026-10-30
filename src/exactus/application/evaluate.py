"""One bounded batch, including adapters that have no batch-capable external computer."""

from .compute import Operation, ComputeError


def evaluate_spans(external_computer, spans, data=None, *, output_format="dec"):
    batch = getattr(external_computer, "evaluate_many", None)
    if callable(batch):
        values = batch([s.expression for s in spans], data, output_format=output_format)
    else:
        values = [
            external_computer.evaluate(s.expression, data, output_format=output_format)
            for s in spans
        ]
    if len(values) != len(spans):
        raise ComputeError("external computer returned the wrong batch size")
    records = []
    for span, value in zip(spans, values):
        if not hasattr(value, "render"):
            raise ComputeError("external computer returned a non-value result")
        records.append(
            Operation(span.expression, value.render(), span.final, value.format)
        )
    return records
