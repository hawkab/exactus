"""Protocol microbenchmark, not a native/MTP or agent-graph speed claim."""

from __future__ import annotations

__author__ = "Olshansky Grigory"

import sys
import time as time
from pathlib import Path

sys.path[:0] = [
    str(Path(__file__).resolve().parents[1] / "src"),
    str(Path(__file__).resolve().parent),
]
import marker_rounds
import tool_rounds
from arithmetic_run import run_benchmark
from bench_live_support import (
    CAPTURED_SEGMENTS as CAPTURED_SEGMENTS,
)
from bench_live_support import (
    DEFAULT_MAX_TOKENS,
    chat,
    pack,
    score_alone,
    score_marker,
)
from bench_live_support import (
    SEED as SEED,
)
from bench_live_support import (
    independent_eval as independent_eval,
)
from bench_live_support import (
    qualification as qualification,
)
from bench_live_support import (
    tasks as tasks,
)
from bench_manifest import (
    QualificationBusy as QualificationBusy,
)
from bench_manifest import (
    QualificationGate as QualificationGate,
)
from bench_manifest import (
    load_runtime as load_runtime,
)
from bench_manifest import (
    source_provenance as source_provenance,
)
from bench_report import (
    aggregate as aggregate,
)
from bench_report import (
    metric_row as metric_row,
)
from bench_report import (
    paired as paired,
)
from bench_report import (
    replay as replay,
)
from bench_report import (
    stats as stats,
)

from exactus.prompts import MARKER_HINT


def run_marker_loop(port, model, t, external_computer, max_tokens=DEFAULT_MAX_TOKENS):
    """Run injected-marker rounds through the selected chat transport."""
    return marker_rounds.run_marker_loop(
        port, model, t, external_computer, max_tokens, chat=chat
    )


def run_toolcall(port, model, t, max_tokens=DEFAULT_MAX_TOKENS):
    """Run calculator-tool rounds through the selected chat transport."""
    return tool_rounds.run_toolcall(port, model, t, max_tokens, chat=chat)


def main():
    """Run arithmetic comparisons with this module's replaceable dependencies."""
    return run_benchmark(sys.modules[__name__])


def run_alone(port, model, t, max_tokens=DEFAULT_MAX_TOKENS):
    """Measure an unassisted model response against the independent task answer."""
    r = chat(port, model, [{"role": "user", "content": t["q"]}], max_tokens=max_tokens)
    return pack(t, score_alone(r["text"]), r, False, True)


def run_marker(port, model, t, external_computer, max_tokens=DEFAULT_MAX_TOKENS):
    """Measure one marker-assisted response and independently verify its computations."""
    r = chat(
        port,
        model,
        [
            {"role": "system", "content": MARKER_HINT},
            {"role": "user", "content": t["q"]},
        ],
        max_tokens=max_tokens,
    )
    emitted, value, parity = score_marker(r["text"], external_computer)
    return pack(t, value, r, emitted, parity)


if __name__ == "__main__":
    main()
