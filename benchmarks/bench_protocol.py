"""Cost-model illustration: generated-token accounting without and with compute.

This is NOT a model benchmark: it counts UTF-8 bytes of fixed strings and
multiplies them by a caller-supplied TPOT constant. No model, tokenizer,
GPU, prefill or MTP acceptance participates. Byte counts are not model
tokens. Use it only to reason about token-path accounting; use a paired
live benchmark with real usage/finish_reason and an independent scorer
for any speedup claim.

Author: Olshansky Grigory
"""

from __future__ import annotations

import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from exactus.adapters.digest import HashlibHasher
from exactus.adapters.in_process import InProcessExternalComputer
from exactus.adapters.tokenizer import ByteTokenizer
from exactus.application.compute import ComputeService

__author__ = "Olshansky Grigory"

DEFAULT_TPOT_S = 0.0204
DEFAULT_REASONING = 12
TASKS = [
    "37*184500",
    "123456789*987654321",
    "2^100",
    "0xff xor 0b1010",
    "999999999+1",
    "(10^27-1)+1",
    "-7//-2 * 3",
    "12345*6789+1",
]


def run(tpot: float, reasoning: int) -> None:
    """Compare token counts and wall time for both arms over the task set."""
    tokenizer = ByteTokenizer()
    service = ComputeService(InProcessExternalComputer(HashlibHasher()))
    rows = []
    for expression in TASKS:
        started = time.perf_counter()
        result = service.run(
            f"<calc-final>{expression}</calc-final>", require_final=True
        )
        compute_s = time.perf_counter() - started
        generated_with = len(tokenizer.encode(f"<calc-final>{expression}</calc-final>"))
        injected = len(tokenizer.encode(result.text))
        answer_only = len(tokenizer.encode(result.text))
        rows.append(
            (expression, answer_only, reasoning, generated_with, injected, compute_s)
        )
    print(
        f"tasks={len(rows)} tokenizer=byte tpot={tpot:.4f}s "
        f"reasoning={reasoning} tok/task (parameterized verbal derivation)"
    )
    print(f"{'task':<22} {'no:ans':>6} {'no:rsn':>7} {'with:gen':>9} {'inject':>7}")
    for row in rows:
        print(f"{row[0]:<22} {row[1]:>6} {row[2]:>7} {row[3]:>9} {row[4]:>7}")
    no_total = [row[1] + row[2] for row in rows]
    with_total = [row[3] for row in rows]
    compute_s = [row[5] for row in rows]
    print()
    print(
        f"generated tokens/task  without={statistics.mean(no_total):.1f}  "
        f"with={statistics.mean(with_total):.1f}"
    )
    print(
        f"decode wall/task       without={statistics.mean(no_total) * tpot:.3f}s  "
        f"with={statistics.mean(with_total) * tpot + statistics.mean(compute_s):.3f}s"
    )
    print(
        f"compute wall/task      {statistics.mean(compute_s) * 1e3:.3f} ms "
        f"(injected digits are known tokens, never sampled)"
    )
    print("NOTE: numbers above are byte-count model arithmetic, not measured")
    print(
        "model latency. Archival live runs (different package, scored by the external computer"
    )
    print("diagnostic, 26/120 responses truncated) measured 242.9 vs 123.7")
    print("completion tokens per task on nvidia/qwen3.8-flash-next-nvfp4.")


def main() -> None:
    """Run the comparison with optional TPOT and reasoning overrides."""
    tpot = float(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_TPOT_S
    reasoning = int(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_REASONING
    run(tpot, reasoning)


if __name__ == "__main__":
    main()
