"""Benchmark: external computer adapter latency without and with a persistent worker.

Author: Olshansky Grigory
"""

from __future__ import annotations

import statistics
import platform
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from exactus.adapters.digest import HashlibHasher
from exactus.adapters.in_process import InProcessExternalComputer
from exactus.adapters.worker import WorkerExternalComputer

__author__ = "Olshansky Grigory"

CORPUS = [
    "37*184500",
    "2^256 + 1",
    "123456789*987654321",
    "-7/2 + -7//2 + -7%2",
    "0xff xor 0b1010",
    "sha256(@m) % 1000000",
    "(10^27-1)+1",
    "1<<4000",
]
DATA = {"m": b"benchmark-payload"}


def run_arm(name: str, external_computer, rounds: int) -> None:
    """Time one external computer adapter over the corpus and print one table row."""
    timings = []
    for _ in range(rounds):
        for expression in CORPUS:
            started = time.perf_counter()
            external_computer.evaluate(expression, DATA)
            timings.append(time.perf_counter() - started)
    per_op_us = statistics.median(timings) * 1e6
    ordered = sorted(timings)
    print(
        f"{name:<28} median={per_op_us:.2f} us mean={statistics.fmean(timings) * 1e6:.2f} us "
        f"p95={ordered[int(len(ordered) * 0.95)] * 1e6:.2f} us "
        f"max={max(timings) * 1e6:.2f} us n={len(timings)}"
    )


def main() -> None:
    """Compare in-process, persistent worker and spawn-per-call adapters."""
    rounds = int(sys.argv[1]) if len(sys.argv) > 1 else 50
    if rounds < 1:
        raise ValueError("rounds must be positive")
    print(
        f"host={platform.machine()} python={platform.python_version()} "
        "persistent max_calls=100000 (warm-policy microbenchmark, not default recycling)"
    )
    hasher = HashlibHasher()
    in_process = InProcessExternalComputer(hasher)
    persistent = WorkerExternalComputer(max_calls=100000)
    default = WorkerExternalComputer()
    spawn_per_call = WorkerExternalComputer(max_calls=1)
    try:
        run_arm("in-process", in_process, rounds)
        run_arm("worker persistent", persistent, rounds)
        run_arm(f"worker default recycle={default._max_calls}", default, rounds)
        run_arm("worker spawn-per-call", spawn_per_call, min(rounds, 10))
    finally:
        persistent.close()
        default.close()
        spawn_per_call.close()


if __name__ == "__main__":
    main()
