"""Parse arithmetic benchmark options without making model calls."""

from __future__ import annotations

__author__ = "Olshansky Grigory"

import argparse

from bench_live_support import DEFAULT_MAX_TOKENS


def arguments():
    """Parse command-line controls without performing inference."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=18001)
    ap.add_argument("--model", default="qwen3.8-flash-next-nvidia")
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--out", default="evidence/bench-live.json")
    ap.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    ap.add_argument("--arms", default="alone,marker,marker-loop,toolcall")
    ap.add_argument(
        "--warmup", type=int, default=2, help="excluded warmup calls per arm"
    )
    ap.add_argument(
        "--runtime-manifest",
        help="JSON with actual model/tokenizer/container/vLLM SHAs and flags",
    )
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--replay", help="rebuild summary from raw JSONL, without network")
    ap.add_argument("--capacity-url", help="read-only ingress qualification endpoint")
    ap.add_argument(
        "--engine-metrics-url", help="read-only engine qualification endpoint"
    )
    args = ap.parse_args()
    return ap, args
