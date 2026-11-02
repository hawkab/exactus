"""Parse operational comparison arguments without starting inference."""

from __future__ import annotations

__author__ = "Olshansky Grigory"

import argparse
import json


def arguments(replay):
    """Parse command-line controls without performing inference."""
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "before",
        "after",
        "out",
        "capacity-url",
        "engine-metrics-url",
        "before-source",
        "after-source",
        "runtime-metadata",
    ):
        parser.add_argument("--" + name)
    parser.add_argument(
        "--replay", help="summarize retained raw, including censored work; no network"
    )
    parser.add_argument("--model", default="qwen3.8-flash-next-nvidia")
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=7403)
    parser.add_argument(
        "--wait-idle",
        action="store_true",
        help="wait rather than refuse a busy qualification gate",
    )
    args = parser.parse_args()
    if args.replay:
        print(json.dumps(replay(args.replay), ensure_ascii=False, indent=2))
        return parser, args
    if args.rounds < 1 or args.rounds > 1000:
        parser.error("rounds must be in 1..1000")
    if any(
        getattr(args, name) is None
        for name in (
            "before",
            "after",
            "out",
            "capacity_url",
            "engine_metrics_url",
            "before_source",
            "after_source",
            "runtime_metadata",
        )
    ):
        parser.error(
            "live benchmark requires endpoints, out, source paths, runtime metadata and idle endpoints"
        )
    return parser, args
