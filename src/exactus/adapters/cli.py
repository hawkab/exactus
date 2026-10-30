"""Command line interface for the Exactus external computer.

Author: Olshansky Grigory
"""

from __future__ import annotations

import argparse
import base64
import json
import sys

from ..domain.errors import ExactusError
from .digest import HashlibHasher
from .in_process import InProcessExternalComputer
from .worker import WorkerExternalComputer

__author__ = "Olshansky Grigory"


def main(argv: list[str] | None = None) -> int:
    """Parse arguments and run the selected command."""
    parser = argparse.ArgumentParser(prog="exactus")
    sub = parser.add_subparsers(dest="command", required=True)
    evaluate = sub.add_parser("eval", help="evaluate one exact expression")
    evaluate.add_argument("expression")
    evaluate.add_argument("--format", choices=("dec", "hex", "bin"), default="dec")
    evaluate.add_argument(
        "--data",
        action="append",
        default=[],
        help="name=base64 payload for a data reference",
    )
    evaluate.add_argument(
        "--worker", action="store_true", help="use the sandboxed worker"
    )
    proxy = sub.add_parser(
        "proxy", help="run the OpenAI-compatible exact-compute proxy"
    )
    proxy.add_argument(
        "--upstream", required=True, help="backend URL, e.g. http://127.0.0.1:8000"
    )
    proxy.add_argument("--host", default="127.0.0.1")
    proxy.add_argument("--port", type=int, default=8090)
    proxy.add_argument(
        "--no-hint", action="store_true", help="do not inject the marker hint"
    )
    proxy.add_argument("--require-final", action="store_true")
    sub.add_parser("demo", help="run the exact-span protocol demo")
    sub.add_parser(
        "external-computers", help="read a JSON {tasks:[...]} batch from stdin; no LLM"
    )
    sub.add_parser("plan", help="execute JSON {plan,inputs,checks} from stdin; no LLM")
    args = parser.parse_args(argv)
    try:
        if args.command == "eval":
            return _eval(args)
        if args.command == "proxy":
            return _proxy(args)
        if args.command in ("external-computers", "plan"):
            raw = sys.stdin.buffer.read(65537)
            if len(raw) > 65536:
                parser.error("external computer request exceeds 64 KiB")
            expected = (
                "expected JSON {tasks:[...]}"
                if args.command == "external-computers"
                else "expected JSON {plan,inputs,checks}"
            )
            try:
                payload = json.loads(raw)
                if type(payload) is not dict:
                    raise ValueError(expected)
                if args.command == "external-computers":
                    valid = set(payload) == {"tasks"}
                else:
                    valid = "plan" in payload and not set(payload) - {
                        "plan",
                        "inputs",
                        "checks",
                    }
                if not valid:
                    raise ValueError(expected)
            except (ValueError, RecursionError):
                parser.error(expected)
            worker = WorkerExternalComputer()
            try:
                result = (
                    worker.run_external_computers(payload["tasks"])
                    if args.command == "external-computers"
                    else worker.run_plan(
                        payload["plan"],
                        inputs=payload.get("inputs"),
                        checks=payload.get("checks"),
                    )
                )
                print(json.dumps(result, ensure_ascii=False))
            finally:
                worker.close()
            return 0
        return _demo()
    except ExactusError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


def _proxy(args) -> int:
    """Run the exact-compute proxy in front of one upstream."""
    from ..proxy import serve

    serve(
        args.upstream,
        InProcessExternalComputer(HashlibHasher()),
        host=args.host,
        port=args.port,
        hint="" if args.no_hint else None,
        require_final=args.require_final,
    )
    return 0


def _eval(args) -> int:
    """Evaluate one expression and print its rendered value."""
    data = {}
    for item in args.data:
        name, _, payload = item.partition("=")
        data[name] = base64.b64decode(payload)
    external_computer = (
        WorkerExternalComputer()
        if args.worker
        else InProcessExternalComputer(HashlibHasher())
    )
    try:
        value = external_computer.evaluate(
            args.expression, data, output_format=args.format
        )
    finally:
        if isinstance(external_computer, WorkerExternalComputer):
            external_computer.close()
    print(value.render())
    return 0


def _demo() -> int:
    """Run the exact-span protocol over a sample transcript."""
    from ..application.compute import ComputeService

    service = ComputeService(InProcessExternalComputer(HashlibHasher()))
    transcript = "budget: <calc>37*184500</calc> checksum: <calc-final>sha256(@blob) % 1000</calc-final>"
    result = service.run(
        transcript, {"blob": b"exactus"}, require_final=True, output_format="dec"
    )
    from dataclasses import asdict

    print(
        json.dumps(
            {
                "text": result.text,
                "exact_final": result.exact_final,
                "operations": [asdict(op) for op in result.operations],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
