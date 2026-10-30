# Exactus

An external computer for LLMs. The model emits an expression or a
compute plan; Exactus executes it and renders the result. Moving mechanical
audits from LLM generation to CPU execution can reduce model calls and tokens.
Execution verifies the supplied program, not whether it solves the user's task.

An external computer runs outside the model's prediction process, on the same
host or another host. The Python port is `ExternalComputer`; implementations
are `InProcessExternalComputer` and `WorkerExternalComputer`.

## Install and run

Python 3.10+, Linux for the resource-limited worker, and an OpenAI-compatible backend.

```bash
git clone https://github.com/hawkab/exactus.git
cd exactus
python3 -m venv .venv
source .venv/bin/activate
pip install .
exactus proxy --upstream http://127.0.0.1:18001 --port 8090
```

Point a non-streaming chat client at `http://127.0.0.1:8090/v1`.
The proxy injects a hint for `<calc>EXPR</calc>` and
`<calc-final>EXPR</calc-final>`, then substitutes evaluated values.
Tool calls and null tool-message content pass through; tools are executed by
the client, not by Exactus. Tool turns are not final answers; a backend's
`stop` with nonempty tool calls is normalized to `tool_calls`, preserving
`upstream_finish_reason`. Streaming is rejected before inference.
Standard `usage` and `upstream_usage` both count upstream generation,
not the length of rewritten text.

Modes: `preserve-text` (default), `audit`, `strict`, `numeric-final`.
`exact_final` means a final expression was evaluated; `task_verified`
is null unless an independent calculator-prompt check is available.
Strict/numeric-final reject a known mismatching final. Arbitrary prose is not verified.
`data: {"name":"<base64>"}` supplies request-scoped bytes for `@name`.

## Compute APIs

- `POST /v1/exact/external-computers`, body `{"tasks":[...]}`: kinds `calc`,
  `python_expr`, `sql_read`; optional dependencies on earlier task results.
  Python: `run_external_computers(tasks, external_computer)` from
  `exactus.external_computers`, or `worker.run_external_computers(tasks)`.
- `POST /v1/exact/plans`, body `{"plan":{"tasks":[...],"answer":...},"inputs":{},"checks":[]}`.
  `$input` binds source data; `$ref` binds task results, with optional
  `path` selectors. Answer templates support `$text` and boolean `$if/then/else`.
  Plans are bounded DAGs with request-local duplicate reuse and caller checks.
- Chat `compute_plan:true`, `plan_inputs`, `plan_checks` generates a plan in one
  model call, then renders the result without another generation.
  Optional `plan_input_view:"schema"` hides cell values, not field names.
- Chat `external_computer_tasks:[...]` runs a task batch before generation;
  execution results are returned in `exactus.external_computers` metadata.
- `exactus external-computers` and `exactus plan` accept the same JSON on stdin.
  `exactus eval '37*184500' --worker` evaluates one integer expression.

Technical APIs run in a resource-limited subprocess: at most 16 tasks,
64 KiB JSON, depth 32, 4096-bit integers; SQL at most 2048 input rows and
512 output rows. This is not a seccomp sandbox for arbitrary code.

`python_expr` interprets JSON values, integer arithmetic, comparisons,
single-generator comprehensions and pure sum/len/min/max/sorted/range/abs/all/any.
No eval/exec, imports, attributes, statements or floats. Generators are lazy.
Calc `^` means power, `/` and `%` truncate toward zero; Python operators
retain Python semantics.

SQL supports read-only SQLite queries over explicit in-memory tables.
REAL, CAST, avg/round and unchecked arithmetic operators are rejected.
Use checked int64 `int_add/sub/mul/div/mod(a,b)` or `python_expr` for
numeric work; int_div floors, integer sum rejects non-integers/overflow.
SQL string/comparison operations retain SQLite semantics. This restricted
numeric contract replaces 0.5's permissive arithmetic.

## Inside vLLM

Install `.[server]` into the existing vLLM environment and add `exactus` to
the existing `VLLM_PLUGINS` allowlist. The registered entry point is
`exactus.adapters.default_plugin:ExactusEndpointPlugin`.
Enable only one provider of the Exactus routes.

`POST /v1/exact/completions` accepts
`{"exact_compute":true,"prompt":"Compute: 6*7","mode":"numeric-final"}`.
Intermediate values re-enter through normal prefill with engine-owned MTP.
The offline logits helper is not a vLLM plugin; speculative custom forcing
is unsupported. Native GPU performance and KV/recurrent-cache reuse require
qualification on the exact runtime; arithmetic HTTP results do not establish them.

## Validation

`pip install '.[server,test]'` then
`python -m unittest discover -s tests -t .`.
Benchmark sources are in `benchmarks/`; runtime manifests and measurement
artifacts stay outside git. `bench_external_computers.py --help` describes the paired
workflow and idle gates; `--replay RAW.jsonl` accounts for all launched work,
including censored graphs. Facts, typed action contracts, prose quality,
queue waiting and successful-pair latency are separate measurements.

Author: Olshansky Grigory
