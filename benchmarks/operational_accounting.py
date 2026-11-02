"""Account for all launched requests and censored graph executions."""

from __future__ import annotations

__author__ = "Olshansky Grigory"


class RunAccounting:
    """Scalar ledger of ALL launched work, including incomplete/censored graphs."""

    def __init__(self):
        """Initialize Run Accounting state and dependencies."""
        self.requests, self.started_graphs, self.finished_graphs = {}, set(), set()

    def record(self, row):
        """Record one observation while preserving request and graph correlation."""
        kind, ident = row.get("record_type"), row.get("request_id")
        if kind == "graph_started":
            self.started_graphs.add(row["graph_id"])
        elif kind == "graph":
            self.finished_graphs.add(row["graph_id"])
        elif kind == "request_started":
            self.requests[ident] = {
                "model": row["url"].endswith("/chat/completions"),
                "completed": False,
                "tokens": None,
                "status": None,
            }
        elif kind == "request_completed" and ident in self.requests:
            state = self.requests[ident]
            body = row.get("response", {})
            usage = (
                body.get("upstream_usage", body.get("usage", {}))
                if type(body) is dict
                else {}
            )
            tokens = usage.get("completion_tokens") if type(usage) is dict else None
            state.update(
                completed=True,
                status=row["status"],
                tokens=tokens if type(tokens) is int and tokens >= 0 else None,
            )

    def summary(self):
        """Summarize recorded observations without treating missing usage as zero."""
        model = [r for r in self.requests.values() if r["model"]]
        missing = sum(r["tokens"] is None for r in model)
        known = sum(r["tokens"] for r in model if r["tokens"] is not None)
        return {
            "http_started": len(self.requests),
            "http_completed": sum(r["completed"] for r in self.requests.values()),
            "http_failures": sum(
                r["completed"] and r["status"] != 200 for r in self.requests.values()
            ),
            "llm_started": len(model),
            "completion_tokens": None if missing else known,
            "known_completion_tokens_partial": known,
            "llm_usage_unknown": missing,
            "graphs_started": len(self.started_graphs),
            "graphs_completed": len(self.finished_graphs),
            "censored_graph_ids": sorted(self.started_graphs - self.finished_graphs),
        }
