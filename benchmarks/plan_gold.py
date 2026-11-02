"""Define independent reference programs for CPU plan qualification."""

__author__ = "Olshansky Grigory"


def gold_plan(item):
    """Reference program for CPU qualification, not supplied to the LLM eval."""
    if item["id"].startswith("capacity"):
        tasks = [
            {
                "id": "total",
                "kind": "python_expr",
                "expression": "sum(xs)",
                "inputs": {"xs": {"$input": "jobs"}},
            },
            {
                "id": "demand",
                "kind": "python_expr",
                "expression": "(n+w-1)//w",
                "inputs": {"n": {"$ref": "total"}, "w": {"$input": "window_s"}},
            },
            {
                "id": "healthy",
                "kind": "python_expr",
                "expression": "sum(sum(n) for n in ns)",
                "inputs": {"ns": {"$input": "nodes"}},
            },
            {
                "id": "n1",
                "kind": "python_expr",
                "expression": "min(sum(n) for n in ns)",
                "inputs": {"ns": {"$input": "nodes"}},
            },
            {
                "id": "safe",
                "kind": "python_expr",
                "expression": "capacity >= demand",
                "inputs": {"capacity": {"$ref": "n1"}, "demand": {"$ref": "demand"}},
            },
        ]
        answer = {
            k: {"$ref": name}
            for k, name in zip(
                item["expected"], ["total", "demand", "healthy", "n1", "safe"]
            )
        }
    elif item["id"].startswith("delivery"):
        query = "WITH unique_events AS (SELECT DISTINCT root, attempt_id, delivered, tokens FROM events) SELECT count(DISTINCT root) AS roots, int_sub(count(*),count(DISTINCT root)) AS retries, count(DISTINCT CASE WHEN delivered=1 THEN root END) AS delivered_roots, int_sub(count(DISTINCT root),count(DISTINCT CASE WHEN delivered=1 THEN root END)) AS undelivered, sum(CASE WHEN delivered=1 THEN tokens ELSE 0 END) AS delivered_tokens FROM unique_events"
        tasks = [
            {
                "id": "delivery",
                "kind": "sql_read",
                "query": query,
                "tables": {"$input": "tables"},
            }
        ]
        answer = {
            k: {"$ref": "delivery", "path": ["rows", 0, i]}
            for i, k in enumerate(item["expected"])
        }
    else:
        tasks = [
            {
                "id": "disk",
                "kind": "python_expr",
                "expression": "sum(d[0]*d[1]*d[2]*d[3] for d in ds)",
                "inputs": {"ds": {"$input": "documents"}},
            },
            {
                "id": "peak",
                "kind": "python_expr",
                "expression": "max(d[0]*d[1]*d[2]*min(d[3],batch) for d in ds)",
                "inputs": {
                    "ds": {"$input": "documents"},
                    "batch": {"$input": "batch_pages"},
                },
            },
            {
                "id": "mib",
                "kind": "python_expr",
                "expression": "(n+1048575)//1048576",
                "inputs": {"n": {"$ref": "peak"}},
            },
            {
                "id": "fits",
                "kind": "python_expr",
                "expression": "n <= ram",
                "inputs": {"n": {"$ref": "peak"}, "ram": {"$input": "ram_bytes"}},
            },
        ]
        answer = {
            k: {"$ref": name}
            for k, name in zip(item["expected"], ["disk", "peak", "mib", "fits"])
        }
    return {"tasks": tasks, "answer": answer}
