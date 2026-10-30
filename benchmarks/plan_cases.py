"""Operational fixtures with independently calculated, exact-type gold answers."""

import random


def cases(seed, *, compact=False):
    rng = random.Random(seed)
    nodes = [
        [rng.randrange(50, 65), rng.randrange(47, 64)],
        [rng.randrange(48, 64), rng.randrange(50, 66)],
    ]
    jobs = [rng.randrange(1231, 11991) for _ in range(8 if compact else 27)]
    total = 0
    for value in jobs:
        total += value
    required = (total + 1199) // 1200
    rates = [nodes[0][0] + nodes[0][1], nodes[1][0] + nodes[1][1]]
    capacity_gold = {
        "requested_tokens": total,
        "required_tps": required,
        "healthy_tps": rates[0] + rates[1],
        "worst_n1_tps": min(rates),
        "n1_ok": min(rates) >= required,
    }
    capacity = {
        "id": f"capacity-{seed}",
        "inputs": {"nodes": nodes, "jobs": jobs, "window_s": 1200},
        "question": "You are the lead engineer rolling out a model-router on two Spark nodes. nodes contains two decode rates per Spark; they are already measured and additive, and must not be multiplied by C3. jobs are requested tokens per job over window_s. Compute JSON with exactly the fields requested_tokens (sum of jobs), required_tps (ceil(requested_tokens/window_s)), healthy_tps (sum of all rates), worst_n1_tps (minimum summed rate of one Spark), n1_ok (whether worst_n1_tps covers required_tps). JSON only, numbers are integers, n1_ok is boolean.",
        "expected": capacity_gold,
    }
    rows = []
    unique = {}
    for i in range(9 if compact else 21):
        root = f"req{i:02}"
        attempts = 2 if i % 3 == 0 else 1
        for attempt in range(attempts):
            ident = f"{root}-a{attempt}"
            delivered = int(i % 5 != 0 and attempt == attempts - 1)
            row = [root, ident, delivered, rng.randrange(101, 11911)]
            unique[ident] = row
            rows.append(row)
            if i % 4 == 0:
                rows.append(row.copy())  # duplicate observation, not a new attempt
    roots, delivered_roots, delivered_tokens = set(), set(), 0
    for root, attempt, delivered, tokens in unique.values():
        roots.add(root)
        if delivered:
            delivered_roots.add(root)
            delivered_tokens += tokens
    delivery_gold = {
        "unique_requests": len(roots),
        "retry_attempts": len(unique) - len(roots),
        "delivered_unique": len(delivered_roots),
        "undelivered": len(roots - delivered_roots),
        "delivered_tokens": delivered_tokens,
    }
    delivery = {
        "id": f"delivery-{seed}",
        "inputs": {
            "tables": {
                "events": {
                    "columns": ["root", "attempt_id", "delivered", "tokens"],
                    "rows": rows,
                }
            }
        },
        "question": "You are the delivery auditor of a model-router. events contains repeated observations of the same attempt_id: deduplicate them first. delivered=1 means an answered delivery; tokens belong to the attempt. Compute JSON with exactly the fields unique_requests (unique roots), retry_attempts (unique attempt_id minus unique roots), delivered_unique (roots with at least one delivery), undelivered (roots without delivery), delivered_tokens (sum of tokens over unique delivered attempt_id). JSON only, integer values.",
        "expected": delivery_gold,
    }
    docs = [
        [
            rng.randrange(1100, 2400),
            rng.randrange(1400, 3300),
            rng.choice([1, 3]),
            rng.randrange(9, 63),
        ]
        for _ in range(4 if compact else 11)
    ]
    disk, peak = 0, 0
    for width, height, channels, pages in docs:
        single = width * height * channels
        disk += single * pages
        candidate = single * min(pages, 3)
        if candidate > peak:
            peak = candidate
    ocr_gold = {
        "disk_bytes": disk,
        "peak_batch_bytes": peak,
        "peak_batch_mib": (peak + 1048575) // 1048576,
        "fits_ram": peak <= 67108864,
    }
    ocr = {
        "id": f"ocr-{seed}",
        "inputs": {"documents": docs, "batch_pages": 3, "ram_bytes": 67108864},
        "question": "You are the OCR resource auditor. Each documents record is [width,height,channels,pages]. Uncompressed uint8 occupies width*height*channels bytes per page. Documents are processed sequentially; within one document up to min(pages,batch_pages) pages run at once. Ignore auxiliary buffers. Compute JSON with exactly the fields disk_bytes (sum of bytes over all pages), peak_batch_bytes (maximum bytes of one concurrent batch), peak_batch_mib (ceil(bytes/1048576)), fits_ram (peak_batch_bytes <= ram_bytes). JSON only, bytes and MiB are integers, fits_ram is boolean.",
        "expected": ocr_gold,
    }
    for item in (capacity, delivery, ocr):
        item["checks"] = [
            {
                "path": [key],
                "op": "type",
                "value": "bool" if type(value) is bool else "int",
            }
            for key, value in item["expected"].items()
        ]
    return [capacity, delivery, ocr]


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
