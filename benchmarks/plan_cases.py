"""Operational fixtures with independently calculated, exact-type gold answers."""

__author__ = "Olshansky Grigory"

import random

from plan_gold import gold_plan as gold_plan


def cases(seed, *, compact=False):
    """Generate operational plan fixtures and their exact expected answers."""
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
                rows.append(row.copy())
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
