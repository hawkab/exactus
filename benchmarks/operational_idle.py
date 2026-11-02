"""Read shared backend admission and engine activity evidence."""

from __future__ import annotations

__author__ = "Olshansky Grigory"

import math
import re


def idle(capacity, metrics):
    """Ingress counters alone miss direct backend users; require both."""
    fields = ("active_weight", "waiting", "reserved", "admitted")
    if (
        capacity.get("backend_ready") is not True
        or capacity.get("admission_enabled") is not True
    ):
        return False
    if any(type(capacity.get(k)) is not int or capacity[k] != 0 for k in fields):
        return False
    found = set()
    for line in metrics.splitlines():
        match = re.match(
            r"^vllm:num_requests_(running|waiting)(?:\{[^\n]*\})?\s+(\S+)", line
        )
        if match:
            try:
                value = float(match[2])
            except ValueError:
                return False
            if not math.isfinite(value) or value != 0:
                return False
            found.add(match[1])
    return found == {"running", "waiting"}
