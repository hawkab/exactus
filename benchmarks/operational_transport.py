"""Capture correlated HTTP calls and decode completed model replies."""

from __future__ import annotations

__author__ = "Olshansky Grigory"

import json
import time
import urllib.error
import urllib.request
import uuid


def post(url, payload, timeout=600, capture=None):
    """Send a request to the local test endpoint and capture its result."""
    started = time.monotonic()
    request_id = "exactus-external-computer-bench-" + uuid.uuid4().hex
    if capture:
        capture(
            {
                "record_type": "request_started",
                "request_id": request_id,
                "url": url,
                "payload": payload,
                "at": time.time(),
            }
        )
    request = urllib.request.Request(
        url,
        json.dumps(payload).encode(),
        {"Content-Type": "application/json", "X-Request-ID": request_id},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = json.loads(response.read(1048577))
            status = response.status
        row = {
            "request": payload,
            "response": body,
            "elapsed_s": time.monotonic() - started,
            "status": status,
        }
    except urllib.error.HTTPError as exc:
        raw = exc.read(1048577)
        try:
            body = json.loads(raw)
        except ValueError:
            body = {"error": raw[:1024].decode("utf-8", "replace")}
        row = {
            "request": payload,
            "response": body,
            "elapsed_s": time.monotonic() - started,
            "status": exc.code,
            "error": "HTTP " + str(exc.code),
        }
    except (OSError, ValueError) as exc:
        row = {
            "request": payload,
            "error": str(exc),
            "elapsed_s": time.monotonic() - started,
            "status": 0,
        }
    row["request_id"] = request_id
    if capture:
        capture({"record_type": "request_completed", **row})
    return row


def content(row):
    """Decode completed model JSON only after validating HTTP and finish status."""
    if row.get("status") != 200:
        raise ValueError("HTTP request did not succeed")
    response = row.get("response", {})
    choice = response.get("choices", [{}])[0]
    if type(choice) is not dict:
        raise ValueError("invalid model choice")
    if choice.get("finish_reason") != "stop":
        raise ValueError("incomplete model output")
    text = choice.get("message", {}).get("content", "")
    if type(text) is not str:
        raise ValueError("invalid model content")
    if text.startswith("```"):
        lines = text.splitlines()
        text = "\n".join(lines[1:-1])
    return json.loads(text)


def same_json(a, b):
    """Compare canonical JSON while distinguishing booleans from integers."""
    return json.dumps(a, sort_keys=True, ensure_ascii=False) == json.dumps(
        b, sort_keys=True, ensure_ascii=False
    )


def chat(url, prompt, model, capture=None):
    """Send a model request and retain its actual usage and completion status."""
    return post(
        url + "/v1/chat/completions",
        {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
        },
        capture=capture,
    )
