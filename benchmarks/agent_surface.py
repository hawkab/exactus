"""Measure the actual raw/proxy/native HTTP surface, including client CPU work."""

import json
import time

from exactus.http_io import request_bytes


class SurfaceFailure(RuntimeError):
    def __init__(self, message, partial):
        super().__init__(message)
        self.partial = partial


class Surface:
    def __init__(self, url, kind, model, max_tokens, params=None, api_key=None):
        if kind not in ("raw", "proxy", "native"):
            raise ValueError("unknown surface")
        self.url, self.kind, self.model, self.max_tokens = (
            url.rstrip("/"),
            kind,
            model,
            max_tokens,
        )
        self.params, self.api_key = dict(params or {}), api_key

    def __call__(self, prompt):
        started = time.perf_counter()
        if self.kind == "native":
            body = {
                "model": self.model,
                "exact_compute": True,
                "prompt": prompt,
                "mode": "preserve-text",
                "max_tokens": self.max_tokens,
            }
            path = "/v1/exact/completions"
        else:
            body = {
                **self.params,
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0,
                "stream": False,
                "max_tokens": self.max_tokens,
            }
            if self.kind == "proxy":
                body["mode"] = "preserve-text"
            path = "/v1/chat/completions"
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = "Bearer " + self.api_key
        partial = {"request": body, "surface": self.kind, "url": self.url + path}
        try:
            status, raw = request_bytes(
                self.url + path, json.dumps(body).encode(), headers=headers, timeout=600
            )
            result = json.loads(raw)
            partial.update(status=status, response=result)
            if status != 200:
                raise ValueError(f"HTTP {status}")
            if self.kind == "native":
                text, finish = result["text"], result["finish_reason"]
                usage = {
                    "completion_tokens": result["sampled_tokens"],
                    "prompt_tokens": result["prompt_tokens"],
                }
                if result.get("generation_complete") is not True:
                    raise ValueError("incomplete native generation")
                if result.get("sampled_tokens_exact") is not True:
                    raise ValueError(
                        "estimated tokens cannot be used as inference evidence"
                    )
            else:
                choice = result["choices"][0]
                text, finish = choice["message"]["content"], choice.get("finish_reason")
                usage = result.get("upstream_usage", result.get("usage", {}))
            if finish != "stop":
                raise ValueError("incomplete generation: " + str(finish))
            if any(
                type(usage.get(k)) is not int or usage[k] < 0
                for k in ("completion_tokens", "prompt_tokens")
            ):
                raise ValueError("invalid usage")
            partial.update(
                text=text,
                finish=finish,
                usage=usage,
                wall=time.perf_counter() - started,
            )
            partial["external_operations"] = (
                len(result.get("operations", []))
                if self.kind == "native"
                else result.get("exactus", {}).get("operations", 0)
            )
            partial["engine_segments"] = result.get("engine_segments", 1)
            return partial
        except Exception as exc:
            partial["wall"] = time.perf_counter() - started
            raise SurfaceFailure(f"{type(exc).__name__}: {exc}", partial) from exc
