"""Compose and run the standalone Exactus proxy server."""

from __future__ import annotations

__author__ = "Olshansky Grigory"

from .proxy_transport import build_server


def serve(
    upstream: str,
    external_computer,
    host: str = "127.0.0.1",
    port: int = 8090,
    hint: str | None = None,
    require_final: bool = False,
    deadline_s: float = 600.0,
    max_concurrent: int = 16,
) -> None:
    """Run the proxy until interrupted."""
    from .proxy import ExactusProxy

    proxy = ExactusProxy(
        upstream,
        external_computer,
        hint=hint,
        require_final=require_final,
        deadline_s=deadline_s,
    )
    server = build_server(proxy, host, port, max_concurrent=max_concurrent)
    print(f"exactus proxy: http://{host}:{port} -> {upstream}")
    try:
        server.serve_forever()
    finally:
        server.server_close()
        proxy.close()
