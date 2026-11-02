"""Bound header parsing and retain admission ownership across disconnects."""

__author__ = "Olshansky Grigory"

import socket
import threading
import time
from http.server import BaseHTTPRequestHandler

from .transport_work import WorkLease


class DeadlineHeaders(BaseHTTPRequestHandler):
    """Share admission and deadline ownership across HTTP header processing."""

    def setup(self):
        """Initialize socket state, ownership, and the header deadline timer."""
        super().setup()
        self._deadline = time.monotonic() + self._timeout
        self.connection.settimeout(self._timeout)
        self._admitted = self._admission.enter()
        self._lease = WorkLease(self._admission.exit) if self._admitted else None
        self._header_timer = threading.Timer(self._timeout, self._close_headers)
        self._header_timer.daemon = True
        if self._admitted:
            self._header_timer.start()

    def _close_headers(self):
        """Interrupt a connection whose header deadline has expired."""
        try:
            self.connection.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass

    def handle(self):
        """Handle the connection while tolerating normal disconnects and expired headers."""
        try:
            if self._admitted:
                super().handle()
            else:
                self.connection.sendall(
                    b"HTTP/1.0 503 Service Unavailable\r\nContent-Type: application/json\r\n"
                    b'Connection: close\r\n\r\n{"error":"server is at capacity","error_code":"overloaded"}'
                )
        except OSError:
            self.close_connection = True

    def finish(self):
        """Close handler streams and release transport ownership."""
        self._header_timer.cancel()
        try:
            super().finish()
        finally:
            if self._lease is not None:
                self._lease.release()
