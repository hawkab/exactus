"""Bound header parsing and retain admission ownership across disconnects."""

from http.server import BaseHTTPRequestHandler
import socket
import threading
import time

from .transport_work import WorkLease


class DeadlineHeaders(BaseHTTPRequestHandler):
    def setup(self):
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
        try:
            self.connection.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass

    def handle(self):
        try:
            if self._admitted:
                super().handle()
            else:
                self.connection.sendall(
                    b"HTTP/1.0 503 Service Unavailable\r\nContent-Type: application/json\r\n"
                    b'Connection: close\r\n\r\n{"error":"server is at capacity","error_code":"overloaded"}'
                )
                # Closing with an unread upload can reset TCP while a normal
                # client is still sending its body, hiding the typed 503 behind
                # BrokenPipe. Half-close the reply, then boundedly drain input.
                self.connection.shutdown(socket.SHUT_WR)
                self.connection.settimeout(min(0.1, self._timeout))
                remaining = 65536
                while remaining:
                    chunk = self.connection.recv(min(remaining, 8192))
                    if not chunk:
                        break
                    remaining -= len(chunk)
        except OSError:
            self.close_connection = True  # normal client disconnect or header deadline

    def finish(self):
        self._header_timer.cancel()
        try:
            super().finish()
        finally:
            if self._lease is not None:
                self._lease.release()
