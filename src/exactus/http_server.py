"""Bound accepted sockets BEFORE creating threads; reuse bounded work threads."""

from concurrent.futures import ThreadPoolExecutor
from http.server import ThreadingHTTPServer
import threading
from .cancellation import Cancellation


class BoundedHTTPServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address, handler, *, limit):
        self._connections = threading.BoundedSemaphore(max(1, limit))
        self._cancellation = Cancellation()
        self._executor = ThreadPoolExecutor(
            max_workers=max(1, limit), thread_name_prefix="exactus"
        )
        handler._executor = self._executor
        handler._cancellation = self._cancellation
        super().__init__(address, handler)

    def process_request(self, request, address):
        if not self._connections.acquire(blocking=False):
            try:
                request.settimeout(0.05)
                request.sendall(
                    b"HTTP/1.0 503 Service Unavailable\r\nContent-Type: application/json\r\n"
                    b'Connection: close\r\n\r\n{"error":"connection capacity","error_code":"overloaded"}'
                )
            except OSError:
                pass
            finally:
                self.shutdown_request(request)
            return
        try:
            super().process_request(request, address)
        except BaseException:
            self._connections.release()
            raise

    def process_request_thread(self, request, address):
        try:
            super().process_request_thread(request, address)
        finally:
            self._connections.release()

    def server_close(self):
        self._cancellation.cancel()
        super().server_close()
        self._executor.shutdown(wait=False)
