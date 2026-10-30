"""Loopback-only llmos queue port: correlated calls with an end-to-end CPU budget."""
import json
import time
from urllib.parse import urlsplit
from uuid import uuid4

from ..domain import errors
from ..domain.errors import DataError, WorkerError
from ..domain.value import ComputeValue
from ..http_io import request_bytes
from .worker_session import encode_data


class LLMOSUnavailable(WorkerError):
    def __init__(self, code):
        self.code = code
        super().__init__('llmos ' + code + '; continue without tools')


class LLMOSExternalComputer:
    """No shared session lock, no retries and no fallback to another CPU executor."""

    def __init__(self, url, *, budget_ms=50):
        target = urlsplit(url)
        if (target.scheme != 'http' or target.hostname not in ('127.0.0.1', '::1')
                or target.username or target.password or target.path not in ('', '/')
                or target.query or target.fragment):
            raise ValueError('llmos requires a literal loopback HTTP origin')
        if type(budget_ms) is not int or not 1 <= budget_ms <= 100:
            raise ValueError('llmos budget must be 1..100 ms')
        self.url, self.budget_ms = url.rstrip('/'), budget_ms

    def for_request(self, data, request_id):
        return LLMOSRequest(self, data, request_id)

    def request(self, data=None):
        return self.for_request(data, uuid4().hex)

    def evaluate_many(self, expressions, data=None, *, output_format='dec'):
        with self.request(data) as request:
            return request.evaluate_many(expressions, output_format=output_format)

    def evaluate(self, expression, data=None, *, output_format='dec'):
        return self.evaluate_many([expression], data, output_format=output_format)[0]

    def run_external_computers(self, tasks):
        with self.request() as request:
            return request.call('tasks', {'tasks': tasks})

    def run_plan(self, plan, *, inputs=None, checks=None):
        with self.request() as request:
            return request.call('plan', {'plan': plan, 'inputs': inputs, 'checks': checks})

    def close(self):
        pass  # llmos owns workers; an API worker owns no remote persistent state.


class LLMOSRequest:
    def __init__(self, client, data, request_id):
        self.client, self.data, self.request_id = client, encode_data(data or {}), request_id
        self.sequence, self.closed = 0, False
        self.calls = []

    def call(self, operation, payload):
        if self.closed:
            raise DataError('llmos request is closed')
        self.sequence += 1
        cid = str(self.sequence)
        end = time.monotonic_ns() + self.client.budget_ms * 1000000
        body = json.dumps({'request_id': self.request_id, 'call_id': cid,
            'operation': operation, 'payload': payload, 'budget_ms': self.client.budget_ms,
            'deadline_ns': end}, allow_nan=False, separators=(',', ':')).encode()
        if len(body) > 32768:
            raise DataError('llmos compute request exceeds 32 KiB')
        attempt = {'request_id': self.request_id, 'call_id': cid, 'ok': False}
        self.calls.append(attempt)
        try:
            status, raw = request_bytes(self.client.url + '/v1/compute', body,
                headers={'Content-Type': 'application/json'}, timeout=self.client.budget_ms / 1000,
                limit=16384)
            reply = json.loads(raw)
        except (OSError, TimeoutError, ValueError) as exc:
            code = 'timeout' if time.monotonic_ns() >= end else 'transport_error'
            attempt['error_code'] = code
            raise LLMOSUnavailable(code) from exc
        if (type(reply) is not dict or reply.get('request_id') != self.request_id
                or reply.get('call_id') != cid):
            raise LLMOSUnavailable('correlation_error')
        attempt.update({k: reply[k] for k in ('ok', 'error_code', 'queue_ms', 'execution_ms') if k in reply})
        if not reply.get('ok'):
            code = reply.get('error_code', 'unavailable')
            if reply.get('fallback') or status >= 500:
                raise LLMOSUnavailable(code)
            cls = getattr(errors, reply.get('error_type', ''), DataError)
            if cls not in (errors.GrammarError, errors.BudgetError, errors.DataError,
                           errors.FormatError, errors.VerificationError):
                cls = DataError
            raise cls(reply.get('error', 'llmos rejected computation'))
        if status != 200 or type(reply.get('result')) is not dict:
            raise LLMOSUnavailable('invalid_response')
        return reply['result']

    def evaluate_many(self, expressions, data=None, *, output_format='dec'):
        if data is not None:
            raise DataError('request data is immutable')
        if isinstance(expressions, (str, bytes)):
            raise DataError('expected an expression sequence')
        expressions = list(expressions)
        if not expressions:
            return []
        result = self.call('eval_many', {'expressions': expressions, 'data': self.data,
                                         'output_format': output_format})
        rows = result.get('values')
        if type(rows) is not list or len(rows) != len(expressions):
            raise LLMOSUnavailable('invalid_values')
        try:
            return [ComputeValue(int(row['value']), row['format']) for row in rows]
        except (KeyError, TypeError, ValueError) as exc:
            raise LLMOSUnavailable('invalid_values') from exc

    def evaluate(self, expression, data=None, *, output_format='dec'):
        return self.evaluate_many([expression], data, output_format=output_format)[0]

    def close(self):
        self.closed = True
        self.data.clear()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
