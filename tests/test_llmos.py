"""llmos port, one-way fallback, strict contracts and native no-tools generation."""
import asyncio
import json
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

from exactus.adapters.llmos import LLMOSExternalComputer, LLMOSUnavailable
from exactus.adapters.endpoint_adapter import ExactusCoordinator
from exactus.adapters.endpoint_route import _serialize
from exactus.adapters.native_engine import NativeEngine
from exactus.adapters.tokenizer import ByteTokenizer
from exactus.domain.ports import Segment


class LLMOSTests(unittest.TestCase):
    def test_ids_values_and_one_payload_batch(self):
        calls = []
        def post(url, body, **kwargs):
            request = json.loads(body)
            calls.append(request)
            return 200, json.dumps({'ok': True, 'request_id': request['request_id'],
                'call_id': request['call_id'], 'result': {'values': [{'value': '42', 'format': 'dec'}]}}).encode()
        with patch('exactus.adapters.llmos.request_bytes', side_effect=post):
            with LLMOSExternalComputer('http://127.0.0.1:8800').for_request(None, 'parent') as r:
                self.assertEqual(r.evaluate('6*7').integer, 42)
                self.assertEqual(r.evaluate('40+2').integer, 42)
        self.assertEqual([r['request_id'] for r in calls], ['parent', 'parent'])
        self.assertEqual([r['call_id'] for r in calls], ['1', '2'])
        self.assertTrue(all('deadline_ns' in r for r in calls))

    def test_wrong_response_id_is_not_a_value(self):
        with patch('exactus.adapters.llmos.request_bytes', return_value=(200, b'{"ok":true,"request_id":"someone-else","call_id":"1"}')):
            with self.assertRaises(LLMOSUnavailable):
                LLMOSExternalComputer('http://127.0.0.1:8800').evaluate('6*7')

    def test_remote_and_unbounded_budgets_rejected(self):
        for url in ('http://10.8.1.111:8800', 'http://localhost:8800', 'http://u:p@127.0.0.1'):
            with self.assertRaises(ValueError):
                LLMOSExternalComputer(url)
        for budget in (0, 101, True):
            with self.assertRaises(ValueError):
                LLMOSExternalComputer('http://127.0.0.1:8800', budget_ms=budget)

    def test_timeout_disables_tools_without_retry_and_marks_unverified(self):
        class Engine:
            calls = 0
            plain = 0
            async def generate(self, tokens, budget, rid):
                self.calls += 1
                text = '<calc>6*7</calc>'
                return Segment(text, tuple(text.encode()), len(text), 'stop')
            async def generate_without_tools(self, tokens, budget, rid):
                self.plain += 1
                self.assertion = bytes(tokens).decode()
                return Segment('42', (52, 50), 2, 'stop')
        engine = Engine()
        with patch('exactus.adapters.llmos.request_bytes', side_effect=TimeoutError):
            result = asyncio.run(ExactusCoordinator(engine, ByteTokenizer(),
                LLMOSExternalComputer('http://127.0.0.1:8800')).run(
                    [65], max_tokens=512, require_final=False, mode='preserve-text'))
        self.assertEqual(result.text, '42')
        self.assertEqual(engine.calls, 1)
        self.assertEqual(engine.plain, 1)
        self.assertIn('Tools are disabled', engine.assertion)
        self.assertFalse(result.exact_final)
        serialized = _serialize(result, 'preserve-text')
        self.assertTrue(serialized['tools_disabled'])
        self.assertIsNone(serialized['task_verified'])
        self.assertEqual(serialized['engine_segments'], 2)

    def test_strict_does_not_silently_downgrade(self):
        class Engine:
            async def generate(self, *args):
                return Segment('<calc-final>6*7</calc-final>', (), 8, 'stop')
        with patch('exactus.adapters.llmos.request_bytes', side_effect=TimeoutError):
            with self.assertRaises(LLMOSUnavailable):
                asyncio.run(ExactusCoordinator(Engine(), ByteTokenizer(),
                    LLMOSExternalComputer('http://127.0.0.1:8800')).run([65], max_tokens=512, mode='strict'))

    def test_native_fallback_has_no_compute_stop(self):
        class Client:
            async def generate(self, prompt, params, rid):
                self.params = params
                yield NS(finished=True, outputs=[NS(text='42', token_ids=[1], finish_reason='stop')])
        client = Client()
        asyncio.run(NativeEngine(client, sampling_factory=lambda **kw: kw).generate_without_tools([1], 8, 'r'))
        self.assertEqual(client.params['stop'], [])


if __name__ == '__main__':
    unittest.main()
