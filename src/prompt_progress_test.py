"""Run with: python src/prompt_progress_test.py"""
from __future__ import annotations

import json
import os
import sys
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace
from urllib.parse import parse_qs, urlparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from prompt_progress import (  # noqa: E402
    PromptProgress,
    EDGE_OBJECT,
    format_prompt_status,
    is_cloud_host,
    llm_api_key,
    llm_base_url,
    pick_progress,
    probe_progress,
    progress_stream_url,
    progress_urls,
    reset_progress_caches,
    stream_chat,
)


SAMPLE = {
    'object': EDGE_OBJECT,
    'version': 1,
    'generated_at': 1756670123.45,
    'active': True,
    'progress': 0.3131,
    'models': [
        {
            'id': 'MiniMax-M2.7-ConfigI-MLX',
            'engine': 'lm',
            'phase': 'prefill',
            'status': 'processing',
            'stream': True,
            'progress': 0.3131,
            'prompt': {
                'processed_tokens': 2048,
                'total_tokens': 6540,
                'ratio': 0.3131,
                'cached_tokens': None,
                'started_at': 1756670120.1,
                'updated_at': 1756670122.4,
                'tokens_per_second': 820.1,
            },
            'generation': {
                'tokens': 0,
                'started_at': None,
                'updated_at': None,
                'tokens_per_second': None,
            },
            'error': None,
        }
    ],
}


class _Secret:
    def __init__(self, value: str) -> None:
        self._value = value

    def get_secret_value(self) -> str:
        return self._value

    def __str__(self) -> str:
        return '**********'


class FormatStatusTest(unittest.TestCase):
    """Status line the Spur badge splits on."""

    def test_zero_is_bare_label(self):
        self.assertEqual(format_prompt_status(0), 'Processing Prompt…')

    def test_one_decimal(self):
        self.assertEqual(format_prompt_status(0.466), 'Processing Prompt… 46.6%')

    def test_whole_percent(self):
        self.assertEqual(format_prompt_status(0.5), 'Processing Prompt… 50%')

    def test_caps_before_first_token(self):
        self.assertEqual(format_prompt_status(1.0), 'Processing Prompt… 99.9%')

    def test_readme_ratio(self):
        self.assertEqual(format_prompt_status(0.3131), 'Processing Prompt… 31.3%')


class PickProgressTest(unittest.TestCase):
    """edge.progress snapshot → PromptProgress."""

    def test_readme_snapshot(self):
        pp = pick_progress(SAMPLE, 'MiniMax-M2.7-ConfigI-MLX')
        self.assertIsNotNone(pp)
        self.assertAlmostEqual(pp.fraction, 0.3131)
        self.assertEqual(pp.processed, 2048)
        self.assertEqual(pp.total, 6540)
        self.assertEqual(pp.phase, 'prefill')

    def test_case_and_basename_match(self):
        pp = pick_progress(SAMPLE, 'minimax-m2.7-configi-mlx')
        self.assertIsNotNone(pp)
        pp = pick_progress(SAMPLE, 'mlx-community/MiniMax-M2.7-ConfigI-MLX')
        self.assertIsNotNone(pp)

    def test_other_model_ignored(self):
        self.assertIsNone(pick_progress(
            {**SAMPLE, 'progress': None}, 'qwen3-8b',
        ))

    def test_models_progress_float_without_ratio(self):
        snap = {
            'object': EDGE_OBJECT,
            'active': True,
            'progress': 0.466,
            'models': [{
                'id': 'MiniMax-M2.7-ConfigI-MLX',
                'phase': 'prefill',
                'status': 'processing',
                'progress': 0.466,
                'prompt': {'ratio': None},
            }],
        }
        pp = pick_progress(snap, 'MiniMax-M2.7-ConfigI-MLX')
        self.assertAlmostEqual(pp.fraction, 0.466)

    def test_top_level_progress(self):
        snap = {
            'object': EDGE_OBJECT,
            'active': True,
            'progress': 0.25,
            'models': [],
        }
        pp = pick_progress(snap)
        self.assertAlmostEqual(pp.fraction, 0.25)
        self.assertEqual(pp.phase, 'prefill')

    def test_ratio_from_counts_when_missing(self):
        snap = {
            'object': EDGE_OBJECT,
            'models': [{
                'id': 'm',
                'phase': 'prefill',
                'status': 'processing',
                'prompt': {'processed_tokens': 10, 'total_tokens': 40},
            }],
        }
        pp = pick_progress(snap, 'm')
        self.assertAlmostEqual(pp.fraction, 0.25)

    def test_rejects_lm_studio_fake_200(self):
        self.assertIsNone(pick_progress({
            'error': 'Unexpected endpoint or method. Returning 200 anyway',
        }))

    def test_rejects_unrelated_json(self):
        self.assertIsNone(pick_progress({'object': 'list', 'data': []}))
        self.assertIsNone(pick_progress(None))

    def test_idle_has_no_ratio(self):
        snap = {
            'object': EDGE_OBJECT,
            'active': False,
            'progress': 0.0,
            'models': [{
                'id': 'm',
                'phase': 'idle',
                'status': 'ready',
                'progress': 0.0,
                'prompt': {
                    'processed_tokens': 0,
                    'total_tokens': None,
                    'ratio': 0.0,
                },
            }],
        }
        self.assertIsNone(pick_progress(snap, 'm'))


class UrlHelpersTest(unittest.TestCase):
    """Host classification and ChatOpenAI attribute unwrapping."""

    def test_progress_urls_strip_v1(self):
        self.assertEqual(
            progress_urls('http://127.0.0.1:8080/v1'),
            ['http://127.0.0.1:8080/v1/progress'],
        )
        self.assertEqual(
            progress_urls('http://llm:1234'),
            ['http://llm:1234/v1/progress', 'http://llm:1234/progress'],
        )

    def test_stream_url(self):
        self.assertEqual(
            progress_stream_url('http://127.0.0.1:8080/v1/progress'),
            'http://127.0.0.1:8080/v1/progress/stream',
        )

    def test_cloud_hosts_skipped(self):
        self.assertTrue(is_cloud_host('https://api.openai.com/v1'))
        self.assertTrue(is_cloud_host('https://api.x.ai/v1'))
        self.assertFalse(is_cloud_host('http://127.0.0.1:8080/v1'))
        self.assertFalse(is_cloud_host('http://llm:1234/v1'))

    def test_secret_str_unwrap(self):
        llm = SimpleNamespace(
            openai_api_base=_Secret('http://127.0.0.1:8080/v1'),
            openai_api_key=_Secret('sk-test'),
            model_name='MiniMax-M2.7-ConfigI-MLX',
        )
        self.assertEqual(llm_base_url(llm), 'http://127.0.0.1:8080/v1')
        self.assertEqual(llm_api_key(llm), 'sk-test')


def _edge_body(ratio: float, ident: str) -> dict:
    return {
        'object': EDGE_OBJECT,
        'active': True,
        'progress': ratio,
        'models': [{
            'id': ident,
            'phase': 'prefill',
            'status': 'processing',
            'progress': ratio,
            'prompt': {
                'processed_tokens': int(ratio * 6540),
                'total_tokens': 6540,
                'ratio': ratio,
            },
        }],
    }


class _ProgressServer(ThreadingHTTPServer):
    allow_reuse_address = True

    def __init__(self, kind: str, *args, **kwargs) -> None:
        self.kind = kind
        self.hits = 0
        self.lock = threading.Lock()
        super().__init__(*args, **kwargs)


class _Handler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        server: _ProgressServer = self.server  # type: ignore[assignment]
        with server.lock:
            server.hits += 1
            n = server.hits
        path = parsed.path.rstrip('/') or '/'
        needle = (parse_qs(parsed.query).get('model') or [''])[0]
        ident = needle or 'MiniMax-M2.7-ConfigI-MLX'
        if path in {'/v1/progress/stream', '/progress/stream'}:
            self._sse(server, ident)
            return
        if path not in {'/v1/progress', '/progress'}:
            self.send_response(404)
            self.end_headers()
            return
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.end_headers()
        if server.kind == 'lmstudio':
            self.wfile.write(b'{"error":"Unexpected endpoint or method. Returning 200 anyway"}')
            self.wfile.flush()
            return
        if server.kind == 'idle':
            body = {
                'object': EDGE_OBJECT,
                'active': False,
                'progress': 0.0,
                'models': [{'id': ident, 'phase': 'idle',
                            'status': 'ready', 'progress': 0.0,
                            'prompt': {'ratio': 0.0}}],
            }
            self.wfile.write(json.dumps(body).encode())
            self.wfile.flush()
            return
        if server.kind == 'poll-only':
            ratio = min(0.9, 0.2 * n)
            self.wfile.write(json.dumps(_edge_body(ratio, ident)).encode())
            self.wfile.flush()
            return
        ratio = min(0.9, 0.2 * n)
        self.wfile.write(json.dumps(_edge_body(ratio, ident)).encode())
        self.wfile.flush()

    def _sse(self, server: _ProgressServer, ident: str) -> None:
        if server.kind in {'lmstudio', 'poll-only'}:
            self.send_response(404)
            self.end_headers()
            return
        self.send_response(200)
        self.send_header('Content-Type', 'text/event-stream')
        self.send_header('Cache-Control', 'no-cache')
        self.send_header('Connection', 'close')
        self.end_headers()
        try:
            n = 0
            while True:
                n += 1
                ratio = min(0.9, 0.2 * n)
                blob = json.dumps(_edge_body(ratio, ident))
                self.wfile.write(f'data: {blob}\n\n'.encode())
                self.wfile.flush()
                time.sleep(0.07)
        except (BrokenPipeError, ConnectionResetError, TimeoutError, OSError):
            return

    def log_message(self, *_args) -> None:
        return


def _serve(kind: str) -> tuple[_ProgressServer, str]:
    server = _ProgressServer(kind, ('127.0.0.1', 0), _Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address[:2]
    return server, f'http://{host}:{port}/v1'


def _stop(server: _ProgressServer) -> None:
    server.shutdown()
    server.server_close()


class _FakeLLM:
    def __init__(self, base_url: str, delay: float = 0.35, chunks: list[str] | None = None):
        self.openai_api_base = base_url
        self.model_name = 'MiniMax-M2.7-ConfigI-MLX'
        self.api_key = 'none'
        self.delay = delay
        self.chunks = chunks if chunks is not None else ['hello']

    def stream(self, _messages):
        time.sleep(self.delay)
        for text in self.chunks:
            yield SimpleNamespace(content=text)


class StreamChatTest(unittest.TestCase):
    """Sideband EventSource while llm.stream() is blocked on the first token."""

    def setUp(self) -> None:
        reset_progress_caches()

    def tearDown(self) -> None:
        reset_progress_caches()

    def test_yields_progress_from_sse(self):
        server, base = _serve('edge')
        try:
            llm = _FakeLLM(base, delay=0.35)
            pieces = list(stream_chat(llm, [], interval=0.05, timeout=0.3))
        finally:
            _stop(server)
        progress = [p for p in pieces if isinstance(p, PromptProgress)]
        chunks = [p for p in pieces if not isinstance(p, PromptProgress)]
        self.assertGreaterEqual(len(progress), 1)
        self.assertGreater(progress[-1].fraction, 0)
        self.assertEqual([c.content for c in chunks], ['hello'])

    def test_falls_back_to_poll_when_stream_missing(self):
        server, base = _serve('poll-only')
        try:
            llm = _FakeLLM(base, delay=0.4)
            pieces = list(stream_chat(llm, [], interval=0.05, timeout=0.3))
        finally:
            _stop(server)
        progress = [p for p in pieces if isinstance(p, PromptProgress)]
        chunks = [p for p in pieces if not isinstance(p, PromptProgress)]
        self.assertGreaterEqual(len(progress), 1)
        self.assertEqual([c.content for c in chunks], ['hello'])

    def test_lm_studio_fake_200_is_a_miss(self):
        server, base = _serve('lmstudio')
        try:
            self.assertIsNone(probe_progress(base, timeout=0.3))
            hits = server.hits
            self.assertGreaterEqual(hits, 1)
            self.assertIsNone(probe_progress(base, timeout=0.3))
            self.assertEqual(server.hits, hits)
            llm = _FakeLLM(base, delay=0.0)
            pieces = list(stream_chat(llm, [], interval=0.05, timeout=0.2))
        finally:
            _stop(server)
        self.assertEqual([c.content for c in pieces], ['hello'])
        self.assertFalse(any(isinstance(p, PromptProgress) for p in pieces))
        self.assertEqual(server.hits, hits)

    def test_cloud_never_polls(self):
        llm = _FakeLLM('https://api.openai.com/v1', delay=0.0)
        pieces = list(stream_chat(llm, []))
        self.assertEqual([c.content for c in pieces], ['hello'])
        self.assertFalse(any(isinstance(p, PromptProgress) for p in pieces))


if __name__ == '__main__':
    unittest.main()
