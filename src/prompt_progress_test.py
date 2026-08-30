"""Run with: python src/prompt_progress_test.py"""
from __future__ import annotations

import json
import os
import sys
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from prompt_progress import (  # noqa: E402  # pylint: disable=wrong-import-position
    PromptProgress,
    TokenChunk,
    _is_local_host,
    _line_json,
    _payload_for,
    delta_text,
    format_prompt_status,
    parse_progress,
    parse_progress_text,
    parse_slots_progress,
    stream_chat,
)


class _Msg:
    def __init__(self, type_, content):
        self.type = type_
        self.content = content


class _LLM:
    def __init__(self, base, extra=None, key='none'):
        self.openai_api_base = base
        self.openai_api_key = key
        self.model_name = 'minimax-m2.7'
        self.extra_body = extra or {}
        self.streamed = 0

    def stream(self, messages):
        self.streamed += 1
        del messages
        yield TokenChunk(content='fallback')


class ParseProgressTest(unittest.TestCase):
    """LM Studio log line, native event, llama.cpp prompt_progress."""

    def test_developer_log_line(self):
        line = '[minimax-m2.7] Prompt processing progress: 46.6%'
        self.assertAlmostEqual(parse_progress_text(line), 0.466, places=4)
        self.assertEqual(format_prompt_status(0.466), 'Processing Prompt… 46.6%')

    def test_timestamped_log_blob(self):
        blob = '2026-08-30 14:21:09  [INFO]\n [minimax-m2.7] Prompt processing progress: 46.6%'
        self.assertAlmostEqual(parse_progress_text(blob), 0.466, places=4)

    def test_native_event(self):
        frac = parse_progress({'type': 'prompt_processing.progress', 'progress': 0.5})
        self.assertEqual(frac, 0.5)
        self.assertEqual(format_prompt_status(0.5), 'Processing Prompt… 50%')

    def test_progress_as_percent_number(self):
        frac = parse_progress({'type': 'prompt_processing.progress', 'progress': 46.6})
        self.assertAlmostEqual(frac, 0.466, places=4)

    def test_llama_prompt_progress(self):
        frac = parse_progress({
            'prompt_progress': {'total': 1000, 'cache': 100, 'processed': 466, 'time_ms': 12},
        })
        self.assertAlmostEqual(frac, 0.466, places=4)

    def test_sse_comment_line(self):
        obj = _line_json(': Prompt processing progress: 12.5%')
        self.assertEqual(obj['type'], 'prompt_processing.progress')
        self.assertAlmostEqual(obj['progress'], 0.125, places=4)

    def test_slots_list(self):
        frac = parse_slots_progress([
            {'prompt_progress': {'total': 200, 'processed': 50}},
            {'prompt_progress': {'total': 200, 'processed': 80}},
        ])
        self.assertEqual(frac, 0.4)

    def test_status_hides_zero(self):
        self.assertEqual(format_prompt_status(0), 'Processing Prompt…')
        self.assertEqual(PromptProgress(0.994).pct, 99)

    def test_delta_text(self):
        content, reason = delta_text({
            'choices': [{'delta': {'content': 'Hi', 'reasoning_content': 'plan'}}],
        })
        self.assertEqual(content, 'Hi')
        self.assertEqual(reason, 'plan')

    def test_local_host_only_gets_return_progress(self):
        local = _LLM('http://localhost:1234/v1')
        cloud = _LLM('https://api.openai.com/v1')
        self.assertTrue(_is_local_host('http://127.0.0.1:1234/v1/chat/completions'))
        self.assertIn('return_progress', _payload_for(local, [_Msg('human', 'hi')], True))
        self.assertNotIn('return_progress', _payload_for(cloud, [_Msg('human', 'hi')], True))


class _Quiet(BaseHTTPRequestHandler):
    """Stub HTTP handler. Verb names are BaseHTTPRequestHandler's contract."""

    def log_message(self, fmt, *args):  # pylint: disable=arguments-differ
        del fmt, args


class StreamChatTest(unittest.TestCase):
    """Raw SSE + /slots against a local stub of LM Studio / llama.cpp."""

    # Nested BaseHTTPRequestHandler verbs must be do_GET / do_POST.
    # pylint: disable=invalid-name,missing-class-docstring,missing-function-docstring

    def _serve(self, handler):
        httpd = ThreadingHTTPServer(('127.0.0.1', 0), handler)
        thread = threading.Thread(target=httpd.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(httpd.shutdown)
        self.addCleanup(httpd.server_close)
        return httpd.server_address[1]

    def test_native_progress_then_token(self):
        class Handler(_Quiet):
            def do_POST(self):
                raw = self.rfile.read(int(self.headers.get('Content-Length', '0')))
                json.loads(raw.decode())
                self.send_response(200)
                self.send_header('Content-Type', 'text/event-stream')
                self.end_headers()
                self.wfile.write(
                    b'data: {"type":"prompt_processing.progress","progress":0.466}\n\n'
                )
                self.wfile.flush()
                self.wfile.write(
                    b'data: {"choices":[{"delta":{"content":"Hi"}}]}\n\n'
                    b'data: [DONE]\n\n'
                )

            def do_GET(self):
                self.send_response(404)
                self.end_headers()

        port = self._serve(Handler)
        llm = _LLM(f'http://127.0.0.1:{port}/v1')
        items = list(stream_chat(llm, [_Msg('human', 'hi')]))
        self.assertTrue(any(isinstance(x, PromptProgress) for x in items))
        progress = next(x for x in items if isinstance(x, PromptProgress))
        self.assertAlmostEqual(progress.fraction, 0.466, places=4)
        tokens = [x for x in items if isinstance(x, TokenChunk)]
        self.assertEqual(''.join(x.content for x in tokens), 'Hi')
        self.assertTrue(llm.openai_api_base.endswith('/v1'))

    def test_slots_while_sse_silent(self):
        started = threading.Event()

        class Handler(_Quiet):
            def do_POST(self):
                self.rfile.read(int(self.headers.get('Content-Length', '0')))
                self.send_response(200)
                self.send_header('Content-Type', 'text/event-stream')
                self.end_headers()
                started.set()
                time.sleep(0.7)
                self.wfile.write(
                    b'data: {"choices":[{"delta":{"content":"ok"}}]}\n\n'
                    b'data: [DONE]\n\n'
                )

            def do_GET(self):
                if self.path.endswith('/slots'):
                    self.send_response(200)
                    self.send_header('Content-Type', 'application/json')
                    self.end_headers()
                    payload = [{'prompt_progress': {'total': 100, 'processed': 40}}]
                    self.wfile.write(json.dumps(payload).encode())
                    return
                self.send_response(404)
                self.end_headers()

        port = self._serve(Handler)
        llm = _LLM(f'http://127.0.0.1:{port}/v1')
        items = list(stream_chat(llm, [_Msg('human', 'hi')]))
        self.assertTrue(started.wait(2))
        progress = [x for x in items if isinstance(x, PromptProgress)]
        self.assertTrue(progress)
        self.assertGreaterEqual(progress[0].fraction, 0.4)

    def test_drops_return_progress_on_400(self):
        seen = []

        class Handler(_Quiet):
            def do_POST(self):
                raw = self.rfile.read(int(self.headers.get('Content-Length', '0')))
                body = json.loads(raw.decode())
                seen.append('return_progress' in body)
                if body.get('return_progress'):
                    self.send_response(400)
                    self.end_headers()
                    self.wfile.write(b'{"error":"unknown field"}')
                    return
                self.send_response(200)
                self.send_header('Content-Type', 'text/event-stream')
                self.end_headers()
                self.wfile.write(
                    b'data: {"choices":[{"delta":{"content":"ok"}}]}\n\n'
                    b'data: [DONE]\n\n'
                )

            def do_GET(self):
                self.send_response(404)
                self.end_headers()

        port = self._serve(Handler)
        llm = _LLM(f'http://127.0.0.1:{port}/v1')
        items = list(stream_chat(llm, [_Msg('human', 'hi')]))
        self.assertEqual(seen, [True, False])
        self.assertEqual(
            ''.join(x.content for x in items if isinstance(x, TokenChunk)),
            'ok',
        )
        self.assertEqual(llm.streamed, 0)


if __name__ == '__main__':
    unittest.main()
