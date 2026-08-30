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
    _is_cloud,
    _origin,
    format_prompt_status,
    parse_progress,
    parse_progress_text,
    parse_slots_progress,
    reset_probe_cache,
    stream_chat,
)


class _Msg:
    def __init__(self, type_, content):
        self.type = type_
        self.content = content


class _LLM:
    def __init__(self, base, reply='fallback', hold=0.0, key='none'):
        self.openai_api_base = base
        self.openai_api_key = key
        self.model_name = 'minimax-m2.7'
        self.reply = reply
        self.hold = hold
        self.streamed = 0

    def stream(self, messages):
        self.streamed += 1
        del messages
        if self.hold:
            time.sleep(self.hold)
        yield TokenChunk(content=self.reply)


class ParseProgressTest(unittest.TestCase):
    """LM Studio log line, native event, llama.cpp prompt_progress."""

    def test_developer_log_line(self):
        line = '[minimax-m2.7] Prompt processing progress: 46.6%'
        self.assertAlmostEqual(parse_progress_text(line), 0.466, places=4)
        self.assertEqual(format_prompt_status(0.466), 'Processing Prompt… 46.6%')

    def test_timestamped_log_blob(self):
        blob = (
            '2026-08-30 14:21:09  [INFO]\n'
            ' [minimax-m2.7] Prompt processing progress: 46.6%'
        )
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
            'prompt_progress': {
                'total': 1000, 'cache': 100, 'processed': 466, 'time_ms': 12,
            },
        })
        self.assertAlmostEqual(frac, 0.466, places=4)

    def test_slots_list(self):
        frac = parse_slots_progress([
            {'prompt_progress': {'total': 200, 'processed': 50}},
            {'prompt_progress': {'total': 200, 'processed': 80}},
        ])
        self.assertEqual(frac, 0.4)

    def test_status_hides_zero(self):
        self.assertEqual(format_prompt_status(0), 'Processing Prompt…')
        self.assertEqual(PromptProgress(0.994).pct, 99)

    def test_origin_from_llm_hostname(self):
        self.assertEqual(_origin('http://llm:1234/v1'), 'http://llm:1234')
        self.assertEqual(
            _origin('http://llm:1234/v1/chat/completions'),
            'http://llm:1234',
        )
        self.assertFalse(_is_cloud('http://llm:1234/v1/chat/completions'))
        self.assertTrue(_is_cloud('https://api.openai.com/v1'))


class _Quiet(BaseHTTPRequestHandler):
    """Stub HTTP handler. Verb names are BaseHTTPRequestHandler's contract."""

    def log_message(self, fmt, *args):  # pylint: disable=arguments-differ
        del fmt, args


class StreamChatTest(unittest.TestCase):
    """Probe GET /slots; only then poll while llm.stream() runs."""

    # Nested BaseHTTPRequestHandler verbs must be do_GET / do_POST.
    # pylint: disable=invalid-name,missing-class-docstring,missing-function-docstring

    def setUp(self):
        reset_probe_cache()

    def _serve(self, handler):
        httpd = ThreadingHTTPServer(('127.0.0.1', 0), handler)
        thread = threading.Thread(target=httpd.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(httpd.shutdown)
        self.addCleanup(httpd.server_close)
        return httpd.server_address[1]

    def test_no_slots_leaves_langchain_alone(self):
        class Handler(_Quiet):
            def do_GET(self):
                self.send_response(404)
                self.end_headers()

        port = self._serve(Handler)
        llm = _LLM(f'http://127.0.0.1:{port}/v1', reply='plain')
        items = list(stream_chat(llm, [_Msg('human', 'hi')]))
        self.assertEqual(llm.streamed, 1)
        self.assertFalse(any(isinstance(x, PromptProgress) for x in items))
        self.assertEqual(''.join(getattr(x, 'content', '') for x in items), 'plain')

    def test_cloud_never_probes(self):
        llm = _LLM('https://api.openai.com/v1', reply='cloud')
        items = list(stream_chat(llm, [_Msg('human', 'hi')]))
        self.assertEqual(llm.streamed, 1)
        self.assertEqual(items[0].content, 'cloud')

    def test_slots_while_waiting_for_token(self):
        busy = threading.Event()

        class Handler(_Quiet):
            def do_GET(self):
                if not self.path.endswith('/slots'):
                    self.send_response(404)
                    self.end_headers()
                    return
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                if busy.is_set():
                    payload = [{'prompt_progress': {'total': 100, 'processed': 40}}]
                else:
                    payload = []
                self.wfile.write(json.dumps(payload).encode())

        port = self._serve(Handler)
        llm = _LLM(f'http://127.0.0.1:{port}/v1', reply='ok', hold=0.55)

        def delayed(messages):
            busy.set()
            time.sleep(0.55)
            yield TokenChunk(content='ok')
            del messages

        llm.stream = delayed  # type: ignore[method-assign]
        items = list(stream_chat(llm, [_Msg('human', 'hi')]))
        progress = [x for x in items if isinstance(x, PromptProgress)]
        self.assertTrue(progress)
        self.assertGreaterEqual(progress[0].fraction, 0.4)
        tokens = [x for x in items if isinstance(x, TokenChunk)]
        self.assertEqual(''.join(x.content for x in tokens), 'ok')

    def test_llm_hostname_probes_origin_not_chat_path(self):
        seen = []

        class Handler(_Quiet):
            def do_GET(self):
                seen.append(self.path)
                self.send_response(404)
                self.end_headers()

        port = self._serve(Handler)
        llm = _LLM(f'http://127.0.0.1:{port}/v1')
        list(stream_chat(llm, [_Msg('human', 'hi')]))
        self.assertTrue(any(path in {'/slots', '/v1/slots'} for path in seen))
        self.assertFalse(any('chat/completions' in path for path in seen))


if __name__ == '__main__':
    unittest.main()
