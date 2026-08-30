"""Run with: python src/prompt_progress_test.py"""
from __future__ import annotations

import json
import os
import socket
import struct
import sys
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from prompt_progress import (  # noqa: E402  # pylint: disable=wrong-import-position,protected-access
    PromptProgress,
    TokenChunk,
    _API_CACHE,
    _is_cloud,
    _line_json,
    _origin,
    _payload_for,
    delta_text,
    find_api_host,
    format_prompt_status,
    is_lmstudio_catchall,
    looks_like_slots,
    parse_progress,
    parse_progress_text,
    parse_slots_progress,
    progress_from_obj,
    reset_progress_caches,
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

    def test_lmstudio_catchall_is_not_slots(self):
        payload = {
            'error': 'Unexpected endpoint or method. (GET /slots). Returning 200 anyway',
        }
        self.assertTrue(is_lmstudio_catchall(payload))
        self.assertFalse(looks_like_slots(payload))
        self.assertIsNone(parse_slots_progress(payload))
        self.assertFalse(looks_like_slots({}))
        self.assertFalse(looks_like_slots([]))
        self.assertFalse(looks_like_slots({'slots': []}))

    def test_diagnostics_envelope(self):
        blob = {
            'type': 'channelMessage',
            'channelId': 0,
            'message': {
                'type': 'log',
                'log': {
                    'timestamp': 1,
                    'data': {
                        'type': 'server.log',
                        'content': '[minimax-m2.7] Prompt processing progress: 60.9%',
                    },
                },
            },
        }
        self.assertAlmostEqual(progress_from_obj(blob), 0.609, places=4)


    def test_status_hides_zero(self):
        self.assertEqual(format_prompt_status(0), 'Processing Prompt…')
        self.assertEqual(PromptProgress(0.994).pct, 99)

    def test_delta_text(self):
        content, reason = delta_text({
            'choices': [{'delta': {'content': 'Hi', 'reasoning_content': 'plan'}}],
        })
        self.assertEqual(content, 'Hi')
        self.assertEqual(reason, 'plan')

    def test_origin_from_llm_hostname(self):
        self.assertEqual(_origin('http://llm:1234/v1'), 'http://llm:1234')
        self.assertEqual(
            _origin('http://llm:1234/v1/chat/completions'),
            'http://llm:1234',
        )
        self.assertFalse(_is_cloud('http://llm:1234/v1/chat/completions'))
        self.assertTrue(_is_cloud('https://api.openai.com/v1'))
        local = _LLM('http://llm:1234/v1')
        cloud = _LLM('https://api.openai.com/v1')
        self.assertIn('return_progress', _payload_for(local, [_Msg('human', 'hi')], True))
        self.assertNotIn('return_progress', _payload_for(cloud, [_Msg('human', 'hi')], True))


class _Quiet(BaseHTTPRequestHandler):
    """Stub HTTP handler. Verb names are BaseHTTPRequestHandler's contract."""

    def log_message(self, fmt, *args):  # pylint: disable=arguments-differ
        del fmt, args


class StreamChatTest(unittest.TestCase):
    """Raw SSE + /slots against a stub of LM Studio / llama.cpp."""

    # Nested BaseHTTPRequestHandler verbs must be do_GET / do_POST.
    # pylint: disable=invalid-name,missing-class-docstring,missing-function-docstring

    def setUp(self):
        reset_progress_caches()

    def _serve(self, handler):
        httpd = ThreadingHTTPServer(('127.0.0.1', 0), handler)
        thread = threading.Thread(target=httpd.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(httpd.shutdown)
        self.addCleanup(httpd.server_close)
        return httpd.server_address[1]

    def test_native_progress_without_slots(self):
        """LM Studio has no /slots — still show % if the SSE carries it."""

        class Handler(_Quiet):
            def do_POST(self):
                self.rfile.read(int(self.headers.get('Content-Length', '0')))
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
        self.assertEqual(llm.streamed, 0)
        progress = [x for x in items if isinstance(x, PromptProgress)]
        self.assertTrue(progress)
        self.assertAlmostEqual(progress[0].fraction, 0.466, places=4)
        tokens = [x for x in items if isinstance(x, TokenChunk)]
        self.assertEqual(''.join(x.content for x in tokens), 'Hi')

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

    def test_cloud_uses_langchain(self):
        llm = _LLM('https://api.openai.com/v1')
        items = list(stream_chat(llm, [_Msg('human', 'hi')]))
        self.assertEqual(llm.streamed, 1)
        self.assertEqual(items[0].content, 'fallback')

    def test_catchall_200_does_not_keep_polling(self):
        """LM Studio 200-anyway on GET /slots is not llama.cpp — poll once."""
        gets = []

        class Handler(_Quiet):
            def do_POST(self):
                self.rfile.read(int(self.headers.get('Content-Length', '0')))
                self.send_response(200)
                self.send_header('Content-Type', 'text/event-stream')
                self.end_headers()
                time.sleep(0.7)
                self.wfile.write(
                    b'data: {"choices":[{"delta":{"content":"ok"}}]}\n\n'
                    b'data: [DONE]\n\n'
                )

            def do_GET(self):
                gets.append(self.path)
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                body = (
                    '{"error":"Unexpected endpoint or method. '
                    '(GET /slots). Returning 200 anyway"}'
                )
                self.wfile.write(body.encode())

        port = self._serve(Handler)
        llm = _LLM(f'http://127.0.0.1:{port}/v1')
        items = list(stream_chat(llm, [_Msg('human', 'hi')]))
        self.assertEqual(
            ''.join(x.content for x in items if isinstance(x, TokenChunk)),
            'ok',
        )
        slot_hits = [p for p in gets if p.endswith('/slots')]
        self.assertEqual(len(slot_hits), 1, gets)
        # Second turn must not touch /slots again (cached miss).
        gets.clear()
        list(stream_chat(llm, [_Msg('human', 'hi')]))
        self.assertFalse([p for p in gets if p.endswith('/slots')], gets)

    def test_greeting_is_not_catchall(self):
        class Handler(_Quiet):
            def do_GET(self):
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                if self.path.endswith('/lmstudio-greeting'):
                    self.wfile.write(b'{"lmstudio":true}')
                    return
                self.wfile.write(b'{"error":"Unexpected endpoint"}')

            def do_POST(self):
                self.send_response(404)
                self.end_headers()

        port = self._serve(Handler)
        origin = f'http://127.0.0.1:{port}'
        self.assertEqual(find_api_host(origin, {}), f'127.0.0.1:{port}')
        self.assertFalse(is_lmstudio_catchall({'lmstudio': True}))

    def test_diagnostics_websocket_progress(self):
        """LM Studio diagnostics.streamLogs carries the Developer-log %."""
        ws_port = _start_diagnostics_stub()

        class Handler(_Quiet):
            def do_POST(self):
                self.rfile.read(int(self.headers.get('Content-Length', '0')))
                self.send_response(200)
                self.send_header('Content-Type', 'text/event-stream')
                self.end_headers()
                time.sleep(0.8)
                self.wfile.write(
                    b'data: {"choices":[{"delta":{"content":"Hi"}}]}\n\n'
                    b'data: [DONE]\n\n'
                )

            def do_GET(self):
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                self.wfile.write(
                    b'{"error":"Unexpected endpoint or method. Returning 200 anyway"}'
                )

        http_port = self._serve(Handler)
        origin = f'http://127.0.0.1:{http_port}'
        _API_CACHE[origin] = f'127.0.0.1:{ws_port}'
        llm = _LLM(f'http://127.0.0.1:{http_port}/v1')
        items = list(stream_chat(llm, [_Msg('human', 'hi')]))
        progress = [x for x in items if isinstance(x, PromptProgress)]
        self.assertTrue(progress, items)
        self.assertAlmostEqual(progress[0].fraction, 0.609, places=3)
        tokens = [x for x in items if isinstance(x, TokenChunk)]
        self.assertEqual(''.join(x.content for x in tokens), 'Hi')


def _start_diagnostics_stub() -> int:
    """One-shot websocket that speaks LM Studio diagnostics.streamLogs."""
    ready = threading.Event()
    port_box: list[int] = []

    def server() -> None:
        listener = socket.socket()
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind(('127.0.0.1', 0))
        listener.listen(1)
        listener.settimeout(3)
        port_box.append(listener.getsockname()[1])
        ready.set()
        try:
            conn, _ = listener.accept()
        except socket.timeout:
            listener.close()
            return
        conn.settimeout(3)
        buf = b''
        while b'\r\n\r\n' not in buf:
            buf += conn.recv(1024)
        conn.sendall(
            b'HTTP/1.1 101 Switching Protocols\r\n'
            b'Upgrade: websocket\r\n'
            b'Connection: Upgrade\r\n'
            b'\r\n'
        )
        _recv_ws_frame(conn)
        _send_ws_text(conn, '{"success":true}')
        _recv_ws_frame(conn)
        event = json.dumps({
            'type': 'channelMessage',
            'channelId': 0,
            'message': {
                'type': 'log',
                'log': {
                    'data': {
                        'type': 'server.log',
                        'content': (
                            '[minimax-m2.7] Prompt processing progress: 60.9%'
                        ),
                    },
                },
            },
        })
        _send_ws_text(conn, event)
        time.sleep(0.4)
        try:
            conn.close()
        except OSError:
            pass
        listener.close()

    threading.Thread(target=server, daemon=True).start()
    if not ready.wait(2):
        raise RuntimeError('diagnostics stub did not bind')
    return port_box[0]


def _send_ws_text(conn, text: str) -> None:
    payload = text.encode('utf-8')
    header = bytearray([0x81])
    length = len(payload)
    if length < 126:
        header.append(length)
    else:
        header.append(126)
        header.extend(struct.pack('!H', length))
    conn.sendall(bytes(header) + payload)


def _recv_ws_frame(conn) -> bytes:
    hdr = conn.recv(2)
    length = hdr[1] & 0x7F
    masked = bool(hdr[1] & 0x80)
    if length == 126:
        length = struct.unpack('!H', conn.recv(2))[0]
    mask = conn.recv(4) if masked else b''
    payload = conn.recv(length) if length else b''
    if masked and payload:
        payload = bytes(b ^ mask[i % 4] for i, b in enumerate(payload))
    return payload



if __name__ == '__main__':
    unittest.main()
