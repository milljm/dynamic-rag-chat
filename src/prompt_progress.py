"""Surface prompt-processing % during the wait-for-first-token.

LM Studio's Developer log (``Prompt processing progress: 46.6%``) is the
native ``prompt_processing.progress`` event. That event is documented on
``POST /api/v1/chat``, which cannot take assistant history, so we do not
switch generation to it.

We stay on OpenAI-compat ``/v1/chat/completions`` (same messages ChatOpenAI
would send) and:

1. Stream it ourselves so we can see llama.cpp ``prompt_progress`` / LM
   Studio ``prompt_processing.progress`` / the log line if they leak
   (``return_progress: true``, never on cloud OpenAI).
2. Poll ``GET /slots`` only when the body is real llama.cpp slots. LM Studio
   answers unknown paths with HTTP 200 + an error JSON — that is not slots,
   and hammering it just spams the Developer log.
3. If slots is absent, subscribe to LM Studio's diagnostics websocket
   (``/lmstudio-greeting`` on the REST port or the SDK API ports) and parse
   the same Developer-log line ``lms log stream --source server`` would show.
4. Fall back to LangChain ``llm.stream()`` if the raw POST fails.
"""
from __future__ import annotations

import base64
import json
import os
import queue
import re
import socket
import struct
import threading
import urllib.error
import urllib.request
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Iterator
from urllib.parse import urlparse


_PROGRESS_LOG = re.compile(
    r'Prompt processing progress:\s*([0-9]*\.?[0-9]+)\s*%',
    re.I,
)
_CATCHALL = re.compile(
    r'unexpected endpoint|returning 200 anyway',
    re.I,
)
_CLOUD = (
    'api.openai.com',
    'api.anthropic.com',
    'generativelanguage.googleapis.com',
    'api.x.ai',
    'openai.azure.com',
    'api.groq.com',
    'api.mistral.ai',
    'openrouter.ai',
)
# llama.cpp slot objects carry at least one of these; LM Studio's fake 200 does not.
_SLOT_MARKERS = frozenset({
    'n_ctx', 'n_predict', 'id_task', 'is_processing',
    'prompt_progress', 'next_token', 'slot_id', 'task_id',
})
# LM Studio SDK / daemon API (not the OpenAI REST port).
_API_PORTS = (41343, 52993, 16141, 39414, 22931)

# origin -> True (llama.cpp slots), False (definitely not). Missing = unknown.
_SLOTS_CACHE: dict[str, bool] = {}
# origin -> "host:port" of a real LM Studio API server, or None after a miss.
_API_CACHE: dict[str, str | None] = {}
_CACHE_LOCK = threading.Lock()


@dataclass
class PromptProgress:
    """Fraction of the prompt the backend has ingested (0..1)."""

    fraction: float

    @property
    def pct(self) -> int:
        """Whole-percent, clamped 1..99 until the first token."""
        value = int(round(max(0.0, min(1.0, self.fraction)) * 100))
        return max(0, min(99, value))


@dataclass
class TokenChunk:
    """Duck-types LangChain AIMessageChunk for think_tags / reveal_thinking."""

    content: str = ''
    reasoning_content: str = ''
    additional_kwargs: dict = field(default_factory=dict)


def reset_progress_caches() -> None:
    """Test helper: forget /slots and API-host probes."""
    with _CACHE_LOCK:
        _SLOTS_CACHE.clear()
        _API_CACHE.clear()


def format_prompt_status(fraction: float) -> str:
    """``Processing Prompt… 46.6%`` — Spur puts model/route/context after."""
    frac = _clamp(float(fraction))
    if frac <= 0:
        return 'Processing Prompt…'
    shown = min(99.9, frac * 100.0)
    if abs(shown - round(shown)) < 0.05:
        whole = int(round(shown))
        if whole <= 0:
            return 'Processing Prompt…'
        return f'Processing Prompt… {min(99, whole)}%'
    return f'Processing Prompt… {shown:.1f}%'


def parse_progress_text(text: str) -> float | None:
    """Parse LM Studio's Developer-log line into 0..1."""
    if not text:
        return None
    match = _PROGRESS_LOG.search(text)
    if not match:
        return None
    return _clamp(float(match.group(1)) / 100.0)


def parse_progress(obj: Any) -> float | None:
    """Extract 0..1 from an SSE JSON object, or None."""
    if isinstance(obj, str):
        return parse_progress_text(obj)
    if not isinstance(obj, dict):
        return None
    kind = str(obj.get('type') or '')
    if kind == 'prompt_processing.progress':
        return _as_fraction(obj.get('progress'))
    blob = obj.get('prompt_progress')
    if isinstance(blob, dict):
        total = float(blob.get('total') or 0)
        processed = float(blob.get('processed') or 0)
        if total > 0:
            return _clamp(processed / total)
    if not obj.get('choices') and obj.get('total') and obj.get('processed') is not None:
        total = float(obj.get('total') or 0)
        if total > 0:
            return _clamp(float(obj.get('processed') or 0) / total)
    for key in ('message', 'text', 'log', 'content'):
        value = obj.get(key)
        if isinstance(value, str):
            frac = parse_progress_text(value)
            if frac is not None:
                return frac
    return None


def progress_from_obj(obj: Any) -> float | None:
    """Walk nested JSON (diagnostics envelopes) for a progress fraction."""
    direct = parse_progress(obj)
    if direct is not None:
        return direct
    if isinstance(obj, dict):
        for value in obj.values():
            frac = progress_from_obj(value)
            if frac is not None:
                return frac
    elif isinstance(obj, list):
        for value in obj:
            frac = progress_from_obj(value)
            if frac is not None:
                return frac
    return None


def is_lmstudio_catchall(payload: Any) -> bool:
    """LM Studio's 'Unexpected endpoint… Returning 200 anyway' body."""
    if not isinstance(payload, dict):
        return False
    err = payload.get('error')
    parts = [str(payload.get('message') or '')]
    if isinstance(err, str):
        parts.append(err)
    elif isinstance(err, dict):
        parts.append(str(err.get('message') or err.get('error') or ''))
    return bool(_CATCHALL.search(' '.join(parts)))


def _slot_rows(payload: Any) -> list:
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        inner = payload.get('slots') or payload.get('data')
        if isinstance(inner, list):
            return inner
        return [payload]
    return []


def looks_like_slots(payload: Any) -> bool:
    """True only for llama.cpp GET /slots, never LM Studio's fake 200."""
    if payload is None or is_lmstudio_catchall(payload):
        return False
    for row in _slot_rows(payload):
        if not isinstance(row, dict):
            continue
        if _SLOT_MARKERS & row.keys():
            return True
    return False


def parse_slots_progress(payload: Any) -> float | None:
    """Best fraction from llama.cpp GET /slots."""
    if not looks_like_slots(payload):
        return None
    best: float | None = None
    for row in _slot_rows(payload):
        if not isinstance(row, dict):
            continue
        frac = parse_progress(row)
        if frac is None:
            continue
        best = frac if best is None else max(best, frac)
    return best


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, value))


def _as_fraction(value: Any) -> float | None:
    """Accept 0..1 or 0..100 (LM Studio log uses 46.6)."""
    if isinstance(value, str):
        try:
            value = float(value)
        except ValueError:
            return None
    if not isinstance(value, (int, float)):
        return None
    number = float(value)
    if number > 1.0:
        number = number / 100.0
    return _clamp(number)


def _is_cloud(url: str) -> bool:
    text = (url or '').lower()
    return any(host in text for host in _CLOUD)


def _origin(base_url: str) -> str:
    """http://llm:1234/v1/chat/completions → http://llm:1234."""
    text = (base_url or '').strip().rstrip('/')
    changed = True
    while changed and text:
        changed = False
        for suffix in (
            '/chat/completions', '/completions',
            '/api/v1/models', '/api/v0/models', '/v1/models', '/models', '/v1',
        ):
            if text.endswith(suffix):
                text = text[: -len(suffix)].rstrip('/')
                changed = True
                break
    return text


def _chat_url(llm: Any) -> str:
    base = (
        getattr(llm, 'openai_api_base', None)
        or getattr(llm, 'base_url', None)
        or ''
    )
    base = str(base).rstrip('/')
    if not base:
        return ''
    if base.endswith('/chat/completions'):
        return base
    return f'{base}/chat/completions'


def messages_to_openai(messages: Any) -> list[dict]:
    """LangChain messages → OpenAI dicts."""
    out: list[dict] = []
    for msg in messages or []:
        kind = getattr(msg, 'type', '') or ''
        role = 'user'
        if kind in {'system', 'ai', 'assistant', 'tool'}:
            role = {'ai': 'assistant'}.get(kind, kind)
        elif msg.__class__.__name__ in {'SystemMessage'}:
            role = 'system'
        elif msg.__class__.__name__ in {'AIMessage', 'AIMessageChunk'}:
            role = 'assistant'
        elif msg.__class__.__name__ in {'ToolMessage'}:
            role = 'tool'
        content = getattr(msg, 'content', msg)
        out.append({'role': role, 'content': content})
    return out


def _payload_for(llm: Any, messages: Any, with_progress: bool) -> dict:
    """Request body, preferring ChatOpenAI's own serializer."""
    getter = getattr(llm, '_get_request_payload', None)
    payload: dict
    if callable(getter):
        try:
            payload = dict(getter(messages))  # pylint: disable=protected-access
        except Exception:  # pylint: disable=broad-exception-caught
            payload = {}
    else:
        payload = {}
    if not payload.get('messages'):
        payload['messages'] = messages_to_openai(messages)
    payload['stream'] = True
    payload.setdefault('model', getattr(llm, 'model_name', None) or getattr(llm, 'model', ''))
    extra = getattr(llm, 'extra_body', None)
    if isinstance(extra, dict):
        for key, value in extra.items():
            payload.setdefault(key, value)
    if with_progress and not _is_cloud(_chat_url(llm)):
        payload['return_progress'] = True
    return payload


def _headers(llm: Any, accept: str = 'application/json') -> dict[str, str]:
    key = str(getattr(llm, 'openai_api_key', None) or getattr(llm, 'api_key', None) or 'none')
    return {
        'Authorization': f'Bearer {key}',
        'Content-Type': 'application/json',
        'Accept': accept,
    }


def _json_get(url: str, headers: dict, timeout: float = 0.6) -> Any | None:
    """GET JSON, or None on 404 / network / non-JSON."""
    try:
        req = urllib.request.Request(url, headers=headers, method='GET')
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode('utf-8', errors='replace')
        return json.loads(raw)
    except (urllib.error.URLError, json.JSONDecodeError, TimeoutError, OSError):
        return None
    except Exception:  # pylint: disable=broad-exception-caught
        return None


def _iter_sse_json(body) -> Iterator[dict]:
    """Yield parsed objects from an HTTP SSE body."""
    buf = b''
    while True:
        chunk = body.read(256)
        if not chunk:
            if buf.strip():
                obj = _line_json(buf.decode('utf-8', errors='replace'))
                if obj is not None:
                    yield obj
            break
        buf += chunk
        while b'\n' in buf:
            raw, buf = buf.split(b'\n', 1)
            obj = _line_json(raw.decode('utf-8', errors='replace'))
            if obj is not None:
                yield obj


def _line_json(line: str) -> dict | None:
    text = line.strip()
    if not text:
        return None
    frac = parse_progress_text(text)
    if frac is not None:
        return {'type': 'prompt_processing.progress', 'progress': frac}
    if text.startswith(':') or text == 'data: [DONE]':
        return None
    if text.lower().startswith('data:'):
        text = text[5:].strip()
    if text.lower().startswith('event:'):
        return None
    if not text or text == '[DONE]':
        return None
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        return None
    return obj if isinstance(obj, dict) else None


class _Closable:
    """Lets ``_abort_llm_stream`` close the raw HTTP body."""

    def __init__(self, response, inner: Iterator):
        self._response = response
        self._inner = inner

    def __iter__(self):
        return self

    def __next__(self):
        return next(self._inner)

    def close(self) -> None:
        """Abort the in-flight HTTP body."""
        closer = getattr(self._response, 'close', None)
        if callable(closer):
            try:
                closer()
            except Exception:  # pylint: disable=broad-exception-caught
                pass


def _host_port(origin: str) -> tuple[str, int]:
    parsed = urlparse(origin if '://' in (origin or '') else f'http://{origin}')
    host = parsed.hostname or ''
    port = parsed.port or (443 if parsed.scheme == 'https' else 80)
    return host, port


def _auth_packet() -> dict:
    return {
        'authVersion': 1,
        'clientIdentifier': f'guest:{uuid.uuid4()}',
        'clientPasskey': str(uuid.uuid4()),
    }


def find_api_host(origin: str, headers: dict) -> str | None:
    """Return host:port of an LM Studio SDK API, or None.

    REST ``http://llm:1234`` is not this — probe ``/lmstudio-greeting``
    (must be ``{"lmstudio": true}``, not the catch-all 200).
    """
    with _CACHE_LOCK:
        if origin in _API_CACHE:
            return _API_CACHE[origin]
    host, port = _host_port(origin)
    if not host:
        with _CACHE_LOCK:
            _API_CACHE[origin] = None
        return None
    seen: list[int] = []
    for candidate in (port, *_API_PORTS):
        if candidate not in seen:
            seen.append(candidate)
    found: list[str] = []
    lock = threading.Lock()

    def probe(listen: int) -> None:
        if found:
            return
        url = f'http://{host}:{listen}/lmstudio-greeting'
        payload = _json_get(url, headers, timeout=0.35)
        if isinstance(payload, dict) and payload.get('lmstudio') is True:
            with lock:
                if not found:
                    found.append(f'{host}:{listen}')

    workers = [threading.Thread(target=probe, args=(p,), daemon=True) for p in seen]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join(0.5)
    result = found[0] if found else None
    with _CACHE_LOCK:
        _API_CACHE[origin] = result
    return result


def _ws_open(url: str, timeout: float = 1.2) -> socket.socket | None:
    """HTTP Upgrade to websocket. None if the peer is not a WS server."""
    parsed = urlparse(url)
    host = parsed.hostname
    port = parsed.port or 80
    path = parsed.path or '/'
    if not host:
        return None
    key = base64.b64encode(os.urandom(16)).decode('ascii')
    req = (
        f'GET {path} HTTP/1.1\r\n'
        f'Host: {host}:{port}\r\n'
        'Upgrade: websocket\r\n'
        'Connection: Upgrade\r\n'
        f'Sec-WebSocket-Key: {key}\r\n'
        'Sec-WebSocket-Version: 13\r\n'
        '\r\n'
    )
    try:
        sock = socket.create_connection((host, port), timeout=timeout)
        sock.sendall(req.encode('ascii'))
        buf = b''
        while b'\r\n\r\n' not in buf:
            chunk = sock.recv(1024)
            if not chunk:
                sock.close()
                return None
            buf += chunk
            if len(buf) > 8192:
                sock.close()
                return None
        status = buf.split(b'\r\n', 1)[0]
        if b'101' not in status:
            sock.close()
            return None
        return sock
    except (OSError, TimeoutError):
        return None


def _ws_send_text(sock: socket.socket, payload: bytes) -> None:
    mask = os.urandom(4)
    masked = bytes(byte ^ mask[i % 4] for i, byte in enumerate(payload))
    header = bytearray([0x81])
    length = len(payload)
    if length < 126:
        header.append(0x80 | length)
    elif length < 65536:
        header.append(0x80 | 126)
        header.extend(struct.pack('!H', length))
    else:
        header.append(0x80 | 127)
        header.extend(struct.pack('!Q', length))
    sock.sendall(bytes(header) + mask + masked)


def _ws_send_json(sock: socket.socket, obj: dict) -> None:
    _ws_send_text(sock, json.dumps(obj).encode('utf-8'))


def _read_exact(sock: socket.socket, count: int, stop: threading.Event) -> bytes | None:
    data = b''
    while len(data) < count:
        if stop.is_set():
            return None
        try:
            chunk = sock.recv(count - len(data))
        except socket.timeout:
            continue
        except OSError:
            return None
        if not chunk:
            return None
        data += chunk
    return data


def _ws_recv_frame(sock: socket.socket, stop: threading.Event) -> tuple[int, bytes] | None:
    hdr = _read_exact(sock, 2, stop)
    if not hdr:
        return None
    opcode = hdr[0] & 0x0F
    masked = bool(hdr[1] & 0x80)
    length = hdr[1] & 0x7F
    if length == 126:
        ext = _read_exact(sock, 2, stop)
        if not ext:
            return None
        length = struct.unpack('!H', ext)[0]
    elif length == 127:
        ext = _read_exact(sock, 8, stop)
        if not ext:
            return None
        length = struct.unpack('!Q', ext)[0]
    mask = _read_exact(sock, 4, stop) if masked else b''
    payload = _read_exact(sock, length, stop) if length else b''
    if payload is None or (masked and mask is None):
        return None
    if masked and payload and mask:
        payload = bytes(byte ^ mask[i % 4] for i, byte in enumerate(payload))
    return opcode, payload or b''


def _ws_recv_json(sock: socket.socket, stop: threading.Event) -> Any | None:
    """Next text JSON object, or None on close / stop. Replies to ping."""
    while not stop.is_set():
        frame = _ws_recv_frame(sock, stop)
        if frame is None:
            return None
        opcode, payload = frame
        if opcode == 0x8:
            return None
        if opcode == 0x9:
            try:
                # pong, masked
                mask = os.urandom(4)
                masked = bytes(byte ^ mask[i % 4] for i, byte in enumerate(payload))
                sock.sendall(bytes([0x8A, 0x80 | len(payload)]) + mask + masked)
            except OSError:
                return None
            continue
        if opcode == 0xA:
            continue
        if opcode != 0x1 or not payload:
            continue
        text = payload.decode('utf-8', errors='replace')
        frac = parse_progress_text(text)
        if frac is not None:
            return {'type': 'prompt_processing.progress', 'progress': frac}
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            continue
    return None


def _stream_diagnostics(api_host: str, sink: Callable[[float], None],
                        stop: threading.Event) -> None:
    """Subscribe to diagnostics.streamLogs and parse Developer-log lines."""
    sock = _ws_open(f'ws://{api_host}/diagnostics')
    if sock is None:
        return
    try:
        sock.settimeout(0.3)
        _ws_send_json(sock, _auth_packet())
        greeting = _ws_recv_json(sock, stop)
        if not isinstance(greeting, dict) or not greeting.get('success'):
            return
        _ws_send_json(sock, {
            'type': 'channelCreate',
            'endpoint': 'streamLogs',
            'channelId': 0,
        })
        while not stop.is_set():
            obj = _ws_recv_json(sock, stop)
            if obj is None:
                return
            frac = progress_from_obj(obj)
            if frac is not None:
                sink(frac)
    except (OSError, TimeoutError, ValueError):
        return
    finally:
        try:
            sock.close()
        except OSError:
            pass


def _poll_slots(origin: str, headers: dict, sink: Callable[[float], None],
                stop: threading.Event) -> bool:
    """llama.cpp /slots while the completion is in-flight. False if not slots."""
    with _CACHE_LOCK:
        cached = _SLOTS_CACHE.get(origin)
    if cached is False:
        return False
    urls = [f'{origin}/slots', f'{origin}/v1/slots']
    confirmed = bool(cached)
    misses = 0
    delay = 0.0
    while not stop.wait(delay):
        delay = 0.2
        payload = None
        for url in urls:
            payload = _json_get(url, headers)
            if payload is None:
                continue
            if looks_like_slots(payload):
                with _CACHE_LOCK:
                    _SLOTS_CACHE[origin] = True
                confirmed = True
                frac = parse_slots_progress(payload)
                if frac is not None:
                    sink(frac)
                break
            with _CACHE_LOCK:
                _SLOTS_CACHE[origin] = False
            return False
        else:
            if confirmed:
                continue
            misses += 1
            if misses >= 2:
                return False
    return confirmed


def _sideband(origin: str, headers: dict, sink: Callable[[float], None],
              stop: threading.Event) -> None:
    """Slots if llama.cpp; else LM Studio diagnostics logs."""
    if not origin:
        return
    if _poll_slots(origin, headers, sink, stop):
        return
    if stop.is_set():
        return
    api_host = find_api_host(origin, headers)
    if not api_host:
        return
    _stream_diagnostics(api_host, sink, stop)


def _post_stream(url: str, payload: dict, headers: dict):
    """POST the completion. Drop return_progress on 400 and retry once."""
    body = json.dumps(payload).encode('utf-8')
    req = urllib.request.Request(url, data=body, headers=headers, method='POST')
    try:
        return urllib.request.urlopen(req, timeout=600)
    except urllib.error.HTTPError as exc:
        try:
            exc.read()
        except Exception:  # pylint: disable=broad-exception-caught
            pass
        if exc.code in {400, 404, 415, 422} and payload.pop('return_progress', None):
            return _post_stream(url, payload, headers)
        return None
    except Exception:  # pylint: disable=broad-exception-caught
        return None


def _maybe_progress(frac: float, last_label: str) -> tuple[str, PromptProgress | None]:
    """Yield only when the status string actually changes."""
    label = format_prompt_status(frac)
    if not label or label == last_label or label == 'Processing Prompt…':
        return last_label, None
    return label, PromptProgress(frac)


def _token_chunk(content: str, reasoning: str) -> TokenChunk:
    extra = {'reasoning_content': reasoning} if reasoning else {}
    return TokenChunk(
        content=content, reasoning_content=reasoning, additional_kwargs=extra,
    )


def delta_text(obj: dict) -> tuple[str, str]:
    """(content, reasoning) from an OpenAI or native chat-completion chunk."""
    kind = str(obj.get('type') or '')
    if kind == 'message.delta':
        piece = obj.get('content') or ''
        return (piece if isinstance(piece, str) else str(piece), '')
    if kind == 'reasoning.delta':
        piece = obj.get('content') or ''
        return ('', piece if isinstance(piece, str) else str(piece))
    choices = obj.get('choices')
    if not isinstance(choices, list) or not choices:
        return '', ''
    choice = choices[0] if isinstance(choices[0], dict) else {}
    delta = choice.get('delta') or choice.get('message') or {}
    if not isinstance(delta, dict):
        return '', ''
    content = delta.get('content')
    if content is None:
        content = ''
    elif not isinstance(content, str):
        content = str(content)
    extra = (
        delta.get('reasoning_content')
        or delta.get('reasoning')
        or ''
    )
    if not isinstance(extra, str):
        extra = str(extra) if extra is not None else ''
    return content, extra


def _drain(box: queue.Queue, stop: threading.Event) -> Iterator[Any]:
    """Merge SSE objects, /slots fractions, and LangChain chunks."""
    # pylint: disable=too-many-branches
    last_label = ''
    seen_token = False
    try:
        while True:
            try:
                kind, payload = box.get(timeout=0.2)
            except queue.Empty:
                if stop.is_set():
                    break
                continue
            if kind == 'eof':
                break
            if kind == 'err':
                raise payload
            if kind == 'progress':
                if seen_token:
                    continue
                last_label, item = _maybe_progress(payload, last_label)
                if item is not None:
                    yield item
                continue
            if kind == 'chunk':
                seen_token = True
                yield payload
                continue
            frac = parse_progress(payload)
            if frac is not None and not seen_token:
                last_label, item = _maybe_progress(frac, last_label)
                if item is not None:
                    yield item
            content, reasoning = delta_text(payload)
            if content or reasoning:
                seen_token = True
                yield _token_chunk(content, reasoning)
    finally:
        stop.set()


def _merge_stream(response, origin: str, headers: dict, stop: threading.Event,
                  box: queue.Queue) -> None:
    """SSE reader thread + /slots or LM Studio diagnostics."""
    def sink(frac: float) -> None:
        box.put(('progress', frac))

    def read_sse() -> None:
        try:
            for obj in _iter_sse_json(response):
                box.put(('sse', obj))
        except Exception as exc:  # pylint: disable=broad-exception-caught
            box.put(('err', exc))
        finally:
            box.put(('eof', None))

    threading.Thread(target=read_sse, daemon=True).start()
    if origin:
        threading.Thread(
            target=_sideband, args=(origin, headers, sink, stop), daemon=True,
        ).start()


def stream_chat(llm: Any, messages: Any) -> Iterator[Any]:
    """Yield PromptProgress then TokenChunk (or LangChain chunks on fallback)."""
    url = _chat_url(llm)
    if not url or _is_cloud(url):
        yield from llm.stream(messages)
        return
    payload = _payload_for(llm, messages, with_progress=True)
    headers = _headers(llm, accept='text/event-stream')
    response = _post_stream(url, payload, headers)
    if response is None:
        yield from llm.stream(messages)
        return

    box: queue.Queue = queue.Queue()
    stop = threading.Event()
    origin = _origin(url)
    _merge_stream(response, origin, _headers(llm), stop, box)
    wrapped = _Closable(response, _drain(box, stop))
    try:
        yield from wrapped
    finally:
        stop.set()
        wrapped.close()
