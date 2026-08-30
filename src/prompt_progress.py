"""Surface prompt-processing % during the wait-for-first-token.

LM Studio's Developer log (``Prompt processing progress: 46.6%``) is not on
the OpenAI-compat ``/v1/chat/completions`` path we use. Native
``POST /api/v1/chat`` *does* emit ``prompt_processing.progress``, but that
API is stateful and cannot take a full assistant history.

So we keep ChatOpenAI's messages and:

1. Stream ``/v1/chat/completions`` ourselves and catch llama.cpp
   ``prompt_progress`` / LM Studio ``prompt_processing.progress`` if they
   leak through (``return_progress: true`` on local hosts only).
2. Parse the same ``Prompt processing progress: 46.6%`` text if it shows
   up as an SSE comment or JSON message.
3. Poll ``GET /slots`` on the same origin (llama.cpp) as a fallback.
4. Fall back to LangChain ``llm.stream()`` if the raw POST fails.
"""
from __future__ import annotations

import json
import queue
import re
import threading
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Callable, Iterator


_PROGRESS_LOG = re.compile(
    r'Prompt processing progress:\s*([0-9]*\.?[0-9]+)\s*%',
    re.I,
)


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


def format_prompt_status(fraction: float) -> str:
    """Status line Spur already knows: Processing Prompt… 46.6%."""
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


def parse_slots_progress(payload: Any) -> float | None:
    """Best fraction from llama.cpp GET /slots."""
    rows = payload
    if isinstance(payload, dict):
        rows = payload.get('slots') or payload.get('data') or payload
    if not isinstance(rows, list):
        rows = [payload] if isinstance(payload, dict) else []
    best: float | None = None
    for row in rows:
        if not isinstance(row, dict):
            continue
        frac = parse_progress(row)
        if frac is None:
            continue
        best = frac if best is None else max(best, frac)
    return best


def delta_text(obj: dict) -> tuple[str, str]:
    """(content, reasoning) from an OpenAI chat-completion chunk."""
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


def _is_local_host(url: str) -> bool:
    text = (url or '').lower()
    return any(
        token in text
        for token in ('127.0.0.1', 'localhost', '0.0.0.0', ':1234', ':11434')
    )


def _origin(base_url: str) -> str:
    """http://host:1234/v1 → http://host:1234."""
    text = (base_url or '').rstrip('/')
    if text.endswith('/v1'):
        return text[:-3]
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
    if with_progress and _is_local_host(_chat_url(llm)):
        payload['return_progress'] = True
    return payload


def _headers(llm: Any) -> dict[str, str]:
    key = str(getattr(llm, 'openai_api_key', None) or getattr(llm, 'api_key', None) or 'none')
    return {
        'Authorization': f'Bearer {key}',
        'Content-Type': 'application/json',
        'Accept': 'text/event-stream',
    }


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


def _poll_slots(origin: str, headers: dict, sink: Callable[[float], None],
                stop: threading.Event) -> None:
    """Best-effort llama.cpp /slots while the completion is in-flight."""
    urls = [f'{origin}/slots', f'{origin}/v1/slots']
    seen_ok = False
    while not stop.wait(0.2 if seen_ok else 0.0):
        hit = False
        for url in urls:
            try:
                req = urllib.request.Request(url, headers=headers, method='GET')
                with urllib.request.urlopen(req, timeout=0.8) as resp:
                    payload = json.loads(resp.read().decode('utf-8', errors='replace'))
                frac = parse_slots_progress(payload)
                if frac is not None:
                    sink(frac)
                    hit = True
                    seen_ok = True
                    break
            except Exception:  # pylint: disable=broad-exception-caught
                continue
        if not hit and not seen_ok:
            return


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


def _drain(box: queue.Queue, stop: threading.Event) -> Iterator[Any]:
    """Merge SSE objects and /slots fractions until the completion ends."""
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


def stream_chat(llm: Any, messages: Any) -> Iterator[Any]:
    """Yield PromptProgress then TokenChunk (or LangChain chunks on fallback)."""
    url = _chat_url(llm)
    if not url:
        yield from llm.stream(messages)
        return
    payload = _payload_for(llm, messages, with_progress=True)
    headers = _headers(llm)
    response = _post_stream(url, payload, headers)
    if response is None:
        yield from llm.stream(messages)
        return

    box: queue.Queue = queue.Queue()
    stop = threading.Event()

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
    if _is_local_host(url):
        threading.Thread(
            target=_poll_slots,
            args=(_origin(url), headers, sink, stop),
            daemon=True,
        ).start()

    wrapped = _Closable(response, _drain(box, stop))
    try:
        yield from wrapped
    finally:
        stop.set()
        wrapped.close()
