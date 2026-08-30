"""Surface prompt-processing % only when the backend actually exposes it.

LM Studio's Developer log (``Prompt processing progress: 46.6%``) lives on
native ``POST /api/v1/chat`` as ``prompt_processing.progress``. That endpoint
is stateful and cannot take assistant history, so we do **not** switch
generation to it.

The OpenAI-compat socket Spur already uses (``/v1/chat/completions``) does
not document those events. llama.cpp does expose ``GET /slots``.

At stream time we probe the configured origin — ``http://llm:1234`` counts,
not just localhost:

1. ``GET /slots`` (or ``/v1/slots``) returns JSON → poll it while
   ``llm.stream()`` runs and update ``Processing Prompt… 46.6%``.
2. Otherwise → leave LangChain alone. Not supported.
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
_PROBE: dict[str, bool] = {}
_PROBE_LOCK = threading.Lock()


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


def reset_probe_cache() -> None:
    """Tests: forget GET /slots results."""
    with _PROBE_LOCK:
        _PROBE.clear()


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
    return str(base).rstrip('/')


def _headers(llm: Any) -> dict[str, str]:
    key = str(getattr(llm, 'openai_api_key', None) or getattr(llm, 'api_key', None) or 'none')
    return {
        'Authorization': f'Bearer {key}',
        'Content-Type': 'application/json',
        'Accept': 'application/json',
    }


def _json_get(url: str, headers: dict) -> Any | None:
    """GET JSON, or None on 404 / network / non-JSON."""
    try:
        req = urllib.request.Request(url, headers=headers, method='GET')
        with urllib.request.urlopen(req, timeout=0.6) as resp:
            raw = resp.read().decode('utf-8', errors='replace')
        return json.loads(raw)
    except (urllib.error.URLError, json.JSONDecodeError, TimeoutError, OSError):
        return None
    except Exception:  # pylint: disable=broad-exception-caught
        return None


def slots_supported(origin: str, headers: dict) -> bool:
    """True when GET /slots (or /v1/slots) returns JSON. Cached per origin."""
    if not origin:
        return False
    with _PROBE_LOCK:
        cached = _PROBE.get(origin)
        if cached is not None:
            return cached
    ok = False
    for path in ('/slots', '/v1/slots'):
        if _json_get(f'{origin}{path}', headers) is not None:
            ok = True
            break
    with _PROBE_LOCK:
        _PROBE[origin] = ok
    return ok


class _Closable:
    """Lets ``_abort_llm_stream`` close the LangChain HTTP body."""

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
    """llama.cpp /slots while the completion is in-flight."""
    urls = [f'{origin}/slots', f'{origin}/v1/slots']
    delay = 0.0
    while not stop.wait(delay):
        delay = 0.2
        for url in urls:
            payload = _json_get(url, headers)
            if payload is None:
                continue
            frac = parse_slots_progress(payload)
            if frac is not None:
                sink(frac)
            break


def _maybe_progress(frac: float, last_label: str) -> tuple[str, PromptProgress | None]:
    """Yield only when the status string actually changes."""
    label = format_prompt_status(frac)
    if not label or label == last_label or label == 'Processing Prompt…':
        return last_label, None
    return label, PromptProgress(frac)


def _drain(box: queue.Queue, stop: threading.Event) -> Iterator[Any]:
    """Merge /slots fractions with LangChain chunks."""
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
            seen_token = True
            yield payload
    finally:
        stop.set()


def _stream_with_slots(llm: Any, messages: Any, origin: str, headers: dict) -> Iterator[Any]:
    """LangChain tokens plus /slots progress."""
    stream = llm.stream(messages)
    box: queue.Queue = queue.Queue()
    stop = threading.Event()

    def sink(frac: float) -> None:
        box.put(('progress', frac))

    def read() -> None:
        try:
            for chunk in stream:
                box.put(('chunk', chunk))
        except Exception as exc:  # pylint: disable=broad-exception-caught
            box.put(('err', exc))
        finally:
            box.put(('eof', None))

    threading.Thread(target=_poll_slots, args=(origin, headers, sink, stop), daemon=True).start()
    threading.Thread(target=read, daemon=True).start()
    wrapped = _Closable(stream, _drain(box, stop))
    try:
        yield from wrapped
    finally:
        stop.set()
        wrapped.close()


def stream_chat(llm: Any, messages: Any) -> Iterator[Any]:
    """Yield PromptProgress then LLM chunks, or just ``llm.stream()``."""
    url = _chat_url(llm)
    if not url or _is_cloud(url):
        yield from llm.stream(messages)
        return
    origin = _origin(url)
    headers = _headers(llm)
    if not slots_supported(origin, headers):
        yield from llm.stream(messages)
        return
    yield from _stream_with_slots(llm, messages, origin, headers)
