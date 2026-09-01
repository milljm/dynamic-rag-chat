"""Prompt-processing % from mlx-edge ``GET /v1/progress/stream``.

OpenAI ``/v1/chat/completions`` stays on LangChain. While that HTTP call is
blocked on prefill (the slow part on long contexts), a sideband subscribes
to the Edge EventSource:

    GET {base}/v1/progress/stream

and reads the 0.0..1.0 float at ``models[0].progress`` (fallback:
``snapshot.progress``, then ``prompt.ratio``). Spur/Streamlit show
``Processing Prompt… 46.6%``.

If the stream is missing, poll ``GET /v1/progress``. LM Studio's catch-all
HTTP 200 is a miss — ``object`` must be ``edge.progress`` — and that miss
is cached. Cloud OpenAI-style hosts are never probed.

This is not PR #79: no ``GET /slots``, no Developer-log scrape, no
``return_progress`` rewrite of the chat POST.
"""
from __future__ import annotations

import http.client
import json
import queue
import socket
import threading
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Any, Iterator
from urllib.parse import urlparse

EDGE_OBJECT = 'edge.progress'

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

# origin -> progress URL, or None after a confirmed miss.
_CACHE: dict[str, str | None] = {}
_CACHE_LOCK = threading.Lock()
_MISSING = object()

# In-flight OpenAI ``/v1/chat/completions`` body for this turn.
_GEN_LOCK = threading.Lock()
_GEN_HTTP: Any = None
_GEN_STOP = threading.Event()


@dataclass
class PromptProgress:
    """Fraction of the prompt mlx-edge has prefills (0..1)."""

    fraction: float
    phase: str = 'prefill'
    model: str = ''
    processed: int | None = None
    total: int | None = None


def reset_progress_caches() -> None:
    """Test helper: forget /v1/progress probes."""
    with _CACHE_LOCK:
        _CACHE.clear()


def begin_generation() -> None:
    """Clear a previous Stop so this turn can stream."""
    global _GEN_HTTP
    _GEN_STOP.clear()
    with _GEN_LOCK:
        _GEN_HTTP = None


def abort_generation() -> bool:
    """Close the in-flight OpenAI ``/v1/chat/completions`` HTTP body.

    Chat Completions has no cancel POST. Closing the stream is the stop
    command — LM Studio logs ``Client disconnected. Stopping generation…``.
    """
    global _GEN_HTTP
    _GEN_STOP.set()
    with _GEN_LOCK:
        stream = _GEN_HTTP
        _GEN_HTTP = None
    if stream is None:
        return False
    _close_stream(stream)
    return True


def generation_stopped() -> bool:
    """True after the user hit Stop this turn."""
    return _GEN_STOP.is_set()


def _attach_http(stream: Any) -> None:
    global _GEN_HTTP
    with _GEN_LOCK:
        _GEN_HTTP = stream


def _detach_http(stream: Any) -> None:
    global _GEN_HTTP
    with _GEN_LOCK:
        if _GEN_HTTP is stream:
            _GEN_HTTP = None


def format_prompt_status(fraction: float) -> str:
    """``Processing Prompt… 46.6%`` — Spur puts model/route/context after.

    Starts at ``0%``. Never publishes ``100%`` here: that would be a leftover
    decode snapshot from the previous turn (capped as 99.9% and looking
    like we finished before we began).
    """
    frac = _clamp(float(fraction))
    shown = min(99.9, frac * 100.0)
    if shown <= 0:
        return 'Processing Prompt… 0%'
    if abs(shown - round(shown)) < 0.05:
        whole = int(round(shown))
        if whole <= 0:
            return 'Processing Prompt… 0%'
        return f'Processing Prompt… {min(99, whole)}%'
    return f'Processing Prompt… {shown:.1f}%'


def progress_urls(base: str) -> list[str]:
    """Candidate snapshot URLs for a ChatOpenAI ``base_url``."""
    root = (base or '').strip().rstrip('/')
    if not root:
        return []
    if root.endswith('/v1'):
        return [f'{root}/progress']
    return [f'{root}/v1/progress', f'{root}/progress']


def progress_stream_url(snapshot_url: str) -> str:
    """``GET /v1/progress`` → ``GET /v1/progress/stream``."""
    root = (snapshot_url or '').rstrip('/')
    if root.endswith('/stream'):
        return root
    return f'{root}/stream'


def llm_base_url(llm: Any) -> str:
    """OpenAI-compat origin ChatOpenAI will POST to."""
    for attr in ('openai_api_base', 'base_url'):
        text = _secret(getattr(llm, attr, None))
        if text:
            return text.rstrip('/')
    for attr in ('root_client', 'client', 'async_client'):
        client = getattr(llm, attr, None)
        if client is None:
            continue
        text = _secret(getattr(client, 'base_url', None))
        if text:
            return str(text).rstrip('/')
    return ''


def llm_model(llm: Any) -> str:
    """Basename ChatOpenAI will send as ``model``."""
    for attr in ('model_name', 'model'):
        text = str(getattr(llm, attr, '') or '').strip()
        if text and text.lower() not in {'none', 'not_set', 'null'}:
            return text
    return ''


def llm_api_key(llm: Any) -> str:
    """Bearer token, unwrapping LangChain SecretStr."""
    for attr in ('openai_api_key', 'api_key'):
        text = _secret(getattr(llm, attr, None))
        if text:
            return text
    return ''


def is_cloud_host(base: str) -> bool:
    """True for hosted OpenAI-style APIs that will never speak edge.progress."""
    host = (urlparse(base).hostname or '').lower()
    if not host:
        return False
    return any(host == name or host.endswith('.' + name) for name in _CLOUD)


def probe_progress(
    base: str,
    headers: dict[str, str] | None = None,
    timeout: float = 0.4,
) -> str | None:
    """Return the snapshot URL when ``GET /v1/progress`` is ``edge.progress``.

    Caches a miss so LM Studio's catch-all HTTP 200 is only asked once.
    """
    if not base or is_cloud_host(base):
        return None
    url, _snap = _probe(base, headers or {}, timeout)
    return url if isinstance(url, str) else None


def pick_progress(
    snapshot: dict | None, model: str = '',
) -> PromptProgress | None:
    """Read the 0..1 float mlx-edge 0.8+ always publishes as ``progress``.

    Order: matching ``models[].progress``, then ``models[0].progress``,
    then top-level ``snapshot.progress``, then ``prompt.ratio`` / counts.
    """
    if not _is_edge(snapshot):
        return None
    rows = snapshot.get('models') or []
    if not isinstance(rows, list):
        rows = []
    row = _choose_row(rows, model)
    ident = ''
    phase = ''
    prompt: dict[str, Any] = {}
    frac: float | None = None
    if row is not None:
        ident = str(row.get('id') or '')
        phase = str(row.get('phase') or '')
        prompt = row.get('prompt') if isinstance(row.get('prompt'), dict) else {}
        frac = _coerce_fraction(row.get('progress'))
        if frac is None:
            frac = _coerce_fraction(prompt.get('ratio'))
        if frac is None:
            frac = _ratio_from_counts(prompt)
    if frac is None:
        frac = _coerce_fraction(snapshot.get('progress'))
    if frac is None or frac <= 0:
        return None
    if not phase:
        # Don't relabel a leftover decode/done snapshot as prefill just
        # because top-level ``progress`` is 1.0 and ``active`` is still true.
        if snapshot.get('active') and frac < 0.999:
            phase = 'prefill'
        else:
            phase = 'idle'
    processed = prompt.get('processed_tokens') if prompt else None
    total = prompt.get('total_tokens') if prompt else None
    return PromptProgress(
        fraction=frac,
        phase=phase,
        model=ident,
        processed=int(processed) if isinstance(processed, (int, float)) else None,
        total=int(total) if isinstance(total, (int, float)) else None,
    )


def stream_chat(
    llm: Any,
    messages: Any,
    *,
    interval: float = 0.12,
    timeout: float = 0.4,
) -> Iterator:
    """Yield ``PromptProgress`` then LangChain chunks.

    Prefill events come from ``GET /v1/progress/stream`` (EventSource) on a
    side thread so the generator can emit percents while ``llm.stream()`` is
    still blocked on the first token. After the first chunk (or when the
    snapshot leaves ``prefill``) the sideband stops. Hosts that are not
    mlx-edge fall through to ``llm.stream()`` alone.
    """
    if _GEN_STOP.is_set():
        return
    base = llm_base_url(llm)
    if not base or is_cloud_host(base) or _cached(_origin(base)) is None:
        yield from _plain_stream(llm, messages)
        return

    box: queue.Queue[tuple[str, Any]] = queue.Queue()
    stop = threading.Event()
    inner = {'stream': None}

    def read_llm() -> None:
        stream = None
        try:
            if _GEN_STOP.is_set():
                box.put(('done', None))
                return
            stream = llm.stream(messages)
            inner['stream'] = stream
            _attach_http(stream)
            for chunk in stream:
                if _GEN_STOP.is_set() or stop.is_set():
                    break
                box.put(('chunk', chunk))
            box.put(('done', None))
        except Exception as exc:  # pylint: disable=broad-exception-caught
            if _GEN_STOP.is_set():
                box.put(('done', None))
            else:
                box.put(('err', exc))
        finally:
            _detach_http(stream)

    def read_progress() -> None:
        _watch_progress(
            base=base,
            model=llm_model(llm),
            headers=_headers(llm),
            box=box,
            stop=stop,
            interval=interval,
            timeout=timeout,
        )

    progress_worker = threading.Thread(target=read_progress, daemon=True)
    llm_worker = threading.Thread(target=read_llm, daemon=True)
    # Connect the EventSource before the chat POST so the first prefill
    # snapshots are not lost behind the completion request.
    progress_worker.start()
    stop.wait(0.05)
    llm_worker.start()
    workers = (llm_worker, progress_worker)
    saw_token = False
    try:
        while True:
            try:
                kind, payload = box.get(timeout=0.2)
            except queue.Empty:
                if stop.is_set() or _GEN_STOP.is_set() or not llm_worker.is_alive():
                    break
                continue
            if kind == 'progress':
                if not saw_token:
                    yield payload
            elif kind == 'chunk':
                saw_token = True
                yield payload
            elif kind == 'done':
                break
            else:
                raise payload
    finally:
        stop.set()
        _close_stream(inner.get('stream'))
        for worker in workers:
            worker.join(timeout=0.6)


def _plain_stream(llm: Any, messages: Any) -> Iterator:
    if _GEN_STOP.is_set():
        return
    stream = llm.stream(messages)
    _attach_http(stream)
    try:
        for chunk in stream:
            if _GEN_STOP.is_set():
                break
            yield chunk
    finally:
        _detach_http(stream)
        _close_stream(stream)


def _watch_progress(
    *,
    base: str,
    model: str,
    headers: dict[str, str],
    box: queue.Queue,
    stop: threading.Event,
    interval: float,
    timeout: float,
) -> None:
    origin = _origin(base)
    url = _cached(origin)
    if url is None:
        return
    if not isinstance(url, str):
        url, _snap = _probe(base, headers, timeout)
        if not isinstance(url, str):
            return
    # Start at 0% so a leftover decode snapshot (progress=1.0 → 99.9%)
    # from the previous turn cannot flash before this prefill begins.
    box.put(('progress', PromptProgress(
        fraction=0.0, phase='prefill', model=model,
    )))
    if stop.is_set():
        return
    if _subscribe_sse(
        progress_stream_url(url), headers, model, box, stop, timeout=max(2.0, timeout),
    ):
        return
    _poll_progress(url, model, headers, box, stop, interval, timeout)


def _poll_progress(
    url: str,
    model: str,
    headers: dict[str, str],
    box: queue.Queue,
    stop: threading.Event,
    interval: float,
    timeout: float,
) -> None:
    while not stop.is_set():
        snap = _get_json(url, headers, timeout)
        if snap is None:
            if stop.wait(interval):
                return
            continue
        if not _is_edge(snap):
            return
        _emit_progress(snap, model, box)
        if stop.wait(interval):
            return


def _subscribe_sse(
    url: str,
    headers: dict[str, str],
    model: str,
    box: queue.Queue,
    stop: threading.Event,
    timeout: float,
) -> bool:
    """EventSource ``GET /v1/progress/stream``. False = fall back to poll."""
    parsed = urlparse(url)
    if not parsed.hostname:
        return False
    path = parsed.path or '/'
    if parsed.query:
        path = f'{path}?{parsed.query}'
    conn: http.client.HTTPConnection | None = None
    try:
        conn = _http_conn(parsed, timeout)
        req_headers = dict(headers)
        req_headers['Accept'] = 'text/event-stream'
        req_headers['Cache-Control'] = 'no-cache'
        conn.request('GET', path, headers=req_headers)
        resp = conn.getresponse()
        if resp.status != 200:
            return False
        ctype = (resp.getheader('Content-Type') or '').lower()
        if 'event-stream' not in ctype and 'json' in ctype:
            return False
        buf = b''
        while not stop.is_set():
            try:
                chunk = resp.read(256)
            except (TimeoutError, socket.timeout):
                continue
            except OSError:
                break
            if not chunk:
                break
            buf += chunk
            while b'\n\n' in buf:
                frame, buf = buf.split(b'\n\n', 1)
                snap = _parse_sse_frame(frame.decode('utf-8', errors='replace'))
                if snap is None:
                    continue
                if not _is_edge(snap):
                    return False
                _emit_progress(snap, model, box)
        return True
    except (http.client.HTTPException, TimeoutError, socket.timeout, OSError):
        return False
    finally:
        if conn is not None:
            try:
                conn.close()
            except OSError:
                pass


def _emit_progress(snap: Any, model: str, box: queue.Queue) -> None:
    """Push a live prefill ratio. Ignore idle/decode leftovers and 1.0.

    mlx-edge lingers at ``progress: 1.0`` after the last token. Publishing
    that as Processing Prompt makes the bar jump to 99.9% on the next turn.
    """
    pp = pick_progress(snap if isinstance(snap, dict) else None, model)
    if pp is None or pp.phase in {'decode', 'done', 'error', 'idle'}:
        return
    if pp.fraction >= 0.999:
        return
    box.put(('progress', pp))


def _parse_sse_frame(frame: str) -> dict | None:
    data_lines: list[str] = []
    for raw in frame.splitlines():
        line = raw.strip('\r')
        if line.startswith('data:'):
            data_lines.append(line[5:].lstrip())
    if not data_lines:
        return None
    try:
        data = json.loads('\n'.join(data_lines))
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None


def _http_conn(parsed: urllib.parse.ParseResult, timeout: float) -> http.client.HTTPConnection:
    host = parsed.hostname or '127.0.0.1'
    port = parsed.port or (443 if parsed.scheme == 'https' else 80)
    if parsed.scheme == 'https':
        return http.client.HTTPSConnection(host, port, timeout=timeout)
    return http.client.HTTPConnection(host, port, timeout=timeout)


def _probe(
    base: str, headers: dict[str, str], timeout: float,
) -> tuple[str | None, Any]:
    """Find a live ``edge.progress`` URL. ``(_MISSING)`` means try again."""
    origin = _origin(base)
    with _CACHE_LOCK:
        if origin in _CACHE:
            cached = _CACHE[origin]
            if cached is None:
                return None, None
            return cached, _MISSING
    network_fail = False
    found_url = None
    found_snap: Any = None
    for url in progress_urls(base):
        snap = _get_json(url, headers, timeout)
        if snap is None:
            network_fail = True
            continue
        if _is_edge(snap):
            found_url, found_snap = url, snap
            break
        _remember(origin, None)
        return None, None
    if found_url:
        _remember(origin, found_url)
        return found_url, found_snap
    if network_fail:
        return None, _MISSING
    return None, None


def _choose_row(rows: list, model: str) -> dict | None:
    """Prefer the named model; otherwise the first processing row / models[0]."""
    first: dict | None = None
    processing: dict | None = None
    for row in rows:
        if not isinstance(row, dict):
            continue
        if first is None:
            first = row
        ident = str(row.get('id') or '')
        if model and ident and _ids_match(ident, model):
            return row
        phase = str(row.get('phase') or '')
        status = str(row.get('status') or '')
        if processing is None and (phase == 'prefill' or status == 'processing'):
            processing = row
    if model:
        return None
    return processing or first


def _coerce_fraction(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return _clamp(float(value))


def _ratio_from_counts(prompt: dict[str, Any]) -> float | None:
    processed = prompt.get('processed_tokens')
    total = prompt.get('total_tokens')
    if isinstance(processed, (int, float)) and isinstance(total, (int, float)) and total:
        return _clamp(processed / float(total))
    return None


def _is_edge(snapshot: Any) -> bool:
    return isinstance(snapshot, dict) and snapshot.get('object') == EDGE_OBJECT


def _ids_match(left: str, right: str) -> bool:
    a = left.strip().replace('\\', '/').rstrip('/').lower()
    b = right.strip().replace('\\', '/').rstrip('/').lower()
    if not a or not b:
        return False
    if a == b:
        return True
    if a.endswith('/' + b) or b.endswith('/' + a):
        return True
    return a.split('/')[-1] == b.split('/')[-1]


def _origin(base: str) -> str:
    parsed = urlparse(base)
    host = (parsed.hostname or '').lower()
    port = parsed.port or (443 if parsed.scheme == 'https' else 80)
    return f'{host}:{port}'


def _cached(origin: str) -> Any:
    with _CACHE_LOCK:
        if origin not in _CACHE:
            return _MISSING
        return _CACHE[origin]


def _remember(origin: str, url: str | None) -> None:
    with _CACHE_LOCK:
        _CACHE[origin] = url


def _headers(llm: Any) -> dict[str, str]:
    key = llm_api_key(llm) or 'none'
    return {'Authorization': f'Bearer {key}'}


def _get_json(
    url: str, headers: dict[str, str], timeout: float,
) -> dict | None:
    try:
        req = urllib.request.Request(url, headers=headers, method='GET')
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode('utf-8', errors='replace')
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            return {}
        return data if isinstance(data, dict) else {}
    except urllib.error.HTTPError:
        return {}
    except (urllib.error.URLError, TimeoutError, OSError):
        return None


def _secret(value: Any) -> str:
    if value is None:
        return ''
    getter = getattr(value, 'get_secret_value', None)
    if callable(getter):
        try:
            value = getter()
        except Exception:  # pylint: disable=broad-exception-caught
            return ''
    text = str(value).strip()
    if text.lower() in {'', 'none', 'not_set', 'null', '~'}:
        return ''
    if set(text) <= {'*'}:
        return ''
    return text


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _close_stream(stream: Any) -> None:
    """Close an in-flight LLM HTTP body so the next completion can start."""
    cur = stream
    seen: set[int] = set()
    while cur is not None and id(cur) not in seen:
        seen.add(id(cur))
        closer = getattr(cur, 'close', None)
        if callable(closer):
            try:
                closer()
            except Exception:  # pylint: disable=broad-exception-caught
                pass
        nxt = None
        for attr in ('response', '_response', 'http_response'):
            cand = getattr(cur, attr, None)
            if cand is not None and cand is not cur:
                nxt = cand
                break
        cur = nxt
