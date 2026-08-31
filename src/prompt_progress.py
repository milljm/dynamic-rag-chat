"""Prompt-processing % from mlx-edge ``GET /v1/progress``.

OpenAI ``/v1/chat/completions`` stays on LangChain. While that HTTP call is
blocked on prefill (the slow part on long contexts), a sideband polls the
Edge snapshot:

    GET {base}/v1/progress?model=…

and yields ``PromptProgress`` so Spur/Streamlit can show
``Processing Prompt… 46.6%``.

LM Studio answers unknown paths with HTTP 200 + an error JSON. We only treat
a body as progress when ``object == "edge.progress"``, and we cache a miss so
we do not hammer a catch-all. Cloud OpenAI-style hosts are never probed.

This is not PR #79: no ``GET /slots``, no Developer-log scrape, no
``return_progress`` rewrite of the chat POST.
"""
from __future__ import annotations

import json
import queue
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


def progress_urls(base: str) -> list[str]:
    """Candidate snapshot URLs for a ChatOpenAI ``base_url``."""
    root = (base or '').strip().rstrip('/')
    if not root:
        return []
    if root.endswith('/v1'):
        return [f'{root}/progress']
    return [f'{root}/v1/progress', f'{root}/progress']


def llm_base_url(llm: Any) -> str:
    """OpenAI-compat origin ChatOpenAI will POST to."""
    for attr in ('openai_api_base', 'base_url'):
        text = _secret(getattr(llm, attr, None))
        if text:
            return text.rstrip('/')
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
    """Read a prefill ratio out of an ``edge.progress`` snapshot."""
    if not _is_edge(snapshot):
        return None
    rows = snapshot.get('models') or []
    if not isinstance(rows, list):
        return None
    fallback: PromptProgress | None = None
    for row in rows:
        pp = _row_progress(row, model)
        if pp is None:
            continue
        status = str(row.get('status') or '') if isinstance(row, dict) else ''
        if pp.phase == 'prefill' or status == 'processing':
            return pp
        if fallback is None:
            fallback = pp
    return fallback


def stream_chat(
    llm: Any,
    messages: Any,
    *,
    interval: float = 0.12,
    timeout: float = 0.4,
) -> Iterator:
    """Yield ``PromptProgress`` then LangChain chunks.

    Prefill polls run on a side thread so the generator can emit percents
    while ``llm.stream()`` is still blocked on the first token. After the
    first chunk (or when the snapshot leaves ``prefill``) polling stops.
    Hosts that are not mlx-edge fall through to ``llm.stream()`` alone.
    """
    base = llm_base_url(llm)
    if not base or is_cloud_host(base) or _cached(_origin(base)) is None:
        yield from _plain_stream(llm, messages)
        return

    box: queue.Queue[tuple[str, Any]] = queue.Queue()
    stop = threading.Event()
    inner = {'stream': None}

    def read_llm() -> None:
        try:
            stream = llm.stream(messages)
            inner['stream'] = stream
            for chunk in stream:
                if stop.is_set():
                    break
                box.put(('chunk', chunk))
            box.put(('done', None))
        except Exception as exc:  # pylint: disable=broad-exception-caught
            box.put(('err', exc))

    def read_progress() -> None:
        _poll_progress(
            base=base,
            model=llm_model(llm),
            headers=_headers(llm),
            box=box,
            stop=stop,
            interval=interval,
            timeout=timeout,
        )

    workers = (
        threading.Thread(target=read_llm, daemon=True),
        threading.Thread(target=read_progress, daemon=True),
    )
    for worker in workers:
        worker.start()
    saw_token = False
    try:
        while True:
            try:
                kind, payload = box.get(timeout=0.2)
            except queue.Empty:
                if stop.is_set() or not workers[0].is_alive():
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
    stream = llm.stream(messages)
    try:
        yield from stream
    finally:
        _close_stream(stream)


def _poll_progress(
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
    last: float | None = None
    while not stop.is_set():
        if not isinstance(url, str):
            url, snap = _probe(base, headers, timeout)
            if url is None:
                if snap is _MISSING:
                    if stop.wait(interval):
                        return
                    continue
                return
        else:
            snap = _get_json(_with_model(url, model), headers, timeout)
            if snap is None:
                if stop.wait(interval):
                    return
                continue
            if not _is_edge(snap):
                return
        pp = pick_progress(snap if isinstance(snap, dict) else None, model)
        if pp is not None and pp.phase in {'decode', 'done', 'error'}:
            return
        if pp is not None and pp.fraction > 0:
            key = round(pp.fraction, 3)
            if key != last:
                last = key
                box.put(('progress', pp))
        if stop.wait(interval):
            return


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


def _row_progress(row: Any, model: str) -> PromptProgress | None:
    if not isinstance(row, dict):
        return None
    ident = str(row.get('id') or '')
    if model and ident and not _ids_match(ident, model):
        return None
    prompt = row.get('prompt') if isinstance(row.get('prompt'), dict) else {}
    ratio = prompt.get('ratio')
    if not isinstance(ratio, (int, float)):
        processed = prompt.get('processed_tokens')
        total = prompt.get('total_tokens')
        if isinstance(processed, (int, float)) and isinstance(total, (int, float)) and total:
            ratio = processed / float(total)
        else:
            return None
    frac = _clamp(float(ratio))
    processed = prompt.get('processed_tokens')
    total = prompt.get('total_tokens')
    return PromptProgress(
        fraction=frac,
        phase=str(row.get('phase') or 'prefill'),
        model=ident,
        processed=int(processed) if isinstance(processed, (int, float)) else None,
        total=int(total) if isinstance(total, (int, float)) else None,
    )


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
    return {'Authorization': f'Bearer {key}', 'Accept': 'application/json'}


def _with_model(url: str, model: str) -> str:
    if not model:
        return url
    parsed = urlparse(url)
    query = dict(urllib.parse.parse_qsl(parsed.query, keep_blank_values=True))
    query['model'] = model
    return urllib.parse.urlunparse(
        parsed._replace(query=urllib.parse.urlencode(query)),
    )


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
    return max(0.0, min(1.0, value))


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
