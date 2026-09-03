"""Optional cross-encoder call: POST /v1/rerank (Cohere / Edge)."""
from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any


def configured(opts: Any) -> bool:
    """True when a rerank model + host are set (host may inherit llm_server)."""
    model = str(getattr(opts, 'rerank_llm', '') or '').strip()
    host = str(getattr(opts, 'rerank_host', '') or '').strip()
    if not host:
        return False
    return model.lower() not in {'', 'none', 'not_set', 'null', '~'}


def rerank_url(host: str) -> str:
    """``http://edge/v1`` → ``/rerank``; bare host → ``/v1/rerank``."""
    base = (host or '').strip().rstrip('/')
    if not base:
        return ''
    if base.endswith('/v1'):
        return base + '/rerank'
    return base + '/v1/rerank'


def _score_map(payload: dict) -> dict[int, float]:
    rows = payload.get('results') or payload.get('data') or []
    out: dict[int, float] = {}
    if not isinstance(rows, list):
        return out
    for row in rows:
        if not isinstance(row, dict):
            continue
        try:
            idx = int(row.get('index'))
        except (TypeError, ValueError):
            continue
        score = row.get('relevance_score', row.get('score'))
        try:
            out[idx] = float(score)
        except (TypeError, ValueError):
            out[idx] = 0.0
    return out


def post_rerank(
    host: str,
    model: str,
    query: str,
    documents: list[str],
    top_n: int,
    api_key: str = 'none',
    timeout: float = 8.0,
) -> list[int] | None:
    """Return document indices, best first. None on transport/parse failure."""
    url = rerank_url(host)
    if not url or not documents:
        return None
    keep = max(1, min(int(top_n), len(documents)))
    body = json.dumps({
        'model': model,
        'query': query,
        'documents': documents,
        'top_n': keep,
    }).encode('utf-8')
    headers = {'Content-Type': 'application/json'}
    key = (api_key or 'none').strip() or 'none'
    headers['Authorization'] = f'Bearer {key}'
    req = urllib.request.Request(url, data=body, headers=headers, method='POST')
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            payload = json.loads(resp.read().decode('utf-8'))
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError):
        return None
    if not isinstance(payload, dict):
        return None
    scores = _score_map(payload)
    if not scores:
        return None
    ranked = sorted(scores, key=lambda i: scores[i], reverse=True)
    return [i for i in ranked if 0 <= i < len(documents)]


def reorder(documents: list, indices: list[int], keep: int) -> list:
    """Apply rerank indices; drop unknowns; cap at keep."""
    seen = set()
    out = []
    for idx in indices:
        if idx in seen or idx < 0 or idx >= len(documents):
            continue
        seen.add(idx)
        out.append(documents[idx])
        if len(out) >= keep:
            return out
    return out[:keep]
