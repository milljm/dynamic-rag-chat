"""Patch model/server keys in ``.chat.yaml`` without flattening comments."""
from __future__ import annotations

import json
import re
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import yaml

# Canonical yaml names the Settings page reads/writes.
CORE_KEYS = ('llm_server', 'api_key', 'model', 'pre_llm', 'embedding_llm')

ROUTE_ROWS: tuple[tuple[str, str, str], ...] = (
    ('vision', 'vision_llm', 'vision_server'),
    ('agent', 'agent_llm', 'agent_server'),
    ('coder', 'coder_llm', 'coder_server'),
    ('casual', 'casual_llm', 'casual_server'),
    ('general', 'general_llm', 'general_server'),
    ('structured', 'structured_llm', 'structured_server'),
    ('nsfw', 'nsfw_llm', 'nsfw_server'),
    ('polisher', 'polisher_llm', 'polisher_server'),
    ('entity', 'entity_llm', 'entity_server'),
    ('rerank', 'rerank_llm', 'rerank_server'),
    ('pre', 'pre_llm', 'pre_server'),
    ('embedding', 'embedding_llm', 'embedding_server'),
)

# pre/embedding model live in CORE; only their *server* is extra here.
ROUTE_SERVER_KEYS = tuple(server for _, _llm, server in ROUTE_ROWS)
ROUTE_LLM_KEYS = tuple(
    llm for kind, llm, _s in ROUTE_ROWS if kind not in {'pre', 'embedding'}
)

ALL_KEYS = CORE_KEYS + ROUTE_LLM_KEYS + ROUTE_SERVER_KEYS + (
    'tavily_key', 'sd_server', 'sd_model',
)

_BLANK = frozenset({'', 'none', 'not_set', 'null', '~'})

_HOST_ALIASES = ('llm_server', 'model_server')


def blank(value: Any) -> str:
    """Empty / None / 'none' → ''."""
    if value is None:
        return ''
    text = str(value).strip()
    if text.lower() in _BLANK:
        return ''
    return text


def _dump_scalar(value: str) -> str:
    """YAML-safe scalar for a single-line value."""
    if not value:
        return ''
    dumped = yaml.safe_dump(
        value, default_flow_style=True, allow_unicode=True,
    ).strip()
    if dumped.endswith('\n...'):
        dumped = dumped[:-4].strip()
    if dumped.endswith('...'):
        dumped = dumped[:-3].strip()
    return dumped


def _line_pattern(key: str) -> re.Pattern[str]:
    return re.compile(
        rf'^([ \t]*{re.escape(key)}[ \t]*:)[ \t]*([^#\n]*?)([ \t]*)(#.*)?[ \t]*$',
        re.MULTILINE,
    )


def upsert_key(text: str, key: str, value: str) -> str:
    """Set ``key: value`` in a YAML document, keeping an inline comment."""
    rendered = _dump_scalar(value)
    pattern = _line_pattern(key)

    def _sub(match: re.Match[str]) -> str:
        comment = match.group(4) or ''
        pad = match.group(3) if comment else ''
        if rendered and comment and not pad:
            pad = '    '
        space = ' ' if rendered else ''
        return f'{match.group(1)}{space}{rendered}{pad}{comment}'

    if pattern.search(text):
        return pattern.sub(_sub, text, count=1)
    indent = '  '
    new_line = f'{indent}{key}: {rendered}'.rstrip()
    chat = re.search(r'^chat:[ \t]*$', text, re.MULTILINE)
    if chat:
        insert_at = chat.end()
        return text[:insert_at] + '\n' + new_line + text[insert_at:]
    if text and not text.endswith('\n'):
        text += '\n'
    return text + f'chat:\n{new_line}\n'


def upsert_keys(text: str, values: dict[str, str]) -> str:
    """Apply many key updates. ``llm_server`` updates ``model_server`` if that is what the file uses."""
    body = text if text else 'chat:\n'
    host = values.get('llm_server', '')
    if 'llm_server' in values:
        if _line_pattern('model_server').search(body) and not _line_pattern(
            'llm_server',
        ).search(body):
            body = upsert_key(body, 'model_server', host)
        else:
            body = upsert_key(body, 'llm_server', host)
    for key, value in values.items():
        if key in {'llm_server', 'model_server'}:
            continue
        if key not in ALL_KEYS:
            continue
        body = upsert_key(body, key, value)
    return body


def read_values(text: str) -> dict[str, str]:
    """Return canonical setting keys from a yaml document (empty on parse fail)."""
    out = {key: '' for key in ALL_KEYS if key != 'model_server'}
    try:
        raw = yaml.safe_load(text) or {}
    except yaml.YAMLError:
        return out
    chat = raw.get('chat', raw) if isinstance(raw, dict) else {}
    if not isinstance(chat, dict):
        return out
    host = blank(chat.get('llm_server')) or blank(chat.get('model_server'))
    out['llm_server'] = host
    for key in out:
        if key == 'llm_server':
            continue
        out[key] = blank(chat.get(key))
    return out


def load_file(path: Path) -> tuple[dict[str, str], str]:
    """Read settings from disk. Missing file → empty values, empty text."""
    if not path.is_file():
        return read_values(''), ''
    text = path.read_text(encoding='utf-8')
    return read_values(text), text


def save_file(path: Path, values: dict[str, str]) -> None:
    """Create or patch ``.chat.yaml``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    existing = path.read_text(encoding='utf-8') if path.is_file() else 'chat:\n'
    path.write_text(upsert_keys(existing, values), encoding='utf-8')


def origin_of(host: str) -> str:
    """Scheme + host + port, stripping /v1 and /models suffixes."""
    base = (host or '').strip().rstrip('/')
    if not base:
        return ''
    for suffix in (
        '/api/v1/models', '/api/v0/models', '/v1/models', '/models', '/v1',
    ):
        if base.endswith(suffix):
            base = base[: -len(suffix)]
            break
    return base.rstrip('/')


def models_urls(host: str) -> list[str]:
    """Candidate OpenAI-compatible ``/models`` URLs for a typed host."""
    base = (host or '').strip().rstrip('/')
    if not base:
        return []
    if base.endswith('/models'):
        return [base]
    urls = [f'{base}/models']
    if not base.endswith('/v1'):
        urls.append(f'{base}/v1/models')
    return urls


def list_model_urls(host: str) -> list[str]:
    """LM Studio native first (loaded vs downloaded), then OpenAI ``/models``."""
    origin = origin_of(host)
    urls: list[str] = []
    if origin and 'api.openai.com' not in origin.lower():
        urls.extend((f'{origin}/api/v1/models', f'{origin}/api/v0/models'))
    urls.extend(models_urls(host))
    seen: list[str] = []
    for address in urls:
        if address not in seen:
            seen.append(address)
    return seen


def parse_models_payload(payload: Any, url: str = '') -> dict[str, Any]:
    """Normalize LM Studio v1/v0 and OpenAI ``/models`` payloads."""
    details: list[dict[str, Any]] = []
    source = 'openai'
    if isinstance(payload, dict) and isinstance(payload.get('models'), list):
        source = 'lmstudio-v1'
        for item in payload['models']:
            if not isinstance(item, dict):
                continue
            ident = item.get('key') or item.get('id') or item.get('display_name')
            if not ident:
                continue
            other = item.get('id') or item.get('display_name')
            if other:
                ident = prefer_model_id(str(ident), str(other))
            instances = item.get('loaded_instances')
            loaded = bool(instances) if isinstance(instances, list) else None
            details.append({'id': str(ident), 'loaded': loaded})
    elif isinstance(payload, dict) and isinstance(payload.get('data'), list):
        has_state = False
        for item in payload['data']:
            if isinstance(item, str) and item:
                details.append({'id': item, 'loaded': None})
                continue
            if not isinstance(item, dict):
                continue
            ident = item.get('id') or item.get('name') or ''
            if not ident:
                continue
            state = item.get('state')
            loaded = None
            if state is not None:
                has_state = True
                loaded = str(state).strip().lower() == 'loaded'
            details.append({'id': str(ident), 'loaded': loaded})
        if has_state:
            source = 'lmstudio-v0'
    elif isinstance(payload, list):
        for item in payload:
            if isinstance(item, str) and item:
                details.append({'id': item, 'loaded': None})
    details = dedupe_details(details)
    names = [row['id'] for row in details]
    loaded_ids = [row['id'] for row in details if row.get('loaded')]
    return {
        'ok': True,
        'error': None,
        'models': names,
        'details': details,
        'loaded': loaded_ids,
        'knows_loaded': any(row.get('loaded') is not None for row in details),
        'source': source,
        'url': url,
    }


def prefer_model_id(first: str, second: str) -> str:
    """Prefer mixed-case (OpenAI ``/v1/models`` spelling) over lowercase."""
    if not first:
        return second
    if not second:
        return first
    if first == second:
        return first
    first_folded = first == first.lower()
    second_folded = second == second.lower()
    if first_folded != second_folded:
        return first if second_folded else second
    first_up = sum(1 for ch in first if 'A' <= ch <= 'Z')
    second_up = sum(1 for ch in second if 'A' <= ch <= 'Z')
    return second if second_up > first_up else first


def _merge_loaded(first: Any, second: Any) -> Any:
    if first is True or second is True:
        return True
    if first is False or second is False:
        return False
    return first if first is not None else second


def dedupe_details(details: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Collapse case-insensitive duplicate model ids, keeping mixed-case."""
    by_key: dict[str, dict[str, Any]] = {}
    order: list[str] = []
    for row in details:
        ident = str(row.get('id') or '').strip()
        if not ident:
            continue
        key = ident.lower()
        prev = by_key.get(key)
        if prev is None:
            by_key[key] = {'id': ident, 'loaded': row.get('loaded')}
            order.append(key)
            continue
        prev['id'] = prefer_model_id(prev['id'], ident)
        prev['loaded'] = _merge_loaded(prev.get('loaded'), row.get('loaded'))
    return [by_key[key] for key in order]


def overlay_openai_ids(native: dict[str, Any], openai: dict[str, Any]) -> dict[str, Any]:
    """Rewrite LM Studio keys with ``/v1/models`` spelling when they match."""
    by_key: dict[str, dict[str, Any]] = {}
    order: list[str] = []
    for row in native.get('details') or []:
        ident = str(row.get('id') or '').strip()
        if not ident:
            continue
        key = ident.lower()
        if key not in by_key:
            order.append(key)
        by_key[key] = {'id': ident, 'loaded': row.get('loaded')}
    for ident in openai.get('models') or []:
        text = str(ident).strip()
        if not text:
            continue
        key = text.lower()
        if key in by_key:
            by_key[key]['id'] = text
        else:
            by_key[key] = {'id': text, 'loaded': None}
            order.append(key)
    details = [by_key[key] for key in order]
    names = [row['id'] for row in details]
    loaded_ids = [row['id'] for row in details if row.get('loaded')]
    merged = dict(native)
    merged['models'] = names
    merged['details'] = details
    merged['loaded'] = loaded_ids
    return merged


def list_models(host: str, api_key: str = 'none', timeout: float = 3.0) -> dict[str, Any]:
    """List models; prefer LM Studio native so we can mark what is loaded."""
    urls = list_model_urls(host)
    if not urls:
        return {
            'ok': False, 'error': 'Server URL is empty.', 'models': [],
            'details': [], 'loaded': [], 'knows_loaded': False,
        }
    headers = {'Authorization': f'Bearer {blank(api_key) or "none"}'}
    last_error = 'No response'
    native = None
    for address in urls:
        try:
            req = urllib.request.Request(address, headers=headers, method='GET')
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                body = resp.read().decode('utf-8', errors='replace')
            payload = json.loads(body)
            parsed = parse_models_payload(payload, address)
            if not parsed['models']:
                last_error = f'{address} returned no models'
                continue
            if str(parsed.get('source') or '').startswith('lmstudio'):
                native = parsed
                continue
            if native is not None:
                return overlay_openai_ids(native, parsed)
            return parsed
        except urllib.error.HTTPError as exc:
            last_error = f'{exc.code} {exc.reason}'
        except urllib.error.URLError as exc:
            last_error = str(getattr(exc, 'reason', exc))
        except (TimeoutError, json.JSONDecodeError, OSError) as exc:
            last_error = str(exc)
    if native is not None:
        return native
    return {
        'ok': False, 'error': last_error, 'models': [],
        'details': [], 'loaded': [], 'knows_loaded': False,
    }
