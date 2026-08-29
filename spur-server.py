#!/usr/bin/env python3
"""
Spur adapter — drop this next to chat.py.

The React UI is only a view. This process owns Chat / SessionContext /
RenderWindow: branches, JSON history, RAG clone/reset, agent tools,
prepare_turn, stream_response, save_history. LM Studio (or whatever is
in .chat.yaml) is reached the same way the terminal app already does.

  ./chat.py --spur

One process: this adapter on :8765 also serves the built UI from
spur/dist/client. First run builds the UI (needs Node). Rebuild with
./chat.py --spur --spur-rebuild.

Split-dev (optional): python spur-server.py   then
  VITE_CHAT_API=http://127.0.0.1:8765 npm run dev   in spur/
"""
from __future__ import annotations

import asyncio
import base64
import glob
import json
import mimetypes
import os
import queue
import shutil
import sys
import threading
import time
import uuid
from copy import deepcopy
from pathlib import Path
from typing import Any, AsyncIterator, Iterator

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from rich.console import Console

ROOT = os.path.dirname(os.path.abspath(__file__))
if os.path.basename(ROOT) == 'public':
    ROOT = os.path.dirname(ROOT)
sys.path.insert(0, ROOT)

from chat import (  # noqa: E402
    Chat,
    ChatOptions,
    SessionContext,
    parse_args,
    parse_user_input,
    seed_from_string,
)
from src.think_tags import ThinkFeed  # noqa: E402
from src.gold_fetch import GoldNeedFeed, MAX_GOLD_FETCHES, recall_status  # noqa: E402
from src.attachment_store import list_attachments  # noqa: E402
from src.chat_utils import (  # noqa: E402
    HISTORY_META_KEYS,
    CommonUtils,
    load_history_from_dir,
)
from src.settings_yaml import (  # noqa: E402
    ALL_KEYS,
    blank,
    list_models,
    load_file as load_settings_file,
    save_file as save_settings_file,
)

LOCKED_BRANCHES = frozenset({'assistant', 'story'})
# Metadata keys in the history file — not message lists.
RESERVED_NAMES = frozenset(
    {'current', 'assistant', 'story', 'assistant_mode', 'branch_modes', 'version'}
)

app = FastAPI(title='Spur')
app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_methods=['*'],
    allow_headers=['*'],
)

console = Console(highlight=True)
_chat: Chat | None = None
_stream_lock = threading.Lock()
_streams = 0
YAML_PATH = Path(ROOT) / '.chat.yaml'


def get_chat() -> Chat:
    global _chat
    if _chat is None:
        opts = ChatOptions.from_yaml(ROOT)
        args = parse_args(sys.argv[1:], opts)
        built = ChatOptions.from_args(ROOT, args, opts)
        built.seed = seed_from_string(str(built.seed))
        session = SessionContext.from_args(console, built)
        _chat = Chat(session, built)
        _sync_chat_object(_chat)
    return _chat


def _opts_snapshot(opts: ChatOptions) -> dict[str, str]:
    """Effective running values (after inherit)."""
    return {
        'llm_server': blank(opts.host),
        'api_key': blank(opts.api_key) or 'none',
        'model': blank(opts.model),
        'pre_llm': blank(opts.preconditioner),
        'embedding_llm': blank(opts.embeddings),
        'pre_server': blank(opts.pre_host),
        'embedding_server': blank(opts.emb_host),
        'vision_llm': blank(opts.vision_llm),
        'vision_server': blank(opts.vision_host),
        'agent_llm': blank(opts.agent_llm),
        'agent_server': blank(opts.agent_host),
        'coder_llm': blank(opts.coder_llm),
        'coder_server': blank(opts.coder_host),
        'casual_llm': blank(opts.casual_llm),
        'casual_server': blank(opts.casual_host),
        'general_llm': blank(opts.general_llm),
        'general_server': blank(opts.general_host),
        'structured_llm': blank(opts.structured_llm),
        'structured_server': blank(opts.structured_host),
        'nsfw_llm': blank(opts.nsfw_llm),
        'nsfw_server': blank(opts.nsfw_host),
        'polisher_llm': blank(opts.polisher_llm),
        'polisher_server': blank(opts.polisher_host),
        'entity_llm': blank(opts.entity_llm),
        'entity_server': blank(opts.entity_host),
        'tavily_key': blank(opts.tavily_key),
    }


def _rebuild_chat_from_yaml() -> None:
    """Replace the live Chat after ``.chat.yaml`` changes. Next turn uses new models."""
    global _chat
    prev = _chat
    opts = ChatOptions.from_yaml(ROOT)
    if prev is not None:
        opts.assistant_mode = bool(prev.opts.assistant_mode)
        opts.vector_dir = prev.opts.vector_dir
        opts.verbose = bool(prev.opts.verbose)
        opts.debug = bool(prev.opts.debug)
        opts.no_rags = bool(prev.opts.no_rags)
        opts.light_mode = bool(prev.opts.light_mode)
        opts.disable_thinking = bool(prev.opts.disable_thinking)
        seed_src = prev.opts.seed
    else:
        seed_src = opts.seed
    if isinstance(seed_src, int):
        opts.seed = seed_src
    else:
        opts.seed = seed_from_string(str(seed_src or ''))
    session = SessionContext.from_args(console, opts)
    new_chat = Chat(session, opts)
    _sync_chat_object(new_chat)
    _chat = new_chat


def sse(obj: dict[str, Any]) -> str:
    # Escape < > so a proxy/browser that HTML-parses the SSE body cannot
    # treat a MiniMax inner <think> token as a real tag and swallow the rest
    # of the stream.
    payload = json.dumps(obj).replace('<', '\\u003c').replace('>', '\\u003e')
    return f'data: {payload}\n\n'


def _status_sse(
    message: str, model: str = '', route: str = '', context: int = 0,
    recalled: list[str] | None = None,
) -> bytes:
    """status event; attach recalled names so Spur can badge the turn."""
    payload: dict[str, Any] = {
        'type': 'status',
        'message': message,
        'model': model or '',
        'route': route or '',
        'context': context or 0,
    }
    if recalled:
        payload['recalled'] = list(recalled)
    return sse(payload).encode()


def _history(chat: Chat) -> dict:
    hist = chat.session.common.load_chat()
    if isinstance(hist, dict):
        return hist
    return chat.session.common.empty_chat_history()


def _vector_dir() -> str:
    if _chat is not None:
        return _chat.opts.vector_dir
    opts = ChatOptions.from_yaml(ROOT)
    vdir = opts.vector_dir
    if not os.path.isabs(vdir):
        vdir = os.path.join(ROOT, vdir)
    return vdir


def read_hist() -> dict:
    """Load JSON (or legacy pickle) without constructing Chat/Chroma."""
    loaded = load_history_from_dir(_vector_dir(), migrate=True)
    if isinstance(loaded, dict):
        return loaded
    if _chat is not None:
        return _history(_chat)
    return {
        'story': [],
        'assistant': [],
        'current': 'story',
        'branch_modes': {},
        'assistant_mode': False,
        'version': 1,
    }


def _canonical_mode(name: str) -> bool | None:
    if name == 'assistant':
        return True
    if name == 'story':
        return False
    return None


def _persisted_mode(hist: dict, name: str, fallback: bool = False) -> bool:
    canon = _canonical_mode(name)
    if canon is not None:
        return canon
    return bool(hist.get('branch_modes', {}).get(name, fallback))


def _turn_count(msgs: list) -> int:
    if not msgs:
        return 0
    return sum(1 for m in msgs if isinstance(m, dict) and m.get('role') == 'user')


def _refresh_mode_runtime(chat: Chat) -> None:
    mode = bool(chat.opts.assistant_mode)
    renderer = chat.session.renderer
    context = chat.session.context
    for holder in (
        getattr(renderer, 'opts', None),
        getattr(renderer, 'args', None),
        getattr(renderer, 'state', None),
        getattr(getattr(renderer, 'prompts', None), 'args', None),
        getattr(context, 'opts', None),
        getattr(context, 'args', None),
        getattr(getattr(context, 'prompts', None), 'args', None),
    ):
        if holder is not None and hasattr(holder, 'assistant_mode'):
            holder.assistant_mode = mode
    if hasattr(renderer, 'state'):
        renderer.state.assistant_mode = mode
    renderer.assistant_prompt = mode
    context.assistant_prompt = mode
    context.mode = 'document_topics' if mode else 'entity'
    if hasattr(renderer, 'prompts') and hasattr(renderer.prompts, 'build_prompts'):
        renderer.prompts.prompt_model = chat.opts.model
        renderer.prompts.build_prompts()
    if hasattr(renderer, 'build_prompts'):
        renderer.prompt_model = chat.opts.model
        renderer.build_prompts()
    if hasattr(context, 'prompts') and hasattr(context.prompts, 'build_prompts'):
        context.prompts.prompt_model = getattr(
            chat.opts, 'preconditioner', chat.opts.model
        )
        context.prompts.build_prompts()


def _sync_chat_object(chat: Chat, hist: dict | None = None) -> tuple[str, bool]:
    hist = hist or _history(chat)
    branch = hist.get('current', 'story')
    mode = _persisted_mode(
        hist, branch, fallback=bool(hist.get('assistant_mode', False))
    )
    chat.chat_branch = branch
    chat.opts.assistant_mode = mode
    hist['assistant_mode'] = mode
    hist.setdefault('branch_modes', {})
    if _canonical_mode(branch) is None:
        hist['branch_modes'][branch] = mode
    chat.session.common.save_chat(hist)
    _refresh_mode_runtime(chat)
    return branch, mode


def _message_text(value: Any) -> str:
    if value is None:
        return ''
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts: list[str] = []
        for item in value:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                parts.append(
                    _message_text(item.get('text') or item.get('content') or '')
                )
        return ''.join(parts)
    return str(value)


def _mode_label(assistant: bool) -> str:
    return 'assistant' if assistant else 'story'


def session_payload(chat: Chat | None = None) -> dict[str, Any]:
    hist = _history(chat) if chat is not None else read_hist()
    if not isinstance(hist, dict):
        hist = {
            'story': [],
            'assistant': [],
            'current': 'story',
            'branch_modes': {},
            'assistant_mode': False,
        }
    if chat is not None:
        current, mode = _sync_chat_object(chat, hist)
    else:
        current = hist.get('current', 'story')
        mode = _persisted_mode(
            hist, current, fallback=bool(hist.get('assistant_mode', False))
        )
    branches: dict[str, Any] = {}
    for name, msgs in hist.items():
        if name in HISTORY_META_KEYS or not isinstance(msgs, list):
            continue
        locked = name in LOCKED_BRANCHES
        assistant = _persisted_mode(hist, name, fallback=mode)
        clean = []
        for m in msgs:
            if isinstance(m, dict):
                row = {
                    'role': 'user' if m.get('role') == 'user' else 'assistant',
                    'content': _message_text(m.get('content')),
                }
                if m.get('reasoning'):
                    row['reasoning'] = m['reasoning']
                if m.get('metrics'):
                    row['metrics'] = m['metrics']
                if m.get('attachments'):
                    row['attachments'] = _hydrate_attachments(
                        _vector_dir(), m['attachments']
                    )
                clean.append(row)
            elif isinstance(m, str) and m.strip():
                clean.append({'role': 'assistant', 'content': m})
        branches[name] = {
            'id': name,
            'name': name,
            'mode': _mode_label(assistant),
            'locked': locked,
            'messages': clean,
        }
    if 'story' not in branches:
        branches['story'] = {
            'id': 'story',
            'name': 'story',
            'mode': 'story',
            'locked': True,
            'messages': [],
        }
    if 'assistant' not in branches:
        branches['assistant'] = {
            'id': 'assistant',
            'name': 'assistant',
            'mode': 'assistant',
            'locked': True,
            'messages': [],
        }
    if current not in branches:
        current = 'story'
    return {'currentId': current, 'branches': branches}


def switch_branch(chat: Chat, name: str) -> tuple[bool, str]:
    hist = _history(chat)
    if not isinstance(hist.get(name), list):
        return False, f"Branch '{name}' does not exist."
    old = hist.get('current', 'story')
    if name == old:
        return True, f"Already on '{name}'."
    new_mode = _persisted_mode(hist, name, fallback=bool(chat.opts.assistant_mode))
    hist['current'] = name
    hist['assistant_mode'] = new_mode
    if _canonical_mode(name) is None:
        hist.setdefault('branch_modes', {})[name] = new_mode
    chat.session.common.save_chat(hist)
    chat.chat_branch = name
    chat.opts.assistant_mode = new_mode
    if hasattr(chat.session.renderer, 'clear_ooc'):
        chat.session.renderer.clear_ooc()
    _refresh_mode_runtime(chat)
    return True, f"Switched to '{name}'."


def create_branch(chat: Chat, name: str, cut_turns: int | None) -> tuple[bool, str]:
    name = (name or '').strip()
    if not name:
        return False, 'Branch name cannot be empty.'
    if name in RESERVED_NAMES:
        return False, f"'{name}' is reserved."
    hist = _history(chat)
    src = hist.get('current', 'story')
    base = hist.get(src, [])
    if not isinstance(base, list):
        return False, f"Source branch '{src}' is invalid."
    if name in hist:
        return switch_branch(chat, name)
    if cut_turns is not None:
        cut_idx = max(0, min(int(cut_turns) * 2, len(base)))
        new_list = deepcopy(base[:cut_idx])
    else:
        new_list = deepcopy(base)
    source_mode = _persisted_mode(hist, src, fallback=bool(chat.opts.assistant_mode))
    hist[name] = new_list
    hist['current'] = name
    hist.setdefault('branch_modes', {})[name] = source_mode
    hist['assistant_mode'] = source_mode
    try:
        chat.session.common.save_chat(hist)
        if hasattr(chat.session.renderer, 'clear_ooc'):
            chat.session.renderer.clear_ooc()
        if cut_turns is None:
            chat.session.rag.clone_collection(src, name, overwrite=False)
        elif hasattr(chat.session.rag, 'build_collection_from_texts'):
            texts = [
                m.get('content', '') if isinstance(m, dict) else str(m)
                for m in new_list
            ]
            chat.session.rag.build_collection_from_texts(name, texts, overwrite=True)
        chat.chat_branch = name
        chat.opts.assistant_mode = source_mode
        _refresh_mode_runtime(chat)
        return True, f"Branched to '{name}'."
    except Exception as exc:  # pylint: disable=broad-exception-caught
        hist.pop(name, None)
        hist['current'] = src
        hist.setdefault('branch_modes', {}).pop(name, None)
        chat.session.common.save_chat(hist)
        chat.chat_branch = src
        return False, f'RAG sync failed: {exc}'


def delete_branch(chat: Chat, name: str) -> tuple[bool, str]:
    if name in RESERVED_NAMES or name in LOCKED_BRANCHES:
        return False, f"Cannot delete protected branch '{name}'."
    hist = _history(chat)
    if hist.get('current') == name:
        return False, 'Cannot delete the branch you are on.'
    if name not in hist or not isinstance(hist.get(name), list):
        return False, f"Unknown branch '{name}'."
    hist.pop(name, None)
    hist.get('branch_modes', {}).pop(name, None)
    chat.session.common.save_chat(hist)
    if hasattr(chat.session.rag, 'delete_collection'):
        chat.session.rag.delete_collection(name)
    vector_dir = getattr(chat.opts, 'vector_dir', '')
    if vector_dir:
        for path in glob.glob(f'{vector_dir}{os.path.sep}{name}*'):
            if os.path.isdir(path):
                shutil.rmtree(path, ignore_errors=True)
    return True, f"Deleted '{name}'."


def reset_branch(chat: Chat) -> tuple[bool, str]:
    hist = _history(chat)
    branch = hist.get('current', 'story')
    if hasattr(chat.session.rag, 'delete_collection'):
        chat.session.rag.delete_collection(branch)
    hist[branch] = []
    chat.session.common.save_chat(hist)
    vector_dir = getattr(chat.opts, 'vector_dir', '')
    if vector_dir:
        for path in glob.glob(f'{vector_dir}{os.path.sep}{branch}*'):
            if os.path.isdir(path):
                shutil.rmtree(path, ignore_errors=True)
    if hasattr(chat.session.renderer, 'clear_ooc'):
        chat.session.renderer.clear_ooc()
    return True, f"Reset '{branch}'."


def delete_last_turn(chat: Chat) -> tuple[bool, str]:
    hist = _history(chat)
    branch = hist.get('current', 'story')
    msgs = hist.get(branch, [])
    if not msgs:
        return False, 'History empty.'
    if msgs and isinstance(msgs[-1], dict) and msgs[-1].get('role') == 'assistant':
        msgs.pop()
    if msgs and msgs[-1].get('role') == 'user':
        msgs.pop()
    hist[branch] = msgs
    chat.session.common.save_chat(hist)
    if hasattr(chat.session.renderer, 'clear_ooc'):
        chat.session.renderer.clear_ooc()
    return True, 'Deleted last turn.'


def pop_last_assistant(chat: Chat) -> tuple[bool, str]:
    hist = _history(chat)
    branch = hist.get('current', 'story')
    msgs = hist.get(branch, []) or []
    if msgs and msgs[-1].get('role') == 'assistant':
        msgs.pop()
        hist[branch] = msgs
        chat.session.common.save_chat(hist)
    if hasattr(chat.session.renderer, 'clear_ooc'):
        chat.session.renderer.clear_ooc()
    return True, 'Popped last assistant.'


def rewind_to(chat: Chat, n: int) -> tuple[bool, str]:
    hist = _history(chat)
    branch = hist.get('current', 'story')
    msgs = hist.get(branch, [])
    total = _turn_count(msgs)
    if not 1 <= n <= total:
        return False, f'Rewind needs 1 ≤ N ≤ {total}.'
    hist[branch] = msgs[: n * 2]
    chat.session.common.save_chat(hist)
    if hasattr(chat.session.renderer, 'clear_ooc'):
        chat.session.renderer.clear_ooc()
    return True, f'Rewound to turn {n} of {total}.'


def set_assistant_mode(chat: Chat, enabled: bool) -> tuple[bool, str]:
    hist = _history(chat)
    current = hist.get('current', 'story')
    canon = _canonical_mode(current)
    target = canon if canon is not None else bool(enabled)
    if canon is None:
        hist.setdefault('branch_modes', {})[current] = target
    hist['assistant_mode'] = target
    chat.opts.assistant_mode = target
    chat.chat_branch = current
    chat.session.common.save_chat(hist)
    _refresh_mode_runtime(chat)
    return True, _mode_label(target)


def _slim_attachments(vector_dir: str, attachments: list | None) -> list[dict]:
    """Never store dataUrls in history. Images live under vector_dir/uploads/."""
    if not attachments:
        return []
    upload_dir = os.path.join(vector_dir, 'uploads')
    os.makedirs(upload_dir, exist_ok=True)
    slim: list[dict] = []
    for raw in attachments:
        if not isinstance(raw, dict):
            continue
        rec = {
            k: raw[k]
            for k in ('id', 'name', 'mime', 'kind', 'size')
            if raw.get(k) is not None
        }
        rec.setdefault('id', uuid.uuid4().hex)
        rec.setdefault('kind', 'image' if raw.get('dataUrl') else 'text')
        if rec.get('kind') == 'text' and raw.get('text'):
            rec['text'] = str(raw['text'])[:100_000]
        data = raw.get('dataUrl')
        if rec.get('kind') == 'image' and isinstance(data, str) and data:
            payload = data.split(',', 1)[1] if data.startswith('data:') and ',' in data else data
            try:
                blob = base64.b64decode(payload)
            except Exception:  # pylint: disable=broad-exception-caught
                blob = b''
            if blob:
                mime = str(rec.get('mime') or 'image/png')
                ext = mimetypes.guess_extension(mime.split(';')[0].strip()) or '.png'
                if ext == '.jpe':
                    ext = '.jpg'
                fname = f"{rec['id']}{ext}"
                with open(os.path.join(upload_dir, fname), 'wb') as handle:
                    handle.write(blob)
                rec['file'] = f'uploads/{fname}'
        slim.append(rec)
    return slim


def _hydrate_attachments(vector_dir: str, attachments: list | None) -> list:
    """Re-attach dataUrls from uploads/ so Spur can render them."""
    if not isinstance(attachments, list):
        return []
    out = []
    for rec in attachments:
        if not isinstance(rec, dict):
            continue
        row = dict(rec)
        row.pop('dataUrl', None)
        rel = row.get('file')
        if rel:
            path = os.path.join(vector_dir, rel) if not os.path.isabs(str(rel)) else str(rel)
            try:
                if os.path.isfile(path) and os.path.getsize(path) <= 8_000_000:
                    mime = str(row.get('mime') or 'image/png')
                    b64 = base64.b64encode(open(path, 'rb').read()).decode('ascii')
                    row['dataUrl'] = f'data:{mime};base64,{b64}'
            except Exception:  # pylint: disable=broad-exception-caught
                pass
        out.append(row)
    return out


def persist_turn(
    renderer,
    documents: dict,
    response: str,
    reasoning: str = '',
    metrics: dict | None = None,
    attachments: list | None = None,
) -> None:
    documents['llm_response'] = response
    renderer.save_history(documents, response, reasoning=reasoning)
    common = renderer.common
    hist = common.load_chat()
    if not isinstance(hist, dict):
        hist = common.empty_chat_history()
    branch = hist.get('current', 'story')
    msgs = hist.get(branch) or []
    if not isinstance(msgs, list):
        msgs = []
        hist[branch] = msgs
    if (
        len(msgs) >= 3
        and isinstance(msgs[-1], dict)
        and isinstance(msgs[-2], dict)
        and isinstance(msgs[-3], dict)
        and msgs[-1].get('role') == 'assistant'
        and msgs[-2].get('role') == 'user'
        and msgs[-3].get('role') == 'user'
    ):
        msgs.pop(-2)
        hist[branch] = msgs
        common.save_chat(hist)
    if attachments:
        slim = _slim_attachments(_vector_dir(), attachments)
        for m in reversed(msgs):
            if isinstance(m, dict) and m.get('role') == 'user':
                m['attachments'] = slim
                break
    extra = {}
    if reasoning and reasoning.strip():
        extra['reasoning'] = reasoning
    if metrics:
        extra['metrics'] = metrics
    if msgs and msgs[-1].get('role') == 'assistant':
        # save_history runs sanitize_response which strips ``` fences.
        # Restore the streamed answer so reload still highlights.
        msgs[-1]['content'] = response
        msgs[-1].update(extra)
    common.save_chat(hist)


def apply_includes(chat: Chat, documents: dict, raw: str) -> dict:
    try:
        parsed = parse_user_input(raw)
    except Exception:  # pylint: disable=broad-exception-caught
        return documents
    if not parsed.includes or not hasattr(chat, 'load_content_as_context'):
        return documents
    wrapped = ' '.join(f'{{{{{item}}}}}' for item in parsed.includes)
    try:
        extra = chat.load_content_as_context(wrapped)
    except Exception:  # pylint: disable=broad-exception-caught
        return documents
    documents.setdefault('dynamic_files', '')
    documents.setdefault('dynamic_images', [])
    if extra.get('dynamic_files'):
        documents['dynamic_files'] += extra['dynamic_files']
    if extra.get('dynamic_images'):
        documents['dynamic_images'].extend(extra['dynamic_images'])
    return documents


def fold_uploads(documents: dict, images: list, files: list,
                 attachments: list | None = None) -> dict:
    """Pixels from `images` or image attachments; text files into dynamic_files."""
    documents.setdefault('dynamic_images', [])
    documents.setdefault('dynamic_files', '')
    seen: set[str] = set()
    extras: list = list(images or [])
    for att in attachments or []:
        if not isinstance(att, dict):
            continue
        kind = str(att.get('kind') or '')
        mime = str(att.get('mime') or '')
        if kind != 'image' and not mime.startswith('image/'):
            continue
        url = att.get('dataUrl') or att.get('data_url')
        if isinstance(url, str) and url.strip():
            extras.append(url)
    for img in extras:
        raw = img
        if not isinstance(raw, str) or not raw.strip():
            continue
        raw = raw.strip()
        key = raw[:80] + str(len(raw))
        if key in seen:
            continue
        seen.add(key)
        documents['dynamic_images'].append(raw)
    for item in files or []:
        name = item.get('name', 'file')
        text = item.get('text', '')
        documents['dynamic_files'] += f'\n--- {name} ---\n\n{text}\n\n'
        CommonUtils.record_attachment(documents, name, text=str(text), kind='text')
    return documents


def include_branch(chat: Chat, documents: dict, name: str) -> dict:
    hist = _history(chat)
    msgs = hist.get(name)
    if not isinstance(msgs, list):
        return documents
    lines = []
    for m in msgs:
        if not isinstance(m, dict):
            continue
        role = 'USER' if m.get('role') == 'user' else 'AI'
        lines.append(f"{role}: {m.get('content', '')}")
    documents.setdefault('dynamic_files', '')
    documents['dynamic_files'] += f'\n=== include:{name} ===\n' + '\n'.join(lines)
    return documents


def _op(fn, *args) -> JSONResponse:
    ok, msg = fn(*args)
    return JSONResponse({'ok': ok, 'error': None if ok else msg, 'message': msg})


@app.get('/api/session')
def api_session() -> dict[str, Any]:
    # JSON history — do not construct Chat/Chroma just to list branches.
    return session_payload(_chat)


@app.post('/api/branches/switch')
async def api_switch(request: Request) -> JSONResponse:
    body = await request.json()
    return _op(switch_branch, get_chat(), str(body.get('id') or ''))


@app.post('/api/branches')
async def api_create(request: Request) -> JSONResponse:
    body = await request.json()
    cut = body.get('cutTurns')
    cut_n = int(cut) if cut is not None else None
    return _op(create_branch, get_chat(), str(body.get('name') or ''), cut_n)


@app.post('/api/branches/delete')
async def api_delete(request: Request) -> JSONResponse:
    body = await request.json()
    return _op(delete_branch, get_chat(), str(body.get('id') or ''))


@app.post('/api/history/reset')
def api_reset() -> JSONResponse:
    """Clear history and RAG for the current branch."""
    return _op(reset_branch, get_chat())


@app.post('/api/history/delete-last')
def api_delete_last() -> JSONResponse:
    """Delete the last turn on the current branch."""
    return _op(delete_last_turn, get_chat())


@app.post('/api/history/rewind')
async def api_rewind(request: Request) -> JSONResponse:
    """Rewind the current branch to turn N."""
    body = await request.json()
    return _op(rewind_to, get_chat(), int(body.get('n') or 0))


@app.post('/api/history/pop-assistant')
def api_pop() -> JSONResponse:
    """Drop the last assistant message (for regenerate)."""
    return _op(pop_last_assistant, get_chat())


@app.post('/api/session/mode')
async def api_mode(request: Request) -> JSONResponse:
    """Switch story vs assistant mode."""
    body = await request.json()
    enabled = str(body.get('mode') or '') == 'assistant'
    return _op(set_assistant_mode, get_chat(), enabled)


@app.get('/api/documents')
def api_documents() -> dict[str, Any]:
    """Whole files in vector_dir/attachments."""
    return {'documents': list_attachments(_vector_dir())}


@app.post('/api/documents/delete')
async def api_documents_delete(request: Request) -> JSONResponse:
    """Remove a named file from the cabinet and gold chunks."""
    body = await request.json()
    name = str(body.get('name') or '')
    if not name:
        return JSONResponse({'ok': False, 'error': 'Missing name', 'message': 'Missing name'})
    chat = get_chat()
    ok = chat.session.context.delete_gold_file(name)
    if not ok:
        return JSONResponse({
            'ok': False, 'error': f'Could not delete {name}', 'message': f'Could not delete {name}',
        })
    return JSONResponse({'ok': True, 'error': None, 'message': f'Deleted {name}'})


def _prepare_chat_documents(chat, body: dict) -> tuple[dict, list]:
    """Run prepare_turn / no-context and stamp uploads, includes, agent flags."""
    prompt = str(body.get('text') or '')
    parsed = parse_user_input(prompt)
    if body.get('noContext'):
        documents = chat.no_context(parsed.args or parsed.clean_text or prompt)
        documents['no_context'] = True
        documents['in_line_commands'] = 'Meta: [no-context]'
        meta: list = []
    else:
        atts = body.get('attachments') or []
        has_text = any(
            isinstance(a, dict) and a.get('kind') == 'text' for a in atts
        )
        has_image = bool(body.get('images')) or any(
            isinstance(a, dict) and (
                a.get('kind') == 'image'
                or str(a.get('mime') or '').startswith('image/')
            )
            for a in atts
        )
        documents, meta = chat.prepare_turn(
            parsed.args or parsed.clean_text or prompt,
            extras={
                'has_images': has_image,
                'has_files': bool(body.get('files') or has_text),
            },
        )
        documents = apply_includes(chat, documents, prompt)
    documents = fold_uploads(
        documents,
        body.get('images') or [],
        body.get('files') or [],
        body.get('attachments') or [],
    )
    # Image attachments: stub in gold (pixels stay this-turn for vision).
    for att in body.get('attachments') or []:
        if not isinstance(att, dict):
            continue
        if att.get('kind') == 'image' or str(att.get('mime') or '').startswith('image/'):
            CommonUtils.record_attachment(
                documents, str(att.get('name') or 'image'), kind='image',
            )
    if not documents.get('no_context'):
        chat.session.context.ingest_user_attachments(documents, meta)
    if body.get('includeBranch'):
        documents = include_branch(chat, documents, str(body['includeBranch']))
    force_agent = bool(body.get('useAgent')) or parsed.command == 'agent'
    if force_agent and chat.opts.assistant_mode:
        documents['use_agent'] = True
        documents['agent_ran'] = False
        documents['in_line_commands'] = 'Meta: [agent]'
    if body.get('rare'):
        documents['system_addendum'] = (
            'Story controls for this turn: ' + ', '.join(body['rare'])
        )
    return documents, meta


def _reset_renderer_think(renderer) -> None:
    """Clear ThinkFeed latches so this turn can reason again."""
    stream_state = getattr(getattr(renderer, 'state', None), 'stream', None)
    if stream_state is None:
        return
    stream_state.never_think = False
    stream_state.shadow_think = False
    stream_state.thinking = False
    if hasattr(stream_state, 'think_ns'):
        stream_state.think_ns = ''


def _iter_sse_chunks(
    renderer, packed, documents: dict, stats: dict,
    route: str = '', context: int = 0, meta=None,
) -> Iterator[bytes]:
    """Yield token/reasoning/status/usage SSE frames for one LLM stream.

    If the model emits <NEED_GOLD:file>, fetch that gold file and resume
    in this same turn (assistant mode, capped).
    """
    _reset_renderer_think(renderer)
    started = time.time()
    first = True
    ttft = 0.0
    tokens = 0
    answer = ''
    reasoning = ''
    model = getattr(renderer.llm, 'model_name', '')
    fetches = 0
    recalled: list[str] = []
    assistant = bool(getattr(renderer.opts, 'assistant_mode', False))

    def bump(count: int = 1) -> None:
        nonlocal tokens
        tokens += max(1, count)

    while True:
        parser = ThinkFeed()
        gold_feed = GoldNeedFeed()
        last_gold_channel = 'visible'
        _reset_renderer_think(renderer)
        chunks = renderer.stream_response(packed)
        try:
            for chunk in chunks:
                if first:
                    ttft = time.time() - started
                    first = False
                    yield _status_sse(
                        'Streaming…', model or '', route or '', context or 0,
                        recalled,
                    )
                visible, thought = parser.feed_chunk(chunk)
                if thought:
                    emit_t, hit_t = gold_feed.feed(thought)
                    if emit_t:
                        bump(len(emit_t.split()))
                        reasoning += emit_t
                        yield sse({'type': 'reasoning', 'content': emit_t}).encode()
                    if hit_t:
                        last_gold_channel = 'thought'
                        break
                if visible:
                    emit_v, hit_v = gold_feed.feed(visible)
                    if emit_v:
                        bump(renderer.response_count(emit_v))
                        answer += emit_v
                        yield sse({'type': 'token', 'content': emit_v}).encode()
                    if hit_v:
                        last_gold_channel = 'visible'
                        break
        finally:
            closer = getattr(chunks, 'close', None)
            if callable(closer):
                try:
                    closer()
                except Exception:  # pylint: disable=broad-exception-caught
                    pass
        leftover = gold_feed.flush()
        if leftover and not gold_feed.filename:
            if last_gold_channel == 'thought':
                bump(len(leftover.split()))
                reasoning += leftover
                yield sse({'type': 'reasoning', 'content': leftover}).encode()
            else:
                bump(renderer.response_count(leftover))
                answer += leftover
                yield sse({'type': 'token', 'content': leftover}).encode()
        fname = gold_feed.filename
        if (not fname or not assistant or fetches >= MAX_GOLD_FETCHES
                or meta is None):
            break
        if not renderer.state.context.fetch_gold_file(documents, fname):
            break
        fetches += 1
        recalled.append(fname)
        documents['gold_resume'] = answer
        yield _status_sse(recall_status(recalled), recalled=recalled)
        yield b':\n\n'
        packed = renderer.get_messages(meta, documents)
        model = getattr(renderer.llm, 'model_name', '') or model
        context = renderer.packed_prompt_tokens(packed)
        documents['prompt_tokens'] = context
        first = True
        yield _status_sse(
            'Processing Prompt…', model or '', route or '', context or 0,
            recalled,
        )
        yield b':\n\n'

    gen = time.time() - started
    stats.update({
        'answer': answer,
        'reasoning': reasoning,
        'model': model,
        'tokens': tokens,
        'ttft': ttft,
        'gen': gen,
    })
    yield sse({
        'type': 'usage',
        'model': model,
        'promptTokens': documents.get('prompt_tokens', 0),
        'completionTokens': tokens,
        'tokenSavings': int(documents.get('token_savings') or 0),
        'ttft': ttft,
    }).encode()


@app.post('/api/chat')
async def api_chat(request: Request) -> StreamingResponse:
    """SSE chat turn: RAG, optional agent search, then token stream."""
    body = await request.json()
    chat = get_chat()
    renderer = chat.session.renderer

    async def generate() -> AsyncIterator[bytes]:
        """Yield SSE frames for this request body."""
        global _streams
        with _stream_lock:
            _streams += 1
        notes: queue.Queue[str] = queue.Queue()
        renderer.status_hook = notes.put
        try:
            _sync_chat_object(chat)
            if body.get('regenerate'):
                pop_last_assistant(chat)
            yield sse({
                'type': 'status',
                'message': 'Working — RAG / agent / prompt…',
            }).encode()
            documents, meta = await asyncio.to_thread(
                _prepare_chat_documents, chat, body,
            )
            yield sse({
                'type': 'documents',
                'documents': list_attachments(_vector_dir()),
            }).encode()

            def _pack():
                renderer.set_llm(meta, documents)
                packed_local = renderer.get_messages(meta, documents)
                renderer.set_llm(meta, documents)
                return packed_local

            pack_task = asyncio.create_task(asyncio.to_thread(_pack))
            while not pack_task.done():
                try:
                    msg = notes.get_nowait()
                except queue.Empty:
                    await asyncio.sleep(0.05)
                    continue
                yield sse({'type': 'status', 'message': msg}).encode()
            packed = await pack_task
            while True:
                try:
                    msg = notes.get_nowait()
                except queue.Empty:
                    break
                yield sse({'type': 'status', 'message': msg}).encode()

            model = getattr(renderer.llm, 'model_name', '') or ''
            route = (renderer.orchestrator.name_of(renderer.llm)
                     or renderer.orchestrator.get_route_name(meta, documents))
            context = int(documents.get('prompt_tokens') or 0)
            if not context:
                context = renderer.packed_prompt_tokens(packed)
                documents['prompt_tokens'] = context
            yield sse({
                'type': 'status',
                'message': 'Processing Prompt…',
                'model': model,
                'route': route,
                'context': context,
            }).encode()
            stats: dict = {}
            for frame in _iter_sse_chunks(
                renderer, packed, documents, stats,
                route=route, context=context, meta=meta,
            ):
                yield frame
            if ((stats.get('answer') or stats.get('reasoning'))
                    and not documents.get('no_context')):
                persist_turn(
                    renderer,
                    documents,
                    stats.get('answer', ''),
                    stats.get('reasoning', ''),
                    metrics={
                        'model': stats.get('model', ''),
                        'tokenCount': stats.get('tokens', 0),
                        'generationTime': stats.get('gen', 0.0),
                        'promptTokens': documents.get('prompt_tokens', 0),
                        'tokenSavings': int(documents.get('token_savings') or 0),
                        'ttft': stats.get('ttft', 0.0),
                    },
                    attachments=body.get('attachments') or None,
                )
            yield sse({'type': 'done'}).encode()
        except Exception as exc:  # pylint: disable=broad-exception-caught
            yield sse({'type': 'error', 'error': str(exc)}).encode()
        finally:
            renderer.status_hook = None
            with _stream_lock:
                _streams -= 1
    return StreamingResponse(
        generate(),
        media_type='text/event-stream',
        headers={
            'Cache-Control': 'no-cache, no-transform',
            'Connection': 'keep-alive',
            'X-Accel-Buffering': 'no',
        },
    )




@app.get('/api/health')
def health() -> JSONResponse:
    """Liveness probe for Spur."""
    return JSONResponse({'ok': True, 'backend': 'chat.py'})


@app.get('/api/settings')
def api_settings_get() -> dict[str, Any]:
    """Current ``.chat.yaml`` values plus what Chat is actually running."""
    file_values, _text = load_settings_file(YAML_PATH)
    try:
        opts = _chat.opts if _chat is not None else ChatOptions.from_yaml(ROOT)
        effective = _opts_snapshot(opts)
    except Exception as exc:  # pylint: disable=broad-exception-caught
        effective = dict(file_values)
        return {
            'ok': False,
            'error': str(exc),
            'path': str(YAML_PATH),
            'values': file_values,
            'effective': effective,
            'busy': _streams > 0,
        }
    return {
        'ok': True,
        'path': str(YAML_PATH),
        'values': file_values,
        'effective': effective,
        'busy': _streams > 0,
    }


@app.post('/api/settings')
async def api_settings_save(request: Request) -> JSONResponse:
    """Write ``.chat.yaml`` and rebuild Chat so the next turn uses the new models."""
    if _streams > 0:
        return JSONResponse(
            {'ok': False, 'error': 'Wait for the current turn to finish.'},
            status_code=409,
        )
    body = await request.json()
    incoming = body.get('values') if isinstance(body.get('values'), dict) else body
    values = {key: blank(incoming.get(key)) for key in ALL_KEYS if key != 'model_server'}
    values['api_key'] = values.get('api_key') or 'none'
    if incoming.get('llm_server') is None and incoming.get('model_server'):
        values['llm_server'] = blank(incoming.get('model_server'))
    missing = [
        key for key in ('llm_server', 'model', 'pre_llm', 'embedding_llm')
        if not values.get(key)
    ]
    if missing:
        return JSONResponse(
            {
                'ok': False,
                'error': 'Need server + generator + pre-conditioner + embeddings.',
                'missing': missing,
            },
            status_code=400,
        )
    try:
        save_settings_file(YAML_PATH, values)
        _rebuild_chat_from_yaml()
    except Exception as exc:  # pylint: disable=broad-exception-caught
        return JSONResponse({'ok': False, 'error': str(exc)}, status_code=400)
    opts = _chat.opts if _chat is not None else ChatOptions.from_yaml(ROOT)
    return JSONResponse({
        'ok': True,
        'path': str(YAML_PATH),
        'values': load_settings_file(YAML_PATH)[0],
        'effective': _opts_snapshot(opts),
        'message': 'Saved. Next turn uses these models.',
    })


@app.post('/api/settings/ping')
async def api_settings_ping(request: Request) -> JSONResponse:
    """List models on an OpenAI-compatible server (LM Studio / Ollama)."""
    body = await request.json()
    host = blank(body.get('host') or body.get('llm_server'))
    api_key = blank(body.get('api_key')) or 'none'
    result = list_models(host, api_key)
    return JSONResponse(result, status_code=200 if result.get('ok') else 502)


STATIC = os.environ.get('SPUR_STATIC') or ''
_UI_MOUNTED = False


def ui_root() -> str:
    """Folder with index.html (built SPA). Empty string if none."""
    env = os.environ.get('SPUR_STATIC') or ''
    if env and os.path.isfile(os.path.join(env, 'index.html')):
        return env
    for candidate in (
        os.path.join(ROOT, 'spur', 'dist', 'client'),
        os.path.join(ROOT, 'spur', 'dist'),
        os.path.join(ROOT, 'spur-ui'),
        os.path.join(ROOT, 'spur', '.output', 'public'),
    ):
        if os.path.isfile(os.path.join(candidate, 'index.html')):
            return candidate
    return ''


def mount_ui(root: str | None = None) -> None:
    """Serve the built SPA from the same origin as /api (idempotent)."""
    global _UI_MOUNTED, STATIC
    if _UI_MOUNTED:
        return
    folder = root or ui_root()
    if not folder or not os.path.isdir(folder):
        return
    STATIC = folder
    app.mount('/', StaticFiles(directory=folder, html=True), name='ui')
    _UI_MOUNTED = True


if os.environ.get('SPUR_NO_MOUNT') != '1':
    mount_ui()


if __name__ == '__main__':
    import uvicorn

    uvicorn.run(app, host='127.0.0.1', port=8765)
