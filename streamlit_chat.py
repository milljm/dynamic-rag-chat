#!/usr/bin/env python3
"""
Streamlit front-end for Dynamic RAG Chat.
"""
from __future__ import annotations
import base64
import html
import io
import os
import re
import sys
import tempfile
import time
import glob
import shutil
from copy import deepcopy
from dataclasses import dataclass
from typing import Generator

import streamlit as st
from PIL import Image
from rich.console import Console

from chat import Chat, ChatOptions, SessionContext, parse_args, seed_from_string
from src import RenderWindow
from src.chat_utils import CommonUtils, RAGTag, load_pdf
from src.prompt_progress import PromptProgress, format_prompt_status

# ─────────────────────────────────────────────────────────────────────────────
# Page config MUST be the first Streamlit call
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title='Dynamic RAG Chat',
    page_icon='💬',
    layout='wide',
    initial_sidebar_state='expanded',
)

# ─────────────────────────────────────────────────────────────────────────────
# Constants / palette
# ─────────────────────────────────────────────────────────────────────────────
INK = '#0a0b0a'
PAPER = '#eceee9'
INK_SOFT = '#141615'
INK_RAISED = '#1c1e1c'
LINE = 'rgba(236,238,233,0.10)'
ACCENT = '#7dce82'
ACCENT_DIM = 'rgba(125,206,130,0.16)'
STORY = '#e8a0bf'
ASSIST = '#8ab4f8'

LOCKED_BRANCHES = frozenset({'assistant', 'story'})
RESERVED_NAMES = frozenset(
    {'current', 'assistant', 'story', 'assistant_mode', 'branch_modes', 'version'}
)

THINK_END_RE = re.compile(
    r'</\s*(?:mm:)?(?:think|thinking|reasoning)\s*>',
    re.IGNORECASE,
)
THINK_START_RE = re.compile(
    r'<\s*(?:mm:)?(?:think|thinking|reasoning)\s*>',
    re.IGNORECASE,
)

# Regex to detect stock-related queries for status display
_STOCK_QUERY = re.compile(
    r'(?i)(stock\s*(price|quote)?|share\s*price|ticker\b|market\s*data)',
)

console = Console(highlight=True)
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

# ─────────────────────────────────────────────────────────────────────────────
# Theme
# ─────────────────────────────────────────────────────────────────────────────
THEME_CSS = f"""
<style>
html, body, [data-testid="stAppViewContainer"] {{
    background: {INK} !important;
    color: {PAPER} !important;
    font-family: 'IBM Plex Sans', ui-sans-serif, system-ui, sans-serif;
}}
[data-testid="stHeader"] {{
    background: {INK} !important;
    border-bottom: 1px solid {LINE};
}}
[data-testid="stSidebar"] {{
    background: {INK_SOFT} !important;
    border-right: 1px solid {LINE};
}}
[data-testid="stSidebar"] * {{
    color: {PAPER} !important;
}}
section.main > div {{
    padding-top: 0.6rem;
}}

[data-testid="stBottomBlockContainer"] {{
    background: {INK} !important;
    padding-bottom: 0.6rem !important;
}}
[data-testid="stChatInput"] {{
    border: 1px solid {LINE} !important;
    outline: none !important;
    box-shadow: none !important;
    background: {INK_RAISED} !important;
    background-image: none !important;
    border-radius: 18px !important;
    padding: 0.35rem 0.45rem 0.35rem 0.6rem !important;
}}
[data-testid="stChatInput"] > div,
[data-testid="stChatInput"] [data-baseweb="textarea"],
[data-testid="stChatInput"] [data-baseweb="base-input"] {{
    background: transparent !important;
    background-image: none !important;
    border: none !important;
    box-shadow: none !important;
    outline: none !important;
}}
[data-testid="stChatInput"] textarea {{
    color: {PAPER} !important;
    background: transparent !important;
    font-family: 'IBM Plex Sans', sans-serif !important;
    caret-color: {PAPER} !important;
    min-height: 5.2rem !important;
    height: 5.2rem !important;
    line-height: 1.45 !important;
    padding: 0.55rem 0.2rem !important;
}}
[data-testid="stChatInput"]:focus-within,
[data-testid="stChatInput"] textarea:focus {{
    border-color: {LINE} !important;
    outline: none !important;
    box-shadow: none !important;
}}
[data-testid="stChatInputSubmitButton"],
[data-testid="stChatInput"] button {{
    align-self: flex-end !important;
    margin: 0 0 0.15rem 0.25rem !important;
}}

.imsg {{
    display: flex !important;
    width: 100% !important;
    margin: 0.4rem 0 0.75rem 0 !important;
}}
.imsg.user {{ justify-content: flex-end !important; }}
.imsg.assistant {{ justify-content: flex-start !important; }}
.imsg-col {{
    max-width: min(68%, 620px);
    display: flex;
    flex-direction: column;
}}
.imsg.user .imsg-col {{ align-items: flex-end; }}
.imsg.assistant .imsg-col {{ align-items: flex-start; }}
.imsg-bubble {{
    padding: 0.7rem 0.95rem;
    line-height: 1.5;
    font-size: 0.97rem;
    word-wrap: break-word;
    overflow-wrap: anywhere;
}}
.imsg.user .imsg-bubble {{
    background: #173322;
    border: 1px solid rgba(125,206,130,0.35);
    border-radius: 18px 18px 4px 18px;
}}
.imsg.assistant .imsg-bubble {{
    background: {INK_RAISED};
    border: 1px solid {LINE};
    border-radius: 18px 18px 18px 4px;
}}
.imsg-foot {{
    font-size: 0.72rem;
    opacity: 0.5;
    margin-top: 4px;
    font-family: 'IBM Plex Mono', monospace;
}}
.reason-panel {{
    max-width: min(68%, 620px);
    margin: 0.15rem 0 0.45rem 0;
    border: 1px solid {LINE};
    border-radius: 10px;
    padding: 0.35rem 0.7rem 0.45rem 0.7rem;
    background: {INK};
    font-size: 0.86rem;
}}
.reason-panel summary {{
    cursor: pointer;
    opacity: 0.8;
    font-weight: 500;
}}
.reason-body {{
    margin-top: 0.4rem;
    opacity: 0.75;
    white-space: pre-wrap;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.78rem;
    line-height: 1.4;
}}
.attach-chip {{
    display: inline-block;
    font-size: 0.78rem;
    padding: 2px 8px;
    border-radius: 999px;
    background: {ACCENT_DIM};
    border: 1px solid rgba(125,206,130,0.3);
    margin: 0 6px 8px 0;
}}
.topbar {{
    position: sticky;
    top: 0;
    z-index: 20;
    background: {INK};
    padding: 0.55rem 0 0.7rem 0;
    border-bottom: 1px solid {LINE};
    margin-bottom: 0.8rem;
}}
.topbar h1 {{
    font-size: 1.15rem;
    font-weight: 600;
    margin: 0;
    letter-spacing: -0.02em;
}}
.topbar .sub {{
    font-size: 0.78rem;
    opacity: 0.55;
    font-family: 'IBM Plex Mono', monospace;
    margin-top: 2px;
}}
.mode-badge {{
    display: inline-block;
    font-size: 0.66rem;
    padding: 1px 7px;
    border-radius: 999px;
    margin-left: 6px;
    font-weight: 600;
    letter-spacing: 0.04em;
    text-transform: uppercase;
}}
.mode-assist {{ background: rgba(138,180,248,0.18); color: {ASSIST}; }}
.mode-story  {{ background: rgba(232,160,191,0.16); color: {STORY}; }}
.slash-list {{
    margin-top: 0.35rem;
    font-size: 0.72rem;
    opacity: 0.72;
}}
.slash-row {{
    display: flex;
    justify-content: space-between;
    align-items: baseline;
    gap: 0.6rem;
    margin: 0.12rem 0;
}}
.slash-row code {{
    font-family: 'IBM Plex Mono', ui-monospace, monospace;
    font-size: 0.72rem;
    color: {ACCENT};
    background: transparent;
    flex: 0 0 auto;
}}
.slash-row span {{
    text-align: right;
    opacity: 0.7;
    font-size: 0.68rem;
    line-height: 1.25;
}}

[data-testid="stSidebar"] [data-testid="stVerticalBlock"] {{
    gap: 0.3rem !important;
}}
[data-testid="stSidebar"] .stButton {{
    margin: 0 !important;
}}
[data-testid="stSidebar"] .stButton > button,
[data-testid="stSidebar"] button[kind="secondary"],
[data-testid="stSidebar"] [data-testid="stBaseButton-secondary"] {{
    display: flex !important;
    flex-direction: column !important;
    align-items: flex-start !important;
    justify-content: flex-start !important;
    text-align: left !important;
    white-space: pre-line !important;
    height: auto !important;
    min-height: 46px !important;
    padding: 7px 10px !important;
    border-radius: 10px !important;
    cursor: pointer !important;
    background: {INK_RAISED} !important;
    line-height: 1.25 !important;
}}
[data-testid="stSidebar"] .stButton > button > div,
[data-testid="stSidebar"] .stButton > button p,
[data-testid="stSidebar"] button[kind="secondary"] > div,
[data-testid="stSidebar"] button[kind="secondary"] p {{
    display: block !important;
    width: 100% !important;
    max-width: 100% !important;
    margin: 0 !important;
    margin-inline: 0 !important;
    text-align: left !important;
    justify-content: flex-start !important;
    align-items: flex-start !important;
}}
[data-testid="stSidebar"] .stButton > button *,
[data-testid="stSidebar"] button[kind="secondary"] * {{
    white-space: pre-line !important;
    text-align: left !important;
    margin-left: 0 !important;
}}
[data-testid="stSidebar"] .stButton > button:disabled,
[data-testid="stSidebar"] .stButton > button[disabled] {{
    background: rgba(125,206,130,0.10) !important;
    border: 1px solid rgba(125,206,130,0.50) !important;
    box-shadow: inset 2px 0 0 {ACCENT} !important;
    color: {PAPER} !important;
    opacity: 1 !important;
    cursor: default !important;
}}
[data-testid="stSidebar"] [data-testid="stHorizontalBlock"] {{
    gap: 6px;
    align-items: center;
}}
[data-testid="stSidebar"] [data-testid="stHorizontalBlock"]
    [data-testid="column"]:last-child .stButton > button {{
    min-height: 26px !important;
    height: 26px !important;
    width: 26px !important;
    padding: 0 !important;
    border-radius: 999px !important;
    font-size: 0.8rem !important;
    opacity: 0.5;
    align-items: center !important;
    justify-content: center !important;
    text-align: center !important;
    background: {INK} !important;
}}
[data-testid="stSidebar"] [data-testid="stHorizontalBlock"]
    [data-testid="column"]:last-child .stButton > button:hover {{
    opacity: 1;
    border-color: #e07a7a !important;
    color: #e07a7a !important;
}}
div[data-testid="stExpander"] {{
    background: {INK} !important;
    border: 1px solid {LINE} !important;
    border-radius: 10px !important;
}}
.stButton > button {{
    border-radius: 10px !important;
    border: 1px solid {LINE} !important;
    background: {INK} !important;
    color: {PAPER} !important;
}}
.stButton > button:hover {{
    border-color: {ACCENT} !important;
    color: {ACCENT} !important;
}}
.stButton > button[kind="primary"] {{
    background: {ACCENT} !important;
    color: {INK} !important;
    border-color: {ACCENT} !important;
    font-weight: 600 !important;
}}
#MainMenu {{ visibility: hidden; }}
footer {{ visibility: hidden; }}
</style>
"""
st.markdown(THEME_CSS, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Session bootstrap
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource
def get_chat_session() -> Chat:
    """Construct and cache the Chat / SessionContext stack."""
    opts = ChatOptions.from_yaml(CURRENT_DIR)
    args = parse_args(sys.argv[1:], opts)
    _opts = ChatOptions.from_args(CURRENT_DIR, args, opts)
    _opts.seed = seed_from_string(_opts.seed)
    session = SessionContext.from_args(console, _opts)
    return Chat(session, _opts)


@dataclass
class StreamMetrics:
    """Per-turn generation stats shown under the assistant bubble."""

    model: str = ''
    token_count: int = 0
    generation_time: float = 0.0
    turn_count: int = 0
    prompt_tokens: int = 0
    token_savings: int = 0
    pre_process_time: float = 0.0
    ttft: float = 0.0


@dataclass
class ThinkParser:
    """
    Per-turn thinking splitter. Same state machine as the original
    streamlit is_thinking() / render_window.reveal_thinking():

    - First empty chunk(s) = shadow think (model is reasoning with no tags).
      The next non-empty chunk is the answer, not more reasoning.
    - First chunk with a <think> tag = tagged reasoning until </think>.
    - First chunk that is already visible text = no reasoning at all.
    """

    is_thinking: bool = False
    never_think: bool = False
    shadow_think: bool = False
    thinking: str = ''
    answer: str = ''
    saw_reason: bool = False

    def feed_chunk(self, chunk: object) -> None:
        """Ingest a LangChain stream chunk, including hidden reasoning fields."""
        extra = _chunk_reasoning(chunk)
        if extra:
            self.saw_reason = True
            self.is_thinking = True
            self.thinking += extra
        content = getattr(chunk, 'content', None)
        if content is None:
            content = ''
        elif not isinstance(content, str):
            content = str(content)
        self.feed(content)

    def feed(self, piece: str | None) -> None:
        """Ingest a raw text piece from the token stream."""
        chunk = piece if piece is not None else ''
        visible = self._classify(chunk)
        if self.is_thinking and not self.never_think and visible:
            self.thinking += visible
            self.saw_reason = True
        elif visible:
            self.answer += visible

    def _classify(self, content: str) -> str:
        """Return the chunk text that belongs to the current bucket."""
        if self.never_think:
            return content

        is_start_tag = bool(THINK_START_RE.search(content))
        is_end_tag = bool(THINK_END_RE.search(content))

        if not self.is_thinking and (not content or is_start_tag):
            if not content:
                self.shadow_think = True
            self.is_thinking = True
            self.saw_reason = True
            if is_start_tag:
                return THINK_START_RE.split(content, maxsplit=1)[-1]
            return ''

        if self.shadow_think and content:
            if is_start_tag:
                self.shadow_think = False
                self.is_thinking = True
                self.saw_reason = True
                return THINK_START_RE.split(content, maxsplit=1)[-1]
            self.is_thinking = False
            self.never_think = True
            return content

        if not self.is_thinking:
            self.never_think = True
            return content

        if self.is_thinking and is_end_tag:
            before = THINK_END_RE.split(content, maxsplit=1)[0]
            after = THINK_END_RE.split(content, maxsplit=1)[-1]
            self.never_think = True
            self.is_thinking = False
            if before:
                self.thinking += before
            return after

        return content

    def flush(self) -> None:
        """No buffered tail in this parser; kept for call-site symmetry."""
        return

    @property
    def reasoning_active(self) -> bool:
        """True while we are still inside a think / shadow-think span."""
        return self.is_thinking and not self.never_think


def _chunk_reasoning(chunk: object) -> str:
    """Pull hidden reasoning tokens off LangChain / OpenAI-style chunks."""
    keys = ('reasoning_content', 'reasoning', 'thinking', 'reasoning_text')
    parts: list[str] = []

    def _take(value) -> None:
        if not value:
            return
        if isinstance(value, dict):
            for key in keys:
                _take(value.get(key))
            _take(value.get('text'))
            return
        if isinstance(value, list):
            for item in value:
                _take(item)
            return
        text = str(value)
        if text and text not in parts:
            parts.append(text)

    for attr in keys:
        _take(getattr(chunk, attr, None))
    extra = getattr(chunk, 'additional_kwargs', None)
    _take(extra if isinstance(extra, dict) else None)
    meta = getattr(chunk, 'response_metadata', None)
    _take(meta if isinstance(meta, dict) else None)
    content = getattr(chunk, 'content', None)
    if isinstance(content, list):
        for item in content:
            if isinstance(item, dict) and item.get('type') in {
                'reasoning',
                'thinking',
                'reason',
            }:
                _take(item)
    return ''.join(parts)


# ─────────────────────────────────────────────────────────────────────────────
# History / branch helpers
# ─────────────────────────────────────────────────────────────────────────────
def _history(chat: Chat) -> dict:
    """Load the on-disk chat history document."""
    return chat.session.common.load_chat()


def _canonical_mode(name: str) -> bool | None:
    """Fixed mode for protected branches: True=assistant, False=story."""
    if name == 'assistant':
        return True
    if name == 'story':
        return False
    return None


def _persisted_mode(hist: dict, name: str, fallback: bool = False) -> bool:
    """Return the stored assistant-mode flag for a branch."""
    canon = _canonical_mode(name)
    if canon is not None:
        return canon
    return bool(hist.get('branch_modes', {}).get(name, fallback))


def _active_branch(chat: Chat) -> str:
    """Return history['current'], defaulting to story."""
    return _history(chat).get('current', 'story')


def _turn_count(msgs: list) -> int:
    """Count user turns in a role/content message list."""
    if not msgs:
        return 0
    return sum(1 for m in msgs if isinstance(m, dict) and m.get('role') == 'user')


def _sync_chat_object(chat: Chat, hist: dict | None = None) -> tuple[str, bool]:
    """
    Keep Chat.chat_branch and Chat.opts.assistant_mode aligned with disk.

    chat.get_documents() reads self.chat_branch — if we only mutate the
    JSON file and forget this field, every turn silently uses the stale
    branch.
    """
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
    st.session_state.current_branch = branch
    _refresh_mode_runtime(chat)
    return branch, mode


def _refresh_mode_runtime(chat: Chat) -> None:
    """
    PromptManager / ContextManager bake assistant_mode in at construct time.

    _match_model() returns 'nostory' whenever args.assistant_mode is True,
    and build_prompts() caches plot_prompt_system / plot_prompt_human.
    Flipping chat.opts.assistant_mode alone does nothing — rebuild files.
    """
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


def _prime_toggle(mode: bool) -> None:
    """Set toggle state only before the widget exists, or via a pending flag."""
    if '_pending_assistant_toggle' in st.session_state:
        st.session_state.assistant_toggle = bool(
            st.session_state.pop('_pending_assistant_toggle')
        )
    elif 'assistant_toggle' not in st.session_state:
        st.session_state.assistant_toggle = bool(mode)


def set_assistant_mode(chat: Chat, enabled: bool) -> None:
    """Persist the mode of the current user branch (protected branches ignore)."""
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


def switch_branch(chat: Chat, name: str) -> tuple[bool, str]:
    """Switch to an existing branch and restore its recorded mode."""
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

    try:
        chat.session.common.save_chat(hist)
        chat.chat_branch = name
        chat.opts.assistant_mode = new_mode
        st.session_state['_pending_assistant_toggle'] = new_mode
        if hasattr(chat.session.renderer, 'clear_ooc'):
            try:
                chat.session.renderer.clear_ooc()
            except Exception:  # pylint: disable=broad-exception-caught
                pass
        return True, f"Switched to '{name}'."
    except Exception as exc:  # pylint: disable=broad-exception-caught
        hist['current'] = old
        chat.session.common.save_chat(hist)
        return False, f'Switch failed: {exc}'


def create_branch(chat: Chat, name: str, cut_turns: int | None) -> tuple[bool, str]:
    """Fork a new branch from the current branch (full clone or first N turns)."""
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
        return False, f"Branch '{name}' already exists."

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
            try:
                chat.session.renderer.clear_ooc()
            except Exception:  # pylint: disable=broad-exception-caught
                pass
        if cut_turns is None:
            chat.session.rag.clone_collection(src, name, overwrite=False)
        elif hasattr(chat.session.rag, 'build_collection_from_texts'):
            chat.session.rag.build_collection_from_texts(name, new_list, overwrite=True)
        else:
            chat.chat_branch = name
            chat.opts.assistant_mode = source_mode
            return True, (
                f"Created '{name}' but RAG rebuild is unsupported — "
                'retrieval may be empty until rebuilt.'
            )
        chat.chat_branch = name
        chat.opts.assistant_mode = source_mode
        st.session_state['_pending_assistant_toggle'] = source_mode
        return True, (
            f"Branched to '{name}' "
            f"({cut_turns if cut_turns is not None else 'full clone'})."
        )
    except Exception as exc:  # pylint: disable=broad-exception-caught
        hist.pop(name, None)
        hist['current'] = src
        hist.setdefault('branch_modes', {}).pop(name, None)
        chat.session.common.save_chat(hist)
        chat.chat_branch = src
        return False, f'RAG sync failed: {exc}'


def delete_branch(chat: Chat, name: str) -> tuple[bool, str]:
    """Delete a non-protected, non-current branch and its RAG collection."""
    if name in RESERVED_NAMES or name in LOCKED_BRANCHES:
        return False, f"Cannot delete protected branch '{name}'."
    hist = _history(chat)
    if hist.get('current') == name:
        return False, 'Cannot delete the branch you are on. Switch first, or reset it.'
    if name not in hist or not isinstance(hist.get(name), list):
        return False, f"Unknown branch '{name}'."
    try:
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
    except Exception as exc:  # pylint: disable=broad-exception-caught
        return False, f'Delete failed: {exc}'


def reset_branch(chat: Chat) -> tuple[bool, str]:
    """Clear history and RAG for the current branch."""
    hist = _history(chat)
    branch = hist.get('current', 'story')
    try:
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
            try:
                chat.session.renderer.clear_ooc()
            except Exception:  # pylint: disable=broad-exception-caught
                pass
        return True, f"Reset '{branch}'."
    except Exception as exc:  # pylint: disable=broad-exception-caught
        return False, f'Reset failed: {exc}'


def delete_last_turn(chat: Chat) -> tuple[bool, str]:
    """Remove the last user/assistant pair from the current branch."""
    hist = _history(chat)
    branch = hist.get('current', 'story')
    msgs = hist.get(branch, [])
    if not msgs:
        return False, 'History empty.'
    if msgs and msgs[-1].get('role') == 'assistant':
        msgs.pop()
    if msgs and msgs[-1].get('role') == 'user':
        msgs.pop()
    hist[branch] = msgs
    chat.session.common.save_chat(hist)
    if hasattr(chat.session.renderer, 'clear_ooc'):
        try:
            chat.session.renderer.clear_ooc()
        except Exception:  # pylint: disable=broad-exception-caught
            pass
    return True, 'Deleted last turn.'


def rewind_to(chat: Chat, n: int) -> tuple[bool, str]:
    """Keep only the first n turns of the current branch."""
    hist = _history(chat)
    branch = hist.get('current', 'story')
    msgs = hist.get(branch, [])
    total = _turn_count(msgs)
    if not 1 <= n <= total:
        return False, f'Rewind needs 1 ≤ N ≤ {total}.'
    hist[branch] = msgs[: n * 2]
    chat.session.common.save_chat(hist)
    if hasattr(chat.session.renderer, 'clear_ooc'):
        try:
            chat.session.renderer.clear_ooc()
        except Exception:  # pylint: disable=broad-exception-caught
            pass
    return True, f'Rewound to turn {n} of {total}.'


def list_branches(chat: Chat) -> dict:
    """Return {name: {count, preview}} for every real history branch."""
    hist = _history(chat)
    out = {}
    for _name, _msgs in hist.items():
        if _name == 'current' or not isinstance(_msgs, list):
            continue
        _turns = _turn_count(_msgs)
        _last_asst = next(
            (
                _m.get('content', '')
                for _m in reversed(_msgs)
                if _m.get('role') == 'assistant'
            ),
            '',
        )
        _flat = ' '.join(_last_asst.split())
        _preview = (_flat[:48] + '…') if len(_flat) > 48 else _flat
        out[_name] = {'count': _turns, 'preview': _preview}
    return out


def load_ui_messages(chat: Chat) -> list[dict]:
    """Build the Streamlit transcript from the active branch on disk."""
    hist = _history(chat)
    branch = hist.get('current', 'story')
    msgs = hist.get(branch, []) or []
    ui = []
    for m in msgs:
        if not isinstance(m, dict) or not m.get('content'):
            continue
        item = {
            'role': m.get('role', 'assistant'),
            'content': m.get('content', ''),
            'attachments': m.get('attachments', []),
        }
        if m.get('reasoning'):
            item['reasoning'] = m['reasoning']
        if m.get('footer'):
            item['footer'] = m['footer']
        ui.append(item)
    return ui


def persist_turn(
    renderer: RenderWindow,
    documents: dict,
    response: str,
    reasoning: str = '',
    footer: str = '',
) -> None:
    """
    Write the completed turn through RenderWindow.save_history, then stamp
    optional extras onto the last assistant message.

    History schema stays a list of {role, content, ...} dicts. chat.py /
    context_manager / save_history only read role + content, so extra keys
    are ignored by the terminal UI and by RAG stringify.
    """
    documents['llm_response'] = response
    renderer.save_history(documents, response)
    extra = {}
    if reasoning and reasoning.strip():
        extra['reasoning'] = reasoning
    if footer:
        extra['footer'] = footer
    if not extra:
        return
    common = renderer.common
    hist = common.load_chat()
    branch = hist.get('current', 'story')
    msgs = hist.get(branch) or []
    if msgs and isinstance(msgs[-1], dict) and msgs[-1].get('role') == 'assistant':
        msgs[-1].update(extra)
        common.save_chat(hist)


def handle_attachments(documents: dict) -> dict:
    """Fold Streamlit uploads into documents['dynamic_images'/'dynamic_files']."""
    attachments = st.session_state.get('attachments') or []
    if not attachments:
        return documents
    documents.setdefault('dynamic_images', [])
    documents.setdefault('dynamic_files', '')

    for att in attachments:
        name = att.get('name', 'unknown')
        mime = att.get('type', '') or ''
        data = att.get('data', b'') or b''

        if mime.startswith('image/'):
            fmt = mime.split('/')[-1].upper()
            if fmt == 'JPG':
                fmt = 'JPEG'
            if fmt not in {'JPEG', 'PNG', 'WEBP', 'GIF'}:
                ext = name.rsplit('.', 1)[-1].lower() if '.' in name else 'png'
                fmt = {'jpg': 'JPEG', 'jpeg': 'JPEG', 'png': 'PNG'}.get(ext, 'PNG')
            try:
                with Image.open(io.BytesIO(data)) as img:
                    img = img.convert('RGB')
                    out = io.BytesIO()
                    img.save(out, format=fmt if fmt != 'GIF' else 'PNG')
                    documents['dynamic_images'].append(
                        base64.b64encode(out.getvalue()).decode('utf-8')
                    )
                CommonUtils.record_attachment(documents, name, kind='image')
            except Exception:  # pylint: disable=broad-exception-caught
                documents['dynamic_files'] += f'\n--- {name} ---\n<unreadable image>\n'
        elif mime == 'application/pdf':
            tmp_path = None
            try:
                with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp:
                    tmp.write(data)
                    tmp_path = tmp.name
                text = ''.join(doc.page_content for doc in load_pdf(tmp_path))
                documents['dynamic_files'] += f'\n--- {name} ---\n\n{text}\n\n'
                CommonUtils.record_attachment(documents, name, text=text, kind='text')
            except Exception as exc:  # pylint: disable=broad-exception-caught
                documents['dynamic_files'] += f'\n--- {name} ---\n<pdf error: {exc}>\n'
            finally:
                if tmp_path:
                    try:
                        os.unlink(tmp_path)
                    except OSError:
                        pass
        else:
            try:
                text = data.decode('utf-8')
            except UnicodeDecodeError:
                try:
                    text = data.decode('latin-1')
                except Exception:  # pylint: disable=broad-exception-caught
                    text = '<binary content, could not decode>'
            documents['dynamic_files'] += f'\n--- {name} ---\n\n{text}\n\n'
            CommonUtils.record_attachment(documents, name, text=text, kind='text')
    return documents


def call_llm_stream(
    renderer: RenderWindow,
    documents: dict,
    meta: RAGTag,
    metrics: StreamMetrics,
    status,
) -> Generator:
    """Stream LLM tokens and fill StreamMetrics."""
    status.markdown('RAG Processing…')
    renderer.set_llm(meta, documents)
    if renderer.orchestrator.requires_agent(meta, documents):
        # Determine which tool to show in status
        query = documents.get('user_query', '') or ''
        if _STOCK_QUERY.search(query):
            tool_name = 'yfinance'
        else:
            key = (renderer.opts.tavily_key or '').strip().lower()
            tool_name = 'tavily' if (key and key != 'none') else 'duckduckgo'
        status.markdown(f'Agent [{tool_name}]…')
    messages = renderer.get_messages(meta, documents)
    renderer.set_llm(meta, documents)
    if documents.get('sd_ran') and not messages:
        status.empty()
        return
    status.markdown('Processing Prompt…')
    metrics.model = renderer.llm.model_name
    metrics.prompt_tokens = documents.get('prompt_tokens', 0)
    metrics.token_savings = documents.get('token_savings', 0)
    started = time.time()
    first = True
    for token in renderer.stream_response(messages):
        if isinstance(token, PromptProgress):
            status.markdown(format_prompt_status(token.fraction))
            continue
        status.empty()
        if first:
            metrics.ttft = time.time() - started
            first = False
        metrics.token_count += renderer.response_count(
            getattr(token, 'content', '') or ''
        )
        yield token
    metrics.generation_time = time.time() - started


CMD_RE = re.compile(r'^[ \t]*\\(?P<cmd>[A-Za-z0-9_\-\?]+)(?:[ \t]+(?P<args>.*))?$')


def try_handle_command(chat: Chat, raw: str) -> bool:
    """
    Handle a leading slash command.

    Return True if the input was consumed (no LLM call).
    """
    first = raw.strip().splitlines()[0] if raw.strip() else ''
    matched = CMD_RE.match(first)
    if not matched:
        return False
    cmd = matched.group('cmd').lower()
    args = (matched.group('args') or '').strip()

    if cmd in {'?', 'help'}:
        st.info(
            'Commands: `\\reset`  `\\delete-last`  `\\rewind N`  '
            '`\\dbranch NAME`  `\\branch`  `\\turn`  `\\no-context msg`  `\\agent msg`'
        )
        return True

    if cmd == 'turn':
        hist = _history(chat)
        st.info(f'Turn {_turn_count(hist.get(chat.chat_branch, []))}')
        return True

    if cmd == 'delete-last':
        ok, msg = delete_last_turn(chat)
        (st.success if ok else st.warning)(msg)
        if ok:
            st.session_state.messages = load_ui_messages(chat)
            st.rerun()
        return True

    if cmd == 'rewind':
        try:
            n = int(args)
        except ValueError:
            st.error('usage: \\rewind N')
            return True
        ok, msg = rewind_to(chat, n)
        (st.success if ok else st.error)(msg)
        if ok:
            st.session_state.messages = load_ui_messages(chat)
            st.rerun()
        return True

    if cmd == 'reset':
        ok, msg = reset_branch(chat)
        (st.success if ok else st.error)(msg)
        if ok:
            st.session_state.messages = load_ui_messages(chat)
            st.rerun()
        return True

    if cmd == 'dbranch':
        ok, msg = delete_branch(chat, args)
        (st.success if ok else st.error)(msg)
        if ok:
            st.rerun()
        return True

    if cmd == 'branch' and not args:
        names = ', '.join(sorted(list_branches(chat)))
        st.info(f"Branches: {names or '(none)'}")
        return True

    return False


def apply_inline_flags(documents: dict, raw: str) -> dict:
    """Stamp \\agent / \\no-context onto the documents dict when present."""
    first = raw.strip().splitlines()[0] if raw.strip() else ''
    matched = CMD_RE.match(first)
    if not matched:
        return documents
    cmd = matched.group('cmd').lower()
    if cmd == 'agent':
        documents['use_agent'] = True
        documents['agent_ran'] = False
        documents['in_line_commands'] = 'Meta: [agent]'
    elif cmd == 'no-context':
        documents['no_context'] = True
        documents['in_line_commands'] = 'Meta: [no-context]'
    return documents


def render_bubble(
    role: str,
    content: str,
    footer: str = '',
    attachments: list | None = None,
    slot=None,
) -> None:
    """Paint an iMessage-style row. slot=st.empty() to update in place."""
    side = 'user' if role == 'user' else 'assistant'
    chips = ''
    if attachments:
        chips = (
            ' '.join(
                f'<span class="attach-chip">📎 {html.escape(str(a.get('name', 'file')))}</span>'
                for a in attachments
            )
            + '\n\n'
        )
    foot = f'<div class="imsg-foot">{html.escape(footer)}</div>' if footer else ''
    markup = (
        f'<div class="imsg {side}"><div class="imsg-col">'
        f'<div class="imsg-bubble">\n\n{chips}{content}\n\n</div>'
        f'{foot}</div></div>'
    )
    target = slot if slot is not None else st
    target.markdown(markup, unsafe_allow_html=True)


def render_reasoning(slot, text: str, model_name: str = '', live: bool = False) -> None:
    """Collapsed reasoning panel. Look stays the same; body is click-to-reveal."""
    title = 'Reasoning…' if live else 'Reasoning'
    if model_name:
        title = f'{title} [{html.escape(str(model_name))}]'
    if text and text.strip():
        body = html.escape(text).replace('\n', '<br>')
    elif live:
        body = '<em>Working…</em>'
    else:
        body = '<em>This model did not stream its reasoning tokens.</em>'
    slot.markdown(
        f'<details class="reason-panel">'
        f'<summary>🤔 {title}</summary>'
        f'<div class="reason-body">{body}</div></details>',
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# App
# ─────────────────────────────────────────────────────────────────────────────
_chat: Chat = get_chat_session()
_renderer = _chat.session.renderer
_hist = _history(_chat)
_branch, _mode = _sync_chat_object(_chat, _hist)
_prime_toggle(_mode)

if 'messages' not in st.session_state:
    st.session_state.messages = load_ui_messages(_chat)
if 'attachments' not in st.session_state:
    st.session_state.attachments = []
if 'pending_notice' not in st.session_state:
    st.session_state.pending_notice = ''

_mode_label = 'Assistant' if _mode else 'Story'
st.markdown(
    f"""
    <div class="topbar">
      <h1>Dynamic RAG Chat</h1>
      <div class="sub">{_branch} · {_mode_label}</div>
    </div>
    """,
    unsafe_allow_html=True,
)

if st.session_state.pending_notice:
    st.caption(st.session_state.pending_notice)
    st.session_state.pending_notice = ''

for _msg in st.session_state.messages:
    _role = _msg.get('role', 'assistant')
    if _role == 'assistant' and 'reasoning' in _msg:
        render_reasoning(st, _msg.get('reasoning') or '', live=False)
    render_bubble(
        _role,
        _msg.get('content', ''),
        footer=_msg.get('footer', ''),
        attachments=_msg.get('attachments') or [],
    )

st.sidebar.markdown('### Mode')
_locked = _branch in LOCKED_BRANCHES


def _on_toggle():
    """Persist the sidebar mode toggle."""
    set_assistant_mode(_chat, bool(st.session_state.assistant_toggle))


st.sidebar.toggle(
    ('🧩 Assistant' if _mode else '🎭 Story') + ('  🔒' if _locked else ''),
    key='assistant_toggle',
    on_change=_on_toggle,
    disabled=_locked,
    help=(
        'Protected branch — fork a new one to change flavor.'
        if _locked
        else 'Assistant = tools/RAG helper. Story = roleplay system prompt.'
    ),
)
_chat.opts.assistant_mode = bool(st.session_state.assistant_toggle)
st.sidebar.divider()

st.sidebar.markdown('### Attachments')
_uploads = st.sidebar.file_uploader(
    'Images, PDFs, text',
    type=['png', 'jpg', 'jpeg', 'webp', 'gif', 'pdf', 'txt', 'py', 'md'],
    accept_multiple_files=True,
    label_visibility='collapsed',
)
if _uploads:
    st.session_state.attachments = [
        {'name': _file.name, 'type': _file.type, 'data': _file.getvalue()}
        for _file in _uploads
    ]
    for _uploaded in _uploads:
        st.sidebar.caption(f'📎 {_uploaded.name}')
elif not st.session_state.get('_hold_attachments'):
    st.session_state.attachments = []
st.sidebar.divider()

st.sidebar.markdown('### Branches')
st.sidebar.caption('Click a card to switch.')
_meta = list_branches(_chat)
_modes = _hist.get('branch_modes', {})


def _switch_to(_name: str) -> None:
    """Button callback: switch branch and reload the transcript."""
    _ok, _message = switch_branch(_chat, _name)
    st.session_state.pending_notice = _message
    if _ok:
        st.session_state.messages = load_ui_messages(_chat)


def _delete_named(_name: str) -> None:
    """Button callback: delete a branch."""
    _ok, _message = delete_branch(_chat, _name)
    st.session_state.pending_notice = _message


if not _meta:
    st.sidebar.caption('No branches yet — send a message.')
else:
    for _name in sorted(_meta, key=lambda n: (n != _branch, n)):
        _info = _meta[_name]
        _is_assist = _canonical_mode(_name) is True or _modes.get(_name) is True
        _badge = 'assist' if _is_assist else 'story'
        _preview = _info.get('preview') or ''
        _active = _name == _branch
        _meta_line = f"{_info['count']} turns"
        _label = f"{'●' if _active else '○'}  {_name}  · {_badge}\u2028 {_meta_line}"
        _show_delete = (not _active) and _name not in LOCKED_BRANCHES
        _row = st.sidebar.container()
        if _show_delete:
            _card_col, _del_col = _row.columns([9, 1], vertical_alignment='center')
        else:
            _card_col = _row.container()
            _del_col = None
        with _card_col:
            st.button(
                _label,
                key=f'br_{_name}',
                use_container_width=True,
                disabled=_active,
                help=_preview or (None if _active else f"Switch to '{_name}'"),
                on_click=None if _active else _switch_to,
                args=() if _active else (_name,),
            )
        if _del_col is not None:
            with _del_col:
                st.button(
                    '×',
                    key=f'del_{_name}',
                    use_container_width=True,
                    help=f"Delete '{_name}'",
                    on_click=_delete_named,
                    args=(_name,),
                )

with st.sidebar.expander('Create branch', expanded=False):
    _new_name = st.text_input('Name', placeholder='alternate-story', key='new_branch_name')
    _fork_turns = st.number_input(
        'Fork after turn (0 = full clone)',
        min_value=0,
        value=0,
        step=1,
        key='fork_turns',
    )
    if st.button('Create', key='create_branch_button', use_container_width=True, type='primary'):
        _cut = None if int(_fork_turns) == 0 else int(_fork_turns)
        _ok, _message = create_branch(_chat, _new_name, _cut)
        if _ok:
            st.session_state.messages = load_ui_messages(_chat)
            st.success(_message)
            st.rerun()
        else:
            st.error(_message)

with st.sidebar.expander('History tools', expanded=False):
    _c1, _c2 = st.columns(2)
    if _c1.button('Delete last', use_container_width=True):
        _ok, _message = delete_last_turn(_chat)
        if _ok:
            st.session_state.messages = load_ui_messages(_chat)
            st.rerun()
        else:
            st.warning(_message)
    if _c2.button('Reset branch', use_container_width=True):
        _ok, _message = reset_branch(_chat)
        if _ok:
            st.session_state.messages = load_ui_messages(_chat)
            st.rerun()
        else:
            st.error(_message)
    _total = _turn_count(_history(_chat).get(_branch, []))
    _rewind_n = st.number_input(
        'Rewind to turn', min_value=0, max_value=max(_total, 0), value=0, step=1
    )
    if st.button('Rewind', use_container_width=True, disabled=_total == 0):
        if _rewind_n <= 0:
            st.warning('Pick a turn ≥ 1.')
        else:
            _ok, _message = rewind_to(_chat, int(_rewind_n))
            if _ok:
                st.session_state.messages = load_ui_messages(_chat)
                st.rerun()
            else:
                st.error(_message)

st.sidebar.markdown('### Slash commands')
st.sidebar.markdown(
    '''
    <div class="slash-list">
      <div class="slash-row"><code>\\reset</code><span>reset history/RAG for this branch</span></div>
      <div class="slash-row"><code>\\delete-last</code><span>delete last turn from history</span></div>
      <div class="slash-row"><code>\\rewind N</code><span>rewind to turn N (keep 0..N)</span></div>
      <div class="slash-row"><code>\\agent msg</code><span>force agent (web search)</span></div>
      <div class="slash-row"><code>\\no-context msg</code><span>query with no RAG context</span></div>
    </div>
    ''',
    unsafe_allow_html=True,
)

_incoming = st.chat_input(
    'Message  ·  Esc-style commands start with \\',
    accept_file='multiple',
    file_type=['png', 'jpg', 'jpeg', 'webp', 'gif', 'pdf', 'txt', 'py', 'md'],
)
if _incoming:
    if isinstance(_incoming, str):
        _prompt = _incoming
        _inbound_files = []
    else:
        _prompt = (getattr(_incoming, 'text', None) or '').strip()
        _inbound_files = list(getattr(_incoming, 'files', None) or [])
    if _inbound_files:
        st.session_state.attachments = [
            {
                'name': _file.name,
                'type': getattr(_file, 'type', '') or '',
                'data': _file.getvalue(),
            }
            for _file in _inbound_files
        ]
    if not _prompt and not st.session_state.attachments:
        st.stop()
    if _prompt and try_handle_command(_chat, _prompt):
        st.stop()
    if not _prompt:
        _prompt = '(attachment)'

    render_bubble('user', _prompt, attachments=st.session_state.attachments)

    st.session_state.messages.append(
        {
            'role': 'user',
            'content': _prompt,
            'attachments': list(st.session_state.attachments),
        }
    )

    _sync_chat_object(_chat)

    try:
        _documents, _meta_data = _chat.prepare_turn(_prompt)
    except Exception as exc:  # pylint: disable=broad-exception-caught
        st.error(f'Pre-processor failed: {exc}')
        st.stop()

    if not _documents:
        st.error('Pre-processor returned nothing. Re-submit the query.')
        st.stop()

    _documents = handle_attachments(_documents)
    _documents = apply_inline_flags(_documents, _prompt)
    if not _documents.get('no_context'):
        _chat.session.context.ingest_user_attachments(_documents, _meta_data)

    if _documents.get('no_context') and hasattr(_chat, 'no_context'):
        _body = _prompt
        _matched = CMD_RE.match(_prompt.strip().splitlines()[0])
        if _matched:
            _body = (_matched.group('args') or '').strip() or '\n'.join(
                _prompt.splitlines()[1:]
            ).strip()
        _documents = _chat.no_context(_body or _prompt)
        _documents['no_context'] = True
        _documents['in_line_commands'] = 'Meta: [no-context]'
        _documents = handle_attachments(_documents)

    _metrics = StreamMetrics(
        turn_count=_turn_count(_history(_chat).get(_chat.chat_branch, []))
    )
    _parser = ThinkParser()
    _footer = ''

    _status = st.empty()
    _think_box = st.empty()
    _answer_box = st.empty()
    try:
        _stream = call_llm_stream(_renderer, _documents, _meta_data, _metrics, _status)
        _model_name = ''
        for _chunk in _stream:
            if not _model_name:
                _model_name = (
                    _metrics.model
                    or getattr(getattr(_renderer, 'llm', None), 'model_name', '')
                    or ''
                )
            _parser.feed_chunk(_chunk)
            if _parser.reasoning_active or _parser.shadow_think or _parser.saw_reason:
                render_reasoning(
                    _think_box,
                    _parser.thinking,
                    model_name=_model_name,
                    live=_parser.reasoning_active and not _parser.answer,
                )
            if _parser.answer:
                render_bubble('assistant', _parser.answer, slot=_answer_box)
        _parser.flush()
        if _parser.thinking.strip() or _parser.saw_reason or _parser.shadow_think:
            render_reasoning(
                _think_box,
                _parser.thinking,
                model_name=_model_name,
                live=False,
            )
        _gen_duration = max(0.0, _metrics.generation_time - _metrics.ttft)
        _tps = _metrics.token_count / _gen_duration if _gen_duration > 0 else 0
        _routed = getattr(getattr(_renderer, 'llm', None), 'model_name', '') or _metrics.model
        _pretty = _routed
        if hasattr(_renderer, '_format_model_name') and _routed:
            try:
                # pylint: disable-next=protected-access     # I see no way around this
                _pretty = _renderer._format_model_name(_routed)
            except Exception:  # pylint: disable=broad-exception-caught
                _pretty = _routed
        _footer = (
            f'{_pretty} · TTFT {_metrics.ttft:.2f}s · gen {_gen_duration:.2f}s · '
            f'{_metrics.token_count} tok · {_tps:.1f} T/s'
        )
        render_bubble(
            'assistant',
            _parser.answer or '_Empty response from model._',
            footer=_footer,
            slot=_answer_box,
        )
    except Exception as exc:  # pylint: disable=broad-exception-caught
        _status.empty()
        st.error(f'LLM error: {exc}')
        _parser.flush()

    _saved = {
        'role': 'assistant',
        'content': _parser.answer,
        'footer': _footer if _parser.answer else '',
    }
    if _parser.thinking or _parser.saw_reason or _parser.shadow_think:
        _saved['reasoning'] = _parser.thinking
    st.session_state.messages.append(_saved)

    if _parser.answer and not _documents.get('no_context'):
        try:
            persist_turn(
                _renderer,
                _documents,
                _parser.answer,
                reasoning=_parser.thinking,
                footer=_footer,
            )
        except Exception as exc:  # pylint: disable=broad-exception-caught
            st.warning(f'Saved UI turn, but disk/RAG persist failed: {exc}')

    st.session_state.attachments = []
    # pylint: disable-next=protected-access     # I see no way around this
    st.session_state._hold_attachments = False
    st.rerun()
