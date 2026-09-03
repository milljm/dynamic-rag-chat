"""Think-tag splitter used by the TUI and the Spur HTTP adapter.

Supported reasoning modes (these should always work):

- **Namespaced tags** (MiniMax-M3): ``<mm:think>…</mm:think>``.
  Bare ``<think>`` / ``</think>`` inside that block are prose — the model
  talking about Python, not a closer. Only a matching ``</mm:think>`` ends it.
- **Null / blank first tokens** (gpt-oss-120b): ``content`` is None/'' while
  ``reasoning_content`` streams. First non-blank token is the answer, unless
  it opens a namespaced block.

Bare ``<think>…</think>`` (Qwen etc.) still splits. If those models *talk
about* ``<think>`` while still inside a bare block, that is unsolvable —
do not try.

Reset the parser when the user sends a new turn.
"""
from __future__ import annotations

import re
from typing import Any, Mapping

# Namespaced (MiniMax). Do not make `mm:` optional — that is what used to
# treat an inner <think> as a real tag.
NS_START_RE = re.compile(r'<\s*mm:(think|thinking|reasoning)\s*>', re.I)
NS_END_RE = re.compile(r'</\s*mm:(think|thinking|reasoning)\s*>', re.I)
# Bare tags. Must NOT match <mm:think>.
BARE_START_RE = re.compile(r'<\s*(think|thinking|reasoning)\s*>', re.I)
BARE_END_RE = re.compile(r'</\s*(think|thinking|reasoning)\s*>', re.I)

# Kept for TUI callers that still import these names.
THINK_START_RE = re.compile(
    r'<\s*(mm:)?(think|thinking|reasoning)\s*>', re.I
)
THINK_END_RE = re.compile(
    r'</\s*(mm:)?(think|thinking|reasoning)\s*>', re.I
)


def tag_ns(match: re.Match[str] | None) -> str:
    """Return the captured think-tag name (mm: vs bare), lowercased."""
    if not match:
        return ''
    return (match.group(1) or '').lower()


def _blank_content(piece: Any) -> bool:
    """LangChain sends None, '', or [] on reasoning-only chunks."""
    return piece is None or piece == '' or piece == []


def _delta_reasoning(delta: Any) -> str:
    """Pull reasoning_content off an OpenAI delta dict or pydantic object."""
    if delta is None:
        return ''
    if not isinstance(delta, dict):
        extra = getattr(delta, 'reasoning_content', None) or getattr(
            delta, 'reasoning', None
        )
        if not extra:
            dumped = getattr(delta, 'model_extra', None) or {}
            if isinstance(dumped, dict):
                extra = dumped.get('reasoning_content') or dumped.get('reasoning')
        if not extra:
            try:
                dumped = delta.model_dump()
            except Exception:  # pylint: disable=broad-exception-caught
                dumped = None
            if isinstance(dumped, dict):
                extra = dumped.get('reasoning_content') or dumped.get('reasoning')
        return extra if isinstance(extra, str) else ''
    extra = delta.get('reasoning_content') or delta.get('reasoning') or ''
    return extra if isinstance(extra, str) else ''


def reasoning_from_openai_chunk(chunk: Any) -> str:
    """Read delta.reasoning_content from a Chat Completions chunk dict."""
    if chunk is None:
        return ''
    if not isinstance(chunk, dict):
        try:
            chunk = chunk.model_dump()
        except Exception:  # pylint: disable=broad-exception-caught
            return _delta_reasoning(getattr(chunk, 'delta', None))
    if not isinstance(chunk, dict):
        return ''
    choices = chunk.get('choices') or []
    if not choices or not isinstance(choices[0], dict):
        return _delta_reasoning(chunk)
    return _delta_reasoning(choices[0].get('delta'))


def _as_text(value: Any) -> str:
    if value is None:
        return ''
    if isinstance(value, str):
        return value
    return str(value)


def _split_content_parts(piece: Any) -> tuple[str, str]:
    """LangChain v1 blocks: type=text vs type=reasoning."""
    if not isinstance(piece, list):
        return _as_text(piece), ''
    texts: list[str] = []
    thoughts: list[str] = []
    for part in piece:
        if isinstance(part, dict):
            typ = str(part.get('type') or '').lower()
            body = _as_text(
                part.get('text') or part.get('reasoning') or part.get('content')
            )
            if typ in {'reasoning', 'thinking', 'thought', 'reasoning_content'}:
                thoughts.append(body)
            else:
                texts.append(body)
        elif part is not None:
            texts.append(_as_text(part))
    return ''.join(texts), ''.join(thoughts)


def chunk_text(chunk: Any) -> tuple[str, str]:
    """Return (content, reasoning_extra) even when LangChain sends content=None."""
    piece = getattr(chunk, 'content', None)
    extra = ''
    if _blank_content(piece):
        piece = ''
    elif isinstance(piece, list):
        piece, extra = _split_content_parts(piece)
    elif not isinstance(piece, str):
        piece = str(piece)
    found = getattr(chunk, 'reasoning_content', None)
    if not found:
        kwargs = getattr(chunk, 'additional_kwargs', None) or {}
        if isinstance(kwargs, dict):
            found = kwargs.get('reasoning_content') or kwargs.get('reasoning')
    if not found:
        meta = getattr(chunk, 'response_metadata', None) or {}
        if isinstance(meta, dict):
            found = meta.get('reasoning_content') or meta.get('reasoning')
    if isinstance(found, str) and found:
        extra = found + extra
    elif found and not extra:
        extra = str(found)
    return piece, extra


def _earliest(*matches: re.Match[str] | None) -> re.Match[str] | None:
    found = [m for m in matches if m]
    if not found:
        return None
    return min(found, key=lambda m: m.start())


def split_think(
    text: str, in_think: bool, ns: str = '', never_think: bool = False
) -> tuple[str, str, bool, str, bool]:
    """Strip think blocks.

    ``ns == "mm:"`` means MiniMax-style: ignore every bare think tag until
    ``</mm:think>``. Empty text does not latch never_think (shadow think).
    """
    if never_think:
        return text or '', '', False, '', True

    content: list[str] = []
    reasoning: list[str] = []
    rest = text or ''
    ns = (ns or '').lower()
    while rest:
        if in_think:
            end = NS_END_RE.search(rest) if ns == 'mm:' else BARE_END_RE.search(rest)
            if not end:
                reasoning.append(rest)
                break
            reasoning.append(rest[: end.start()])
            rest = rest[end.end() :]
            in_think = False
            ns = ''
            never_think = True
            if rest:
                content.append(rest)
            break
        ns_open = NS_START_RE.search(rest)
        bare_open = BARE_START_RE.search(rest)
        match = _earliest(ns_open, bare_open)
        if not match:
            content.append(rest)
            never_think = True
            break
        content.append(rest[: match.start()])
        ns = 'mm:' if NS_START_RE.match(match.group(0)) else ''
        rest = rest[match.end() :]
        in_think = True
    return ''.join(content), ''.join(reasoning), in_think, ns, never_think


class ThinkFeed:
    """Turn-scoped parser for null-token and namespaced think modes."""

    def __init__(
        self,
        in_think: bool = False,
        ns: str = '',
        never_think: bool = False,
        shadow_think: bool = False,
    ) -> None:
        self.in_think = in_think
        self.ns = ns
        self.never_think = never_think
        self.shadow_think = shadow_think

    def feed(self, piece: str) -> tuple[str, str]:
        """Return (visible, thought) for one content piece."""
        text = piece or ''
        if self.never_think:
            return text, ''

        if not self.in_think and not self.shadow_think and not text:
            self.shadow_think = True
            return '', ''

        if self.shadow_think:
            if not text:
                return '', ''
            # First non-blank after null tokens: gpt-oss answer — unless
            # MiniMax is opening a namespaced block on this same chunk.
            stripped = text.lstrip()
            if NS_START_RE.match(stripped):
                self.shadow_think = False
            else:
                self.shadow_think = False
                self.in_think = False
                self.ns = ''
                self.never_think = True
                return text, ''

        visible, thought, self.in_think, self.ns, self.never_think = split_think(
            text, self.in_think, self.ns, self.never_think
        )
        return visible, thought

    def feed_chunk(self, chunk: Any) -> tuple[str, str]:
        """Return (visible, thought) including reasoning_content extras."""
        piece, extra = chunk_text(chunk)
        visible, thought = self.feed(piece)
        if extra:
            thought = extra + thought
        return visible, thought


def _attach_reasoning(message: Any, extra: str) -> None:
    if not extra or message is None:
        return
    kwargs = getattr(message, 'additional_kwargs', None)
    if not isinstance(kwargs, dict):
        try:
            message.additional_kwargs = {'reasoning_content': extra}
        except Exception:  # pylint: disable=broad-exception-caught
            return
        return
    prev = kwargs.get('reasoning_content') or ''
    if extra not in prev:
        kwargs['reasoning_content'] = prev + extra if prev else extra


def _patch_openai_choice_delta() -> None:
    """Keep non-OpenAI delta fields (reasoning_content) on the pydantic model."""
    try:
        from openai.types.chat.chat_completion_chunk import ChoiceDelta
    except Exception:  # pylint: disable=broad-exception-caught
        return
    if getattr(ChoiceDelta, '_spur_reasoning', False):
        return
    try:
        cfg = ChoiceDelta.model_config
        extra = cfg.get('extra') if hasattr(cfg, 'get') else getattr(cfg, 'extra', None)
        if extra != 'allow':
            if isinstance(cfg, dict):
                cfg['extra'] = 'allow'
            else:
                try:
                    cfg.extra = 'allow'
                except Exception:  # pylint: disable=broad-exception-caught
                    pass
            if hasattr(ChoiceDelta, 'model_rebuild'):
                ChoiceDelta.model_rebuild(force=True)
        ChoiceDelta._spur_reasoning = True  # type: ignore[attr-defined]
    except Exception:  # pylint: disable=broad-exception-caught
        pass


def _patch_langchain_delta() -> None:
    """LangChain drops delta.reasoning_content; copy it onto additional_kwargs."""
    try:
        import langchain_openai.chat_models.base as base
    except Exception:  # pylint: disable=broad-exception-caught
        return
    orig = getattr(base, '_convert_delta_to_message_chunk', None)
    if orig is None or getattr(orig, '_spur_patched', False):
        return

    def wrapped(_dict: Mapping[str, Any], default_class: Any) -> Any:
        msg = orig(_dict, default_class)
        _attach_reasoning(msg, _delta_reasoning(_dict))
        return msg

    wrapped._spur_patched = True  # type: ignore[attr-defined]
    base._convert_delta_to_message_chunk = wrapped


def _patch_langchain_chunk() -> None:
    try:
        from langchain_openai.chat_models.base import BaseChatOpenAI
    except Exception:  # pylint: disable=broad-exception-caught
        try:
            from langchain_openai.chat_models.base import ChatOpenAI as BaseChatOpenAI
        except Exception:  # pylint: disable=broad-exception-caught
            return
    orig = getattr(BaseChatOpenAI, '_convert_chunk_to_generation_chunk', None)
    if orig is None or getattr(orig, '_spur_patched', False):
        return

    def wrapped(self: Any, chunk: Any, default_chunk_class: Any,
                base_generation_info: Any = None) -> Any:
        gen = orig(self, chunk, default_chunk_class, base_generation_info)
        extra = reasoning_from_openai_chunk(chunk)
        if gen is not None and extra:
            _attach_reasoning(getattr(gen, 'message', None), extra)
        return gen

    wrapped._spur_patched = True  # type: ignore[attr-defined]
    BaseChatOpenAI._convert_chunk_to_generation_chunk = wrapped


def install_reasoning_patches() -> None:
    """Idempotent: keep OpenAI-compat reasoning_content through LangChain."""
    _patch_openai_choice_delta()
    _patch_langchain_delta()
    _patch_langchain_chunk()


install_reasoning_patches()
