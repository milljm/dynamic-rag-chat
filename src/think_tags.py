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
from typing import Any

# Namespaced (MiniMax). Do not make `mm:` optional — that is what used to
# treat an inner <think> as a real tag.
NS_START_RE = re.compile(r"<\s*mm:(think|thinking|reasoning)\s*>", re.I)
NS_END_RE = re.compile(r"</\s*mm:(think|thinking|reasoning)\s*>", re.I)
# Bare tags. Must NOT match <mm:think>.
BARE_START_RE = re.compile(r"<\s*(think|thinking|reasoning)\s*>", re.I)
BARE_END_RE = re.compile(r"</\s*(think|thinking|reasoning)\s*>", re.I)

# Kept for TUI callers that still import these names.
THINK_START_RE = re.compile(
    r"<\s*(mm:)?(think|thinking|reasoning)\s*>", re.I
)
THINK_END_RE = re.compile(
    r"</\s*(mm:)?(think|thinking|reasoning)\s*>", re.I
)


def tag_ns(match: re.Match[str] | None) -> str:
    if not match:
        return ""
    return (match.group(1) or "").lower()


def _blank_content(piece: Any) -> bool:
    """LangChain sends None, '', or [] on reasoning-only chunks."""
    return piece is None or piece == "" or piece == []


def chunk_text(chunk: Any) -> tuple[str, str]:
    """Return (content, reasoning_extra) even when LangChain sends content=None."""
    piece = getattr(chunk, "content", None)
    if _blank_content(piece):
        piece = ""
    elif not isinstance(piece, str):
        piece = str(piece)
    extra = getattr(chunk, "reasoning_content", None)
    if not extra:
        kwargs = getattr(chunk, "additional_kwargs", None) or {}
        if isinstance(kwargs, dict):
            extra = kwargs.get("reasoning_content") or kwargs.get("reasoning")
    if not isinstance(extra, str):
        extra = "" if extra is None else str(extra)
    return piece, extra


def _earliest(*matches: re.Match[str] | None) -> re.Match[str] | None:
    found = [m for m in matches if m]
    if not found:
        return None
    return min(found, key=lambda m: m.start())


def split_think(
    text: str, in_think: bool, ns: str = "", never_think: bool = False
) -> tuple[str, str, bool, str, bool]:
    """Strip think blocks.

    ``ns == "mm:"`` means MiniMax-style: ignore every bare think tag until
    ``</mm:think>``. Empty text does not latch never_think (shadow think).
    """
    if never_think:
        return text or "", "", False, "", True

    content: list[str] = []
    reasoning: list[str] = []
    rest = text or ""
    ns = (ns or "").lower()
    while rest:
        if in_think:
            end = NS_END_RE.search(rest) if ns == "mm:" else BARE_END_RE.search(rest)
            if not end:
                reasoning.append(rest)
                break
            reasoning.append(rest[: end.start()])
            rest = rest[end.end() :]
            in_think = False
            ns = ""
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
        ns = "mm:" if NS_START_RE.match(match.group(0)) else ""
        rest = rest[match.end() :]
        in_think = True
    return "".join(content), "".join(reasoning), in_think, ns, never_think


class ThinkFeed:
    """Turn-scoped parser for null-token and namespaced think modes."""

    def __init__(
        self,
        in_think: bool = False,
        ns: str = "",
        never_think: bool = False,
        shadow_think: bool = False,
    ) -> None:
        self.in_think = in_think
        self.ns = ns
        self.never_think = never_think
        self.shadow_think = shadow_think

    def feed(self, piece: str) -> tuple[str, str]:
        """Return (visible, thought) for one content piece."""
        text = piece or ""
        if self.never_think:
            return text, ""

        if not self.in_think and not self.shadow_think and not text:
            self.shadow_think = True
            return "", ""

        if self.shadow_think:
            if not text:
                return "", ""
            # First non-blank after null tokens: gpt-oss answer — unless
            # MiniMax is opening a namespaced block on this same chunk.
            stripped = text.lstrip()
            if NS_START_RE.match(stripped):
                self.shadow_think = False
            else:
                self.shadow_think = False
                self.in_think = False
                self.ns = ""
                self.never_think = True
                return text, ""

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
