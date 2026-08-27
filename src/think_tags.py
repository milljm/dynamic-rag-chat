"""Think-tag splitter used by the TUI and the Spur HTTP adapter.

MiniMax emits <mm:think>…</mm:think> and often *mentions* <think> / </think>
inside that block (code samples, prose). Treating `mm:` as optional used to
close reasoning at the inner tag and stall the stream.
"""
from __future__ import annotations

import re
from typing import Any

THINK_START_RE = re.compile(r"<\s*(mm:)?(think|thinking|reasoning)\s*>", re.I)
THINK_END_RE = re.compile(r"</\s*(mm:)?(think|thinking|reasoning)\s*>", re.I)


def tag_ns(match: re.Match[str] | None) -> str:
    if not match:
        return ""
    return (match.group(1) or "").lower()


def chunk_text(chunk: Any) -> tuple[str, str]:
    """Return (content, reasoning_extra) even when LangChain sends content=None."""
    piece = getattr(chunk, "content", None)
    if not isinstance(piece, str):
        piece = "" if piece is None else str(piece)
    extra = getattr(chunk, "reasoning_content", None)
    if not extra:
        kwargs = getattr(chunk, "additional_kwargs", None) or {}
        if isinstance(kwargs, dict):
            extra = kwargs.get("reasoning_content") or kwargs.get("reasoning")
    if not isinstance(extra, str):
        extra = "" if extra is None else str(extra)
    return piece, extra


def split_think(
    text: str, in_think: bool, ns: str = ""
) -> tuple[str, str, bool, str]:
    """Strip think blocks. The closer must match the opener's namespace."""
    content: list[str] = []
    reasoning: list[str] = []
    rest = text or ""
    while rest:
        if in_think:
            match = THINK_END_RE.search(rest)
            if not match:
                reasoning.append(rest)
                break
            if tag_ns(match) != ns:
                reasoning.append(rest[: match.end()])
                rest = rest[match.end() :]
                continue
            reasoning.append(rest[: match.start()])
            rest = rest[match.end() :]
            in_think = False
            ns = ""
        else:
            match = THINK_START_RE.search(rest)
            if not match:
                content.append(rest)
                break
            content.append(rest[: match.start()])
            ns = tag_ns(match)
            rest = rest[match.end() :]
            in_think = True
    return "".join(content), "".join(reasoning), in_think, ns
