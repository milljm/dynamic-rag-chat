"""Think-tag splitter used by the TUI and the Spur HTTP adapter.

MiniMax emits <mm:think>…</mm:think> and often *mentions* <think> / </think>
inside that block (code samples, prose). Treating `mm:` as optional used to
close reasoning at the inner tag and stall the stream.

Thinking is once-per-turn. Two discovery modes:

- Tag-based: first non-empty token contains <think> / <mm:think> / …
- Shadow: first token is blank (None, "", []). gpt-oss-120b and friends
  stream reasoning on another field while content stays empty. The first
  non-blank token is the answer — never_think latches, even if that
  answer *mentions* <think>.

Reset the parser when the user sends a new turn.
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


def split_think(
    text: str, in_think: bool, ns: str = "", never_think: bool = False
) -> tuple[str, str, bool, str, bool]:
    """Strip think blocks. The closer must match the opener's namespace.

    Once a block closes, never_think latches True and the rest of the turn
    (this chunk and later chunks) is visible content — even if the model
    talks about <think> / </think>. Empty text does not latch (shadow think).
    """
    if never_think:
        return text or "", "", False, "", True

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
            never_think = True
            if rest:
                content.append(rest)
            break
        match = THINK_START_RE.search(rest)
        if not match:
            content.append(rest)
            never_think = True
            break
        content.append(rest[: match.start()])
        ns = tag_ns(match)
        rest = rest[match.end() :]
        in_think = True
    return "".join(content), "".join(reasoning), in_think, ns, never_think


class ThinkFeed:
    """Turn-scoped parser. Blank first tokens are gpt-oss-style shadow think."""

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
            # First non-blank token after null reasoning: that's the answer.
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
