"""Mid-turn gold file fetch.

The assistant may emit ``<NEED_GOLD:README.md>`` and stop. The streamer
fetches that file from gold, injects it, and resumes — one history turn.
"""
from __future__ import annotations

import re

MAX_GOLD_FETCHES = 2
RECALL_LIST_MAX = 40
NEED_GOLD_RE = re.compile(r'<NEED_GOLD:\s*([^>\n]+?)\s*>', re.IGNORECASE)
_TAG_OPEN = '<need_gold:'

# Models copy the cookbook: <NEED_GOLD:filename> / <NEED_GOLD:exact-basename>.
_PLACEHOLDERS = frozenset({
    'filename',
    'file',
    'name',
    'basename',
    'exact-basename',
    'filepath',
    'path',
    'example',
    'yourfile',
    'your-file',
    'your_file',
})


def _clean_name(raw: str) -> str:
    """Strip quotes/backticks around a NEED_GOLD capture."""
    return (raw or '').strip().strip('"\'`')


def _is_placeholder(name: str) -> bool:
    """True for cookbook tokens like filename / filename.py — not a real file."""
    raw = (name or '').strip().lower()
    if not raw:
        return True
    if raw in _PLACEHOLDERS:
        return True
    stem = raw.rsplit('.', 1)[0]
    return stem in _PLACEHOLDERS


def recall_status(names: list[str]) -> str:
    """Recalling Documents… [a.py, b.md, c...] — list clipped to 40 chars."""
    listed = ', '.join(str(n).strip() for n in names if str(n).strip())
    if len(listed) > RECALL_LIST_MAX:
        listed = listed[:RECALL_LIST_MAX - 3].rstrip(' ,') + '...'
    if listed:
        return f'Recalling Documents… [{listed}]'
    return 'Recalling Documents…'


def take_need_gold(text: str) -> tuple[str, str | None]:
    """Split ``(visible, filename)``. Filename is None when no real tag."""
    if not text:
        return '', None
    for match in NEED_GOLD_RE.finditer(text):
        name = _clean_name(match.group(1))
        if not name or _is_placeholder(name):
            continue
        visible = (text[:match.start()] + text[match.end():]).rstrip()
        return visible, name
    return text, None


class GoldNeedFeed:
    """Hold back an in-progress <NEED_GOLD:…> so the tag is not streamed."""

    def __init__(self):
        self.buf = ''
        self.filename: str | None = None

    def feed(self, chunk: str) -> tuple[str, bool]:
        """Return (safe_to_emit, tag_complete)."""
        if self.filename or not chunk:
            return '', bool(self.filename)
        self.buf += chunk
        out: list[str] = []
        while True:
            match = NEED_GOLD_RE.search(self.buf)
            if not match:
                emit, self.buf = _hold_open_tag(self.buf)
                out.append(emit)
                return ''.join(out), False
            name = _clean_name(match.group(1))
            if name and not _is_placeholder(name):
                self.filename = name
                out.append(self.buf[:match.start()])
                self.buf = ''
                return ''.join(out), True
            out.append(self.buf[:match.end()])
            self.buf = self.buf[match.end():]

    def flush(self) -> str:
        """Emit leftovers when the stream ends without a complete tag."""
        leftover = self.buf
        self.buf = ''
        return leftover


def _hold_open_tag(buf: str) -> tuple[str, str]:
    """Keep a possible tag prefix in the hold buffer."""
    lower = buf.lower()
    idx = lower.rfind('<')
    if idx == -1:
        return buf, ''
    frag = lower[idx:]
    if _TAG_OPEN.startswith(frag) or frag.startswith(_TAG_OPEN):
        return buf[:idx], buf[idx:]
    return buf, ''
