"""Mid-turn gold file fetch.

The assistant may emit ``<NEED_GOLD:README.md>`` on its own line and stop.
The streamer fetches that file from gold, injects it, and resumes — one
history turn.

Inline / backticked / in-sentence tags are talk, not a fetch. Models love
to quote the cookbook (``gold-fetch tags like <NEED_GOLD:README.md>``).
"""
from __future__ import annotations

import re

MAX_GOLD_FETCHES = 2
RECALL_LIST_MAX = 40
# Entire line is the tag. Mid-sentence copies of the cookbook do not match.
NEED_GOLD_LINE_RE = re.compile(
    r'^[ \t]*<NEED_GOLD:\s*([^>\n]+?)\s*>[ \t]*$',
    re.IGNORECASE,
)
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


def _own_line_name(line: str) -> str | None:
    """Basename if this line is a real fetch tag, else None."""
    match = NEED_GOLD_LINE_RE.match(line)
    if not match:
        return None
    name = _clean_name(match.group(1))
    if not name or _is_placeholder(name):
        return None
    return name


def _might_become_tag(line: str) -> bool:
    """True if this incomplete line could still become an own-line fetch."""
    if '\n' in line:
        return False
    stripped = line.lstrip(' \t')
    if stripped == '':
        return True
    lower = stripped.lower()
    if not (_TAG_OPEN.startswith(lower) or lower.startswith(_TAG_OPEN)):
        return False
    close = lower.find('>')
    if close == -1:
        return True
    return stripped[close + 1:].strip() == ''


def recall_status(names: list[str]) -> str:
    """Recalling Documents… [a.py, b.md, c...] — list clipped to 40 chars."""
    listed = ', '.join(str(n).strip() for n in names if str(n).strip())
    if len(listed) > RECALL_LIST_MAX:
        listed = listed[:RECALL_LIST_MAX - 3].rstrip(' ,') + '...'
    if listed:
        return f'Recalling Documents… [{listed}]'
    return 'Recalling Documents…'


def take_need_gold(text: str) -> tuple[str, str | None]:
    """Split ``(visible, filename)``. Filename is None when no own-line tag."""
    if not text:
        return '', None
    lines = text.split('\n')
    for i, line in enumerate(lines):
        name = _own_line_name(line)
        if not name:
            continue
        visible = '\n'.join(lines[:i] + lines[i + 1:]).rstrip()
        return visible, name
    return text, None


class GoldNeedFeed:
    """Hold back an in-progress own-line <NEED_GOLD:…> so the tag is not streamed."""

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
            nl = self.buf.find('\n')
            if nl == -1:
                if _might_become_tag(self.buf):
                    return ''.join(out), False
                out.append(self.buf)
                self.buf = ''
                return ''.join(out), False
            line = self.buf[:nl]
            rest = self.buf[nl + 1:]
            name = _own_line_name(line)
            if name:
                self.filename = name
                self.buf = ''
                return ''.join(out), True
            out.append(line + '\n')
            self.buf = rest

    def flush(self) -> str:
        """Emit leftovers, or commit a trailing own-line tag (model stopped)."""
        leftover = self.buf
        self.buf = ''
        if self.filename:
            return ''
        name = _own_line_name(leftover)
        if name:
            self.filename = name
            return ''
        return leftover
