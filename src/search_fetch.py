"""Mid-turn web search, NEED_GOLD-style.

The answering model (after the pre-processor) may emit
``<NEED_SEARCH:query>`` on its own line and stop. The streamer runs a
live lookup, injects WEB_SEARCH into FILES, and resumes — one history
turn. Used when the tagger missed that the question needs the web.

Also holds back ``<NEED_GOLD:…>`` in the same feed so ``<need_`` is not
leaked while the tag is still incomplete.
"""
from __future__ import annotations

import re

MAX_SEARCH_FETCHES = 2
SEARCH_LIST_MAX = 40

_GOLD_OPEN = '<need_gold:'
_SEARCH_OPEN = '<need_search:'

MID_LINE_RE = re.compile(
    r'^[ \t]*<(NEED_GOLD|NEED_SEARCH):\s*([^>\n]+?)\s*>[ \t]*$',
    re.IGNORECASE,
)

_GOLD_PLACEHOLDERS = frozenset({
    'filename', 'file', 'name', 'basename', 'exact-basename',
    'filepath', 'path', 'example', 'yourfile', 'your-file', 'your_file',
})
_SEARCH_PLACEHOLDERS = frozenset({
    'query', 'search', 'example', 'q', 'keywords',
    'search-query', 'search_query', 'your-query', 'your_query',
    'yoursearch', 'question',
})


def _clean(raw: str) -> str:
    """Strip quotes/backticks around a tag capture."""
    return (raw or '').strip().strip('"\'`')


def _is_placeholder(kind: str, value: str) -> bool:
    """True for cookbook tokens — not a real file or query."""
    raw = (value or '').strip().lower()
    if not raw:
        return True
    names = _GOLD_PLACEHOLDERS if kind == 'gold' else _SEARCH_PLACEHOLDERS
    if raw in names:
        return True
    if kind == 'gold':
        stem = raw.rsplit('.', 1)[0]
        return stem in names
    return False


def _own_line(line: str) -> tuple[str, str] | None:
    """Return ``('gold'|'search', value)`` if this line is a real fetch tag."""
    match = MID_LINE_RE.match(line)
    if not match:
        return None
    tag = match.group(1).lower()
    kind = 'gold' if tag == 'need_gold' else 'search'
    value = _clean(match.group(2))
    if not value or _is_placeholder(kind, value):
        return None
    return kind, value


def _might_become_mid_tag(line: str) -> bool:
    """True if this incomplete line could still become NEED_GOLD or NEED_SEARCH."""
    if '\n' in line:
        return False
    stripped = line.lstrip(' \t')
    if stripped == '':
        return True
    lower = stripped.lower()
    for open_tag in (_GOLD_OPEN, _SEARCH_OPEN):
        if not (open_tag.startswith(lower) or lower.startswith(open_tag)):
            continue
        close = lower.find('>')
        if close == -1:
            return True
        return stripped[close + 1:].strip() == ''
    return False


def search_status(queries: list[str]) -> str:
    """Searching web… [query] — list clipped to 40 chars."""
    listed = ', '.join(str(q).strip() for q in queries if str(q).strip())
    if len(listed) > SEARCH_LIST_MAX:
        listed = listed[:SEARCH_LIST_MAX - 3].rstrip(' ,') + '...'
    if listed:
        return f'Searching web… [{listed}]'
    return 'Searching web…'


def take_need_search(text: str) -> tuple[str, str | None]:
    """Split ``(visible, query)``. Query is None when no own-line NEED_SEARCH."""
    if not text:
        return '', None
    lines = text.split('\n')
    for i, line in enumerate(lines):
        hit = _own_line(line)
        if not hit or hit[0] != 'search':
            continue
        visible = '\n'.join(lines[:i] + lines[i + 1:]).rstrip()
        return visible, hit[1]
    return text, None


class MidTurnFeed:
    """
    ### MidTurnFeed

    Hold back an in-progress own-line ``<NEED_GOLD:file>`` or
    ``<NEED_SEARCH:query>`` so the tag is not streamed. ``kind`` is
    ``gold`` or ``search`` when complete; ``value`` is the filename or
    query.

    *Class init args:*
        .. code-block:: python
            (none)

    *Usage:*
        - per stream:
            .. code-block:: python
                feed = MidTurnFeed()
                visible, done = feed.feed(chunk)
                if done:
                    kind, value = feed.kind, feed.value
    """

    def __init__(self):
        self.buf = ''
        self.kind: str | None = None
        self.value: str | None = None

    @property
    def filename(self) -> str | None:
        """NEED_GOLD basename, else None."""
        return self.value if self.kind == 'gold' else None

    @property
    def query(self) -> str | None:
        """NEED_SEARCH query, else None."""
        return self.value if self.kind == 'search' else None

    def feed(self, chunk: str) -> tuple[str, bool]:
        """Return (safe_to_emit, tag_complete)."""
        if self.kind or not chunk:
            return '', bool(self.kind)
        self.buf += chunk
        out: list[str] = []
        while True:
            nl = self.buf.find('\n')
            if nl == -1:
                if _might_become_mid_tag(self.buf):
                    return ''.join(out), False
                out.append(self.buf)
                self.buf = ''
                return ''.join(out), False
            line = self.buf[:nl]
            rest = self.buf[nl + 1:]
            hit = _own_line(line)
            if hit:
                self.kind, self.value = hit
                self.buf = ''
                return ''.join(out), True
            out.append(line + '\n')
            self.buf = rest

    def flush(self) -> str:
        """Emit leftovers, or commit a trailing own-line tag (model stopped)."""
        leftover = self.buf
        self.buf = ''
        if self.kind:
            return ''
        hit = _own_line(leftover)
        if hit:
            self.kind, self.value = hit
            return ''
        return leftover
