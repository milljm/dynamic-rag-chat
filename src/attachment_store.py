"""Whole-file cabinet next to Chroma.

Gold RAG stays the semantic index (import-dir chunks, attachment chunks).
Chat-attached files also live here as themselves so NEED_GOLD / filename
mentions / the Spur Documents widget do not have to reverse-engineer
ParentDocumentRetriever.
"""
from __future__ import annotations

import os
import re
from pathlib import Path

_SAFE = re.compile(r'[^A-Za-z0-9._-]+')


def attachments_root(vector_dir: str) -> Path:
    """``vector_dir/attachments``."""
    return Path(vector_dir) / 'attachments'


def safe_filename(name: str) -> str:
    """Basename only; no path traversal."""
    base = os.path.basename(name or '').strip() or 'file'
    return _SAFE.sub('_', base)[:180]


def put_attachment(vector_dir: str, name: str, text: str) -> str:
    """Write UTF-8 text; return the stored basename."""
    root = attachments_root(vector_dir)
    root.mkdir(parents=True, exist_ok=True)
    dest = root / safe_filename(name)
    dest.write_text(text or '', encoding='utf-8')
    return dest.name


def get_attachment(vector_dir: str, name: str) -> str | None:
    """Return file text, case-insensitive, or None."""
    root = attachments_root(vector_dir)
    if not root.is_dir():
        return None
    want = safe_filename(name).lower()
    for path in root.iterdir():
        if path.is_file() and path.name.lower() == want:
            try:
                return path.read_text(encoding='utf-8')
            except OSError:
                return None
    return None


def list_attachments(vector_dir: str) -> list[dict]:
    """[{name, chars}] sorted by name."""
    root = attachments_root(vector_dir)
    if not root.is_dir():
        return []
    out = []
    for path in sorted(root.iterdir(), key=lambda p: p.name.lower()):
        if not path.is_file() or path.name.startswith('.'):
            continue
        try:
            chars = path.stat().st_size
        except OSError:
            chars = 0
        out.append({'name': path.name, 'chars': chars})
    return out


def delete_attachment(vector_dir: str, name: str) -> bool:
    """Unlink the file. True if something was removed."""
    root = attachments_root(vector_dir)
    if not root.is_dir():
        return False
    want = safe_filename(name).lower()
    removed = False
    for path in list(root.iterdir()):
        if path.is_file() and path.name.lower() == want:
            try:
                path.unlink()
                removed = True
            except OSError:
                pass
    return removed
