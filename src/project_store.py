"""Persistent coding workspace under vector_dir/projects/workspace.

Named fences land here when Coding is on. ``<RUN:path>`` / ``<READ:path>``
are own-line tags (same shape as NEED_GOLD). Execution is argv-only:
python3 for .py, node for .js — no shell, no path escape.
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any

MAX_FILE_BYTES = 512_000
MAX_OUTPUT = 32_000
RUN_TIMEOUT = 20
MAX_PROJECT_OPS = 4
WORKSPACE = 'workspace'

_FILE_TOKEN = re.compile(
    r'^(?:\./)?[\w.@+-]+(?:/[\w.@+-]+)*\.[A-Za-z0-9]{1,8}$',
)
_FENCE_OPEN = re.compile(r'^( {0,3})(`{3,}|~{3,})([^\n]*)$')
_RUN_READ = re.compile(
    r'^[ \t]*<(RUN|READ):\s*([^>\n]+?)\s*>[ \t]*$',
    re.I,
)
_PLACEHOLDERS = frozenset({
    'filename', 'file', 'name', 'path', 'example',
    'script', 'yourfile', 'your-file', 'your_file',
})
_INTERPRETERS = {
    '.py': ('python3', 'python'),
    '.js': ('node',),
    '.mjs': ('node',),
}


def project_root(vector_dir: str) -> Path:
    """``vector_dir/projects/workspace``."""
    return Path(vector_dir) / 'projects' / WORKSPACE


def is_filename(raw: str) -> bool:
    """Relative path with an extension; no spaces or ``..``."""
    token = (raw or '').strip().strip('"\'`')
    if not token or len(token) > 180 or ' ' in token:
        return False
    return bool(_FILE_TOKEN.match(token))


def safe_relpath(raw: str) -> str | None:
    """Jail a user/model path to a relative file inside the workspace."""
    text = (raw or '').strip().strip('"\'`').replace('\\', '/')
    if text.startswith('./'):
        text = text[2:]
    text = text.lstrip('/')
    if not text or text.startswith('.') or '\x00' in text:
        return None
    parts = [p for p in text.split('/') if p]
    if not parts or any(p == '..' or p.startswith('.') for p in parts):
        return None
    joined = '/'.join(parts)
    if not is_filename(joined):
        return None
    return joined


def resolve(vector_dir: str, rel: str) -> Path | None:
    """Absolute path inside the workspace, or None."""
    name = safe_relpath(rel)
    if not name:
        return None
    root = project_root(vector_dir).resolve()
    dest = (root / name).resolve()
    if dest == root or root not in dest.parents:
        return None
    return dest


def list_files(vector_dir: str) -> list[dict]:
    """[{path, chars}] sorted by path."""
    root = project_root(vector_dir)
    if not root.is_dir():
        return []
    out: list[dict] = []
    for path in sorted(root.rglob('*')):
        if not path.is_file() or path.name.startswith('.'):
            continue
        rel = path.relative_to(root).as_posix()
        try:
            chars = path.stat().st_size
        except OSError:
            chars = 0
        out.append({'path': rel, 'chars': chars})
    return out


def tree_listing(vector_dir: str) -> str:
    """Human listing for the prompt, or ``(empty workspace)``."""
    rows = list_files(vector_dir)
    if not rows:
        return '(empty workspace)'
    lines = [f'- {row["path"]} ({row["chars"]} bytes)' for row in rows]
    return '\n'.join(lines)


def read_file(vector_dir: str, rel: str) -> str | None:
    """UTF-8 file text, or None."""
    dest = resolve(vector_dir, rel)
    if dest is None or not dest.is_file():
        return None
    try:
        return dest.read_text(encoding='utf-8')
    except (OSError, UnicodeDecodeError):
        return None


def write_file(vector_dir: str, rel: str, text: str) -> str | None:
    """Write UTF-8 text. Return stored relative path, or None."""
    dest = resolve(vector_dir, rel)
    if dest is None:
        return None
    payload = text or ''
    if len(payload.encode('utf-8')) > MAX_FILE_BYTES:
        return None
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(payload, encoding='utf-8')
    return dest.relative_to(project_root(vector_dir).resolve()).as_posix()


def delete_file(vector_dir: str, rel: str) -> bool:
    """Unlink a workspace file. True if something was removed."""
    dest = resolve(vector_dir, rel)
    if dest is None or not dest.is_file():
        return False
    try:
        dest.unlink()
    except OSError:
        return False
    root = project_root(vector_dir)
    parent = dest.parent
    while parent != root and parent.is_dir() and not any(parent.iterdir()):
        try:
            parent.rmdir()
        except OSError:
            break
        parent = parent.parent
    return True


def extract_named_fences(text: str) -> list[dict[str, str]]:
    """``[{file, text}]`` from fences that name a file in the info line."""
    source = text or ''
    lines = source.split('\n')
    out: list[dict[str, str]] = []
    used: set[str] = set()
    i = 0
    while i < len(lines):
        open_ = _FENCE_OPEN.match(lines[i])
        if not open_:
            i += 1
            continue
        marker = open_.group(2)
        info = open_.group(3) or ''
        close = re.compile(rf'^ {{0,3}}{re.escape(marker)}[ \t]*$')
        body: list[str] = []
        i += 1
        while i < len(lines) and not close.match(lines[i]):
            body.append(lines[i])
            i += 1
        if i < len(lines):
            i += 1
        name = _filename_from_info(info)
        code = '\n'.join(body)
        if name and name.lower() not in used and code.strip():
            used.add(name.lower())
            out.append({'file': name, 'text': code.rstrip('\n') + '\n'})
    return out


def persist_named_fences(vector_dir: str, text: str) -> list[str]:
    """Write named fences into the workspace. Return stored paths."""
    written: list[str] = []
    for art in extract_named_fences(text):
        stored = write_file(vector_dir, art['file'], art['text'])
        if stored:
            written.append(stored)
    return written


def _filename_from_info(info: str) -> str | None:
    raw = (info or '').strip()
    named = re.search(r'(?:filename|file|title|path)\s*[:=]\s*["\']?([^\s"\']+)', raw, re.I)
    if named:
        return safe_relpath(named.group(1))
    colon = re.match(r'^[A-Za-z0-9_+-]+\s*:\s*(\S+)$', raw)
    if colon:
        return safe_relpath(colon.group(1))
    for part in raw.split():
        hit = safe_relpath(part)
        if hit:
            return hit
    return None


def interpreter_for(rel: str) -> str | None:
    """First available binary for this extension, or None."""
    ext = Path(rel).suffix.lower()
    for candidate in _INTERPRETERS.get(ext, ()):
        if shutil.which(candidate):
            return candidate
    return None


def run_file(vector_dir: str, rel: str) -> dict[str, Any]:
    """Run a workspace file. Never a shell. cwd is the workspace root."""
    name = safe_relpath(rel)
    dest = resolve(vector_dir, rel) if name else None
    if not name or dest is None or not dest.is_file():
        return _run_result(name or rel, '', f'No such file: {rel}', 127, '')
    exe = interpreter_for(name)
    if not exe:
        return _run_result(name, '', f'Cannot run {name} (need python3 or node).', 127, '')
    root = project_root(vector_dir).resolve()
    env = {
        'PATH': os.environ.get('PATH', '/usr/bin'),
        'HOME': str(root),
        'PYTHONUNBUFFERED': '1',
        'LANG': 'C.UTF-8',
    }
    try:
        proc = subprocess.run(  # noqa: S603  # argv list, no shell
            [exe, name],
            cwd=str(root),
            capture_output=True,
            text=True,
            timeout=RUN_TIMEOUT,
            env=env,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        out = _clip(getattr(exc, 'stdout', None) or '')
        err = _clip((getattr(exc, 'stderr', None) or '') + f'\n(timed out after {RUN_TIMEOUT}s)')
        return _run_result(name, out, err, 124, f'{exe} {name}')
    except OSError as exc:
        return _run_result(name, '', str(exc), 127, f'{exe} {name}')
    return _run_result(
        name,
        _clip(proc.stdout or ''),
        _clip(proc.stderr or ''),
        int(proc.returncode),
        f'{exe} {name}',
    )


def format_run(result: dict[str, Any]) -> str:
    """Block the model sees after ``<RUN:…>``."""
    path = result.get('path') or ''
    cmd = result.get('cmd') or ''
    code = result.get('code')
    stdout = result.get('stdout') or ''
    stderr = result.get('stderr') or ''
    parts = [f'=== PROJECT_RUN {path}  cmd={cmd}  exit={code} ===']
    if stdout:
        parts.append(stdout.rstrip())
    if stderr:
        parts.append('--- stderr ---')
        parts.append(stderr.rstrip())
    if not stdout and not stderr:
        parts.append('(no output)')
    return '\n'.join(parts)


def format_read(rel: str, text: str) -> str:
    """Block the model sees after ``<READ:…>``."""
    return f'=== PROJECT_READ {rel} ===\n{text}'


def take_project_tag(text: str) -> tuple[str, str | None, str | None]:
    """Split ``(visible, action, path)``. Action is run/read or None."""
    if not text:
        return '', None, None
    lines = text.split('\n')
    for i, line in enumerate(lines):
        action, name = _own_line_tag(line)
        if not name:
            continue
        visible = '\n'.join(lines[:i] + lines[i + 1:]).rstrip()
        return visible, action, name
    return text, None, None


class ProjectNeedFeed:
    """Hold back an in-progress own-line ``<RUN:>`` / ``<READ:>`` tag."""

    def __init__(self):
        self.buf = ''
        self.action: str | None = None
        self.path: str | None = None

    def feed(self, chunk: str) -> tuple[str, bool]:
        """Return (safe_to_emit, tag_complete)."""
        if self.path or not chunk:
            return '', bool(self.path)
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
            action, name = _own_line_tag(line)
            if name:
                self.action = action
                self.path = name
                self.buf = ''
                return ''.join(out), True
            out.append(line + '\n')
            self.buf = rest

    def flush(self) -> str:
        """Emit leftovers, or commit a trailing own-line tag."""
        leftover = self.buf
        self.buf = ''
        if self.path:
            return ''
        action, name = _own_line_tag(leftover)
        if name:
            self.action = action
            self.path = name
            return ''
        return leftover


def _own_line_tag(line: str) -> tuple[str | None, str | None]:
    match = _RUN_READ.match(line or '')
    if not match:
        return None, None
    action = match.group(1).lower()
    name = safe_relpath(match.group(2))
    if not name:
        return None, None
    stem = Path(name).stem.lower()
    if stem in _PLACEHOLDERS:
        return None, None
    return action, name


def _might_become_tag(line: str) -> bool:
    if '\n' in line:
        return False
    stripped = line.lstrip(' \t')
    if stripped == '':
        return True
    lower = stripped.lower()
    for prefix in ('<run:', '<read:'):
        if prefix.startswith(lower) or lower.startswith(prefix):
            close = lower.find('>')
            if close == -1:
                return True
            return stripped[close + 1:].strip() == ''
    return False


def _clip(text: str) -> str:
    if text is None:
        return ''
    blob = str(text)
    if len(blob) <= MAX_OUTPUT:
        return blob
    return blob[:MAX_OUTPUT] + '\n…(truncated)'


def _run_result(path: str, stdout: str, stderr: str, code: int, cmd: str) -> dict[str, Any]:
    return {
        'path': path,
        'stdout': stdout,
        'stderr': stderr,
        'code': code,
        'cmd': cmd,
    }
