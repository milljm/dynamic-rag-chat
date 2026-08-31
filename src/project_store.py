# pylint: disable=too-many-lines  # project + tools + git in one store
"""Persistent coding workspace, plus imported project directories.

Default root is ``vector_dir/projects/workspace``. ``add_project`` registers
an existing directory in place (a git repo stays a git repo). Named fences
land in the active root when Coding is on. ``<RUN:path args>`` /
``<READ:path>`` are own-line tags (same shape as NEED_GOLD). Execution is
argv-only: python3 for .py, node for .js — no shell, no path escape.
Workers under ``agents/`` are ordinary scripts in the project.
Persistent **tools** live in ``vector_dir/tools/`` (outside the project) and
grow over time. Write them with ``tool:name.py`` fences; run with
``<TOOL:name.py>``. cwd is the active project so installs land there.
``<GIT:status>`` is a local git agent (no remotes). Non-git projects must
be initialized only after the user agrees.
"""
from __future__ import annotations

import json
import os
import re
import shlex
import shutil
import subprocess
from pathlib import Path
from typing import Any

MAX_FILE_BYTES = 512_000
MAX_OUTPUT = 32_000
RUN_TIMEOUT = 45
TOOL_TIMEOUT = 120
MAX_PROJECT_OPS = 8
MAX_RUN_ARGS = 16
MAX_ARG_LEN = 200
MAX_LIST_FILES = 400
MAX_TREE_LINES = 80
WORKSPACE = 'workspace'
SCRATCH_ID = 'workspace'
TOOLS = 'tools'
REGISTRY = 'registry.json'
SKIP_DIRS = frozenset({
    '.git', 'node_modules', '__pycache__', '.venv', 'venv',
    'dist', '.tox', '.mypy_cache', '.pytest_cache', '.ruff_cache',
    '.next', '.cache', '.turbo', 'eggs', '.eggs',
})

_FILE_TOKEN = re.compile(
    r'^(?:\./)?[\w.@+-]+(?:/[\w.@+-]+)*\.[A-Za-z0-9]{1,8}$',
)
_FENCE_OPEN = re.compile(r'^( {0,3})(`{3,}|~{3,})([^\n]*)$')
_PROJECT_TAG = re.compile(
    r'^[ \t]*<(RUN|READ|GIT|TOOL):\s*([^>\n]+?)\s*>[ \t]*$',
    re.I,
)
_SLUG = re.compile(r'[^a-z0-9]+')
_PLACEHOLDERS = frozenset({
    'filename', 'file', 'name', 'path', 'example',
    'script', 'yourfile', 'your-file', 'your_file',
})
_INTERPRETERS = {
    '.py': ('python3', 'python'),
    '.js': ('node',),
    '.mjs': ('node',),
}
GIT_ALLOW = frozenset({
    'init', 'status', 'add', 'commit', 'diff', 'log', 'branch',
    'checkout', 'switch', 'restore', 'show', 'stash', 'mv', 'rm',
    'tag', 'blame', 'ls-files', 'rev-parse', 'config',
})


def scratch_root(vector_dir: str) -> Path:
    """``vector_dir/projects/workspace`` — the default scratch project."""
    return Path(vector_dir) / 'projects' / WORKSPACE


def project_root(vector_dir: str) -> Path:
    """Active project's root (scratch workspace or an imported dir)."""
    return Path(_active_record(vector_dir)['path'])


def tools_root(vector_dir: str) -> Path:
    """``vector_dir/tools`` — Coding's toolkit, outside any project."""
    return Path(vector_dir) / TOOLS


def list_projects(vector_dir: str) -> list[dict[str, Any]]:
    """Registered projects; scratch first."""
    return list(_load_registry(vector_dir)['projects'])


def snapshot(vector_dir: str) -> dict[str, Any]:
    """Sidebar payload: active id, projects, files."""
    data = _load_registry(vector_dir)
    files = list_files(vector_dir)
    return {
        'active': data['active'],
        'projects': data['projects'],
        'files': files,
        'tools': list_tools(vector_dir),
        'truncated': len(files) >= MAX_LIST_FILES,
    }


def add_project(vector_dir: str, raw_path: str) -> dict[str, Any]:
    """Register an existing directory in place. Selects it."""
    dest, err = _existing_dir(raw_path)
    if err:
        return {'ok': False, 'error': err}
    data = _load_registry(vector_dir)
    for rec in data['projects']:
        if _same_path(rec.get('path') or '', dest):
            data['active'] = rec['id']
            _save_registry(vector_dir, data)
            return _result(vector_dir, rec)
    rec = _imported_record(dest, {p['id'] for p in data['projects']})
    data['projects'].append(rec)
    data['active'] = rec['id']
    _save_registry(vector_dir, data)
    return _result(vector_dir, rec)


def select_project(vector_dir: str, ident: str) -> dict[str, Any]:
    """Switch the active project. Writes and runs follow."""
    want = (ident or '').strip()
    data = _load_registry(vector_dir)
    rec = next((p for p in data['projects'] if p['id'] == want), None)
    if rec is None:
        return {'ok': False, 'error': f'No such project: {want}'}
    if rec['kind'] == 'imported' and not Path(rec['path']).is_dir():
        return {'ok': False, 'error': f'Not a directory: {rec["path"]}'}
    data['active'] = rec['id']
    _save_registry(vector_dir, data)
    return _result(vector_dir, rec)


def remove_project(vector_dir: str, ident: str) -> dict[str, Any]:
    """Drop a registered import. Does not delete files on disk."""
    want = (ident or '').strip()
    if want == SCRATCH_ID:
        return {'ok': False, 'error': 'Cannot remove the workspace'}
    data = _load_registry(vector_dir)
    kept = [p for p in data['projects'] if p['id'] != want]
    if len(kept) == len(data['projects']):
        return {'ok': False, 'error': f'No such project: {want}'}
    data['projects'] = kept
    if data['active'] == want:
        data['active'] = SCRATCH_ID
    _save_registry(vector_dir, data)
    rec = next(p for p in kept if p['id'] == data['active'])
    return _result(vector_dir, rec)


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
    """Absolute path inside the active project, or None."""
    name = safe_relpath(rel)
    if not name:
        return None
    root = project_root(vector_dir).resolve()
    dest = (root / name).resolve()
    if dest == root or root not in dest.parents:
        return None
    return dest


def list_files(vector_dir: str) -> list[dict]:
    """[{path, chars}] sorted by path. Skips hidden / vendor dirs."""
    root = project_root(vector_dir)
    if not root.is_dir():
        return []
    try:
        root = root.resolve()
    except OSError:
        return []
    out: list[dict] = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [
            name for name in dirnames
            if name not in SKIP_DIRS and not name.startswith('.')
        ]
        dirnames.sort()
        for name in sorted(filenames):
            if name.startswith('.'):
                continue
            path = Path(dirpath) / name
            if not path.is_file():
                continue
            rel = path.relative_to(root).as_posix()
            try:
                chars = path.stat().st_size
            except OSError:
                chars = 0
            out.append({'path': rel, 'chars': chars})
            if len(out) >= MAX_LIST_FILES:
                out.sort(key=lambda row: row['path'])
                return out
    out.sort(key=lambda row: row['path'])
    return out


def tree_listing(vector_dir: str) -> str:
    """Human listing for the prompt, or ``(empty workspace)``."""
    rec = _active_record(vector_dir)
    rows = list_files(vector_dir)
    git = 'yes' if rec.get('git') else 'no'
    header = f'project: {rec["name"]}\nroot: {rec["path"]}\ngit: {git}'
    if not rows:
        empty = (
            '(empty workspace)' if rec['kind'] == 'scratch'
            else '(empty project)'
        )
        return f'{header}\n{empty}'
    lines = [header]
    shown = rows[:MAX_TREE_LINES]
    lines.extend(f'- {row["path"]} ({row["chars"]} bytes)' for row in shown)
    extra = len(rows) - len(shown)
    if extra > 0:
        more = f'{extra}+' if len(rows) >= MAX_LIST_FILES else str(extra)
        lines.append(f'({more} more files)')
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
    """Unlink a project file. True if something was removed."""
    dest = resolve(vector_dir, rel)
    if dest is None or not dest.is_file():
        return False
    try:
        dest.unlink()
    except OSError:
        return False
    root = project_root(vector_dir).resolve()
    parent = dest.parent
    while parent != root and parent.is_dir() and not any(parent.iterdir()):
        try:
            parent.rmdir()
        except OSError:
            break
        parent = parent.parent
    return True


def resolve_tool(vector_dir: str, rel: str) -> Path | None:
    """Absolute path inside ``vector_dir/tools``, or None."""
    name = safe_relpath(rel)
    if not name:
        return None
    root = tools_root(vector_dir).resolve()
    dest = (root / name).resolve()
    if dest == root or root not in dest.parents:
        return None
    return dest


def list_tools(vector_dir: str) -> list[dict]:
    """[{path, chars}] in the tools namespace."""
    root = tools_root(vector_dir)
    if not root.is_dir():
        return []
    out: list[dict] = []
    try:
        root = root.resolve()
    except OSError:
        return []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if not d.startswith('.')]
        dirnames.sort()
        for name in sorted(filenames):
            if name.startswith('.'):
                continue
            path = Path(dirpath) / name
            if not path.is_file():
                continue
            rel = path.relative_to(root).as_posix()
            try:
                chars = path.stat().st_size
            except OSError:
                chars = 0
            out.append({'path': rel, 'chars': chars})
    out.sort(key=lambda row: row['path'])
    return out


def tools_listing(vector_dir: str) -> str:
    """Human listing of the toolkit, or ``(no tools yet)``."""
    rows = list_tools(vector_dir)
    if not rows:
        return '(no tools yet)'
    lines = [f'- {row["path"]} ({row["chars"]} bytes)' for row in rows]
    return '\n'.join(lines)


def read_tool(vector_dir: str, rel: str) -> str | None:
    """UTF-8 tool text, or None."""
    dest = resolve_tool(vector_dir, rel)
    if dest is None or not dest.is_file():
        return None
    try:
        return dest.read_text(encoding='utf-8')
    except (OSError, UnicodeDecodeError):
        return None


def write_tool(vector_dir: str, rel: str, text: str) -> str | None:
    """Write a tool outside the project. Return stored relative path."""
    dest = resolve_tool(vector_dir, rel)
    if dest is None:
        return None
    payload = text or ''
    if len(payload.encode('utf-8')) > MAX_FILE_BYTES:
        return None
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(payload, encoding='utf-8')
    return dest.relative_to(tools_root(vector_dir).resolve()).as_posix()


def delete_tool(vector_dir: str, rel: str) -> bool:
    """Unlink a tool. True if something was removed."""
    dest = resolve_tool(vector_dir, rel)
    if dest is None or not dest.is_file():
        return False
    try:
        dest.unlink()
    except OSError:
        return False
    root = tools_root(vector_dir).resolve()
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
        name, ns = _fence_target(info)
        code = '\n'.join(body)
        key = f'{ns}:{name.lower()}' if name else ''
        if name and key not in used and code.strip():
            used.add(key)
            out.append({'file': name, 'text': code.rstrip('\n') + '\n', 'ns': ns})
    return out


def persist_named_fences(vector_dir: str, text: str) -> list[str]:
    """Write named fences into the project or tools namespace."""
    written: list[str] = []
    for art in extract_named_fences(text):
        if art.get('ns') == 'tool':
            stored = write_tool(vector_dir, art['file'], art['text'])
            if stored:
                written.append(f'tool:{stored}')
            continue
        stored = write_file(vector_dir, art['file'], art['text'])
        if stored:
            written.append(stored)
    return written


def _fence_target(info: str) -> tuple[str | None, str]:
    raw = (info or '').strip()
    tool = re.search(r'(?:^|[\s])tool\s*:\s*([^\s"\']+)', raw, re.I)
    if tool:
        return safe_relpath(tool.group(1)), 'tool'
    named = re.search(r'(?:filename|file|title|path)\s*[:=]\s*["\']?([^\s"\']+)', raw, re.I)
    if named:
        return safe_relpath(named.group(1)), 'project'
    colon = re.match(r'^[A-Za-z0-9_+-]+\s*:\s*(\S+)$', raw)
    if colon:
        return safe_relpath(colon.group(1)), 'project'
    for part in raw.split():
        hit = safe_relpath(part)
        if hit:
            return hit, 'project'
    return None, 'project'


def interpreter_for(rel: str) -> str | None:
    """First available binary for this extension, or None."""
    ext = Path(rel).suffix.lower()
    for candidate in _INTERPRETERS.get(ext, ()):
        if shutil.which(candidate):
            return candidate
    return None


def run_file(
    vector_dir: str,
    rel: str,
    args: list[str] | None = None,
) -> dict[str, Any]:
    """Run a project file. Never a shell. cwd is the active project root."""
    name = safe_relpath(rel)
    dest = resolve(vector_dir, rel) if name else None
    extra = _clean_args(args)
    if not name or dest is None or not dest.is_file():
        return _run_result(name or rel, '', f'No such file: {rel}', 127, '')
    root = project_root(vector_dir).resolve()
    return _exec_script(
        name, dest, extra, cwd=root, pythonpath=str(root), timeout=RUN_TIMEOUT,
    )


def run_tool(
    vector_dir: str,
    rel: str,
    args: list[str] | None = None,
) -> dict[str, Any]:
    """Run a tool. Script lives in tools/; cwd is the active project."""
    name = safe_relpath(rel)
    dest = resolve_tool(vector_dir, rel) if name else None
    extra = _clean_args(args)
    if not name or dest is None or not dest.is_file():
        return _run_result(name or rel, '', f'No such tool: {rel}', 127, '')
    project = project_root(vector_dir).resolve()
    troot = tools_root(vector_dir).resolve()
    pypath = str(troot) + os.pathsep + str(project)
    return _exec_script(
        name, dest, extra, cwd=project, pythonpath=pypath, timeout=TOOL_TIMEOUT,
    )


def _exec_script(
    name: str,
    dest: Path,
    extra: list[str],
    cwd: Path,
    pythonpath: str,
    timeout: int,
) -> dict[str, Any]:
    exe = interpreter_for(name)
    if not exe:
        return _run_result(name, '', f'Cannot run {name} (need python3 or node).', 127, '')
    cwd.mkdir(parents=True, exist_ok=True)
    argv = [exe, str(dest), *extra]
    displayed = _cmd_str([exe, name, *extra])
    path = os.environ.get('PATH', '/usr/bin')
    for bin_dir in (cwd / '.venv' / 'bin', cwd / '.miniforge' / 'bin', cwd / 'bin'):
        if bin_dir.is_dir():
            path = str(bin_dir) + os.pathsep + path
    env = {
        'PATH': path,
        'HOME': str(cwd),
        'PYTHONUNBUFFERED': '1',
        'PYTHONPATH': pythonpath,
        'NODE_PATH': pythonpath,
        'LANG': 'C.UTF-8',
    }
    try:
        proc = subprocess.run(  # noqa: S603  # argv list, no shell
            argv,
            cwd=str(cwd),
            capture_output=True,
            text=True,
            timeout=timeout,
            env=env,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        out = _clip(getattr(exc, 'stdout', None) or '')
        err = _clip((getattr(exc, 'stderr', None) or '') + f'\n(timed out after {timeout}s)')
        return _run_result(name, out, err, 124, displayed)
    except OSError as exc:
        return _run_result(name, '', str(exc), 127, displayed)
    return _run_result(
        name,
        _clip(proc.stdout or ''),
        _clip(proc.stderr or ''),
        int(proc.returncode),
        displayed,
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


def format_git(result: dict[str, Any]) -> str:
    """Block the model sees after ``<GIT:…>``."""
    return _format_block('PROJECT_GIT', result)


def format_tool(result: dict[str, Any]) -> str:
    """Block the model sees after ``<TOOL:…>``."""
    return _format_block('PROJECT_TOOL', result)


def _format_block(kind: str, result: dict[str, Any]) -> str:
    path = result.get('path') or ''
    cmd = result.get('cmd') or ''
    code = result.get('code')
    stdout = result.get('stdout') or ''
    stderr = result.get('stderr') or ''
    label = f'{kind} {path}'.rstrip() if path else kind
    parts = [f'=== {label}  cmd={cmd}  exit={code} ===']
    if stdout:
        parts.append(stdout.rstrip())
    if stderr:
        parts.append('--- stderr ---')
        parts.append(stderr.rstrip())
    if not stdout and not stderr:
        parts.append('(no output)')
    return '\n'.join(parts)


def run_git(vector_dir: str, argv: list[str] | None = None) -> dict[str, Any]:
    """Local git only. cwd is the active project. No remotes."""
    parts = _clean_args(argv)
    if not parts:
        return _run_result('git', '', 'Missing git command', 127, '')
    sub = parts[0].lower()
    extra = parts[1:]
    if sub not in GIT_ALLOW:
        return _run_result(
            'git', '',
            f'git {sub} is not allowed (local only; no remotes).',
            127, '',
        )
    if any(_git_escape(part) for part in extra):
        return _run_result('git', '', 'Refusing git path escape.', 127, '')
    extra = _git_config_args(sub, extra)
    if extra is None:
        return _run_result('git', '', 'git config is local only.', 127, '')
    exe = shutil.which('git')
    if not exe:
        return _run_result('git', '', 'git is not installed.', 127, '')
    root = project_root(vector_dir).resolve()
    cmd = [exe, '-c', 'color.ui=never', '-c', 'core.pager=cat', sub, *extra]
    displayed = _cmd_str(['git', sub, *extra])
    try:
        proc = subprocess.run(  # noqa: S603  # argv list, no shell
            cmd,
            cwd=str(root),
            capture_output=True,
            text=True,
            timeout=RUN_TIMEOUT,
            env=_git_env(root),
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        out = _clip(getattr(exc, 'stdout', None) or '')
        err = _clip((getattr(exc, 'stderr', None) or '') + f'\n(timed out after {RUN_TIMEOUT}s)')
        return _run_result('git', out, err, 124, displayed)
    except OSError as exc:
        return _run_result('git', '', str(exc), 127, displayed)
    return _run_result(
        'git',
        _clip(proc.stdout or ''),
        _clip(proc.stderr or ''),
        int(proc.returncode),
        displayed,
    )


def apply_tag(
    vector_dir: str,
    action: str,
    rel: str,
    args: list[str] | None = None,
) -> tuple[str, str]:
    """Run a coding tag. Return ``(project_result, status)``."""
    extra = args or []
    if action == 'read':
        text = read_file(vector_dir, rel)
        body = (
            format_read(rel, text) if text is not None
            else f'=== PROJECT_READ {rel} ===\n(missing)'
        )
        return body, f'Reading {rel}…'
    if action == 'git':
        return format_git(run_git(vector_dir, [rel, *extra])), f'Git {rel}…'
    if action == 'tool':
        return format_tool(run_tool(vector_dir, rel, extra)), f'Tool {rel}…'
    result = run_file(vector_dir, rel, extra)
    return format_run(result), f'Running {rel}…'


def take_project_tag(text: str) -> tuple[str, str | None, str | None, list[str]]:
    """Split ``(visible, action, path, args)``. Action is run/read/git/tool or None."""
    if not text:
        return '', None, None, []
    lines = text.split('\n')
    for i, line in enumerate(lines):
        action, name, args = _own_line_tag(line)
        if not name:
            continue
        visible = '\n'.join(lines[:i] + lines[i + 1:]).rstrip()
        return visible, action, name, args
    return text, None, None, []


class ProjectNeedFeed:
    """Hold back an in-progress own-line RUN/READ/GIT/TOOL tag."""

    def __init__(self):
        self.buf = ''
        self.action: str | None = None
        self.path: str | None = None
        self.args: list[str] = []

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
            action, name, args = _own_line_tag(line)
            if name:
                self.action = action
                self.path = name
                self.args = args
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
        action, name, args = _own_line_tag(leftover)
        if name:
            self.action = action
            self.path = name
            self.args = args
            return ''
        return leftover


def _own_line_tag(line: str) -> tuple[str | None, str | None, list[str]]:
    match = _PROJECT_TAG.match(line or '')
    if not match:
        return None, None, []
    action = match.group(1).lower()
    raw = match.group(2)
    if action == 'git':
        return _parse_git(raw)
    if action == 'read':
        name = safe_relpath(raw)
        args: list[str] = []
    else:
        name, args = _parse_run(raw)
    if not name:
        return None, None, []
    stem = Path(name).stem.lower()
    if stem in _PLACEHOLDERS:
        return None, None, []
    return action, name, args


def _parse_git(raw: str) -> tuple[str | None, str | None, list[str]]:
    text = (raw or '').strip()
    if not text:
        return None, None, []
    try:
        parts = shlex.split(text, posix=True)
    except ValueError:
        return None, None, []
    if not parts:
        return None, None, []
    sub = parts[0].lower()
    if sub in _PLACEHOLDERS:
        return None, None, []
    extra = parts[1:]
    if len(extra) > MAX_RUN_ARGS:
        return None, None, []
    if any('\x00' in part or len(part) > MAX_ARG_LEN for part in extra):
        return None, None, []
    if any(_git_escape(part) for part in extra):
        return None, None, []
    return 'git', sub, extra


def _parse_run(raw: str) -> tuple[str | None, list[str]]:
    text = (raw or '').strip()
    if not text:
        return None, []
    try:
        parts = shlex.split(text, posix=True)
    except ValueError:
        return None, []
    if not parts:
        return None, []
    name = safe_relpath(parts[0])
    if not name:
        return None, []
    extra = parts[1:]
    if len(extra) > MAX_RUN_ARGS:
        return None, []
    if any('\x00' in part or len(part) > MAX_ARG_LEN for part in extra):
        return None, []
    return name, extra


def _clean_args(args: list[str] | None) -> list[str]:
    out: list[str] = []
    for raw in args or ():
        text = str(raw)
        if not text or '\x00' in text or len(text) > MAX_ARG_LEN:
            continue
        out.append(text)
        if len(out) >= MAX_RUN_ARGS:
            break
    return out


def _cmd_str(parts: list[str]) -> str:
    return ' '.join(shlex.quote(part) for part in parts)


def _has_git(path: Path) -> bool:
    return (path / '.git').exists()


def _git_escape(part: str) -> bool:
    if part in ('-C', '--git-dir', '--work-tree'):
        return True
    if part.startswith('--git-dir=') or part.startswith('--work-tree='):
        return True
    if part.startswith('/') or part.startswith('~'):
        return True
    if part == '..' or part.startswith('../') or '/../' in part:
        return True
    return False


def _git_config_args(sub: str, extra: list[str]) -> list[str] | None:
    if sub != 'config':
        return extra
    blocked = ('--global', '--system', '--file', '-f')
    if any(part in blocked or part.startswith('--file=') for part in extra):
        return None
    return extra if '--local' in extra else ['--local', *extra]


def _git_env(root: Path) -> dict[str, str]:
    home = os.environ.get('HOME') or str(root)
    env = {
        'PATH': os.environ.get('PATH', '/usr/bin'),
        'HOME': home,
        'LANG': 'C.UTF-8',
        'GIT_TERMINAL_PROMPT': '0',
        'GIT_PAGER': 'cat',
        'GIT_EDITOR': 'true',
        'GC_AUTO': '0',
    }
    for key in (
        'GIT_AUTHOR_NAME', 'GIT_AUTHOR_EMAIL',
        'GIT_COMMITTER_NAME', 'GIT_COMMITTER_EMAIL',
    ):
        val = os.environ.get(key)
        if val:
            env[key] = val
    return env


def _might_become_tag(line: str) -> bool:
    if '\n' in line:
        return False
    stripped = line.lstrip(' \t')
    if stripped == '':
        return True
    lower = stripped.lower()
    for prefix in ('<run:', '<read:', '<git:', '<tool:'):
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


def _registry_file(vector_dir: str) -> Path:
    return Path(vector_dir) / 'projects' / REGISTRY


def _scratch_record(vector_dir: str) -> dict[str, Any]:
    path = scratch_root(vector_dir)
    try:
        path = path.resolve()
    except OSError:
        pass
    return {
        'id': SCRATCH_ID,
        'name': WORKSPACE,
        'kind': 'scratch',
        'path': str(path),
        'git': _has_git(path),
    }


def _load_registry(vector_dir: str) -> dict[str, Any]:
    path = _registry_file(vector_dir)
    loaded: Any = None
    if path.is_file():
        try:
            loaded = json.loads(path.read_text(encoding='utf-8'))
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            loaded = None
    if not isinstance(loaded, dict):
        loaded = {'active': SCRATCH_ID, 'projects': []}
    return _normalize_registry(vector_dir, loaded)


def _normalize_registry(vector_dir: str, data: dict) -> dict[str, Any]:
    scratch = _scratch_record(vector_dir)
    imported: list[dict[str, Any]] = []
    seen_paths = {scratch['path']}
    seen_ids = {SCRATCH_ID}
    for rec in data.get('projects') or []:
        if not isinstance(rec, dict):
            continue
        ident = str(rec.get('id') or '').strip()
        raw = str(rec.get('path') or '').strip()
        if rec.get('kind') == 'scratch' or ident == SCRATCH_ID:
            continue
        if not ident or not raw or ident in seen_ids:
            continue
        try:
            dest = Path(raw).resolve()
        except (OSError, RuntimeError):
            dest = Path(raw)
        key = str(dest)
        if key in seen_paths:
            continue
        seen_paths.add(key)
        seen_ids.add(ident)
        imported.append({
            'id': ident,
            'name': str(rec.get('name') or dest.name or ident),
            'kind': 'imported',
            'path': key,
            'git': (dest / '.git').exists(),
        })
    active = str(data.get('active') or SCRATCH_ID)
    if active not in seen_ids:
        active = SCRATCH_ID
    return {'active': active, 'projects': [scratch, *imported]}


def _save_registry(vector_dir: str, data: dict) -> None:
    path = _registry_file(vector_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(_normalize_registry(vector_dir, data), indent=2) + '\n'
    tmp = path.with_name(f'.{REGISTRY}.tmp')
    tmp.write_text(payload, encoding='utf-8')
    tmp.replace(path)


def _active_record(vector_dir: str) -> dict[str, Any]:
    data = _load_registry(vector_dir)
    rec = next((p for p in data['projects'] if p['id'] == data['active']), None)
    return rec or data['projects'][0]


def _existing_dir(raw: str) -> tuple[Path | None, str | None]:
    text = (raw or '').strip().strip('"\'')
    if not text:
        return None, 'Missing path'
    try:
        dest = Path(os.path.expanduser(text)).resolve()
    except (OSError, RuntimeError):
        return None, 'Could not resolve path'
    if dest == Path(dest.anchor):
        return None, 'Refusing to use the filesystem root'
    if not dest.is_dir():
        return None, f'Not a directory: {dest}'
    return dest, None


def _same_path(stored: str, dest: Path) -> bool:
    try:
        return Path(stored).resolve() == dest
    except (OSError, RuntimeError):
        return False


def _imported_record(dest: Path, used: set[str]) -> dict[str, Any]:
    ident = _unique_id(dest.name, used)
    return {
        'id': ident,
        'name': dest.name or ident,
        'kind': 'imported',
        'path': str(dest),
        'git': (dest / '.git').exists(),
    }


def _unique_id(name: str, used: set[str]) -> str:
    base = _SLUG.sub('-', (name or '').lower()).strip('-')[:40] or 'project'
    if base not in used:
        return base
    n = 2
    while f'{base}-{n}' in used:
        n += 1
    return f'{base}-{n}'


def _result(vector_dir: str, rec: dict[str, Any]) -> dict[str, Any]:
    snap = snapshot(vector_dir)
    snap['ok'] = True
    snap['error'] = None
    snap['project'] = rec
    return snap
