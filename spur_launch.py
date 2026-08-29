#!/usr/bin/env python3
"""Boot Spur as one process: FastAPI adapter + built static UI.

Used by ``./chat.py --spur``. Remaining CLI flags (``--assistant-mode``,
model hosts, …) are forwarded to the adapter so it shares ``.chat.yaml``.
"""
from __future__ import annotations

import importlib.util
import os
import shutil
import subprocess
import sys
import time
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SPUR_DIR = ROOT / 'spur'
SERVER_SCRIPT = ROOT / 'spur-server.py'
DEFAULT_URL = 'http://127.0.0.1:8765'


def strip_spur_flags(argv: list[str]) -> list[str]:
    """Drop ``--spur`` / ``--spur-rebuild`` so spur-server argparse is clean."""
    skip = {'--spur', '--spur-rebuild'}
    return [a for a in argv if a not in skip]


# Back-compat alias for tests written against the first launcher.
strip_spur_flag = strip_spur_flags


def find_ui_root(spur_dir: Path = SPUR_DIR) -> Path | None:
    """Return the directory that contains index.html, if any."""
    for folder in (
        spur_dir / 'dist' / 'client',
        spur_dir / 'dist',
        spur_dir / '.output' / 'public',
    ):
        if (folder / 'index.html').is_file():
            return folder
    return None


def _source_mtime(spur_dir: Path) -> float:
    """Newest mtime among UI sources that should trigger a rebuild."""
    newest = 0.0
    for path in (
        spur_dir / 'package.json',
        spur_dir / 'vite.config.ts',
        spur_dir / 'src',
    ):
        if path.is_file():
            newest = max(newest, path.stat().st_mtime)
        elif path.is_dir():
            for child in path.rglob('*'):
                if child.is_file():
                    newest = max(newest, child.stat().st_mtime)
    return newest


def _needs_build(spur_dir: Path, force: bool) -> bool:
    if force:
        return True
    root = find_ui_root(spur_dir)
    if root is None:
        return True
    index = root / 'index.html'
    return _source_mtime(spur_dir) > index.stat().st_mtime


def _ensure_npm(spur_dir: Path) -> str:
    """Install UI deps if node_modules is missing. Return npm path."""
    npm = shutil.which('npm')
    if not npm:
        sys.exit(
            'npm not found. Install Node (the conda recipe uses `nodejs`) '
            'then re-run ./chat.py --spur.'
        )
    if not (spur_dir / 'node_modules').is_dir():
        print('Installing Spur UI dependencies (npm install)…')
        proc = subprocess.run([npm, 'install'], cwd=spur_dir, check=False)
        if proc.returncode != 0:
            sys.exit('npm install failed in spur/.')
    return npm


def ensure_ui(force: bool = False) -> Path:
    """Build the static UI if missing or stale. Return the folder to serve."""
    if not (SPUR_DIR / 'package.json').is_file():
        sys.exit(f'missing Spur UI at {SPUR_DIR}')
    npm = _ensure_npm(SPUR_DIR)
    if _needs_build(SPUR_DIR, force):
        print('Building Spur UI (first run, or sources changed)…')
        proc = subprocess.run([npm, 'run', 'build'], cwd=SPUR_DIR, check=False)
        if proc.returncode != 0:
            sys.exit('Spur UI build failed. See npm output above.')
    root = find_ui_root(SPUR_DIR)
    if root is None:
        sys.exit('Spur UI build produced no index.html')
    return root


def _load_server():
    """Load spur-server.py (hyphenated filename) as a module."""
    spec = importlib.util.spec_from_file_location('spur_server', SERVER_SCRIPT)
    if spec is None or spec.loader is None:
        sys.exit(f'cannot load {SERVER_SCRIPT}')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def launch(argv: list[str] | None = None) -> int:
    """Build the UI if needed, then run the adapter in this process."""
    if argv is None:
        argv = sys.argv[1:]
    force = '--spur-rebuild' in argv or os.environ.get('SPUR_REBUILD') == '1'
    forwarded = strip_spur_flags(argv)

    if not SERVER_SCRIPT.is_file():
        sys.exit(f'missing {SERVER_SCRIPT}')

    ui_root = ensure_ui(force=force)
    os.environ['SPUR_STATIC'] = str(ui_root)

    # spur-server.py reads sys.argv for ChatOptions.
    sys.argv = [str(SERVER_SCRIPT), *forwarded]

    try:
        import uvicorn
    except ImportError:
        sys.exit('uvicorn is required. `uv pip install -r requirements.txt`')

    mod = _load_server()
    if hasattr(mod, 'mount_ui'):
        mod.mount_ui(str(ui_root))

    url = os.environ.get('SPUR_URL', DEFAULT_URL)
    host = os.environ.get('SPUR_HOST', '127.0.0.1')
    port = int(os.environ.get('SPUR_PORT', '8765'))
    print(f'Spur: {url}  ·  API docs: {url}/docs')
    print('Ctrl-C stops the server.')
    if os.environ.get('SPUR_NO_BROWSER') != '1':
        # Give uvicorn a tick to bind, then open.
        def _open() -> None:
            time.sleep(0.6)
            try:
                webbrowser.open(url)
            except Exception:  # pylint: disable=broad-exception-caught
                pass

        import threading
        threading.Thread(target=_open, daemon=True).start()

    uvicorn.run(mod.app, host=host, port=port, log_level='info')
    return 0


def main() -> int:
    """CLI entry for ``python spur_launch.py``."""
    return launch()


if __name__ == '__main__':
    sys.exit(main())
