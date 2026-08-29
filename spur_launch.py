#!/usr/bin/env python3
"""Boot spur-server.py and the React UI from one command.

Used by ``./chat.py --spur``. Remaining CLI flags (``--assistant-mode``,
model hosts, …) are forwarded to the adapter so it shares ``.chat.yaml``.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SPUR_DIR = ROOT / 'spur'
SERVER_SCRIPT = ROOT / 'spur-server.py'
DEFAULT_API = 'http://127.0.0.1:8765'
DEFAULT_UI = 'http://127.0.0.1:8080'


def strip_spur_flag(argv: list[str]) -> list[str]:
    """Drop ``--spur`` so argparse in spur-server.py never sees it."""
    return [a for a in argv if a != '--spur']


def _wait_http(url: str, timeout: float = 30.0) -> bool:
    """Return True once ``url`` answers, else False."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=1.5) as resp:
                if 200 <= getattr(resp, 'status', 200) < 500:
                    return True
        except (urllib.error.URLError, TimeoutError, OSError):
            time.sleep(0.25)
    return False


def _ensure_npm(spur_dir: Path) -> None:
    """Install UI deps if node_modules is missing."""
    if (spur_dir / 'node_modules').is_dir():
        return
    npm = shutil.which('npm')
    if not npm:
        sys.exit(
            'npm not found. Install Node (the conda recipe uses `nodejs`) '
            'then re-run ./chat.py --spur.'
        )
    print('Installing Spur UI dependencies (npm install)…')
    proc = subprocess.run([npm, 'install'], cwd=spur_dir, check=False)
    if proc.returncode != 0:
        sys.exit('npm install failed in spur/.')


def launch(argv: list[str] | None = None) -> int:
    """Start the adapter and Vite. Return the adapter's exit code."""
    if argv is None:
        argv = sys.argv[1:]
    forwarded = strip_spur_flag(argv)

    if not SERVER_SCRIPT.is_file():
        sys.exit(f'missing {SERVER_SCRIPT}')
    if not (SPUR_DIR / 'package.json').is_file():
        sys.exit(f'missing Spur UI at {SPUR_DIR}')

    _ensure_npm(SPUR_DIR)

    api = os.environ.get('SPUR_API', DEFAULT_API)
    ui = os.environ.get('SPUR_UI', DEFAULT_UI)
    python = sys.executable
    server_cmd = [python, str(SERVER_SCRIPT), *forwarded]
    print(f'Starting adapter on {api} …')
    server = subprocess.Popen(server_cmd, cwd=str(ROOT))

    health = api.rstrip('/') + '/api/health'
    if not _wait_http(health, timeout=45.0):
        server.terminate()
        sys.exit(
            f'spur-server.py did not become ready at {health}. '
            'Check the traceback above.'
        )

    npm = shutil.which('npm')
    if not npm:
        server.terminate()
        sys.exit('npm not found; adapter is up but the UI cannot start.')

    env = os.environ.copy()
    env['VITE_CHAT_API'] = api
    print(f'Starting Spur UI on {ui} (VITE_CHAT_API={api}) …')
    ui_proc = subprocess.Popen(
        [npm, 'run', 'dev', '--', '--strictPort'],
        cwd=str(SPUR_DIR),
        env=env,
    )

    _wait_http(ui, timeout=45.0)
    try:
        webbrowser.open(ui)
    except Exception:  # pylint: disable=broad-exception-caught
        pass
    print(f'Spur: {ui}  ·  API docs: {api}/docs')
    print('Ctrl-C stops both processes.')

    try:
        while True:
            s_code = server.poll()
            u_code = ui_proc.poll()
            if s_code is not None:
                ui_proc.terminate()
                return s_code
            if u_code is not None:
                server.terminate()
                return u_code
            time.sleep(0.4)
    except KeyboardInterrupt:
        print('\nStopping Spur…')
        ui_proc.terminate()
        server.terminate()
        return 0


def main() -> int:
    """CLI entry for ``python spur_launch.py``."""
    return launch()


if __name__ == '__main__':
    sys.exit(main())
