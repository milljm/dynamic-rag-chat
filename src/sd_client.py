"""Automatic1111 REST client and a safe ImageMagick wrapper."""
from __future__ import annotations

import base64
import json
import os
import re
import shutil
import subprocess
import urllib.error
import urllib.request
from typing import Any

_BLANK = frozenset({'', 'none', 'not_set', 'null', '~'})
_SAFE_ARG = re.compile(r'^[0-9A-Za-zx%+.,\-]+$')

MAGICK_OPS = (
    'resize', 'rotate', 'blur', 'sharpen', 'grayscale',
    'negate', 'modulate', 'brightness',
)


def sd_enabled(host: str | None) -> bool:
    """True when a Stable Diffusion server URL is configured."""
    return bool(normalize_sd_url(host))


def normalize_sd_url(host: str | None) -> str:
    """Strip trailing /sdapi/v1 so we can append endpoints."""
    base = (host or '').strip().rstrip('/')
    if not base or base.lower() in _BLANK:
        return ''
    for suffix in ('/sdapi/v1', '/sdapi', '/docs'):
        if base.lower().endswith(suffix):
            base = base[: -len(suffix)].rstrip('/')
            break
    return base


def _align(value: int, lo: int = 256, hi: int = 1280) -> int:
    raw = max(lo, min(hi, int(value or lo)))
    return max(lo, (raw // 64) * 64)


def _post(url: str, payload: dict, timeout: float) -> dict[str, Any]:
    body = json.dumps(payload).encode('utf-8')
    req = urllib.request.Request(
        url, data=body, method='POST',
        headers={'Content-Type': 'application/json'},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode('utf-8', errors='replace'))


def _get(url: str, timeout: float = 5.0) -> Any:
    req = urllib.request.Request(url, method='GET')
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode('utf-8', errors='replace'))


def ping_sd(host: str, timeout: float = 5.0) -> dict[str, Any]:
    """List Automatic1111 checkpoints. OpenAI /models will not work here."""
    origin = normalize_sd_url(host)
    if not origin:
        return {'ok': False, 'error': 'Stable Diffusion URL is empty.', 'models': []}
    url = f'{origin}/sdapi/v1/sd-models'
    try:
        payload = _get(url, timeout=timeout)
    except urllib.error.HTTPError as exc:
        return {'ok': False, 'error': f'{exc.code} {exc.reason}', 'models': []}
    except urllib.error.URLError as exc:
        return {'ok': False, 'error': str(getattr(exc, 'reason', exc)), 'models': []}
    except (TimeoutError, json.JSONDecodeError, OSError) as exc:
        return {'ok': False, 'error': str(exc), 'models': []}
    names: list[str] = []
    rows = payload if isinstance(payload, list) else []
    for item in rows:
        if isinstance(item, dict):
            ident = item.get('title') or item.get('model_name') or item.get('filename')
            if ident:
                names.append(str(ident))
        elif isinstance(item, str) and item:
            names.append(item)
    return {'ok': True, 'error': None, 'models': names, 'url': url, 'source': 'automatic1111'}


def txt2img(
    host: str,
    prompt: str,
    negative_prompt: str = '',
    steps: int = 20,
    width: int = 768,
    height: int = 768,
    seed: int = -1,
    timeout: float = 180.0,
) -> bytes:
    """POST /sdapi/v1/txt2img and return the first PNG/JPEG bytes."""
    origin = normalize_sd_url(host)
    if not origin:
        raise RuntimeError('Stable Diffusion URL is empty.')
    payload = {
        'prompt': prompt,
        'negative_prompt': negative_prompt or '',
        'steps': max(1, min(50, int(steps or 20))),
        'width': _align(width),
        'height': _align(height),
        'cfg_scale': 7,
        'seed': int(seed),
        'sampler_name': 'Euler a',
    }
    data = _post(f'{origin}/sdapi/v1/txt2img', payload, timeout)
    images = data.get('images') if isinstance(data, dict) else None
    if not isinstance(images, list) or not images:
        raise RuntimeError('Automatic1111 returned no images.')
    return base64.b64decode(images[0])


def img2img(
    host: str,
    image_bytes: bytes,
    prompt: str,
    negative_prompt: str = '',
    denoising: float = 0.45,
    steps: int = 20,
    timeout: float = 180.0,
) -> bytes:
    """POST /sdapi/v1/img2img using a PNG/JPEG already on disk."""
    origin = normalize_sd_url(host)
    if not origin:
        raise RuntimeError('Stable Diffusion URL is empty.')
    b64 = base64.b64encode(image_bytes).decode('ascii')
    payload = {
        'prompt': prompt,
        'negative_prompt': negative_prompt or '',
        'init_images': [b64],
        'denoising_strength': max(0.05, min(0.95, float(denoising or 0.45))),
        'steps': max(1, min(50, int(steps or 20))),
        'cfg_scale': 7,
        'sampler_name': 'Euler a',
    }
    data = _post(f'{origin}/sdapi/v1/img2img', payload, timeout)
    images = data.get('images') if isinstance(data, dict) else None
    if not isinstance(images, list) or not images:
        raise RuntimeError('Automatic1111 img2img returned no images.')
    return base64.b64decode(images[0])


def magick_argv(operation: str, argument: str = '') -> list[str]:
    """Return convert extra args, or raise on unknown/unsafe input."""
    op = (operation or '').strip().lower()
    arg = (argument or '').strip()
    if op not in MAGICK_OPS:
        raise ValueError(f'Unknown ImageMagick op {operation!r}. Use: {", ".join(MAGICK_OPS)}')
    if arg and not _SAFE_ARG.match(arg):
        raise ValueError('ImageMagick argument has disallowed characters.')
    if op == 'resize':
        if not arg:
            raise ValueError('resize needs an argument like 1024x1024')
        return ['-resize', arg]
    if op == 'rotate':
        return ['-rotate', arg or '90']
    if op == 'blur':
        return ['-blur', arg or '0x2']
    if op == 'sharpen':
        return ['-sharpen', arg or '0x1']
    if op == 'grayscale':
        return ['-colorspace', 'Gray']
    if op == 'negate':
        return ['-negate']
    if op == 'modulate':
        return ['-modulate', arg or '100,100,100']
    if op == 'brightness':
        return ['-brightness-contrast', arg or '10x0']
    raise ValueError(f'Unknown ImageMagick op {operation!r}')


def run_magick(src: str, dest: str, operation: str, argument: str = '') -> None:
    """Run convert/magick as argv (no shell)."""
    binary = shutil.which('magick') or shutil.which('convert')
    if not binary:
        raise RuntimeError('ImageMagick not installed (need magick or convert).')
    extra = magick_argv(operation, argument)
    cmd = [binary]
    if os.path.basename(binary) == 'magick':
        cmd.append('convert')
    cmd.extend([src, *extra, dest])
    proc = subprocess.run(cmd, check=False, capture_output=True, text=True)
    if proc.returncode != 0:
        err = (proc.stderr or proc.stdout or 'convert failed').strip()
        raise RuntimeError(err[:400])
