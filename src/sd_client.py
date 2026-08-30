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
_SAFE_CAPTION = re.compile(r'^[A-Za-z0-9 .,!?\-]{1,80}$')

MAGICK_OPS = (
    'resize', 'rotate', 'blur', 'sharpen', 'grayscale',
    'negate', 'modulate', 'brightness', 'border', 'caption',
)

# Fresh generate / explicit image request.
IMAGE_QUERY = re.compile(
    r'(?ix)'
    r'\b(draw|paint|sketch|illustrate|render)\s+(me\s+)?(a|an|the|some)\b'
    r'|\b(draw|paint|sketch|illustrate|render|imagine|generate|create|make)\b'
    r'.{0,80}\b(image|picture|pic\b|photo|illustration|portrait|logo|icon|wallpaper|artwork|poster)\b'
    r'|\b(image|picture|illustration|portrait|logo)\s+of\b'
    r'|\b(txt2img|img2img|stable\s+diffusion)\b'
    r'|\b(redraw|re-?generate|re-?draw)\b'
    r'|\b(the|that|this)\s+(image|picture|photo|drawing)\b'
    r'|\b(add|put)\b.{0,40}\b(border|caption|text|frame)\b'
)
# Follow-up tweaks — only when a last PNG exists.
IMAGE_EDIT = re.compile(
    r'(?ix)'
    r'\b(now\s+)?(make|render|paint|do)\s+(it|that|this|the)\b'
    r'|\b(darker|brighter|warmer|cooler|redder|bluer|softer|sharper|moodier)\b'
    r'|\b(more|less)\s+(contrast|saturation|shadows?|highlights?|vignette)\b'
    r'|\b(add|put|give)\s+(a\s+)?(border|caption|text|frame|watermark)\b'
    r'|\b(crop|resize|rotate|flip|grayscale)\b'
    r'|\b(tweak|adjust|fix|change|edit)\s+(it|that|this)\b'
)
MAGICK_QUERY = re.compile(
    r'(?ix)'
    r'\b(border|caption|watermark|label|resize|rotate|flip|'
    r'grayscale|black\s*and\s*white|blur|sharpen|vignette)\b'
)


def wants_sd(query: str, has_last: bool = False) -> bool:
    """True when this user text should hit the Automatic1111 agent."""
    text = query or ''
    if IMAGE_QUERY.search(text):
        return True
    if has_last and IMAGE_EDIT.search(text):
        return True
    return False


def has_generated_images(vector_dir: str) -> bool:
    """True when vector_dir/generated has at least one picture."""
    folder = os.path.join(str(vector_dir or ''), 'generated')
    if not os.path.isdir(folder):
        return False
    for name in os.listdir(folder):
        if name.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
            return True
    return False


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


def _txt2img_payload(
    prompt: str,
    negative_prompt: str = '',
    steps: int = 20,
    width: int = 768,
    height: int = 768,
    seed: int = -1,
    checkpoint: str = '',
) -> dict[str, Any]:
    """Body for /sdapi/v1/txt2img, including an optional checkpoint swap."""
    payload: dict[str, Any] = {
        'prompt': prompt,
        'negative_prompt': negative_prompt or '',
        'steps': max(1, min(50, int(steps or 20))),
        'width': _align(width),
        'height': _align(height),
        'cfg_scale': 7,
        'seed': int(seed),
        'sampler_name': 'Euler a',
    }
    if (checkpoint or '').strip():
        payload['override_settings'] = {
            'sd_model_checkpoint': checkpoint.strip(),
        }
        payload['override_settings_restore_afterwards'] = False
    return payload


def _img2img_payload(
    b64: str,
    prompt: str,
    negative_prompt: str = '',
    denoising: float = 0.45,
    steps: int = 20,
    checkpoint: str = '',
) -> dict[str, Any]:
    """Body for /sdapi/v1/img2img."""
    payload: dict[str, Any] = {
        'prompt': prompt,
        'negative_prompt': negative_prompt or '',
        'init_images': [b64],
        'denoising_strength': max(0.05, min(0.95, float(denoising or 0.45))),
        'steps': max(1, min(50, int(steps or 20))),
        'cfg_scale': 7,
        'sampler_name': 'Euler a',
    }
    if (checkpoint or '').strip():
        payload['override_settings'] = {
            'sd_model_checkpoint': checkpoint.strip(),
        }
        payload['override_settings_restore_afterwards'] = False
    return payload


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
    current = ''
    try:
        opts = _get(f'{origin}/sdapi/v1/options', timeout=timeout)
        if isinstance(opts, dict):
            current = str(opts.get('sd_model_checkpoint') or '')
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError):
        current = ''
    return {
        'ok': True, 'error': None, 'models': names, 'current': current,
        'url': url, 'source': 'automatic1111',
    }


def txt2img(
    host: str,
    prompt: str,
    negative_prompt: str = '',
    steps: int = 20,
    width: int = 768,
    height: int = 768,
    seed: int = -1,
    checkpoint: str = '',
    timeout: float = 180.0,
) -> bytes:
    """POST /sdapi/v1/txt2img and return the first PNG/JPEG bytes."""
    origin = normalize_sd_url(host)
    if not origin:
        raise RuntimeError('Stable Diffusion URL is empty.')
    payload = _txt2img_payload(
        prompt, negative_prompt=negative_prompt, steps=steps,
        width=width, height=height, seed=seed, checkpoint=checkpoint,
    )
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
    checkpoint: str = '',
    timeout: float = 180.0,
) -> bytes:
    """POST /sdapi/v1/img2img using a PNG/JPEG already on disk."""
    origin = normalize_sd_url(host)
    if not origin:
        raise RuntimeError('Stable Diffusion URL is empty.')
    b64 = base64.b64encode(image_bytes).decode('ascii')
    payload = _img2img_payload(
        b64, prompt, negative_prompt=negative_prompt,
        denoising=denoising, steps=steps, checkpoint=checkpoint,
    )
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
    if op == 'caption':
        if not arg or not _SAFE_CAPTION.match(arg):
            raise ValueError('caption needs short plain text (letters/numbers, max 80).')
        return [
            '-gravity', 'South', '-fill', 'white', '-pointsize', '28',
            '-annotate', '+0+16', arg,
        ]
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
    if op == 'border':
        return ['-bordercolor', 'black', '-border', arg or '16']
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
