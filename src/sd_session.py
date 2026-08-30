"""Accumulated Automatic1111 prompt while Image mode is on."""
from __future__ import annotations

import json
import os
import re
from typing import Any

SESSION_NAME = 'sd_session.json'


def session_path(vector_dir: str) -> str:
    """vector_dir/generated/sd_session.json"""
    return os.path.join(str(vector_dir or ''), 'generated', SESSION_NAME)


def load_session(vector_dir: str) -> dict[str, Any]:
    """Return the prompt stack, or empty dict."""
    path = session_path(vector_dir)
    try:
        with open(path, encoding='utf-8') as handle:
            data = json.load(handle)
    except (OSError, json.JSONDecodeError, TypeError):
        return {}
    if not isinstance(data, dict):
        return {}
    return data


def save_session(vector_dir: str, data: dict[str, Any]) -> None:
    """Write the prompt stack next to generated PNGs."""
    folder = os.path.join(str(vector_dir or ''), 'generated')
    os.makedirs(folder, exist_ok=True)
    path = session_path(vector_dir)
    tmp = path + '.tmp'
    payload = {
        'prompt': str(data.get('prompt') or '').strip(),
        'negative': str(data.get('negative') or '').strip(),
        'width': int(data.get('width') or 768),
        'height': int(data.get('height') or 768),
        'seed': int(data.get('seed') or -1),
    }
    with open(tmp, 'w', encoding='utf-8') as handle:
        json.dump(payload, handle, indent=2)
        handle.write('\n')
    os.replace(tmp, path)


def clear_session(vector_dir: str) -> None:
    """Drop the stack when Image mode turns off."""
    path = session_path(vector_dir)
    try:
        os.remove(path)
    except OSError:
        pass


QUALITY = {
    'pony': (
        'score_9, score_8_up, score_7_up, masterpiece, best quality, highly detailed',
        'score_6, score_5, score_4, worst quality, low quality, jpeg artifacts, blurry, ugly',
    ),
    'illustrious': (
        'masterpiece, best quality, amazing quality, very aesthetic, newest, highly detailed',
        'worst quality, low quality, old, early, jpeg artifacts, blurry, ugly',
    ),
    'sdxl': (
        'masterpiece, best quality, ultra high resolution, highly detailed, sharp focus, 8k',
        'low quality, low resolution, worst quality, jpeg artifacts, blurry, ugly, deformed',
    ),
    'sd15': (
        'masterpiece, best quality, ultra-detailed, highly detailed, sharp focus',
        'low quality, low resolution, worst quality, jpeg artifacts, blurry, ugly, deformed',
    ),
    'flux': (
        '',
        'blurry, low quality, watermark, text',
    ),
}


def checkpoint_flavor(name: str) -> str:
    """Guess Pony / Illustrious / SDXL / SD1.5 / Flux from the checkpoint filename."""
    text = (name or '').lower()
    if any(k in text for k in (
        'pony', 'pdxl', 'ponyv6', 'autismmix', 'prefectpony', 'easypony',
    )):
        return 'pony'
    if any(k in text for k in ('illustrious', 'noobai', 'noobxl')):
        return 'illustrious'
    if 'flux' in text:
        return 'flux'
    if 'sdxl' in text or re.search(r'xl(?:[_.\-]|$)', text):
        return 'sdxl'
    return 'sd15'


def apply_quality(prompt: str, negative: str, flavor: str) -> tuple[str, str]:
    """Prepend flavor quality tags if the agent forgot them."""
    pos, neg = QUALITY.get(flavor) or QUALITY['sdxl']
    text = (prompt or '').strip()
    bad = (negative or '').strip()
    low = text.lower()
    if flavor == 'pony' and 'score_9' not in low and pos:
        text = f'{pos}, {text}' if text else pos
    elif flavor != 'flux' and 'masterpiece' not in low and 'best quality' not in low and pos:
        text = f'{pos}, {text}' if text else pos
    if not bad:
        bad = neg
    elif flavor == 'pony' and 'score_4' not in bad.lower():
        bad = f'{neg}, {bad}'
    elif flavor != 'pony' and neg and 'low quality' not in bad.lower():
        bad = f'{neg}, {bad}'
    return text, bad


def flavor_brief(flavor: str, checkpoint: str) -> str:
    """Agent instructions for this checkpoint family."""
    ckpt = (checkpoint or '').strip() or 'whatever is loaded in A1111'
    if flavor == 'pony':
        return (
            f'CHECKPOINT: {ckpt} (Pony Diffusion XL).\n'
            'WRITE a rich Pony prompt. Do not make the user type quality tags.\n'
            'Positive MUST start with: score_9, score_8_up, score_7_up\n'
            'Then the scene as danbooru-style tags (commas, not prose): '
            'subject, anatomy, clothing, lighting, camera, mood. '
            'source_anime or source_furry if it fits. masterpiece, best quality, highly detailed.\n'
            'Negative: score_6, score_5, score_4, worst quality, low quality, jpeg artifacts, '
            'blurry, ugly, extra fingers.\n'
            'Go to town — expand a short ask into a full tag list.'
        )
    if flavor == 'illustrious':
        return (
            f'CHECKPOINT: {ckpt} (Illustrious / NoobAI).\n'
            'Danbooru tags, commas. Start with masterpiece, best quality, amazing quality, '
            'very aesthetic, newest. Then the scene. Negative: worst quality, low quality, old, early.'
        )
    if flavor == 'flux':
        return (
            f'CHECKPOINT: {ckpt} (Flux). Natural language, not tag soup. '
            'One vivid paragraph: subject, lighting, lens, mood. Short negative.'
        )
    if flavor == 'sd15':
        return (
            f'CHECKPOINT: {ckpt} (SD 1.5).\n'
            'Positive: masterpiece, best quality, ultra-detailed, highly detailed, sharp focus, '
            'then the scene (medium, lighting, camera). '
            'Negative: low quality, low resolution, worst quality, jpeg artifacts, blurry, ugly, deformed.\n'
            'Expand a short ask. Do not dump Pony score_ tags.'
        )
    return (
        f'CHECKPOINT: {ckpt} (SDXL).\n'
        'Positive: masterpiece, best quality, ultra high resolution, highly detailed, '
        'sharp focus, 8k, then the scene. '
        'Negative: low quality, low resolution, worst quality, jpeg artifacts, blurry, ugly, deformed.\n'
        'Expand a short ask into a cinematic prompt. No Pony score_ tags unless this is Pony.'
    )


def merge_prompt(base: str, addition: str) -> str:
    """Keep unique clauses. Restated scenes do not duplicate the forest."""
    prior = _clauses(base)
    extra = _clauses(addition)
    if not extra:
        return ', '.join(prior)
    if not prior:
        return ', '.join(extra)
    have = [p.lower() for p in prior]
    for clause in extra:
        low = clause.lower()
        if low in have:
            continue
        if any(
            (low in h or h in low)
            for h in have
            if min(len(h), len(low)) >= 24
        ):
            continue
        prior.append(clause)
        have.append(low)
    return ', '.join(prior)


def _clauses(text: str) -> list[str]:
    return [part.strip() for part in (text or '').split(',') if part.strip()]
