"""Accumulated Automatic1111 prompt while Image mode is on."""
from __future__ import annotations

import json
import os
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


def merge_prompt(base: str, addition: str) -> str:
    """Keep the scene; append the new ask. If the agent restated the stack, use that."""
    prior = (base or '').strip()
    extra = (addition or '').strip()
    if not extra:
        return prior
    if not prior:
        return extra
    low_prior = prior.lower()
    low_extra = extra.lower()
    if low_prior[:80] in low_extra:
        return extra
    if low_extra in low_prior:
        return prior
    return f'{prior}, {extra}'
