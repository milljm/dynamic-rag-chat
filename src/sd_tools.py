"""LangChain tools that talk to Automatic1111 and ImageMagick."""
from __future__ import annotations

import os
import time
import uuid
from typing import Any, Callable

from langchain_core.tools import StructuredTool

from .sd_client import img2img, is_generated_picture, run_magick, txt2img
from .sd_session import apply_quality, merge_prompt


def _data_url(blob: bytes, mime: str = 'image/png') -> str:
    import base64
    return f'data:{mime};base64,{base64.b64encode(blob).decode("ascii")}'


def _write_png(folder: str, blob: bytes, stem: str = 'sd') -> dict[str, Any]:
    os.makedirs(folder, exist_ok=True)
    name = f'{stem}-{time.strftime("%H%M%S")}-{uuid.uuid4().hex[:6]}.png'
    path = os.path.join(folder, name)
    with open(path, 'wb') as handle:
        handle.write(blob)
    return {
        'name': name,
        'path': path,
        'rel': f'generated/{name}',
        'mime': 'image/png',
        'dataUrl': _data_url(blob),
        'size': len(blob),
        'kind': 'image',
        'id': uuid.uuid4().hex,
    }


def seed_last_generated(folder: str, limit: int = 1) -> list[dict[str, Any]]:
    """Load the newest PNG(s) so a follow-up turn can img2img / magick them."""
    if not os.path.isdir(folder):
        return []
    names = [
        n for n in os.listdir(folder)
        if is_generated_picture(n)
    ]
    paths = sorted(
        (os.path.join(folder, n) for n in names),
        key=os.path.getmtime,
    )
    out: list[dict[str, Any]] = []
    for path in paths[-limit:]:
        try:
            with open(path, 'rb') as handle:
                blob = handle.read()
        except OSError:
            continue
        name = os.path.basename(path)
        out.append({
            'name': name,
            'path': path,
            'rel': f'generated/{name}',
            'mime': 'image/png',
            'dataUrl': _data_url(blob),
            'size': len(blob),
            'kind': 'image',
            'id': uuid.uuid4().hex,
            'prior': True,
        })
    return out


def _find(store: list[dict], image_name: str) -> dict[str, Any]:
    if not store:
        raise RuntimeError('No generated image yet. Call txt2img first.')
    want = (image_name or '').strip()
    if not want or want in {'.', 'last', 'latest'}:
        return store[-1]
    for rec in reversed(store):
        if rec.get('name') == want or rec.get('name', '').endswith(want):
            return rec
    raise RuntimeError(f'No image named {want!r}. Latest is {store[-1].get("name")}.')


def make_sd_tools(
    host: str,
    folder: str,
    store: list[dict],
    status: Callable[[str], None] | None = None,
    emit_image: Callable[[dict], None] | None = None,
    checkpoint: str = '',
    allow_magick: bool = False,
    session: dict | None = None,
    persist: Callable[[dict], None] | None = None,
    fresh: bool = False,
    flavor: str = 'sdxl',
) -> list:
    """txt2img first; later turns img2img with the accumulated prompt stack."""
    stack = session if session is not None else {}
    if fresh:
        stack.clear()
    sd_calls = {'n': 0}
    once_msg = (
        'Already generated once this turn. Stop. The user will ask '
        'for another generate on the next turn.'
    )
    building = bool((stack.get('prompt') or '').strip() and store and not fresh)

    def _status(msg: str) -> None:
        if status:
            status(msg)

    def _emit(rec: dict) -> None:
        store.append(rec)
        if emit_image and not rec.get('prior'):
            emit_image(rec)

    def _remember(
        prompt: str, negative: str, width: int = 0, height: int = 0, seed: int = -1,
    ) -> None:
        stack['prompt'] = prompt
        if negative:
            stack['negative'] = negative
        if width:
            stack['width'] = width
        if height:
            stack['height'] = height
        if seed is not None and int(seed) >= 0:
            stack['seed'] = int(seed)
        if persist:
            persist(stack)

    def do_txt2img(
        prompt: str,
        negative_prompt: str = '',
        steps: int = 20,
        width: int = 768,
        height: int = 768,
    ) -> str:
        """Generate a new image with Automatic1111 txt2img."""
        if building:
            return (
                'A prompt stack already exists. Use img2img and pass ONLY the '
                'new addition (the system keeps the previous scene).'
            )
        if sd_calls['n'] >= 1:
            return once_msg
        sd_calls['n'] += 1
        prompt, negative_prompt = apply_quality(prompt, negative_prompt, flavor)
        _status('Stable Diffusion…')
        blob, meta = txt2img(
            host, prompt, negative_prompt=negative_prompt,
            steps=steps, width=width, height=height, checkpoint=checkpoint,
        )
        rec = _write_png(folder, blob, stem='txt2img')
        _emit(rec)
        _remember(prompt, negative_prompt, width, height, seed=int(meta.get('seed') or -1))
        return f'Generated {rec["name"]} ({width}x{height}). Stop. Do not edit it this turn.'

    def do_img2img(
        prompt: str,
        image_name: str = 'last',
        denoising: float = 0.28,
        negative_prompt: str = '',
        steps: int = 20,
    ) -> str:
        """Re-draw the last picture, keeping the accumulated prompt."""
        if sd_calls['n'] >= 1:
            return once_msg
        rec = _find(store, image_name)
        sd_calls['n'] += 1
        full = merge_prompt(str(stack.get('prompt') or ''), prompt)
        negative = negative_prompt or str(stack.get('negative') or '')
        full, negative = apply_quality(full, negative, flavor)
        denoise = min(0.4, max(0.2, float(denoising or 0.28)))
        seed = int(stack.get('seed') or -1)
        with open(rec['path'], 'rb') as handle:
            src = handle.read()
        _status('Stable Diffusion…')
        blob, meta = img2img(
            host, src, full, negative_prompt=negative,
            denoising=denoise, steps=steps, checkpoint=checkpoint, seed=seed,
        )
        out = _write_png(folder, blob, stem='img2img')
        _emit(out)
        _remember(full, negative, seed=int(meta.get('seed') or seed))
        return (
            f'Re-drew {rec["name"]} → {out["name"]} (denoise {denoise:.2f}). '
            f'Prompt stack: {full[:240]}'
        )

    def do_magick(
        operation: str,
        argument: str = '',
        image_name: str = 'last',
    ) -> str:
        """Cheap ImageMagick: resize, rotate, blur, sharpen, grayscale, negate, modulate, brightness, border, caption."""
        rec = _find(store, image_name)
        dest_name = f'magick-{operation}-{uuid.uuid4().hex[:6]}.png'
        dest = os.path.join(folder, dest_name)
        _status('ImageMagick…')
        run_magick(rec['path'], dest, operation, argument)
        with open(dest, 'rb') as handle:
            blob = handle.read()
        out = {
            'name': dest_name,
            'path': dest,
            'rel': f'generated/{dest_name}',
            'mime': 'image/png',
            'dataUrl': _data_url(blob),
            'size': len(blob),
            'kind': 'image',
            'id': uuid.uuid4().hex,
        }
        _emit(out)
        return f'{operation} on {rec["name"]} → {out["name"]}.'

    tools: list = []
    iterating = building or (bool(store) and not fresh)
    if not iterating:
        tools.append(
            StructuredTool.from_function(
                func=do_txt2img,
                name='txt2img',
                description=(
                    'First picture only. Write a detailed visual prompt (subject, '
                    'medium, lighting, camera). Later turns must use img2img.'
                ),
            ),
        )
    if iterating or store:
        tools.append(
            StructuredTool.from_function(
                func=do_img2img,
                name='img2img',
                description=(
                    'Add to the last picture. Pass ONLY the new ask '
                    '(e.g. "a large soap bubble with rainbow shimmer"). '
                    'The previous scene prompt is prepended. denoising 0.25-0.35 '
                    'keeps the scene; never go above 0.4. Default 0.28.'
                ),
            ),
        )
    if allow_magick:
        tools.append(
            StructuredTool.from_function(
                func=do_magick,
                name='imagemagick',
                description=(
                    'ONLY if the user asked: resize, rotate, blur, sharpen, grayscale, '
                    'negate, modulate, brightness, border (16), caption (short text). '
                    'Do not add a border on your own.'
                ),
            ),
        )
    return tools
