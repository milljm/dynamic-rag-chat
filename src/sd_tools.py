"""LangChain tools that talk to Automatic1111 and ImageMagick."""
from __future__ import annotations

import os
import time
import uuid
from typing import Any, Callable

from langchain_core.tools import StructuredTool

from .sd_client import img2img, run_magick, txt2img


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
        if n.lower().endswith(('.png', '.jpg', '.jpeg', '.webp'))
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
) -> list:
    """txt2img / img2img once per turn; ImageMagick is cheap follow-up."""
    sd_calls = {'n': 0}
    once_msg = (
        'Already generated once this turn. Use imagemagick for a cheap edit '
        '(border, caption, resize, rotate), then stop. The user will ask '
        'for another generate on the next turn.'
    )

    def _status(msg: str) -> None:
        if status:
            status(msg)

    def _emit(rec: dict) -> None:
        store.append(rec)
        if emit_image and not rec.get('prior'):
            emit_image(rec)

    def do_txt2img(
        prompt: str,
        negative_prompt: str = '',
        steps: int = 20,
        width: int = 768,
        height: int = 768,
    ) -> str:
        """Generate a new image with Automatic1111 txt2img."""
        if sd_calls['n'] >= 1:
            return once_msg
        sd_calls['n'] += 1
        _status('Stable Diffusion…')
        blob = txt2img(
            host, prompt, negative_prompt=negative_prompt,
            steps=steps, width=width, height=height, checkpoint=checkpoint,
        )
        rec = _write_png(folder, blob, stem='txt2img')
        _emit(rec)
        return (
            f'Generated {rec["name"]} ({width}x{height}). '
            'You may imagemagick (border/caption/resize). Do not generate again.'
        )

    def do_img2img(
        prompt: str,
        image_name: str = 'last',
        denoising: float = 0.45,
        negative_prompt: str = '',
        steps: int = 20,
    ) -> str:
        """Re-draw an existing generated image with img2img."""
        if sd_calls['n'] >= 1:
            return once_msg
        rec = _find(store, image_name)
        sd_calls['n'] += 1
        with open(rec['path'], 'rb') as handle:
            src = handle.read()
        _status('Stable Diffusion…')
        blob = img2img(
            host, src, prompt, negative_prompt=negative_prompt,
            denoising=denoising, steps=steps, checkpoint=checkpoint,
        )
        out = _write_png(folder, blob, stem='img2img')
        _emit(out)
        return (
            f'Re-drew {rec["name"]} → {out["name"]} (denoise {denoising}). '
            'You may imagemagick. Do not generate again.'
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

    return [
        StructuredTool.from_function(
            func=do_txt2img,
            name='txt2img',
            description=(
                'ONE new image this turn via Stable Diffusion. '
                'Do not call again. Use imagemagick after, or stop.'
            ),
        ),
        StructuredTool.from_function(
            func=do_img2img,
            name='img2img',
            description=(
                'ONE redraw of the last picture this turn (user asked for a change). '
                'Not for "improving" a fresh txt2img. imagemagick after if needed.'
            ),
        ),
        StructuredTool.from_function(
            func=do_magick,
            name='imagemagick',
            description=(
                'Cheap edits: resize (1024x1024), rotate, blur, sharpen, grayscale, '
                'negate, modulate, brightness, border (16), caption (short text).'
            ),
        ),
    ]
