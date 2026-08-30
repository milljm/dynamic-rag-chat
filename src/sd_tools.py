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
    """txt2img, img2img, imagemagick — all write into ``store``."""

    def _status(msg: str) -> None:
        if status:
            status(msg)

    def _emit(rec: dict) -> None:
        store.append(rec)
        if emit_image:
            emit_image(rec)

    def do_txt2img(
        prompt: str,
        negative_prompt: str = '',
        steps: int = 20,
        width: int = 768,
        height: int = 768,
    ) -> str:
        """Generate a new image with Automatic1111 txt2img."""
        _status('Stable Diffusion…')
        blob = txt2img(
            host, prompt, negative_prompt=negative_prompt,
            steps=steps, width=width, height=height, checkpoint=checkpoint,
        )
        rec = _write_png(folder, blob, stem='txt2img')
        _emit(rec)
        return (
            f'Generated {rec["name"]} ({width}x{height}). '
            'Look at it (vision). Adjust with imagemagick or img2img, '
            'or stop if it satisfies the user.'
        )

    def do_img2img(
        prompt: str,
        image_name: str = 'last',
        denoising: float = 0.45,
        negative_prompt: str = '',
        steps: int = 20,
    ) -> str:
        """Re-draw an existing generated image with img2img."""
        rec = _find(store, image_name)
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
            'Stop if the user would be happy.'
        )

    def do_magick(
        operation: str,
        argument: str = '',
        image_name: str = 'last',
    ) -> str:
        """Run a safe ImageMagick op: resize, rotate, blur, sharpen, grayscale, negate, modulate, brightness."""
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
                'Generate an image with Stable Diffusion (Automatic1111). '
                'Write a detailed visual prompt. Call this when the user wants a picture.'
            ),
        ),
        StructuredTool.from_function(
            func=do_img2img,
            name='img2img',
            description=(
                'Adjust a generated image with img2img. '
                'denoising 0.25=subtle, 0.55=strong. image_name=last or a filename.'
            ),
        ),
        StructuredTool.from_function(
            func=do_magick,
            name='imagemagick',
            description=(
                'Crop-less ImageMagick: resize (1024x1024), rotate (90), blur, '
                'sharpen, grayscale, negate, modulate (100,80,100), brightness (10x0).'
            ),
        ),
    ]
