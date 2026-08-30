"""Run with: python src/sd_client_test.py"""
from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sd_client import (  # noqa: E402
    is_generated_picture,
    is_new_scene,
    magick_argv,
    normalize_sd_url,
    sd_enabled,
    vision_thumb_data_url,
    wants_sd,
    _txt2img_payload,
)


class SdClientTest(unittest.TestCase):
    """URL + ImageMagick guardrails, no live Automatic1111."""

    def test_normalize_strips_sdapi(self):
        self.assertEqual(
            normalize_sd_url('http://192.168.1.9:7860/sdapi/v1'),
            'http://192.168.1.9:7860',
        )
        self.assertEqual(normalize_sd_url('none'), '')
        self.assertFalse(sd_enabled(''))

    def test_magick_resize_ok(self):
        self.assertEqual(magick_argv('resize', '1024x1024'), ['-resize', '1024x1024'])

    def test_magick_rejects_shell(self):
        with self.assertRaises(ValueError):
            magick_argv('resize', '1024x1024; rm -rf /')
        with self.assertRaises(ValueError):
            magick_argv('resize', '$(reboot)')
        with self.assertRaises(ValueError):
            magick_argv('explode', '1')

    def test_checkpoint_override(self):
        body = _txt2img_payload('a cat', checkpoint='dreamshaper_8')
        self.assertEqual(
            body['override_settings']['sd_model_checkpoint'],
            'dreamshaper_8',
        )
        self.assertFalse(body['override_settings_restore_afterwards'])
        plain = _txt2img_payload('a cat')
        self.assertNotIn('override_settings', plain)

    def test_magick_border_and_caption(self):
        self.assertEqual(
            magick_argv('border', '12'),
            ['-bordercolor', 'black', '-border', '12'],
        )
        self.assertIn('Hello world', magick_argv('caption', 'Hello world'))
        with self.assertRaises(ValueError):
            magick_argv('caption', 'rm -rf / ; hi')

    def test_followup_make_it_darker(self):
        self.assertFalse(wants_sd('Now make it darker', has_last=False))
        self.assertTrue(wants_sd('Now make it darker', has_last=True))
        self.assertTrue(wants_sd('draw me a red fox'))
        self.assertTrue(wants_sd('generate an image of a fox'))
        self.assertTrue(wants_sd('too yellow', has_last=True))
        self.assertTrue(wants_sd('punch up the lighting', has_last=True))
        self.assertFalse(wants_sd('looks great', has_last=True))
        self.assertFalse(wants_sd('what is a fox'))

    def test_new_scene_vs_tweak(self):
        self.assertTrue(is_new_scene(
            'Lets create an image of a soap bubble floating in the air',
        ))
        self.assertTrue(is_new_scene('draw me a red fox'))
        self.assertFalse(is_new_scene(
            'Have one of its legs raised, as if waving to the viewer',
        ))
        self.assertFalse(is_new_scene('now make it darker'))

    def test_vision_sidecars_are_not_pictures(self):
        self.assertTrue(is_generated_picture('txt2img-120000-abc.png'))
        self.assertFalse(is_generated_picture('txt2img-120000-abc.png.v512.png'))
        self.assertFalse(is_generated_picture('txt2img-120000-abc.png.v512.jpg'))

    def test_vision_thumb_returns_data_url(self):
        tiny = (
            'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR4'
            '2mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=='
        )
        url = vision_thumb_data_url(f'data:image/png;base64,{tiny}')
        if url:
            self.assertTrue(url.startswith('data:image/jpeg;base64,'))


if __name__ == '__main__':
    unittest.main()
