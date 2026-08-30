"""Run with: python src/sd_client_test.py"""
from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sd_client import (  # noqa: E402
    magick_argv,
    normalize_sd_url,
    sd_enabled,
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


if __name__ == '__main__':
    unittest.main()
