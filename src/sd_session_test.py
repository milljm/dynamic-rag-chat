"""Run with: python src/sd_session_test.py"""
from __future__ import annotations

import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sd_session import (  # noqa: E402
    clear_session,
    load_session,
    merge_prompt,
    save_session,
)


class SdSessionTest(unittest.TestCase):
    """Prompt stack merge + disk roundtrip."""

    def test_merge_appends_short_ask(self):
        forest = 'damp forest, water droplets on vines, close-up, blurred background'
        out = merge_prompt(forest, 'a large soap bubble with rainbow shimmer')
        self.assertIn('damp forest', out)
        self.assertIn('soap bubble', out)

    def test_merge_keeps_restated_stack(self):
        forest = 'damp forest, water droplets on vines, close-up'
        restated = (
            'damp forest, water droplets on vines, close-up, '
            'a large soap bubble with rainbow shimmer'
        )
        self.assertEqual(merge_prompt(forest, restated), restated)

    def test_roundtrip_and_clear(self):
        with tempfile.TemporaryDirectory() as tmp:
            save_session(tmp, {'prompt': 'a red fox', 'width': 1024})
            data = load_session(tmp)
            self.assertEqual(data['prompt'], 'a red fox')
            self.assertEqual(data['width'], 1024)
            self.assertEqual(data.get('seed', -1), -1)
            clear_session(tmp)
            self.assertEqual(load_session(tmp), {})


if __name__ == '__main__':
    unittest.main()
