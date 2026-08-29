"""Run with: python src/spur_launch_test.py"""
from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from spur_launch import SPUR_DIR, SERVER_SCRIPT, strip_spur_flag  # noqa: E402


class SpurLaunchTest(unittest.TestCase):
    """Flag stripping and tree layout for ./chat.py --spur."""

    def test_strips_spur_anywhere(self):
        self.assertEqual(
            strip_spur_flag(['--spur', '--assistant-mode']),
            ['--assistant-mode'],
        )
        self.assertEqual(
            strip_spur_flag(['--assistant-mode', '--spur', '--verbose']),
            ['--assistant-mode', '--verbose'],
        )
        self.assertEqual(strip_spur_flag(['--assistant-mode']), ['--assistant-mode'])

    def test_ui_and_adapter_live_in_this_repo(self):
        self.assertTrue((SPUR_DIR / 'package.json').is_file(), SPUR_DIR)
        self.assertTrue(SERVER_SCRIPT.is_file())
        self.assertEqual(SPUR_DIR.name, 'spur')
        self.assertTrue(os.path.isdir(SPUR_DIR / 'src'))


if __name__ == '__main__':
    unittest.main()
