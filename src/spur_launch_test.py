"""Run with: python src/spur_launch_test.py"""
from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from spur_launch import (  # noqa: E402
    DEFAULT_URL,
    SERVER_SCRIPT,
    SPUR_DIR,
    find_ui_root,
    lan_ips,
    strip_spur_flag,
    strip_spur_flags,
)


class SpurLaunchTest(unittest.TestCase):
    """Flag stripping and tree layout for ./chat.py --spur."""

    def test_strips_spur_anywhere(self):
        self.assertEqual(
            strip_spur_flag(['--spur', '--assistant-mode']),
            ['--assistant-mode'],
        )
        self.assertEqual(
            strip_spur_flags(['--assistant-mode', '--spur', '--verbose']),
            ['--assistant-mode', '--verbose'],
        )
        self.assertEqual(
            strip_spur_flags(['--spur-rebuild', '--spur', '--assistant-mode']),
            ['--assistant-mode'],
        )
        self.assertEqual(
            strip_spur_flags(['--spur', '--serve', '--assistant-mode']),
            ['--assistant-mode'],
        )
        self.assertEqual(strip_spur_flag(['--assistant-mode']), ['--assistant-mode'])

    def test_lan_ips_are_not_loopback(self):
        for ip in lan_ips():
            self.assertFalse(ip.startswith('127.'), ip)
            self.assertNotIn(':', ip)

    def test_ui_and_adapter_live_in_this_repo(self):
        self.assertTrue((SPUR_DIR / 'package.json').is_file(), SPUR_DIR)
        self.assertTrue(SERVER_SCRIPT.is_file())
        self.assertEqual(SPUR_DIR.name, 'spur')
        self.assertTrue(os.path.isdir(SPUR_DIR / 'src'))
        self.assertTrue(DEFAULT_URL.endswith(':8765'))

    def test_find_ui_root_prefers_dist_client(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            client = root / 'dist' / 'client'
            client.mkdir(parents=True)
            (client / 'index.html').write_text('<html></html>', encoding='utf-8')
            found = find_ui_root(root)
            self.assertEqual(found, client)

    def test_find_ui_root_none_without_index(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertIsNone(find_ui_root(Path(tmp)))


if __name__ == '__main__':
    unittest.main()
