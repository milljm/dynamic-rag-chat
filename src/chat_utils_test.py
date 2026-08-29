"""Run with: python src/chat_utils_test.py"""
from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from chat_utils import active_branch  # noqa: E402


class ActiveBranchTest(unittest.TestCase):
    """Bare ./chat.py is story even if Spur left current=assistant."""

    def test_flag_forces_assistant(self):
        hist = {'current': 'story', 'story': [], 'assistant': []}
        self.assertEqual(active_branch(True, hist), 'assistant')

    def test_bare_cli_does_not_resume_assistant(self):
        hist = {
            'current': 'assistant',
            'assistant_mode': True,
            'story': [{'role': 'user', 'content': 'hi'}],
            'assistant': [{'role': 'user', 'content': 'help'}],
        }
        self.assertEqual(active_branch(False, hist), 'story')

    def test_bare_cli_keeps_story_fork(self):
        hist = {
            'current': 'alt-ending',
            'story': [],
            'assistant': [],
            'alt-ending': [{'role': 'user', 'content': 'once'}],
        }
        self.assertEqual(active_branch(False, hist), 'alt-ending')


if __name__ == '__main__':
    unittest.main()
