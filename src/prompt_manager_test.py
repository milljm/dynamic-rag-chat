"""Run with: python src/prompt_manager_test.py

Avoid `python -m src.prompt_manager_test` — src/__init__.py imports langchain.
"""
from __future__ import annotations

import os
import sys
import unittest
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from prompt_manager import PromptManager  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class _Console:
    def print(self, *args, **kwargs):
        return None


def _args(assistant=False):
    return SimpleNamespace(assistant_mode=assistant, debug=False)


class PlotFileTests(unittest.TestCase):
    def test_assistant_is_nostory_even_when_chat_is_story(self):
        pm = PromptManager(_Console(), ROOT, _args(False), prompt_model='qwen3-8b')
        path = pm.plot_file('assistant', 'system')
        self.assertTrue(path.endswith('plot_prompt_nostory_system.md'))
        self.assertTrue(os.path.isfile(path))

    def test_story_falls_back_to_default(self):
        pm = PromptManager(_Console(), ROOT, _args(False), prompt_model='unknown-llm')
        path = pm.plot_file('story', 'human')
        self.assertTrue(path.endswith('plot_prompt_default_human.md'))
        self.assertTrue(os.path.isfile(path))

    def test_rejects_bad_slots(self):
        pm = PromptManager(_Console(), ROOT, _args(True), prompt_model='qwen')
        with self.assertRaises(ValueError):
            pm.plot_file('other', 'system')
        with self.assertRaises(ValueError):
            pm.plot_file('story', 'footer')

    def test_reload_rereads_disk(self):
        pm = PromptManager(_Console(), ROOT, _args(True), prompt_model='qwen')
        first = pm.plot_prompt_system
        pm.reload()
        self.assertEqual(first, pm.plot_prompt_system)


if __name__ == '__main__':
    unittest.main()
