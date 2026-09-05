"""Run with: python src/prompt_manager_test.py

Avoid `python -m src.prompt_manager_test` — src/__init__.py imports langchain.
"""
from __future__ import annotations

import os
import sys
import tempfile
import unittest
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from prompt_manager import PromptManager  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class _Console:
    def print(self, *args, **kwargs):
        return None


def _args(assistant=False, vector_dir=''):
    return SimpleNamespace(
        assistant_mode=assistant, debug=False, vector_dir=vector_dir,
    )


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

    def test_overlay_does_not_touch_stock(self):
        with tempfile.TemporaryDirectory() as tmp:
            pm = PromptManager(
                _Console(), ROOT, _args(True, tmp), prompt_model='qwen',
            )
            stock = pm.plot_file('assistant', 'system')
            with open(stock, encoding='utf-8') as handle:
                before = handle.read()
            pm.write_plot('assistant', 'system', 'CUSTOM SYSTEM\n')
            pm.reload()
            self.assertEqual(pm.plot_prompt_system, 'CUSTOM SYSTEM\n')
            with open(stock, encoding='utf-8') as handle:
                self.assertEqual(handle.read(), before)
            restored = pm.restore_plot('assistant', 'system')
            self.assertFalse(restored['overlaid'])
            pm.reload()
            self.assertEqual(pm.plot_prompt_system, before)

    def test_write_plot_rejects_human(self):
        with tempfile.TemporaryDirectory() as tmp:
            pm = PromptManager(
                _Console(), ROOT, _args(False, tmp), prompt_model='qwen',
            )
            with self.assertRaises(ValueError):
                pm.write_plot('story', 'human', 'nope')

    def test_compose_includes_need_search_cookbook(self):
        pm = PromptManager(_Console(), ROOT, _args(True), prompt_model='qwen')
        system, _ = pm.compose_nostory_plot({
            'gold_resume': '',
            'search_resume': '',
            'has_documents_index': False,
            'dynamic_files': '',
            'agent_calls': 0,
            'search_fetches': 0,
        })
        self.assertIn('<NEED_SEARCH>', system)
        self.assertIn('<NEED_SEARCH:NVDA share price>', system)

    def test_compose_omits_need_search_cookbook_on_resume(self):
        pm = PromptManager(_Console(), ROOT, _args(True), prompt_model='qwen')
        system, human = pm.compose_nostory_plot({
            'gold_resume': '',
            'search_resume': 'Lead in.',
            'has_documents_index': False,
            'dynamic_files': '=== WEB_SEARCH ===\nhits',
            'agent_calls': 0,
            'search_fetches': 1,
        })
        self.assertNotIn('You may emit one live-lookup tag', system)
        self.assertIn('<SEARCH_RESUME_EVENT>', system)
        self.assertIn('LIVE_LOOKUP', system)

    def test_compose_omits_need_search_when_agent_already_capped(self):
        pm = PromptManager(_Console(), ROOT, _args(True), prompt_model='qwen')
        system, _ = pm.compose_nostory_plot({
            'gold_resume': '',
            'search_resume': '',
            'has_documents_index': False,
            'dynamic_files': '',
            'agent_calls': 2,
            'search_fetches': 0,
        })
        self.assertNotIn('You may emit one live-lookup tag', system)


if __name__ == '__main__':
    unittest.main()
