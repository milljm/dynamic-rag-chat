"""Run with: python src/settings_yaml_test.py"""
from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from settings_yaml import (  # noqa: E402
    list_models,
    load_file,
    models_urls,
    read_values,
    save_file,
    upsert_key,
    upsert_keys,
)


SAMPLE = """\
chat:

  model: gemma3:27b                    # Heavy-weight LLM
  pre_llm: gemma3:1b
  embedding_llm: nomic-embed-text
  model_server: http://localhost:1234/v1
  vision_llm:
"""


class SettingsYamlTest(unittest.TestCase):
    """Comment-preserving yaml patcher for the Settings page."""

    def test_upsert_keeps_inline_comment(self):
        out = upsert_key(SAMPLE, 'model', 'minimax-m3')
        self.assertIn('model: minimax-m3', out)
        self.assertIn('# Heavy-weight LLM', out)
        self.assertIn('pre_llm: gemma3:1b', out)

    def test_read_canonicalizes_model_server(self):
        values = read_values(SAMPLE)
        self.assertEqual(values['llm_server'], 'http://localhost:1234/v1')
        self.assertEqual(values['model'], 'gemma3:27b')
        self.assertEqual(values['vision_llm'], '')

    def test_upsert_keys_updates_existing_model_server(self):
        out = upsert_keys(SAMPLE, {
            'llm_server': 'http://127.0.0.1:1234/v1',
            'model': 'qwen3',
        })
        self.assertIn('model_server: http://127.0.0.1:1234/v1', out)
        self.assertNotIn('llm_server:', out)
        self.assertIn('model: qwen3', out)

    def test_save_roundtrip_creates_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / '.chat.yaml'
            save_file(path, {
                'llm_server': 'http://localhost:11434/v1',
                'model': 'gemma3:27b',
                'pre_llm': 'gemma3:1b',
                'embedding_llm': 'nomic-embed-text',
                'api_key': 'none',
            })
            values, _ = load_file(path)
            self.assertEqual(values['model'], 'gemma3:27b')
            self.assertEqual(values['llm_server'], 'http://localhost:11434/v1')

    def test_models_urls_appends_v1(self):
        self.assertEqual(
            models_urls('http://localhost:1234'),
            [
                'http://localhost:1234/models',
                'http://localhost:1234/v1/models',
            ],
        )
        self.assertEqual(
            models_urls('http://localhost:1234/v1'),
            ['http://localhost:1234/v1/models'],
        )

    def test_list_models_empty_host(self):
        result = list_models('')
        self.assertFalse(result['ok'])
        self.assertEqual(result['models'], [])

    def test_tavily_key_roundtrip(self):
        out = upsert_keys(SAMPLE, {'tavily_key': 'tvly-test'})
        self.assertIn('tavily_key: tvly-test', out)
        self.assertEqual(read_values(out)['tavily_key'], 'tvly-test')
        self.assertEqual(read_values(upsert_keys(out, {'tavily_key': ''}))['tavily_key'], '')


if __name__ == '__main__':
    unittest.main()
