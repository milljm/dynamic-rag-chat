"""Run with: python src/settings_yaml_test.py"""
from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from settings_yaml import (  # noqa: E402
    list_model_urls,
    list_models,
    load_file,
    models_urls,
    parse_models_payload,
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

    def test_list_model_urls_prefers_lmstudio_native(self):
        urls = list_model_urls('http://127.0.0.1:1234/v1')
        self.assertEqual(urls[0], 'http://127.0.0.1:1234/api/v1/models')
        self.assertEqual(urls[1], 'http://127.0.0.1:1234/api/v0/models')
        self.assertIn('http://127.0.0.1:1234/v1/models', urls)

    def test_openai_cloud_skips_native(self):
        urls = list_model_urls('https://api.openai.com/v1')
        self.assertFalse(any(u.endswith('/api/v0/models') for u in urls))
        self.assertTrue(any(u.endswith('/v1/models') for u in urls))


    def test_parse_lmstudio_v0_state(self):
        parsed = parse_models_payload({
            'data': [
                {'id': 'hot-model', 'state': 'loaded'},
                {'id': 'cold-model', 'state': 'not-loaded'},
            ],
        })
        self.assertEqual(parsed['source'], 'lmstudio-v0')
        self.assertEqual(parsed['loaded'], ['hot-model'])
        self.assertTrue(parsed['knows_loaded'])

    def test_parse_lmstudio_v1_instances(self):
        parsed = parse_models_payload({
            'models': [
                {'key': 'google/gemma', 'loaded_instances': [{'id': 'x'}]},
                {'key': 'other', 'loaded_instances': []},
            ],
        })
        self.assertEqual(parsed['source'], 'lmstudio-v1')
        self.assertEqual(parsed['loaded'], ['google/gemma'])

    def test_parse_openai_has_no_loaded(self):
        parsed = parse_models_payload({
            'data': [{'id': 'gpt-4o'}, {'id': 'gpt-4.1'}],
        })
        self.assertEqual(parsed['source'], 'openai')
        self.assertFalse(parsed['knows_loaded'])
        self.assertEqual(parsed['models'], ['gpt-4o', 'gpt-4.1'])


    def test_list_models_empty_host(self):
        result = list_models('')
        self.assertFalse(result['ok'])
        self.assertEqual(result['models'], [])

    def test_tavily_key_roundtrip(self):
        out = upsert_keys(SAMPLE, {'tavily_key': 'tvly-test'})
        self.assertIn('tavily_key: tvly-test', out)
        self.assertEqual(read_values(out)['tavily_key'], 'tvly-test')
        self.assertEqual(read_values(upsert_keys(out, {'tavily_key': ''}))['tavily_key'], '')

    def test_sd_server_roundtrip(self):
        out = upsert_keys(SAMPLE, {'sd_server': 'http://192.168.1.9:7860'})
        self.assertIn('sd_server: http://192.168.1.9:7860', out)
        self.assertEqual(
            read_values(out)['sd_server'], 'http://192.168.1.9:7860',
        )

    def test_sd_model_roundtrip(self):
        out = upsert_keys(SAMPLE, {'sd_model': 'dreamshaper_8'})
        self.assertEqual(read_values(out)['sd_model'], 'dreamshaper_8')



if __name__ == '__main__':
    unittest.main()
