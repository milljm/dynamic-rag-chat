"""Run with: python src/chat_history_test.py

Avoid `python -m src.chat_history_test` — src/__init__.py imports langchain.
"""
from __future__ import annotations

import json
import os
import pickle
import sys
import tempfile
import unittest

try:
    from .chat_utils import (
        HISTORY_JSON,
        HISTORY_PKL,
        HISTORY_VERSION,
        load_history_from_dir,
        _atomic_write_json,
        _read_json_dict,
    )
except ImportError:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from chat_utils import (
        HISTORY_JSON,
        HISTORY_PKL,
        HISTORY_VERSION,
        load_history_from_dir,
        _atomic_write_json,
        _read_json_dict,
    )


def _hist(**extra):
    data = {
        'story': [{'role': 'user', 'content': 'hi'}],
        'assistant': [],
        'current': 'story',
        'branch_modes': {},
        'assistant_mode': False,
    }
    data.update(extra)
    return data


class ChatHistoryJsonTest(unittest.TestCase):
    """JSON history roundtrip and pickle migration."""

    def test_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, HISTORY_JSON)
            payload = _hist()
            _atomic_write_json(path, payload)
            loaded = load_history_from_dir(tmp, migrate=False)
            self.assertEqual(loaded['story'][0]['content'], 'hi')
            self.assertEqual(loaded['version'], HISTORY_VERSION)
            self.assertIsInstance(_read_json_dict(path), dict)

    def test_migrates_pickle(self):
        with tempfile.TemporaryDirectory() as tmp:
            pkl = os.path.join(tmp, HISTORY_PKL)
            with open(pkl, 'wb') as handle:
                pickle.dump(_hist(), handle)
            loaded = load_history_from_dir(tmp, migrate=True)
            self.assertEqual(loaded['story'][0]['role'], 'user')
            json_path = os.path.join(tmp, HISTORY_JSON)
            self.assertTrue(os.path.isfile(json_path))
            self.assertTrue(os.path.isfile(pkl))
            with open(json_path, encoding='utf-8') as handle:
                disk = json.load(handle)
            self.assertEqual(disk['story'][0]['content'], 'hi')

    def test_json_wins_over_pickle(self):
        with tempfile.TemporaryDirectory() as tmp:
            with open(os.path.join(tmp, HISTORY_PKL), 'wb') as handle:
                pickle.dump(_hist(story=[{'role': 'user', 'content': 'old'}]), handle)
            _atomic_write_json(
                os.path.join(tmp, HISTORY_JSON),
                _hist(story=[{'role': 'user', 'content': 'new'}]),
            )
            loaded = load_history_from_dir(tmp, migrate=False)
            self.assertEqual(loaded['story'][0]['content'], 'new')

    def test_corrupt_json_falls_back_to_bak(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, HISTORY_JSON)
            _atomic_write_json(path, _hist(story=[{'role': 'user', 'content': 'old'}]))
            _atomic_write_json(path, _hist(story=[{'role': 'user', 'content': 'ok'}]))
            with open(path, 'w', encoding='utf-8') as handle:
                handle.write('{truncated')
            loaded = load_history_from_dir(tmp, migrate=False)
            self.assertEqual(loaded['story'][0]['content'], 'old')

    def test_rejects_list_payload(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, HISTORY_JSON)
            with open(path, 'w', encoding='utf-8') as handle:
                json.dump([1, 2, 3], handle)
            self.assertIsNone(load_history_from_dir(tmp, migrate=False))

    def test_assistant_reasoning_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            payload = _hist()
            payload['story'].append({
                'role': 'assistant',
                'content': 'Hello.',
                'reasoning': 'The user said hi; greet them.',
            })
            path = os.path.join(tmp, HISTORY_JSON)
            _atomic_write_json(path, payload)
            loaded = load_history_from_dir(tmp, migrate=False)
            asst = loaded['story'][-1]
            self.assertEqual(asst['reasoning'], 'The user said hi; greet them.')
            self.assertEqual(asst['content'], 'Hello.')


if __name__ == '__main__':
    unittest.main()
