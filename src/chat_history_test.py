"""Run with: python src/chat_history_test.py

Avoid `python -m src.chat_history_test` — src/__init__.py imports langchain.
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor

try:
    from .chat_utils import (
        HISTORY_JSON,
        HISTORY_VERSION,
        load_history_from_dir,
        _atomic_write_json,
        _read_json_dict,
        CommonUtils,
    )
except ImportError:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from chat_utils import (
        HISTORY_JSON,
        HISTORY_VERSION,
        load_history_from_dir,
        _atomic_write_json,
        _read_json_dict,
        CommonUtils,
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
    """JSON history roundtrip."""

    def test_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, HISTORY_JSON)
            payload = _hist()
            _atomic_write_json(path, payload)
            loaded = load_history_from_dir(tmp)
            self.assertEqual(loaded['story'][0]['content'], 'hi')
            self.assertEqual(loaded['version'], HISTORY_VERSION)
            self.assertIsInstance(_read_json_dict(path), dict)

    def test_corrupt_json_falls_back_to_bak(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, HISTORY_JSON)
            _atomic_write_json(path, _hist(story=[{'role': 'user', 'content': 'old'}]))
            _atomic_write_json(path, _hist(story=[{'role': 'user', 'content': 'ok'}]))
            with open(path, 'w', encoding='utf-8') as handle:
                handle.write('{truncated')
            loaded = load_history_from_dir(tmp)
            self.assertEqual(loaded['story'][0]['content'], 'old')

    def test_rejects_list_payload(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, HISTORY_JSON)
            with open(path, 'w', encoding='utf-8') as handle:
                json.dump([1, 2, 3], handle)
            self.assertIsNone(load_history_from_dir(tmp))

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
            loaded = load_history_from_dir(tmp)
            asst = loaded['story'][-1]
            self.assertEqual(asst['reasoning'], 'The user said hi; greet them.')
            self.assertEqual(asst['content'], 'Hello.')

    def test_concurrent_writes_leave_valid_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, HISTORY_JSON)
            _atomic_write_json(path, _hist(story=[]))

            def write(i):
                _atomic_write_json(path, _hist(
                    story=[{'role': 'user', 'content': f'turn-{i}'}],
                ))

            with ThreadPoolExecutor(max_workers=8) as pool:
                list(pool.map(write, range(24)))
            loaded = load_history_from_dir(tmp)
            self.assertIsInstance(loaded, dict)
            self.assertTrue(loaded['story'][0]['content'].startswith('turn-'))
            self.assertFalse(any(
                name.endswith('.tmp') for name in os.listdir(tmp)
                if not name.endswith('.lock')
            ))


class AttachmentHelpersTest(unittest.TestCase):
    """History lines and attachment bookkeeping."""

    def test_history_line_includes_filenames(self):
        line = CommonUtils.history_line({
            'role': 'user',
            'content': 'what does this do?',
            'attachments': [{'name': 'spur-server.py', 'kind': 'text'}],
        })
        self.assertIn('USER: what does this do?', line)
        self.assertIn('[attached: spur-server.py]', line)

    def test_history_line_plain(self):
        line = CommonUtils.history_line({'role': 'assistant', 'content': 'hi'})
        self.assertEqual(line, 'AI: hi')

    def test_record_attachment(self):
        docs = {}
        CommonUtils.record_attachment(docs, 'a.py', text='print(1)', kind='text')
        CommonUtils.record_attachment(docs, 'pic.png', kind='image')
        self.assertEqual(len(docs['attachment_texts']), 2)
        self.assertEqual(docs['attachment_texts'][0]['name'], 'a.py')
        self.assertEqual(docs['attachment_texts'][1]['kind'], 'image')

    def test_extract_filenames(self):
        names = CommonUtils.extract_filenames(
            'look at spur-server.py and README.md please',
        )
        self.assertEqual(names, ['spur-server.py', 'readme.md'])
        self.assertEqual(CommonUtils.extract_filenames('no files here'), [])
        self.assertIn('chat_history.json', CommonUtils.extract_filenames(
            'the chat_history.json dump',
        ))
        self.assertEqual(CommonUtils.extract_filenames('python 3.13 rocks'), [])

    def test_collect_filenames_from_dicts_and_paths(self):
        names = CommonUtils.collect_filenames(
            [{'name': 'spur-server.py', 'text': 'SECRET'}],
            [{'name': 'notes.md'}],
            ['/tmp/vector/README.md', 'notes.md'],
        )
        self.assertEqual(names, ['spur-server.py', 'notes.md', 'README.md'])

    def test_preprocessor_payload_keeps_names_drops_bodies(self):
        docs = {
            'user_query': 'what does this do?',
            'dynamic_files': '\n--- spur-server.py ---\n\nSECRET_BODY\n',
            'dynamic_images': ['iVBORw0KGgo='],
            'gold_documents': 'GOLD DUMP',
            'attachment_texts': [{
                'name': 'spur-server.py',
                'text': 'SECRET_BODY',
                'kind': 'text',
            }],
            'chat_history': [{
                'role': 'user',
                'content': 'look at this',
                'attachments': [{
                    'name': 'old.py',
                    'text': 'PREVIOUS_SECRET',
                    'kind': 'text',
                }],
            }, {
                'role': 'assistant',
                'content': 'ok',
            }],
        }
        slim = CommonUtils.preprocessor_payload(docs)
        blob = str(slim)
        self.assertNotIn('SECRET_BODY', blob)
        self.assertNotIn('PREVIOUS_SECRET', blob)
        self.assertNotIn('GOLD DUMP', blob)
        self.assertNotIn('iVBORw0KGgo=', blob)
        self.assertEqual(slim['attached_filenames'], 'spur-server.py')
        self.assertEqual(slim['dynamic_files'], '')
        self.assertEqual(slim['dynamic_images'], [])
        self.assertIn('[attached: old.py]', slim['chat_history'])
        self.assertIn('what does this do?', slim['user_query'])
        # Original documents must keep the bodies for the generator.
        self.assertIn('SECRET_BODY', docs['dynamic_files'])
        self.assertEqual(docs['attachment_texts'][0]['text'], 'SECRET_BODY')


if __name__ == '__main__':
    unittest.main()
