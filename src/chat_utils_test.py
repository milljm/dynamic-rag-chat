"""Run with: python src/chat_utils_test.py"""
from __future__ import annotations

import os
import sys
import unittest
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from chat_utils import (  # noqa: C0413
    ChatOptions,
    CommonUtils,
    RegExp,
    active_branch,
    dedupe_rag_chunks,
    overlap_ratio,
)


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

    def test_assistant_mode_keeps_user_fork(self):
        hist = {
            'current': 'testing',
            'assistant_mode': True,
            'story': [],
            'assistant': [{'role': 'user', 'content': 'help'}],
            'testing': [{'role': 'user', 'content': 'forked'}],
            'branch_modes': {'testing': True},
        }
        self.assertEqual(active_branch(True, hist), 'testing')


class ChatOptionsYamlTest(unittest.TestCase):
    """Settings page writes llm_server; unknown keys must not crash."""

    def test_llm_server_alias(self):
        opts = ChatOptions._build('.', {  # pylint: disable=protected-access
            'llm_server': 'http://127.0.0.1:1234/v1',
            'model': 'minimax-m3',
        })
        self.assertEqual(opts.host, 'http://127.0.0.1:1234/v1')
        self.assertEqual(opts.model, 'minimax-m3')

    def test_unknown_yaml_key_ignored(self):
        opts = ChatOptions._build('.', {  # pylint: disable=protected-access
            'nope': 'whatever',
            'model': 'x',
        })
        self.assertEqual(opts.model, 'x')

    def test_empty_api_key_becomes_none(self):
        opts = ChatOptions._build('.', {  # pylint: disable=protected-access
            'api_key': '',
        })
        self.assertEqual(opts.api_key, 'none')

    def test_empty_vision_stays_sentinel(self):
        opts = ChatOptions._build('.', {  # pylint: disable=protected-access
            'vision_llm': None,
            'agent_llm': '',
            'model': 'minimax-m3',
        })
        self.assertEqual(opts.vision_llm, 'None')
        self.assertEqual(opts.agent_llm, 'None')
        self.assertEqual(opts.model, 'minimax-m3')

    def test_specialized_server_survives_model_inherit(self):
        opts = ChatOptions._build('.', {  # pylint: disable=protected-access
            'llm_server': 'http://main:1234/v1',
            'model': 'big-model',
            'coder_server': 'http://coder:1234/v1',
        })
        self.assertEqual(opts.coder_llm, 'big-model')
        self.assertEqual(opts.coder_host, 'http://coder:1234/v1')
        self.assertEqual(opts.host, 'http://main:1234/v1')

    def test_blank_specialized_server_inherits_main(self):
        opts = ChatOptions._build('.', {  # pylint: disable=protected-access
            'llm_server': 'http://main:1234/v1',
            'model': 'big-model',
            'coder_llm': 'coder-model',
        })
        self.assertEqual(opts.coder_llm, 'coder-model')
        self.assertEqual(opts.coder_host, 'http://main:1234/v1')


def _store_sanitize(text: str) -> str:
    """Same transforms store_data applies: drop fences, lowercase, collapse space."""
    return CommonUtils.normalize_for_dedup(text.replace('```', ''))


class DedupeRagChunksTest(unittest.TestCase):
    """USER_RAG / AI_RAG must not echo CHAT_HISTORY."""

    def test_user_question_case_was_just_under_threshold(self):
        hist = 'Can you create a matplotlib python example that uses Streamlit as a GUI front-end?'
        rag = 'can you create a matplotlib python example that uses streamlit as a gui front-end?'
        raw = overlap_ratio(rag, hist)
        self.assertLess(raw, 0.65)
        self.assertGreater(
            overlap_ratio(_store_sanitize(rag), _store_sanitize(hist)),
            0.65,
        )
        self.assertEqual(
            dedupe_rag_chunks(
                [rag],
                [{'role': 'user', 'content': hist}],
                sanitize=_store_sanitize,
            ),
            [],
        )

    def test_ai_reply_with_fences_vs_stored_parent(self):
        history_ai = (
            "Sure thing. Here's a self-contained example that gives you "
            'interactive controls in the sidebar and two live matplotlib charts. '
            'Save it and run with `streamlit run matplotlib_streamlit_app.py`.\n\n'
            '```python matplotlib_streamlit_app.py\n'
            'import streamlit as st\n'
            'import matplotlib.pyplot as plt\n'
            'import numpy as np\n'
            '```\n'
        )
        stored = _store_sanitize(history_ai)
        self.assertEqual(
            dedupe_rag_chunks(
                [stored],
                [
                    {'role': 'user', 'content': 'Can you create a matplotlib example?'},
                    {'role': 'assistant', 'content': history_ai},
                ],
                sanitize=_store_sanitize,
            ),
            [],
        )

    def test_keeps_older_turn_not_in_window(self):
        kept = dedupe_rag_chunks(
            ['the wizard left town three sessions ago'],
            [{'role': 'assistant', 'content': 'Hello. What now?'}],
            sanitize=_store_sanitize,
        )
        self.assertEqual(kept, ['the wizard left town three sessions ago'])

    def test_short_history_does_not_drop_gold_file(self):
        gold = (
            'matplotlib cookbook\n' * 40
            + 'Can you create a matplotlib python example that uses Streamlit as a GUI front-end?'
        )
        kept = dedupe_rag_chunks(
            [gold],
            [{'role': 'user', 'content':
              'Can you create a matplotlib python example that uses Streamlit as a GUI front-end?'}],
            sanitize=_store_sanitize,
        )
        self.assertEqual(kept, [gold])

    def test_empty_history_keeps_chunks(self):
        self.assertEqual(
            dedupe_rag_chunks(['fresh fact'], [], sanitize=_store_sanitize),
            ['fresh fact'],
        )

    def test_sanitize_response_matches_store_data(self):
        util = CommonUtils.__new__(CommonUtils)
        util.opts = SimpleNamespace(assistant_mode=True, debug=False)
        util.regex = RegExp()
        history = 'Sure thing.\n\n```python hello.py\nprint("hi")\n```\n'
        stored = util.sanitize_response(history, strip=True)
        self.assertNotIn('```', stored)
        self.assertEqual(stored, stored.lower())
        self.assertEqual(
            dedupe_rag_chunks(
                [stored],
                [{'role': 'assistant', 'content': history}],
                sanitize=lambda text: util.sanitize_response(text, strip=True),
            ),
            [],
        )


if __name__ == '__main__':
    unittest.main()
