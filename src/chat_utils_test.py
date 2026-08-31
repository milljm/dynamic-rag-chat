"""Run with: python src/chat_utils_test.py"""
from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from chat_utils import ChatOptions, active_branch  # noqa: C0413


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


if __name__ == '__main__':
    unittest.main()
