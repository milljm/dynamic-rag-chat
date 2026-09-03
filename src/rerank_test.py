"""Run with: python src/rerank_test.py"""
from __future__ import annotations

import json
import os
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rerank import configured, post_rerank, reorder, rerank_url  # noqa: E402


class RerankHelperTest(unittest.TestCase):
    def test_configured(self):
        self.assertFalse(configured(SimpleNamespace(rerank_llm='None', rerank_host='http://x/v1')))
        self.assertFalse(configured(SimpleNamespace(rerank_llm='bge', rerank_host='')))
        self.assertTrue(configured(SimpleNamespace(rerank_llm='bge', rerank_host='http://x/v1')))

    def test_rerank_url(self):
        self.assertEqual(rerank_url('http://edge:8080/v1'), 'http://edge:8080/v1/rerank')
        self.assertEqual(rerank_url('http://edge:8080'), 'http://edge:8080/v1/rerank')

    def test_reorder_caps_and_skips_dupes(self):
        docs = ['a', 'b', 'c']
        self.assertEqual(reorder(docs, [2, 2, 0, 9], 2), ['c', 'a'])

    def test_post_rerank_orders_by_score(self):
        payload = {
            'results': [
                {'index': 0, 'relevance_score': 0.1},
                {'index': 2, 'relevance_score': 0.9},
                {'index': 1, 'relevance_score': 0.5},
            ]
        }

        class _Resp:
            def read(self):
                return json.dumps(payload).encode()
            def __enter__(self):
                return self
            def __exit__(self, *args):
                return False

        with patch('rerank.urllib.request.urlopen', return_value=_Resp()):
            order = post_rerank(
                'http://edge/v1', 'bge', 'q', ['aa', 'bb', 'cc'], top_n=2,
            )
        self.assertEqual(order, [2, 1, 0])


if __name__ == '__main__':
    unittest.main()
