"""Run with: python src/rag_manager_test.py"""
from __future__ import annotations

import os
import sys
import unittest
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    from rag_manager import RAG  # noqa: E402
except ImportError:
    RAG = None


@unittest.skipUnless(RAG, 'rag_manager deps not installed in this env')
class RagManagerTest(unittest.TestCase):
    """Construction without a configured embedding server."""

    def test_missing_emb_host_does_not_crash(self):
        opts = SimpleNamespace(
            emb_host=None,
            embeddings=None,
            api_key='none',
            assistant_mode=False,
            matches=4,
            vector_dir='/tmp',
            debug=False,
            color=236,
        )
        rag = RAG(None, None, opts)
        self.assertIsNone(rag.embeddings)
        self.assertFalse(rag._embeddings_ready())
        self.assertEqual(rag.retrieve('hello', 'assistant_user_documents'), [])


if __name__ == '__main__':
    unittest.main()
