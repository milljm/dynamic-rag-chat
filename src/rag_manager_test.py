"""Run with: python src/rag_manager_test.py"""
from __future__ import annotations

import os
import sys
import tempfile
import unittest
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    from langchain_core.documents import Document
    from rag_manager import RAG  # noqa: E402
except ImportError:
    RAG = None
    Document = None


def _opts(**kw):
    base = dict(
        emb_host=None,
        embeddings=None,
        api_key='none',
        assistant_mode=False,
        matches=4,
        vector_dir='/tmp',
        debug=False,
        color=236,
    )
    base.update(kw)
    return SimpleNamespace(**base)


@unittest.skipUnless(RAG, 'rag_manager deps not installed in this env')
class RagManagerTest(unittest.TestCase):
    """Construction without a configured embedding server."""

    def test_missing_emb_host_does_not_crash(self):
        rag = RAG(None, None, _opts())
        self.assertIsNone(rag.embeddings)
        self.assertFalse(rag._embeddings_ready())
        self.assertEqual(rag.retrieve('hello', 'assistant_user_documents'), [])

    def test_parent_ids_unique_in_order(self):
        docs = [
            Document(page_content='a', metadata={'doc_id': 'p1'}),
            Document(page_content='b', metadata={'doc_id': 'p1'}),
            Document(page_content='c', metadata={'doc_id': 'p2'}),
            Document(page_content='d', metadata={}),
        ]
        self.assertEqual(RAG._parent_ids(docs), ['p1', 'p2'])

    def test_forget_drops_cached_stores(self):
        rag = RAG(None, None, _opts())
        rag._chroma['hello'] = object()
        rag._pdr['hello'] = object()
        rag._forget_collection('hello')
        self.assertNotIn('hello', rag._chroma)
        self.assertNotIn('hello', rag._pdr)

    def test_vector_store_reuses_chroma_client(self):
        with tempfile.TemporaryDirectory() as tmp:
            rag = RAG(None, None, _opts(vector_dir=tmp))
            first = rag._vector_store('col-one')
            second = rag._vector_store('col-one')
            self.assertIs(first, second)
            rag._forget_collection('col-one')
            third = rag._vector_store('col-one')
            self.assertIsNot(first, third)


if __name__ == '__main__':
    unittest.main()
