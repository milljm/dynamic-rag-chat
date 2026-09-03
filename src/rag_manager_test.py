"""Run with: python src/rag_manager_test.py"""
from __future__ import annotations

import os
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    from langchain_core.documents import Document
    from rag_manager import RAG, BM25Retriever  # noqa: E402
except ImportError:
    RAG = None
    BM25Retriever = None
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

    def test_recall_k_widens_when_reranker_set(self):
        rag = RAG(None, None, _opts(
            matches=1,
            rerank_llm='bge-reranker',
            rerank_host='http://127.0.0.1:8080/v1',
        ))
        self.assertEqual(rag._recall_k(False), 24)
        self.assertGreaterEqual(rag._recall_k(True, 2), 24)

    def test_recall_k_stays_matches_without_reranker(self):
        rag = RAG(None, None, _opts(matches=1, rerank_llm='None', rerank_host=''))
        self.assertEqual(rag._recall_k(False), 1)

    def test_apply_rerank_reorders(self):
        rag = RAG(None, None, _opts(
            matches=2,
            rerank_llm='bge',
            rerank_host='http://x/v1',
        ))
        docs = [
            Document(page_content='noise'),
            Document(page_content='hit'),
            Document(page_content='other'),
        ]
        with patch('rag_manager.post_rerank', return_value=[1, 0]):
            out = rag._apply_rerank('q', docs)
        self.assertEqual([d.page_content for d in out], ['hit', 'noise'])

    def test_bm25_tokenize_folds_case_and_punct(self):
        self.assertEqual(
            BM25Retriever.tokenize('Login, login.'),
            ['login', 'login'],
        )
        self.assertEqual(BM25Retriever.tokenize('file.py'), ['file', 'py'])

    def test_promote_parents_dedupes_siblings(self):
        rag = RAG(None, None, _opts())
        children = [
            Document(page_content='c1', metadata={'doc_id': 'p1'}),
            Document(page_content='c2', metadata={'doc_id': 'p1'}),
            Document(page_content='orphan', metadata={}),
        ]
        parent = Document(page_content='PARENT')

        class _Store:
            def mget(self, ids):
                self.ids = ids
                return [parent]

        store = _Store()
        rag._docstore = lambda collection: store
        out = rag._promote_parents(children, 'col')
        self.assertEqual([d.page_content for d in out], ['PARENT', 'orphan'])

    def test_apply_rerank_passes_timeout(self):
        rag = RAG(None, None, _opts(
            matches=1,
            rerank_llm='bge',
            rerank_host='http://x/v1',
            rerank_timeout=2.5,
        ))
        docs = [Document(page_content='a'), Document(page_content='b')]
        with patch('rag_manager.post_rerank', return_value=[1]) as posted:
            rag._apply_rerank('q', docs)
        self.assertEqual(posted.call_args.kwargs['timeout'], 2.5)


if __name__ == '__main__':
    unittest.main()
