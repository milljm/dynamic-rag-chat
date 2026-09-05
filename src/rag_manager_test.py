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

    def test_collection_prefix_fork_is_not_shared_assistant(self):
        self.assertEqual(
            RAG.collection_prefix('scratch', 'user_documents', True),
            'scratch_',
        )
        self.assertEqual(
            RAG.collection_prefix('scratch', 'ai_documents', True),
            'scratch_',
        )
        self.assertEqual(
            RAG.collection_prefix('assistant', 'user_documents', True),
            'assistant_',
        )

    def test_collection_prefix_gold_stays_shared(self):
        self.assertEqual(
            RAG.collection_prefix('scratch', 'gold_documents', True),
            'assistant_',
        )
        self.assertEqual(
            RAG.collection_prefix('story', 'gold_documents', False),
            '',
        )

    def test_is_hnsw_error(self):
        self.assertTrue(RAG._is_hnsw_error(
            RuntimeError(
                'Error executing plan: Internal error: '
                'Error creating hnsw segment reader: Nothing found on disk',
            ),
        ))
        self.assertFalse(RAG._is_hnsw_error(ValueError('bad k')))

    def test_retrieve_skips_empty_collection(self):
        rag = RAG(None, None, _opts(emb_host='http://x', embeddings='m', matches=4))
        rag.embeddings = object()

        class _Col:
            def count(self):
                return 0

        class _Vec:
            _collection = _Col()

            def as_retriever(self, **kwargs):
                raise AssertionError('must not query an empty HNSW index')

        rag._vector_store = lambda collection: _Vec()
        self.assertEqual(rag.retrieve('hello!', 'scratch_user_documents'), [])

    def test_retrieve_heals_hnsw_gap(self):
        rag = RAG(None, None, _opts(emb_host='http://x', embeddings='m', matches=4))
        rag.embeddings = object()
        healed = []

        class _Col:
            def count(self):
                return 3

        class _Ret:
            def invoke(self, query):
                del query
                raise RuntimeError(
                    'Error creating hnsw segment reader: Nothing found on disk',
                )

        class _Vec:
            _collection = _Col()
            _client = object()

            def as_retriever(self, **kwargs):
                return _Ret()

        rag._vector_store = lambda collection: _Vec()
        rag._heal_hnsw = healed.append
        self.assertEqual(rag.retrieve('hello!', 'scratch_ai_documents'), [])
        self.assertEqual(healed, ['scratch_ai_documents'])

    def test_drop_chroma_uses_cached_client(self):
        rag = RAG(None, None, _opts())
        deleted = []

        class _Client:
            def delete_collection(self, name):
                deleted.append(name)

        class _Vec:
            _client = _Client()

        rag._chroma['scratch_user_documents'] = _Vec()
        rag._drop_chroma_collection('scratch_user_documents')
        self.assertEqual(deleted, ['scratch_user_documents'])
        self.assertNotIn('scratch_user_documents', rag._chroma)

    def test_wipe_branch_stores_spares_gold(self):
        with tempfile.TemporaryDirectory() as tmp:
            rag = RAG(None, SimpleNamespace(
                attributes=SimpleNamespace(collections={
                    'user': 'user_documents',
                    'ai': 'ai_documents',
                    'gold': 'gold_documents',
                }),
            ), _opts(vector_dir=tmp))
            dropped = []
            rag._drop_chroma_collection = dropped.append
            for name in (
                'branch_x_user_documents',
                'branch_x_ai_documents',
                'branch_y_user_documents',
                'branch_y_ai_documents',
                'assistant_gold_documents',
                'branch_x_gold_documents',
            ):
                os.makedirs(os.path.join(tmp, name))
            rag.wipe_branch_stores('branch_x')
            self.assertEqual(
                set(dropped),
                {'branch_x_user_documents', 'branch_x_ai_documents'},
            )
            self.assertFalse(os.path.isdir(os.path.join(tmp, 'branch_x_user_documents')))
            self.assertFalse(os.path.isdir(os.path.join(tmp, 'branch_x_ai_documents')))
            self.assertTrue(os.path.isdir(os.path.join(tmp, 'branch_y_user_documents')))
            self.assertTrue(os.path.isdir(os.path.join(tmp, 'branch_y_ai_documents')))
            self.assertTrue(os.path.isdir(os.path.join(tmp, 'assistant_gold_documents')))
            self.assertTrue(os.path.isdir(os.path.join(tmp, 'branch_x_gold_documents')))

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

    def test_delete_entries_removes_children_and_parents(self):
        rag = RAG(None, None, _opts())

        class _Col:
            def __init__(self):
                self.deleted = []

            def get(self, include=None):
                del include
                return {
                    'ids': ['child-1', 'other'],
                    'metadatas': [{'doc_id': 'p1'}, {'doc_id': 'p2'}],
                }

            def delete(self, ids=None):
                self.deleted.extend(ids or [])

        class _Vec:
            def __init__(self):
                self._collection = _Col()

        class _Store:
            def __init__(self):
                self.deleted = []

            def mdelete(self, ids):
                self.deleted.extend(ids)

        vec = _Vec()
        store = _Store()
        rag._vector_store = lambda collection: vec
        rag._docstore = lambda collection: store
        n = rag.delete_entries('story_user_documents', ['p1'])
        self.assertEqual(n, 1)
        self.assertEqual(vec._collection.deleted, ['child-1'])
        self.assertEqual(store.deleted, ['p1'])

    def test_delete_entries_skips_gold(self):
        rag = RAG(None, None, _opts())
        called = []
        rag._vector_store = lambda collection: called.append(collection)
        self.assertEqual(rag.delete_entries('gold_documents', ['p1']), 0)
        self.assertEqual(called, [])

    def test_purge_entry_refs_groups_and_skips_gold(self):
        rag = RAG(None, None, _opts())
        seen = []

        def fake_delete(collection, ids):
            seen.append((collection, list(ids)))
            return len(ids)

        rag.delete_entries = fake_delete
        n = rag.purge_entry_refs([
            ('story_user_documents', 'u1'),
            ('story_ai_documents', 'a1'),
            ('gold_documents', 'g1'),
            ('story_user_documents', 'u2'),
        ])
        self.assertEqual(n, 3)
        self.assertEqual(seen, [
            ('story_user_documents', ['u1', 'u2']),
            ('story_ai_documents', ['a1']),
        ])

    def test_store_data_uses_supplied_ids(self):
        rag = RAG(None, None, _opts())
        rag._embeddings_ready = lambda: True
        rag.common = SimpleNamespace(
            sanitize_response=lambda data, strip=True: data,
            normalize_metadata_for_rag=lambda meta: meta,
            attributes=SimpleNamespace(collections={'ai': 'ai_documents'}),
        )
        captured = {}

        class _Ret:
            def add_documents(self, docs, ids=None):
                captured['ids'] = ids
                captured['docs'] = docs

        rag._parent_retriever = lambda collection: _Ret()
        rag.console = SimpleNamespace(print=lambda *a, **k: None)
        out = rag.store_data(
            'hello',
            tags_metadata=[],
            collection='story_ai_documents',
            ids=['fixed-id'],
        )
        self.assertEqual(out, ['fixed-id'])
        self.assertEqual(captured['ids'], ['fixed-id'])
        self.assertEqual(captured['docs'][0].metadata.get('doc_id'), 'fixed-id')


if __name__ == '__main__':
    unittest.main()
