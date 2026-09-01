"""Run with: python src/import_data_test.py"""
from __future__ import annotations

import os
import sys
import unittest
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from import_data import ImportData  # noqa: E402
except ImportError:
    ImportData = None
    RecursiveCharacterTextSplitter = None


class _FakeRAG:
    def __init__(self):
        self.stored = []
        self.parent_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=500, separators=['\n\n'],
        )
        self.child_splitter = RecursiveCharacterTextSplitter(
            chunk_size=100, chunk_overlap=50, separators=['.'],
        )

    def store_data(self, *args, **kwargs):
        self.stored.append((args, kwargs))


@unittest.skipUnless(ImportData, 'import_data deps not installed in this env')
class ImportDataTest(unittest.TestCase):
    """Children must not be re-embedded one HTTP call at a time."""

    def test_do_childdocs_does_not_store(self):
        rag = _FakeRAG()
        session = SimpleNamespace(
            rag=rag,
            common=SimpleNamespace(
                opts=SimpleNamespace(assistant_mode=False),
                attributes=SimpleNamespace(collections={'gold': 'gold_documents'}),
            ),
        )
        importer = ImportData(session)
        importer.live = None
        importer._do_childdocs(
            ['alpha. ' * 5, 'beta. ' * 5, 'gamma. ' * 5],
            'notes.txt',
            (1, 1),
            [],
        )
        self.assertEqual(rag.stored, [])


if __name__ == '__main__':
    unittest.main()
