"""Run with: python src/filter_builder_test.py

Avoid `python -m src.filter_builder_test` — src/__init__.py imports langchain.
"""
from __future__ import annotations

import os
import sys
import unittest

try:
    from .chat_utils import RAGTag
    from .filter_builder import FilterBuilder, metadata_matches
except ImportError:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from chat_utils import RAGTag
    from filter_builder import FilterBuilder, metadata_matches


class FilterBuilderTest(unittest.TestCase):
    """Field filter must match comma-joined Chroma metadata."""

    def test_skips_list_tags_no_longer(self):
        tags = [
            RAGTag('entity', ['jason', 'mira']),
            RAGTag('player_location', 'tavern'),
        ]
        spec = FilterBuilder().build(tags, 'entity')
        self.assertEqual(spec['field'], 'entity')
        self.assertEqual(spec['values'], ['jason', 'mira'])

    def test_splits_comma_string_and_strips(self):
        tags = [RAGTag('entity', 'Jason, Mira')]
        spec = FilterBuilder().build(tags, 'entity')
        self.assertEqual(spec['values'], ['jason', 'mira'])

    def test_ignores_other_fields_as_must(self):
        tags = [
            RAGTag('entity', 'mira'),
            RAGTag('moving_confidence', '0.9'),
            RAGTag('content_rating', 'sfw'),
        ]
        spec = FilterBuilder().build(tags, 'entity')
        self.assertEqual(spec['values'], ['mira'])
        self.assertNotIn('moving_confidence', spec)

    def test_metadata_matches_member_of_joined_string(self):
        meta = {'entity': 'jason, mira'}
        self.assertTrue(metadata_matches(meta, 'entity', ['mira']))
        self.assertFalse(metadata_matches(meta, 'entity', ['cal']))

    def test_nsfw_from_content_rating(self):
        tags = [RAGTag('content_rating', 'nsfw')]
        self.assertTrue(FilterBuilder.tags_are_nsfw(tags))
        tags = [RAGTag('content_rating', 'sfw')]
        self.assertFalse(FilterBuilder.tags_are_nsfw(tags))
        tags = [RAGTag('scene_mode', 'NSFW')]
        self.assertTrue(FilterBuilder.tags_are_nsfw(tags))

    def test_document_topics_same_path_as_entity(self):
        tags = [
            RAGTag('document_topics', ['ai', 'programming']),
            RAGTag('keywords_entities', ['langchain']),
            RAGTag('language', 'python'),
        ]
        spec = FilterBuilder().build(tags, 'document_topics')
        self.assertEqual(spec['field'], 'document_topics')
        self.assertEqual(spec['values'], ['ai', 'programming'])
        self.assertNotIn('keywords_entities', spec)
        meta = {'document_topics': 'ai, programming, rag'}
        self.assertTrue(metadata_matches(meta, 'document_topics', ['programming']))
        self.assertFalse(metadata_matches(meta, 'document_topics', ['cooking']))

    def test_story_and_assistant_fields_do_not_cross(self):
        tags = [
            RAGTag('entity', ['mira']),
            RAGTag('document_topics', ['ai']),
        ]
        self.assertEqual(FilterBuilder().build(tags, 'entity')['values'], ['mira'])
        self.assertEqual(
            FilterBuilder().build(tags, 'document_topics')['values'], ['ai'],
        )


if __name__ == '__main__':
    unittest.main()
