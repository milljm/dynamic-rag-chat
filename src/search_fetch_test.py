"""Run with: python src/search_fetch_test.py"""
from __future__ import annotations

import os
import sys
import unittest

try:
    from .search_fetch import MidTurnFeed, search_status, take_need_search
except ImportError:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from search_fetch import MidTurnFeed, search_status, take_need_search


class TakeNeedSearchTest(unittest.TestCase):
    """Parse <NEED_SEARCH:query> out of a reply."""

    def test_strips_tag(self):
        vis, query = take_need_search(
            'Let me look that up.\n<NEED_SEARCH:NVDA share price>\n',
        )
        self.assertEqual(query, 'NVDA share price')
        self.assertEqual(vis, 'Let me look that up.')

    def test_no_tag(self):
        vis, query = take_need_search('just an answer')
        self.assertIsNone(query)
        self.assertEqual(vis, 'just an answer')

    def test_backticks_around_query_on_own_line(self):
        _, query = take_need_search('<NEED_SEARCH:`current python version`>')
        self.assertEqual(query, 'current python version')

    def test_placeholder_query_is_ignored(self):
        text = 'The tag looks like <NEED_SEARCH:query> when you explain it.'
        vis, query = take_need_search(text)
        self.assertIsNone(query)
        self.assertEqual(vis, text)

    def test_real_tag_after_placeholder(self):
        vis, query = take_need_search(
            'e.g. <NEED_SEARCH:query>\n<NEED_SEARCH:weather Tampa>\n',
        )
        self.assertEqual(query, 'weather Tampa')
        self.assertIn('<NEED_SEARCH:query>', vis)
        self.assertNotIn('weather Tampa>', vis)

    def test_inline_example_is_talk_not_fetch(self):
        text = 'emit `<NEED_SEARCH:latest news>` and then keep talking'
        vis, query = take_need_search(text)
        self.assertIsNone(query)
        self.assertEqual(vis, text)

    def test_gold_tag_is_not_a_search(self):
        vis, query = take_need_search('<NEED_GOLD:README.md>')
        self.assertIsNone(query)
        self.assertEqual(vis, '<NEED_GOLD:README.md>')


class MidTurnFeedTest(unittest.TestCase):
    """Hold back NEED_GOLD / NEED_SEARCH across chunks."""

    def test_split_search_tag(self):
        feed = MidTurnFeed()
        a, hit = feed.feed('Lead in.\n<NEED_')
        self.assertFalse(hit)
        self.assertEqual(a, 'Lead in.\n')
        b, hit = feed.feed('SEARCH:NVDA price>\n')
        self.assertTrue(hit)
        self.assertEqual(b, '')
        self.assertEqual(feed.kind, 'search')
        self.assertEqual(feed.query, 'NVDA price')

    def test_split_gold_still_works(self):
        feed = MidTurnFeed()
        a, hit = feed.feed('Lead in.\n<NEED_')
        self.assertFalse(hit)
        self.assertEqual(a, 'Lead in.\n')
        b, hit = feed.feed('GOLD:prompt_manager.py>\n')
        self.assertTrue(hit)
        self.assertEqual(b, '')
        self.assertEqual(feed.kind, 'gold')
        self.assertEqual(feed.filename, 'prompt_manager.py')

    def test_need_prefix_does_not_leak_before_kind_is_known(self):
        feed = MidTurnFeed()
        a, hit = feed.feed('<need_s')
        self.assertFalse(hit)
        self.assertEqual(a, '')
        b, hit = feed.feed('earch:tampa weather>\n')
        self.assertTrue(hit)
        self.assertEqual(b, '')
        self.assertEqual(feed.query, 'tampa weather')

    def test_search_inside_reasoning_text(self):
        feed = MidTurnFeed()
        a, hit = feed.feed('need a price\n')
        self.assertFalse(hit)
        b, hit = feed.feed('<NEED_SEARCH:AAPL>')
        self.assertFalse(hit)
        c = feed.flush()
        self.assertEqual(feed.kind, 'search')
        self.assertEqual(feed.query, 'AAPL')
        self.assertEqual((a + b + c).strip(), 'need a price')

    def test_placeholder_does_not_complete(self):
        feed = MidTurnFeed()
        a, hit = feed.feed('copy this: <NEED_SEARCH:query>')
        self.assertFalse(hit)
        self.assertIsNone(feed.kind)
        self.assertEqual(a + feed.flush(), 'copy this: <NEED_SEARCH:query>')

    def test_inline_cookbook_example_does_not_fetch(self):
        feed = MidTurnFeed()
        text = 'search tags like `<NEED_SEARCH:latest news>`'
        a, hit = feed.feed(text)
        self.assertFalse(hit)
        self.assertIsNone(feed.kind)
        self.assertEqual(a + feed.flush(), text)

    def test_false_angle(self):
        feed = MidTurnFeed()
        a, hit = feed.feed('x < 3 and y > 4')
        self.assertFalse(hit)
        self.assertEqual(a + feed.flush(), 'x < 3 and y > 4')


class SearchStatusTest(unittest.TestCase):
    """Status line for Searching web."""

    def test_empty(self):
        self.assertEqual(search_status([]), 'Searching web…')

    def test_lists_query(self):
        self.assertIn('NVDA', search_status(['NVDA share price']))
