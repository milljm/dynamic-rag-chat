"""Run with: python src/gold_fetch_test.py"""
from __future__ import annotations

import os
import sys
import unittest

try:
    from .gold_fetch import GoldNeedFeed, take_need_gold, recall_status
except ImportError:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from gold_fetch import GoldNeedFeed, take_need_gold, recall_status


class TakeNeedGoldTest(unittest.TestCase):
    """Parse <NEED_GOLD:file> out of a reply."""

    def test_strips_tag(self):
        vis, name = take_need_gold(
            'I need the assembler.\n<NEED_GOLD:prompt_manager.py>\n',
        )
        self.assertEqual(name, 'prompt_manager.py')
        self.assertEqual(vis, 'I need the assembler.')

    def test_no_tag(self):
        vis, name = take_need_gold('just an answer')
        self.assertIsNone(name)
        self.assertEqual(vis, 'just an answer')

    def test_backticks(self):
        _, name = take_need_gold('<NEED_GOLD:`spur-server.py`>')
        self.assertEqual(name, 'spur-server.py')


class GoldNeedFeedTest(unittest.TestCase):
    """Hold back the tag across chunks."""

    def test_split_tag(self):
        feed = GoldNeedFeed()
        a, hit = feed.feed('Lead in. <NEED_')
        self.assertFalse(hit)
        self.assertEqual(a, 'Lead in. ')
        b, hit = feed.feed('GOLD:prompt_manager.py>')
        self.assertTrue(hit)
        self.assertEqual(b, '')
        self.assertEqual(feed.filename, 'prompt_manager.py')

    def test_tag_inside_reasoning_text(self):
        feed = GoldNeedFeed()
        a, hit = feed.feed('need the assembler\n')
        self.assertFalse(hit)
        b, hit = feed.feed('<NEED_GOLD:prompt_manager.py>')
        self.assertTrue(hit)
        self.assertEqual(feed.filename, 'prompt_manager.py')
        self.assertEqual((a + b).strip(), 'need the assembler')


    def test_false_angle(self):
        feed = GoldNeedFeed()
        a, hit = feed.feed('x < 3 and y > 4')
        self.assertFalse(hit)
        rest = a + feed.flush()
        self.assertEqual(rest, 'x < 3 and y > 4')


class RecallStatusTest(unittest.TestCase):
    """Status line for Recalling Documents."""

    def test_one_name(self):
        self.assertEqual(
            recall_status(['README.md']),
            'Recalling Documents… [README.md]',
        )

    def test_clips_at_40(self):
        label = recall_status([
            'README.md', 'render_window.py', 'context_manager.py',
        ])
        inner = label.split('[', 1)[1].rstrip(']')
        self.assertLessEqual(len(inner), 40)
        self.assertTrue(inner.endswith('...'))
        self.assertIn('README.md', inner)


if __name__ == '__main__':
    unittest.main()
