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

    def test_backticks_around_name_on_own_line(self):
        _, name = take_need_gold('<NEED_GOLD:`spur-server.py`>')
        self.assertEqual(name, 'spur-server.py')

    def test_placeholder_filename_is_ignored(self):
        text = 'The tag looks like <NEED_GOLD:filename> when you explain it.'
        vis, name = take_need_gold(text)
        self.assertIsNone(name)
        self.assertEqual(vis, text)

    def test_placeholder_with_ext_is_ignored(self):
        vis, name = take_need_gold('<NEED_GOLD:filename.py>')
        self.assertIsNone(name)
        self.assertIn('filename.py', vis)

    def test_real_tag_after_placeholder(self):
        vis, name = take_need_gold(
            'e.g. <NEED_GOLD:filename>\n<NEED_GOLD:README.md>\n',
        )
        self.assertEqual(name, 'README.md')
        self.assertIn('<NEED_GOLD:filename>', vis)
        self.assertNotIn('README.md>', vis)

    def test_inline_readme_is_talk_not_fetch(self):
        text = (
            'gold-fetch tags like `<NEED_GOLD:README.md>` and then keep talking'
        )
        vis, name = take_need_gold(text)
        self.assertIsNone(name)
        self.assertEqual(vis, text)


class GoldNeedFeedTest(unittest.TestCase):
    """Hold back the tag across chunks."""

    def test_split_tag(self):
        feed = GoldNeedFeed()
        a, hit = feed.feed('Lead in.\n<NEED_')
        self.assertFalse(hit)
        self.assertEqual(a, 'Lead in.\n')
        b, hit = feed.feed('GOLD:prompt_manager.py>\n')
        self.assertTrue(hit)
        self.assertEqual(b, '')
        self.assertEqual(feed.filename, 'prompt_manager.py')

    def test_tag_inside_reasoning_text(self):
        feed = GoldNeedFeed()
        a, hit = feed.feed('need the assembler\n')
        self.assertFalse(hit)
        b, hit = feed.feed('<NEED_GOLD:prompt_manager.py>')
        self.assertFalse(hit)
        c = feed.flush()
        self.assertTrue(feed.filename)
        self.assertEqual(feed.filename, 'prompt_manager.py')
        self.assertEqual((a + b + c).strip(), 'need the assembler')

    def test_false_angle(self):
        feed = GoldNeedFeed()
        a, hit = feed.feed('x < 3 and y > 4')
        self.assertFalse(hit)
        rest = a + feed.flush()
        self.assertEqual(rest, 'x < 3 and y > 4')

    def test_placeholder_does_not_complete(self):
        feed = GoldNeedFeed()
        a, hit = feed.feed('copy this: <NEED_GOLD:filename>')
        self.assertFalse(hit)
        self.assertIsNone(feed.filename)
        self.assertEqual(a + feed.flush(), 'copy this: <NEED_GOLD:filename>')

    def test_real_tag_after_placeholder_in_stream(self):
        feed = GoldNeedFeed()
        a, hit = feed.feed('<NEED_GOLD:filename>\n<NEED_GOLD:README.md>\n')
        self.assertTrue(hit)
        self.assertEqual(feed.filename, 'README.md')
        self.assertEqual(a, '<NEED_GOLD:filename>\n')

    def test_inline_cookbook_example_does_not_fetch(self):
        feed = GoldNeedFeed()
        text = 'gold-fetch tags like `<NEED_GOLD:README.md>`'
        a, hit = feed.feed(text)
        self.assertFalse(hit)
        self.assertIsNone(feed.filename)
        self.assertEqual(a + feed.flush(), text)

    def test_inline_then_more_tokens_stay_visible(self):
        feed = GoldNeedFeed()
        a, hit = feed.feed('tags like <NEED_GOLD:README.md>')
        self.assertFalse(hit)
        b, hit = feed.feed(' and keep going')
        self.assertFalse(hit)
        self.assertIsNone(feed.filename)
        self.assertEqual(
            a + b + feed.flush(),
            'tags like <NEED_GOLD:README.md> and keep going',
        )


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
