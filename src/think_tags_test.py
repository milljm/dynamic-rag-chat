"""Run with: python src/think_tags_test.py

Avoid `python -m src.think_tags_test` — src/__init__.py imports langchain.
"""
from __future__ import annotations

import os
import sys
import unittest

try:
    from .think_tags import ThinkFeed, chunk_text, split_think
except ImportError:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from think_tags import ThinkFeed, chunk_text, split_think


class _Chunk:
    def __init__(self, content=None, reasoning_content=None, additional_kwargs=None):
        self.content = content
        self.reasoning_content = reasoning_content
        self.additional_kwargs = additional_kwargs or {}


class ThinkTagsTest(unittest.TestCase):
    """ThinkFeed / split_think cases for MiniMax, gpt-oss, and bare tags."""
    def test_bare_think_still_works(self):
        visible, thought, in_think, ns, never = split_think(
            'Hello <think>secret', False, ''
        )
        self.assertEqual(visible, 'Hello ')
        self.assertEqual(thought, 'secret')
        self.assertTrue(in_think)
        self.assertFalse(never)
        visible, thought, in_think, ns, never = split_think(
            ' plan</think>\nanswer', True, ns, never
        )
        self.assertEqual(thought, ' plan')
        self.assertEqual(visible, '\nanswer')
        self.assertFalse(in_think)
        self.assertTrue(never)

    def test_mm_think_ignores_nested_think_mentions(self):
        raw = (
            '<mm:think>If a `<think>` tag gets split mid-token. '
            'handles `<think>...</think>` style tags.'
            '</mm:think>Took a read-through — solid.'
        )
        visible, thought, in_think, ns, never = split_think(raw, False, '')
        self.assertFalse(in_think)
        self.assertEqual(ns, '')
        self.assertTrue(never)
        self.assertIn('If a `<think>` tag gets split', thought)
        self.assertIn('</think>` style tags', thought)
        self.assertEqual(visible, 'Took a read-through — solid.')

    def test_mm_think_across_chunks(self):
        vis, thought, in_think, ns, never = split_think('<mm:think>If a `', False, '')
        self.assertEqual(vis, '')
        self.assertTrue(in_think)
        self.assertEqual(ns, 'mm:')
        self.assertFalse(never)

        vis, thought, in_think, ns, never = split_think(
            '<think>` tag gets split', True, ns, never
        )
        self.assertEqual(vis, '')
        self.assertEqual(thought, '<think>` tag gets split')
        self.assertTrue(in_think)
        self.assertEqual(ns, 'mm:')

        vis, thought, in_think, ns, never = split_think(
            'handles `<think>...</think>` style tags.</mm:think>Took a read-through — solid.',
            True,
            ns,
            never,
        )
        self.assertFalse(in_think)
        self.assertEqual(ns, '')
        self.assertTrue(never)
        self.assertIn('</think>` style tags', thought)
        self.assertEqual(vis, 'Took a read-through — solid.')

    def test_mm_think_literal_think_token_does_not_stop(self):
        """MiniMax-M3: a streamed chunk that *is* the token <think>."""
        feed = ThinkFeed()
        vis, thought = feed.feed('<mm:think>The uploaded file parses ')
        self.assertEqual(vis, '')
        self.assertTrue(feed.in_think)
        self.assertEqual(feed.ns, 'mm:')
        self.assertFalse(feed.never_think)

        vis, thought = feed.feed('<think>')
        self.assertEqual(vis, '')
        self.assertEqual(thought, '<think>')
        self.assertTrue(feed.in_think)
        self.assertEqual(feed.ns, 'mm:')
        self.assertFalse(feed.never_think)

        vis, thought = feed.feed(' tags in spur-server.py.')
        self.assertEqual(vis, '')
        self.assertEqual(thought, ' tags in spur-server.py.')
        self.assertTrue(feed.in_think)

        vis, thought = feed.feed('</mm:think>\nSolid adapter.')
        self.assertEqual(thought, '')
        self.assertEqual(vis, '\nSolid adapter.')
        self.assertFalse(feed.in_think)
        self.assertTrue(feed.never_think)

    def test_mm_think_inner_bare_closer_is_prose(self):
        feed = ThinkFeed()
        vis, thought = feed.feed('<mm:think>see </think> in the sample')
        self.assertEqual(vis, '')
        self.assertTrue(feed.in_think)
        self.assertEqual(feed.ns, 'mm:')
        self.assertIn('</think>', thought)
        vis, thought = feed.feed('</mm:think>done')
        self.assertEqual(vis, 'done')
        self.assertTrue(feed.never_think)

    def test_shadow_then_mm_think_still_opens(self):
        """Null tokens then a namespaced opener — MiniMax, not gpt-oss."""
        feed = ThinkFeed()
        vis, thought = feed.feed_chunk(_Chunk(content='', reasoning_content='warmup'))
        self.assertTrue(feed.shadow_think)
        self.assertEqual(thought, 'warmup')
        vis, thought = feed.feed('<mm:think>real reasoning')
        self.assertFalse(feed.shadow_think)
        self.assertTrue(feed.in_think)
        self.assertEqual(feed.ns, 'mm:')
        self.assertFalse(feed.never_think)
        self.assertEqual(vis, '')
        self.assertEqual(thought, 'real reasoning')

    def test_never_think_after_close_keeps_answer_mentions(self):
        raw = (
            '<think>Let me give my honest, casual take on this.\n'
            '</think>\n\n'
            'Oh, spur-server.py — clean move.\n'
            '4. The thinking/reasoning tag parsing — split_think() with the '
            'regex to handle `<think>` / `</thinking>` blocks is neat.'
        )
        vis, thought, in_think, ns, never = split_think(raw, False, '')
        self.assertFalse(in_think)
        self.assertTrue(never)
        self.assertIn('honest, casual take', thought)
        self.assertIn('Oh, spur-server.py', vis)
        self.assertIn('`<think>` / `</thinking>`', vis)
        self.assertNotIn('honest, casual take', vis)

    def test_never_think_latches_across_chunks(self):
        vis, thought, in_think, ns, never = split_think(
            '</think>\nOh, spur-server.py', True, '', False
        )
        self.assertTrue(never)
        self.assertFalse(in_think)
        self.assertIn('Oh, spur-server.py', vis)

        vis, thought, in_think, ns, never = split_think(
            ' handle `<think>` / `</thinking>` blocks is neat.',
            in_think,
            ns,
            never,
        )
        self.assertEqual(thought, '')
        self.assertFalse(in_think)
        self.assertIn('`<think>` / `</thinking>`', vis)

    def test_plain_first_token_never_thinks(self):
        vis, thought, in_think, ns, never = split_think('Hello world', False, '')
        self.assertEqual(vis, 'Hello world')
        self.assertTrue(never)
        vis, thought, in_think, ns, never = split_think(
            ' see `<think>` later', False, ns, never
        )
        self.assertEqual(thought, '')
        self.assertIn('`<think>`', vis)

    def test_empty_chunk_does_not_lock(self):
        vis, thought, in_think, ns, never = split_think('', False, '')
        self.assertFalse(never)
        vis, thought, in_think, ns, never = split_think(
            '<think>later', False, ns, never
        )
        self.assertTrue(in_think)
        self.assertEqual(thought, 'later')

    def test_chunk_text_survives_none_and_empty_list(self):
        piece, extra = chunk_text(_Chunk(content=None, reasoning_content='hmm'))
        self.assertEqual(piece, '')
        self.assertEqual(extra, 'hmm')
        piece, extra = chunk_text(_Chunk(content='', reasoning_content='x'))
        self.assertEqual(piece, '')
        self.assertEqual(extra, 'x')
        piece, extra = chunk_text(
            _Chunk(content=[], additional_kwargs={'reasoning_content': 'y'})
        )
        self.assertEqual(piece, '')
        self.assertEqual(extra, 'y')

    def test_gpt_oss_blank_tokens_latch_on_first_nonblank(self):
        feed = ThinkFeed()
        vis, thought = feed.feed_chunk(
            _Chunk(content=None, reasoning_content='The user attached spur-server.')
        )
        self.assertEqual(vis, '')
        self.assertEqual(thought, 'The user attached spur-server.')
        self.assertTrue(feed.shadow_think)
        self.assertFalse(feed.never_think)

        vis, thought = feed.feed_chunk(_Chunk(content='', reasoning_content=' Honest take.'))
        self.assertEqual(vis, '')
        self.assertEqual(thought, ' Honest take.')
        self.assertTrue(feed.shadow_think)

        vis, thought = feed.feed_chunk(
            _Chunk(
                content=(
                    'Oh, spur-server.py — clean move.\n'
                    '4. split_think() to handle `<think>` / `</thinking>` blocks.'
                )
            )
        )
        self.assertTrue(feed.never_think)
        self.assertFalse(feed.shadow_think)
        self.assertEqual(thought, '')
        self.assertIn('Oh, spur-server.py', vis)
        self.assertIn('`<think>` / `</thinking>`', vis)

    def test_qwen_tags_still_open_when_first_token_is_not_blank(self):
        feed = ThinkFeed()
        vis, thought = feed.feed('<think>secret')
        self.assertEqual(vis, '')
        self.assertEqual(thought, 'secret')
        self.assertTrue(feed.in_think)
        self.assertFalse(feed.never_think)
        vis, thought = feed.feed('</think>\nanswer `<think>` mention')
        self.assertEqual(thought, '')
        self.assertTrue(feed.never_think)
        self.assertIn('answer', vis)
        self.assertIn('`<think>` mention', vis)

    def test_spur_server_loop_keeps_emitting_after_think_token(self):
        """What spur-server.py does: feed_chunk per LLM chunk, emit SSE."""
        feed = ThinkFeed()
        events: list[tuple[str, str]] = []
        chunks = [
            _Chunk(content='<mm:think>Reviewing think_tags.py. The '),
            _Chunk(content='<think>'),
            _Chunk(content=' regex must not close mm:think. '),
            _Chunk(content='</think>'),
            _Chunk(content=' is mentioned in the file.'),
            _Chunk(content='</mm:think>Looks good.'),
        ]
        for chunk in chunks:
            vis, thought = feed.feed_chunk(chunk)
            if thought:
                events.append(('reasoning', thought))
            if vis:
                events.append(('token', vis))
        kinds = [k for k, _ in events]
        self.assertIn('reasoning', kinds)
        self.assertEqual(events[-1], ('token', 'Looks good.'))
        thoughts = ''.join(t for k, t in events if k == 'reasoning')
        self.assertIn('<think>', thoughts)
        self.assertIn('</think>', thoughts)
        self.assertTrue(feed.never_think)
        self.assertFalse(feed.in_think)


if __name__ == '__main__':
    unittest.main()
