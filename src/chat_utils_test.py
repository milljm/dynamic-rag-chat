"""Run with: python src/chat_utils_test.py"""
from __future__ import annotations

import os
import sys
import unittest
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from chat_utils import (  # noqa: C0413
    ChatOptions,
    CommonUtils,
    EndMarkerFeed,
    RAGTag,
    RegExp,
    active_branch,
    dedupe_rag_chunks,
    overlap_ratio,
    strip_end_markers,
)


class ActiveBranchTest(unittest.TestCase):
    """Bare ./chat.py is story even if Spur left current=assistant."""

    def test_flag_forces_assistant(self):
        hist = {'current': 'story', 'story': [], 'assistant': []}
        self.assertEqual(active_branch(True, hist), 'assistant')

    def test_bare_cli_does_not_resume_assistant(self):
        hist = {
            'current': 'assistant',
            'assistant_mode': True,
            'story': [{'role': 'user', 'content': 'hi'}],
            'assistant': [{'role': 'user', 'content': 'help'}],
        }
        self.assertEqual(active_branch(False, hist), 'story')

    def test_bare_cli_keeps_story_fork(self):
        hist = {
            'current': 'alt-ending',
            'story': [],
            'assistant': [],
            'alt-ending': [{'role': 'user', 'content': 'once'}],
        }
        self.assertEqual(active_branch(False, hist), 'alt-ending')

    def test_assistant_mode_keeps_user_fork(self):
        hist = {
            'current': 'testing',
            'assistant_mode': True,
            'story': [],
            'assistant': [{'role': 'user', 'content': 'help'}],
            'testing': [{'role': 'user', 'content': 'forked'}],
            'branch_modes': {'testing': True},
        }
        self.assertEqual(active_branch(True, hist), 'testing')


class PromptStackParserTest(unittest.TestCase):
    """The tagger's prompt_stack rides *beside* metadata, never inside it."""

    def test_reads_the_sibling_key(self):
        raw = ('{"metadata": {"entity": ["aeloria"]}, '
               '"prompt_stack": ["mood_combat", "mood_tense"]}')
        self.assertEqual(
            CommonUtils.get_prompt_stack(raw), ['mood_combat', 'mood_tense'])

    def test_metadata_only_yields_empty(self):
        raw = '{"metadata": {"entity": ["aeloria"]}}'
        self.assertEqual(CommonUtils.get_prompt_stack(raw), [])

    def test_nested_in_metadata_is_not_read(self):
        """Nested, it would become a RAGTag: written to Chroma + the scene."""
        raw = '{"metadata": {"prompt_stack": ["mood_combat"]}}'
        self.assertEqual(CommonUtils.get_prompt_stack(raw), [])

    def test_malformed_and_empty_are_safe(self):
        for raw in ('', 'not json', '[]', '{"metadata": {}}', None):
            self.assertEqual(CommonUtils.get_prompt_stack(raw), [])

    def test_strips_think_frame_and_normalises(self):
        raw = ('<think>choosing a mood</think>{"metadata": {}, '
               '"prompt_stack": ["mood_combat", "MOOD_COMBAT", " mood_anger "]}')
        self.assertEqual(
            CommonUtils.get_prompt_stack(raw), ['mood_combat', 'mood_anger'])

    def test_single_string_is_wrapped(self):
        raw = '{"metadata": {}, "prompt_stack": "mood_combat"}'
        self.assertEqual(CommonUtils.get_prompt_stack(raw), ['mood_combat'])

    def test_get_families_reads_its_own_key_only(self):
        self.assertEqual(
            CommonUtils.get_families('{"families": ["danger", "QUIET"]}'),
            ['danger', 'quiet'])
        # Sibling keys must not bleed into one another.
        self.assertEqual(
            CommonUtils.get_prompt_stack('{"families": ["danger"]}'), [])
        self.assertEqual(
            CommonUtils.get_families('{"prompt_stack": ["mood_x"]}'), [])

class TagSanitisingTest(unittest.TestCase):
    """A ~2B tagger sends [[]] and ["none"]; neither may reach the pipeline."""

    def test_nested_empty_list_is_flattened_away(self):
        tags = CommonUtils.parse_tags({'audience': [[]], 'npc_locations': [[]]})
        self.assertEqual(tags, [RAGTag('audience', []),
                                RAGTag('npc_locations', [])])

    def test_placeholder_values_are_dropped(self):
        tags = CommonUtils.parse_tags({
            'audience': ['none'], 'npc_locations': ['unknown', 'n/a'],
        })
        self.assertEqual(tags, [RAGTag('audience', []),
                                RAGTag('npc_locations', [])])

    def test_nested_real_values_survive(self):
        tags = CommonUtils.parse_tags({
            'entity': [['aeloria'], [], 'elara', 'aeloria'],
        })
        self.assertEqual(tags, [RAGTag('entity', ['aeloria', 'elara'])])

    def test_dedupe_survives_a_nested_list(self):
        """This exact key raised 'unhashable type: list' in production."""
        tags = [RAGTag('audience', [[]]), RAGTag('audience', [[]]),
                RAGTag('entity', ['aeloria'])]
        deduped = CommonUtils.dedupe_tags(tags)
        self.assertEqual(len(deduped), 2)

    def test_dedupe_keeps_first_occurrence_order(self):
        tags = [RAGTag('a', ['1']), RAGTag('b', 'x'), RAGTag('a', ['1'])]
        self.assertEqual(CommonUtils.dedupe_tags(tags),
                         [RAGTag('a', ['1']), RAGTag('b', 'x')])


class ChatOptionsYamlTest(unittest.TestCase):
    """Settings page writes llm_server; unknown keys must not crash."""

    def test_llm_server_alias(self):
        opts = ChatOptions._build('.', {  # pylint: disable=protected-access
            'llm_server': 'http://127.0.0.1:1234/v1',
            'model': 'minimax-m3',
        })
        self.assertEqual(opts.host, 'http://127.0.0.1:1234/v1')
        self.assertEqual(opts.model, 'minimax-m3')

    def test_unknown_yaml_key_ignored(self):
        opts = ChatOptions._build('.', {  # pylint: disable=protected-access
            'nope': 'whatever',
            'model': 'x',
        })
        self.assertEqual(opts.model, 'x')

    def test_empty_api_key_becomes_none(self):
        opts = ChatOptions._build('.', {  # pylint: disable=protected-access
            'api_key': '',
        })
        self.assertEqual(opts.api_key, 'none')

    def test_empty_vision_stays_sentinel(self):
        opts = ChatOptions._build('.', {  # pylint: disable=protected-access
            'vision_llm': None,
            'agent_llm': '',
            'model': 'minimax-m3',
        })
        self.assertEqual(opts.vision_llm, 'None')
        self.assertEqual(opts.agent_llm, 'None')
        self.assertEqual(opts.model, 'minimax-m3')

    def test_specialized_server_survives_model_inherit(self):
        opts = ChatOptions._build('.', {  # pylint: disable=protected-access
            'llm_server': 'http://main:1234/v1',
            'model': 'big-model',
            'coder_server': 'http://coder:1234/v1',
        })
        self.assertEqual(opts.coder_llm, 'big-model')
        self.assertEqual(opts.coder_host, 'http://coder:1234/v1')
        self.assertEqual(opts.host, 'http://main:1234/v1')

    def test_blank_specialized_server_inherits_main(self):
        opts = ChatOptions._build('.', {  # pylint: disable=protected-access
            'llm_server': 'http://main:1234/v1',
            'model': 'big-model',
            'coder_llm': 'coder-model',
        })
        self.assertEqual(opts.coder_llm, 'coder-model')
        self.assertEqual(opts.coder_host, 'http://main:1234/v1')


def _store_sanitize(text: str) -> str:
    """Same transforms store_data applies: drop fences, lowercase, collapse space."""
    return CommonUtils.normalize_for_dedup(text.replace('```', ''))


class DedupeRagChunksTest(unittest.TestCase):
    """USER_RAG / AI_RAG must not echo CHAT_HISTORY."""

    def test_user_question_case_was_just_under_threshold(self):
        hist = 'Can you create a matplotlib python example that uses Streamlit as a GUI front-end?'
        rag = 'can you create a matplotlib python example that uses streamlit as a gui front-end?'
        raw = overlap_ratio(rag, hist)
        self.assertLess(raw, 0.65)
        self.assertGreater(
            overlap_ratio(_store_sanitize(rag), _store_sanitize(hist)),
            0.65,
        )
        self.assertEqual(
            dedupe_rag_chunks(
                [rag],
                [{'role': 'user', 'content': hist}],
                sanitize=_store_sanitize,
            ),
            [],
        )

    def test_ai_reply_with_fences_vs_stored_parent(self):
        history_ai = (
            "Sure thing. Here's a self-contained example that gives you "
            'interactive controls in the sidebar and two live matplotlib charts. '
            'Save it and run with `streamlit run matplotlib_streamlit_app.py`.\n\n'
            '```python matplotlib_streamlit_app.py\n'
            'import streamlit as st\n'
            'import matplotlib.pyplot as plt\n'
            'import numpy as np\n'
            '```\n'
        )
        stored = _store_sanitize(history_ai)
        self.assertEqual(
            dedupe_rag_chunks(
                [stored],
                [
                    {'role': 'user', 'content': 'Can you create a matplotlib example?'},
                    {'role': 'assistant', 'content': history_ai},
                ],
                sanitize=_store_sanitize,
            ),
            [],
        )

    def test_keeps_older_turn_not_in_window(self):
        kept = dedupe_rag_chunks(
            ['the wizard left town three sessions ago'],
            [{'role': 'assistant', 'content': 'Hello. What now?'}],
            sanitize=_store_sanitize,
        )
        self.assertEqual(kept, ['the wizard left town three sessions ago'])

    def test_short_history_does_not_drop_gold_file(self):
        gold = (
            'matplotlib cookbook\n' * 40
            + 'Can you create a matplotlib python example that uses Streamlit as a GUI front-end?'
        )
        kept = dedupe_rag_chunks(
            [gold],
            [{'role': 'user', 'content':
              'Can you create a matplotlib python example that uses Streamlit as a GUI front-end?'}],
            sanitize=_store_sanitize,
        )
        self.assertEqual(kept, [gold])

    def test_empty_history_keeps_chunks(self):
        self.assertEqual(
            dedupe_rag_chunks(['fresh fact'], [], sanitize=_store_sanitize),
            ['fresh fact'],
        )

    def test_sanitize_response_matches_store_data(self):
        util = CommonUtils.__new__(CommonUtils)
        util.opts = SimpleNamespace(assistant_mode=True, debug=False)
        util.regex = RegExp()
        history = 'Sure thing.\n\n```python hello.py\nprint("hi")\n```\n'
        stored = util.sanitize_response(history, strip=True)
        self.assertNotIn('```', stored)
        self.assertEqual(stored, stored.lower())
        self.assertEqual(
            dedupe_rag_chunks(
                [stored],
                [{'role': 'assistant', 'content': history}],
                sanitize=lambda text: util.sanitize_response(text, strip=True),
            ),
            [],
        )


class StripEndMarkersTest(unittest.TestCase):
    """Double-post regression: a reply is cut at its end-of-turn marker."""

    def test_marker_on_final_line_is_removed(self):
        self.assertEqual(
            strip_end_markers('Scene ends here.\n<END_TURN>'),
            'Scene ends here.',
        )

    def test_double_post_is_cut_at_first_marker(self):
        text = 'Take one ends.<END_TURN>Take two begins. Also <END_TURN>'
        self.assertEqual(strip_end_markers(text), 'Take one ends.')

    def test_beat_marker_cuts_too(self):
        self.assertEqual(strip_end_markers('Beat.<END_BEAT>continued'), 'Beat.')

    def test_partial_trailing_marker_is_removed(self):
        self.assertEqual(strip_end_markers('Scene ends.<END_TUR'), 'Scene ends.')
        self.assertEqual(strip_end_markers('Scene ends.<END'), 'Scene ends.')

    def test_prose_angle_brackets_survive(self):
        text = 'She scrawled <3 on the note and <something> in the margin'
        self.assertEqual(strip_end_markers(text), text)

    def test_empty_text_passes_through(self):
        self.assertEqual(strip_end_markers(''), '')

    def test_sanitize_response_strips_markers(self):
        util = CommonUtils.__new__(CommonUtils)
        util.opts = SimpleNamespace(assistant_mode=False)
        util.regex = RegExp()
        cleaned = util.sanitize_response('Reply text <END_TURN> second take')
        self.assertNotIn('<END_TURN>', cleaned)
        self.assertIn('Reply text', cleaned)


class EndMarkerFeedTest(unittest.TestCase):
    """Stream filter: markers never surface; hit fires on the full marker."""

    def test_marker_split_across_chunks(self):
        feed = EndMarkerFeed()
        self.assertEqual(feed.feed('Scene ends.\n<EN'), 'Scene ends.\n')
        self.assertFalse(feed.hit)
        self.assertEqual(feed.feed('D_TURN>'), '')
        self.assertTrue(feed.hit)

    def test_marker_with_pre_and_post_text_in_one_chunk(self):
        feed = EndMarkerFeed()
        self.assertEqual(feed.feed('End here.<END_TURN>never shown'), 'End here.')
        self.assertTrue(feed.hit)
        self.assertEqual(feed.feed('more'), '')
        self.assertEqual(feed.flush(), '')

    def test_beat_marker_hits_too(self):
        feed = EndMarkerFeed()
        self.assertEqual(feed.feed('Beat.<END_BEAT>rest'), 'Beat.')
        self.assertTrue(feed.hit)

    def test_clean_text_passes_through(self):
        feed = EndMarkerFeed()
        self.assertEqual(feed.feed('Harrick laughs. '), 'Harrick laughs. ')
        self.assertFalse(feed.hit)

    def test_partial_marker_tail_is_dropped_by_flush(self):
        feed = EndMarkerFeed()
        self.assertEqual(feed.feed('Scene ends.<END_TUR'), 'Scene ends.')
        self.assertFalse(feed.hit)
        self.assertEqual(feed.flush(), '')

    def test_prose_angle_bracket_passes_through(self):
        feed = EndMarkerFeed()
        self.assertEqual(feed.feed('She scrawled <3 on'), 'She scrawled <3 on')
        self.assertEqual(feed.feed(' the note'), ' the note')
        self.assertEqual(feed.flush(), '')

    def test_empty_feed_is_noop(self):
        feed = EndMarkerFeed()
        self.assertEqual(feed.feed(''), '')
        self.assertFalse(feed.hit)

    def test_flush_empty_after_hit(self):
        feed = EndMarkerFeed()
        feed.feed('done.<END_TURN>')
        self.assertEqual(feed.flush(), '')


class ChatOptionsTuningTest(unittest.TestCase):
    """Slider values persist as strings; _build coerces and whitelists."""

    def test_string_floats_are_coerced(self):
        opts = ChatOptions._build('.', {'model_temp': '1.0', 'pre_topp': '0.9'})
        self.assertEqual(opts.model_temp, 1.0)
        self.assertEqual(opts.pre_topp, 0.9)

    def test_blank_numbers_keep_defaults(self):
        opts = ChatOptions._build('.', {'model_temp': None, 'casual_temp': '  '})
        self.assertEqual(opts.model_temp, 1.0)
        self.assertEqual(opts.casual_temp, 0.9)

    def test_effort_is_whitelisted(self):
        opts = ChatOptions._build('.', {
            'model_reasoning_effort': 'MAX',
            'pre_reasoning_effort': 'bogus',
        })
        self.assertEqual(opts.model_reasoning_effort, 'max')
        self.assertEqual(opts.pre_reasoning_effort, '')

    def test_effort_defaults_to_blank(self):
        opts = ChatOptions()
        self.assertEqual(opts.model_reasoning_effort, '')
        self.assertEqual(opts.structured_reasoning_effort, '')


if __name__ == '__main__':
    unittest.main()
