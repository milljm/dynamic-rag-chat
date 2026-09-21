"""Run with: python src/plot_manager_test.py

Avoid `python -m src.plot_manager_test` — src/__init__.py imports langchain.
"""
# pylint: disable=missing-function-docstring  # test names carry the intent
from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from types import SimpleNamespace

try:
    from .plot_manager import PlotManager
except ImportError:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from plot_manager import PlotManager


class _Console:
    """Discard console output so tests stay quiet."""

    def print(self, *args, **kwargs):  # pylint: disable=unused-argument
        del args, kwargs


class _Common:
    """write_debug stand-in."""

    @staticmethod
    def write_debug(*args, **kwargs):  # pylint: disable=unused-argument
        del args, kwargs


class _Prompts:
    """Slot lookups returning renderable (but inert) prompt text."""

    SSCRIBE = (
        'SCRIBE {{user_name}} {{turn_num}} {{user_input}} '
        '{{story_reply}} {{scene_state}} {{current_fable}}'
    )
    DIRECTOR = (
        'DIRECTOR {{user_name}} {{turn_num}} {{user_input}} '
        '{{story_reply}} {{scene_state}} {{current_fable}}'
    )

    def slot(self, role, name):  # pylint: disable=unused-argument
        return self.SSCRIBE if name == 'fable_scribe_human' else ''

    def optional_slot(self, role, name):  # pylint: disable=unused-argument
        return self.DIRECTOR if name == 'fable_director_human' else ''


class _LLM:
    """Canned scribe/director responses, in call order."""

    model_name = 'stub'

    def __init__(self):
        self.scribe = '{}'
        self.director = '{}'
        self.calls = 0

    def invoke(self, prompt):  # pylint: disable=unused-argument
        self.calls += 1
        payload = self.scribe if self.calls % 2 else self.director
        return SimpleNamespace(content=payload)


class _Scene:
    """SceneManager stand-in exposing only what PlotManager reads."""

    def __init__(self, scene=None):
        self.scene = scene or {
            'player_location': 'tavern',
            'entity': ['jason', 'mira'],
            'creature': [],
            'audience': ['mira'],
            'npc_locations': ['mira: tavern'],
            'known_characters': ['jason', 'mira'],
        }

    def get_scene(self):
        return self.scene


def _mgr(tmpdir: str, name: str = 'Jason',
         scene: _Scene | None = None) -> tuple[PlotManager, _LLM]:
    opts = SimpleNamespace(
        user_name=name,
        vector_dir=tmpdir,
        debug=False,
        color=0,
    )
    llm = _LLM()
    mgr = PlotManager(_Console(), _Common(), opts, llm, _Prompts(), scene)
    return mgr, llm


def _docs(turn: int, scribe: str, director: str = '{}',
          regen: bool = False) -> dict:
    return {
        'turn_num': turn,
        'user_query': 'I wait and watch Mira.',
        'llm_response': 'Mira turns the sealed letter over in her hands.',
        'regenerate': regen,
    }, scribe, director


def _record(mgr: PlotManager, llm: _LLM, turn: int,
            scribe: str, director: str = '{}', regen: bool = False) -> list:
    documents, llm.scribe, llm.director = _docs(turn, scribe, director, regen)
    return mgr.record(documents)


def _scribe_json(spine: str = 'The road to the citadel.',
                 loops: str = '[]') -> str:
    return json.dumps({'spine': spine, 'loops': json.loads(loops)})


def _director_json(names: list | None = None) -> str:
    directives = {
        name: {
            'drive': 'greed', 'secret': f'{name} poisoned the well',
            'plan': 'flee at dawn', 'stance': 'cold',
        } for name in (names or ['mira'])
    }
    return json.dumps({
        'npc_directives': directives,
        'dormant_arcs': ['The well sickness spreads.'],
        'director_notes': 'The constable is one day behind.',
    })


class PlotManagerTest(unittest.TestCase):
    """Fable persistence, turn guards, pruning, briefs and forking."""

    def test_record_persists_and_reloads(self):
        with tempfile.TemporaryDirectory() as tmp:
            mgr, llm = _mgr(tmp, scene=_Scene())
            llm.scribe = _scribe_json(loops=json.dumps([
                {'summary': 'the sealed letter', 'status': 'open',
                 'planted': True, 'entity': ['mira']},
            ]))
            llm.director = _director_json()
            self.assertEqual(mgr.record({'turn_num': 5}), [])
            self.assertEqual(mgr.fable['last_fabled_turn'], 5)
            self.assertTrue(mgr.fable['spine'])
            self.assertEqual(
                mgr.fable['loops'][0]['summary'], 'the sealed letter',
            )
            self.assertEqual(
                mgr.fable['npc_directives']['mira']['secret'],
                'mira poisoned the well',
            )
            self.assertTrue(os.path.exists(
                os.path.join(tmp, 'ephemeral_fable_story.json'),
            ))
            again = PlotManager(_Console(), _Common(), mgr.opts,
                                _LLM(), _Prompts(), None)
            self.assertEqual(again.fable['spine'], mgr.fable['spine'])
            self.assertEqual(again.fable['loops'], mgr.fable['loops'])
            self.assertEqual(
                again.fable['npc_directives'], mgr.fable['npc_directives'],
            )

    def test_same_turn_recorded_once(self):
        with tempfile.TemporaryDirectory() as tmp:
            mgr, llm = _mgr(tmp)
            llm.scribe = _scribe_json()
            first = mgr.record({'turn_num': 5})
            second = mgr.record({'turn_num': 5})
            self.assertEqual(first, [])
            self.assertEqual(second, [])
            self.assertEqual(llm.calls, 2)  # one scribe + one director

    def test_regenerate_redoes_the_turn(self):
        with tempfile.TemporaryDirectory() as tmp:
            mgr, llm = _mgr(tmp)
            llm.scribe = _scribe_json(loops=json.dumps([
                {'summary': 'old reply hook', 'status': 'open',
                 'planted': True, 'entity': []},
            ]))
            _record(mgr, llm, 5, llm.scribe)
            self.assertEqual(
                [loop['summary'] for loop in mgr.fable['loops']],
                ['old reply hook'],
            )
            # The rewrite drops the old hook and plants a different one.
            llm.scribe = _scribe_json(loops=json.dumps([
                {'summary': 'new reply hook', 'status': 'open',
                 'planted': True, 'entity': []},
            ]))
            archived = _record(mgr, llm, 5, llm.scribe, regen=True)
            self.assertEqual(archived, [])
            self.assertEqual(
                [loop['summary'] for loop in mgr.fable['loops']],
                ['new reply hook'],
            )

    def test_loops_touch_close_and_archive(self):
        with tempfile.TemporaryDirectory() as tmp:
            mgr, llm = _mgr(tmp)
            llm.scribe = _scribe_json(loops=json.dumps([
                {'summary': 'the sealed letter', 'status': 'closed',
                 'planted': False, 'entity': ['mira']},
            ]))
            mgr.fable['loops'] = [{
                'summary': 'the sealed letter', 'planted_turn': 2,
                'last_seen_turn': 3, 'entity': ['mira'],
            }]
            mgr.fable['last_fabled_turn'] = 3
            archived = mgr.record({'turn_num': 4})
            self.assertEqual(
                [item['summary'] for item in archived],
                ['the sealed letter'],
            )
            self.assertEqual(mgr.fable['loops'], [])

    def test_ttl_retires_untouched_loops(self):
        with tempfile.TemporaryDirectory() as tmp:
            mgr, llm = _mgr(tmp)
            llm.scribe = '{"spine": "still the citadel road."}'
            mgr.fable['loops'] = [{
                'summary': 'a promise nobody kept', 'planted_turn': 1,
                'last_seen_turn': 1, 'entity': ['jason'],
            }]
            mgr.fable['last_fabled_turn'] = 1
            archived = mgr.record({'turn_num': 20})
            self.assertEqual(
                [item['summary'] for item in archived],
                ['a promise nobody kept'],
            )
            self.assertEqual(mgr.fable['loops'], [])

    def test_loop_cap_keeps_latest(self):
        with tempfile.TemporaryDirectory() as tmp:
            mgr, llm = _mgr(tmp)
            llm.scribe = _scribe_json(loops=json.dumps([
                {'summary': f'hook number {i}', 'status': 'open',
                 'planted': True, 'entity': []}
                for i in range(10)
            ]))
            mgr.record({'turn_num': 1})
            self.assertEqual(len(mgr.fable['loops']), 8)
            summaries = [loop['summary'] for loop in mgr.fable['loops']]
            self.assertNotIn('hook number 0', summaries)
            self.assertIn('hook number 9', summaries)

    def test_directive_roster_filter_and_cap(self):
        with tempfile.TemporaryDirectory() as tmp:
            mgr, llm = _mgr(tmp, scene=_Scene())
            llm.scribe = _scribe_json()
            llm.director = _director_json(['mira', 'zorg'])
            mgr.record({'turn_num': 2})
            self.assertIn('mira', mgr.fable['npc_directives'])
            self.assertNotIn('zorg', mgr.fable['npc_directives'])

    def test_brief_hides_absent_npc_directives(self):
        with tempfile.TemporaryDirectory() as tmp:
            mgr, llm = _mgr(tmp, scene=_Scene())
            llm.scribe = _scribe_json(loops=json.dumps([
                {'summary': 'the sealed letter', 'status': 'open',
                 'planted': True, 'entity': ['mira']},
            ]))
            llm.director = _director_json()
            mgr.record({'turn_num': 3})
            here = mgr.brief(['mira'])
            self.assertIn('The road to the citadel', here)
            self.assertIn('the sealed letter', here)
            self.assertIn('mira — drive: greed', here)
            self.assertIn('The constable is one day behind', here)
            gone = mgr.brief(['jason'])
            self.assertIn('the sealed letter', gone)
            self.assertNotIn('drive: greed', gone)

    def test_malformed_llm_output_never_fatal(self):
        with tempfile.TemporaryDirectory() as tmp:
            mgr, llm = _mgr(tmp)
            llm.scribe = 'Here is your fable: not json at all.'
            llm.director = 'also not json {broken'
            archived = mgr.record({'turn_num': 7})
            self.assertEqual(archived, [])
            self.assertEqual(mgr.fable['spine'], '')
            self.assertEqual(mgr.fable['last_fabled_turn'], 7)

    def test_json_wrapped_in_prose_parses(self):
        with tempfile.TemporaryDirectory() as tmp:
            mgr, llm = _mgr(tmp)
            llm.scribe = (
                'Sure! Here is the JSON you asked for:\n'
                + _scribe_json()
                + '\nHope that helps.'
            )
            mgr.record({'turn_num': 2})
            self.assertIn('citadel', mgr.fable['spine'])

    def test_disabled_llm_records_nothing(self):
        with tempfile.TemporaryDirectory() as tmp:
            opts = SimpleNamespace(
                user_name='Jason', vector_dir=tmp, debug=False, color=0,
            )
            llm = _LLM()
            llm.model_name = 'None'
            mgr = PlotManager(_Console(), _Common(), opts, llm,
                              _Prompts(), None)
            self.assertEqual(mgr.record({'turn_num': 1}), [])

    def test_branch_files_are_separate(self):
        with tempfile.TemporaryDirectory() as tmp:
            mgr, llm = _mgr(tmp)
            llm.scribe = _scribe_json()
            mgr.record({'turn_num': 1})
            story_spine = mgr.fable['spine']
            mgr.set_branch('alt')
            self.assertEqual(mgr.fable['spine'], '')
            llm.scribe = _scribe_json(spine='A different book entirely.')
            mgr.record({'turn_num': 1})
            self.assertEqual(
                mgr.fable['spine'], 'A different book entirely.',
            )
            mgr.set_branch('story')
            self.assertEqual(mgr.fable['spine'], story_spine)

    def test_fork_full_clone_and_cut(self):
        with tempfile.TemporaryDirectory() as tmp:
            mgr, llm = _mgr(tmp, scene=_Scene())
            for turn in (1, 2, 3):
                llm.scribe = _scribe_json(
                    spine=f'Spine after turn {turn}.',
                    loops=json.dumps([
                        {'summary': f'hook {turn}', 'status': 'open',
                         'planted': True, 'entity': []},
                    ]),
                )
                mgr.record({'turn_num': turn})
            self.assertEqual(mgr.fable['last_fabled_turn'], 3)

            # Full clone: the whole book travels.
            mgr.fork_branch('story', 'book2')
            with open(os.path.join(tmp, 'ephemeral_fable_book2.json'),
                      encoding='utf-8') as handle:
                clone = json.load(handle)
            self.assertEqual(clone['spine'], 'Spine after turn 3.')

            # Cut fork: the fable as of page two — hook 3 never existed.
            mgr.fork_branch('story', 'book3', cut_turns=2)
            with open(os.path.join(tmp, 'ephemeral_fable_book3.json'),
                      encoding='utf-8') as handle:
                cut = json.load(handle)
            self.assertEqual(cut['last_fabled_turn'], 2)
            self.assertNotIn('Spine after turn 3', cut['spine'])
            summaries = [
                loop['summary'] for loop in cut['loops']
            ]
            self.assertIn('hook 2', summaries)
            self.assertNotIn('hook 3', summaries)


if __name__ == '__main__':
    unittest.main()
