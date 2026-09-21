"""Run with: python src/context_manager_test.py

Avoid `python -m src.context_manager_test` — src/__init__.py imports langchain.
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
    from .chat_utils import RAGTag, RegExp
    from .context_manager import ContextManager
    from .scene_manager import SceneManager
except ImportError:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from chat_utils import RAGTag, RegExp
    from context_manager import ContextManager
    from scene_manager import SceneManager


class _Console:
    """Discard console output so tests stay quiet."""

    def print(self, *args, **kwargs):  # pylint: disable=unused-argument
        del args, kwargs


class _LLM:
    """Entity-LLM stand-in: records prompts, counts sheet requests."""

    model_name = 'stub'

    def __init__(self, replies=None):
        self.calls = 0
        self.prompts: list[str] = []
        self.replies = list(replies or [])

    def invoke(self, prompt):
        self.calls += 1
        self.prompts.append(' '.join(
            str(getattr(message, 'content', message)) for message in prompt
        ))
        if self.replies:
            return SimpleNamespace(content=self.replies.pop(0))
        return SimpleNamespace(content='{"name": "sheet"}')


class _Prompts:
    """Slot lookups whose text names the sheet shape it stands for."""

    @staticmethod
    def slot(role, name):  # pylint: disable=unused-argument
        return (
            'ENTITY-SHEET for {{character_name}}'
            if name == 'entity_human'
            else 'CREATURE-SHEET for {{character_name}}'
        )


class _RAG:
    """store_data stand-in that records every write."""

    def __init__(self):
        self.stored = []

    def store_data(self, text, tags_metadata=None, collection='', ids=None, **kwargs):
        del kwargs
        self.stored.append({
            'text': text,
            'tags': list(tags_metadata or []),
            'collection': collection,
        })
        return []


def _common():
    """Just the regex table and the attributes save_response() reads."""
    return SimpleNamespace(
        regex=RegExp(),
        attributes=SimpleNamespace(collections={'ai': 'ai_documents'}),
        active_branch=lambda history: 'story',
    )


def _cm(tmpdir: str, name: str = 'Aeloria') -> tuple[ContextManager, _LLM]:
    """A hand-built ContextManager: __init__ drags in RAG + prompt files."""
    opts = SimpleNamespace(
        user_name=name,
        vector_dir=tmpdir,
        debug=False,
        color=0,
        assistant_mode=False,
    )
    cm = ContextManager.__new__(ContextManager)
    cm.console = _Console()
    cm.common = _common()
    cm.opts = opts
    cm.debug = False
    cm.scene = SceneManager(_Console(), None, opts)
    cm.scene.set_branch('story')
    cm.prompts = _Prompts()
    cm.entity_llm = _LLM()
    cm.rag = _RAG()
    cm.plot = None
    return cm, cm.entity_llm


def _tags(**kwargs) -> list[RAGTag]:
    return [RAGTag(k, v) for k, v in kwargs.items()]


def _roll_reversal() -> dict:
    """The documents dict save_response() hands to the minter."""
    return {
        'user_query': 'Bren waves from the gate.',
        'chat_history': 'Aeloria tends the garden.',
        'user_name': 'Aeloria',
    }


class MintNewCharactersTest(unittest.TestCase):
    """Sheets mint from the cast, gated on the file, never on the roster."""

    def test_burned_roster_still_mints_missing_sheet(self):
        """The regression: ground_scene() unions the cast into
        known_characters before minting runs — the sheet must be written
        anyway, or an NPC can never get one."""
        with tempfile.TemporaryDirectory() as tmp:
            cm, llm = _cm(tmp)
            cm.scene.ground_scene(_tags(
                entity=['bren'], creature=['mule'],
                player_location='garden', moving_confidence=0.0,
            ))
            # Precondition: the roster already knows Bren (no sheet yet).
            self.assertIn('bren', cm.scene.scene['known_characters'])
            self.assertFalse(
                os.path.exists(os.path.join(tmp, 'entities', 'bren.txt')))

            cm._mint_new_characters(_roll_reversal())

            # People get the entity sheet shape, beasts the creature shape,
            # and each sheet file holds the LLM's response as clean JSON.
            self.assertIn('ENTITY-SHEET for bren', llm.prompts[0])
            self.assertIn('CREATURE-SHEET for mule', llm.prompts[1])
            with open(os.path.join(tmp, 'entities', 'bren.txt'),
                      encoding='utf-8') as handle:
                self.assertEqual(json.load(handle), {'name': 'sheet'})
            self.assertEqual(llm.calls, 2)

    def test_existing_sheet_is_not_minted_twice(self):
        with tempfile.TemporaryDirectory() as tmp:
            cm, llm = _cm(tmp)
            cm.scene.ground_scene(_tags(
                entity=['bren'], player_location='garden',
            ))
            cm._mint_new_characters(_roll_reversal())
            self.assertEqual(llm.calls, 1)
            cm._mint_new_characters(_roll_reversal())
            self.assertEqual(llm.calls, 1)

    def test_player_never_gets_a_sheet(self):
        """The protagonist's sheet is user-supplied; the minter skips them."""
        with tempfile.TemporaryDirectory() as tmp:
            cm, llm = _cm(tmp, name='Aeloria')
            cm.scene.ground_scene(_tags(
                entity=['bren'], player_location='garden',
            ))
            cm._mint_new_characters(_roll_reversal())
            self.assertFalse(
                os.path.exists(os.path.join(tmp, 'entities', 'aeloria.txt')))
            self.assertEqual(llm.calls, 1)  # bren only

    def test_creatures_retry_until_the_sheet_lands(self):
        """A failed entity-LLM call is retried on a later turn, not burned."""
        with tempfile.TemporaryDirectory() as tmp:
            cm, llm = _cm(tmp)
            cm.scene.ground_scene(_tags(
                creature=['mule'], player_location='garden',
            ))
            llm.calls = -1  # force create_character's write to "fail" once

            original_invoke = llm.invoke

            def flaky_invoke(prompt):
                llm.calls += 1
                if llm.calls == 0:
                    raise RuntimeError('model briefly unavailable')
                return original_invoke(prompt)

            llm.invoke = flaky_invoke
            with self.assertRaises(RuntimeError):
                cm._mint_new_characters(_roll_reversal())
            self.assertFalse(
                os.path.exists(os.path.join(tmp, 'entities', 'mule.txt')))

            cm._mint_new_characters(_roll_reversal())
            self.assertTrue(
                os.path.exists(os.path.join(tmp, 'entities', 'mule.txt')))

    def test_prose_refusal_is_not_cached_as_a_sheet(self):
        """A prose refusal must not be cached; the name retries later."""
        refusal = ('(No output - CHAT_HISTORY is empty, providing no '
                   'reliable information to infer traits for "smith")')
        with tempfile.TemporaryDirectory() as tmp:
            cm, _ = _cm(tmp)
            cm.entity_llm = _LLM(replies=[refusal])
            cm.scene.ground_scene(_tags(
                entity=['smith'], player_location='garden',
            ))
            cm._mint_new_characters(_roll_reversal())
            self.assertFalse(
                os.path.exists(os.path.join(tmp, 'entities', 'smith.txt')))

            # A later turn with a real answer mints the sheet.
            cm.entity_llm = _LLM(replies=[
                '{"name": "smith", "gender": "male", "race": "human", '
                '"appearance": "burn-scarred forearms"}'])
            cm._mint_new_characters(_roll_reversal())
            self.assertTrue(
                os.path.exists(os.path.join(tmp, 'entities', 'smith.txt')))

    def test_unknown_sheet_name_takes_the_roster_name(self):
        """A sheet that dodges the name is stored under the roster name."""
        fenced = ('```json\n{"name": "unknown", "gender": "male", '
                  '"race": "human", "appearance": "lean"}\n```')
        with tempfile.TemporaryDirectory() as tmp:
            cm, _ = _cm(tmp)
            cm.entity_llm = _LLM(replies=[fenced])
            cm.scene.ground_scene(_tags(
                entity=['man'], player_location='garden',
            ))
            cm._mint_new_characters(_roll_reversal())
            with open(os.path.join(tmp, 'entities', 'man.txt'),
                      encoding='utf-8') as f:
                sheet = json.load(f)
            self.assertEqual(sheet['name'], 'man')


class SaveResponseTest(unittest.TestCase):
    """The reply path grounds the scene once, then mints from the cast."""

    def test_save_response_grounds_then_mints(self):
        with tempfile.TemporaryDirectory() as tmp:
            cm, llm = _cm(tmp)
            calls = {}

            def fake_pre_processor(query, docs, do_scene=True, direction='query'):
                del query, docs
                calls['do_scene'] = do_scene
                calls['direction'] = direction
                return ('', _tags(entity=['bren'], creature=['mule']), True)

            cm.pre_processor = fake_pre_processor
            documents = {
                'sd_ran': False,
                'llm_response': 'Bren waves from the gate.',
                'history': {'story': []},
                'chat_history': 'Aeloria tends the garden.',
                'user_name': 'Aeloria',
            }
            cm.save_response(documents)

            # The reply path tags without grounding; grounding happens here.
            self.assertEqual(calls, {'do_scene': False, 'direction': 'response'})
            # The cast was grounded into the roster...
            self.assertIn('bren', cm.scene.scene['known_characters'])
            # ...and the missing sheets were minted from the grounded cast.
            self.assertTrue(
                os.path.exists(os.path.join(tmp, 'entities', 'bren.txt')))
            self.assertTrue(
                os.path.exists(os.path.join(tmp, 'entities', 'mule.txt')))
            self.assertEqual(llm.calls, 2)
            # The reply was stored once, under the branch's AI collection,
            # with the grounded scene carried in the tag metadata.
            self.assertEqual(len(cm.rag.stored), 1)
            self.assertEqual(cm.rag.stored[0]['collection'], 'story_ai_documents')
            grounded = {
                tag.tag: str(tag.content) for tag in cm.rag.stored[0]['tags']
            }
            self.assertIn('bren', grounded.get('known_characters', ''))
            self.assertIn('bren', grounded.get('entity', ''))


if __name__ == '__main__':
    unittest.main()