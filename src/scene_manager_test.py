"""Run with: python src/scene_manager_test.py

Avoid `python -m src.scene_manager_test` — src/__init__.py imports langchain.
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from types import SimpleNamespace

try:
    from .chat_utils import RAGTag
    from .scene_manager import SceneManager
except ImportError:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from chat_utils import RAGTag
    from scene_manager import SceneManager


class _Console:
    def print(self, *args, **kwargs):
        del args, kwargs


def _mgr(tmpdir: str, name: str = 'Jason') -> SceneManager:
    opts = SimpleNamespace(
        user_name=name,
        vector_dir=tmpdir,
        debug=False,
        color=0,
    )
    return SceneManager(_Console(), common=None, args=opts)


def _tags(**kwargs) -> list[RAGTag]:
    return [RAGTag(k, v) for k, v in kwargs.items()]


class SceneManagerTest(unittest.TestCase):
    """Scene persistence, carry-forward, and location-change rules."""

    def test_json_load_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            mgr = _mgr(tmp)
            mgr.scene['player_location'] = 'tavern'
            mgr.scene['entity'] = ['jason', 'mira']
            mgr.save_scene()
            path = os.path.join(tmp, 'ephemeral_scene_story.json')
            self.assertTrue(os.path.exists(path))
            again = _mgr(tmp)
            self.assertEqual(again.scene['player_location'], 'tavern')
            self.assertIn('mira', again.scene['entity'])

    def test_legacy_json_loads_file_not_loads_string(self):
        with tempfile.TemporaryDirectory() as tmp:
            legacy = os.path.join(tmp, 'ephemeral_scene.json')
            with open(legacy, 'w', encoding='utf-8') as handle:
                json.dump({'player_location': 'woods', 'entity': ['jason']}, handle)
            again = _mgr(tmp)
            self.assertEqual(again.scene['player_location'], 'woods')

    def test_entities_persist_when_tagger_omits_them(self):
        with tempfile.TemporaryDirectory() as tmp:
            mgr = _mgr(tmp)
            mgr.ground_scene(_tags(
                entity=['jason', 'mira'],
                player_location='tavern',
                moving_confidence=0.2,
            ))
            grounded = mgr.ground_scene(_tags(
                entity=['jason'],
                player_location='tavern',
                moving_confidence=0.2,
            ))
            entity = dict(grounded)['entity']
            self.assertIn('mira', entity)
            self.assertIn('jason', entity)

    def test_location_change_clears_old_cast(self):
        with tempfile.TemporaryDirectory() as tmp:
            mgr = _mgr(tmp)
            mgr.ground_scene(_tags(
                entity=['jason', 'mira'],
                player_location='tavern',
                moving_confidence=0.1,
            ))
            grounded = mgr.ground_scene(_tags(
                entity=['jason', 'cal'],
                player_location='stables',
                moving_confidence=0.95,
            ))
            entity = dict(grounded)['entity']
            self.assertNotIn('mira', entity)
            self.assertIn('cal', entity)
            self.assertIn('mira', mgr.scene['known_characters'])

    def test_high_confidence_same_room_does_not_reset(self):
        with tempfile.TemporaryDirectory() as tmp:
            mgr = _mgr(tmp)
            mgr.ground_scene(_tags(
                entity=['jason', 'mira'],
                player_location='tavern',
                moving_confidence=0.2,
            ))
            grounded = mgr.ground_scene(_tags(
                entity=['jason'],
                player_location='tavern',
                moving_confidence=0.95,
            ))
            self.assertIn('mira', dict(grounded)['entity'])

    def test_pronouns_stripped_and_string_entity_not_exploded(self):
        with tempfile.TemporaryDirectory() as tmp:
            mgr = _mgr(tmp)
            grounded = mgr.ground_scene(_tags(
                entity='she, mira',
                player_location='inn',
                moving_confidence=0.1,
            ))
            entity = dict(grounded)['entity']
            self.assertNotIn('she', entity)
            self.assertIn('mira', entity)
            self.assertIn('jason', entity)
            # iterating entity must not yield letters of a string
            self.assertTrue(all(len(n) > 1 or n == 'i' for n in entity))
            self.assertNotIn('m', entity)

    def test_save_updates_memory(self):
        with tempfile.TemporaryDirectory() as tmp:
            mgr = _mgr(tmp)
            mgr.ground_scene(_tags(
                entity=['jason'],
                player_location='dock',
                moving_confidence=0.9,
            ))
            self.assertEqual(mgr.get_scene()['player_location'], 'dock')

    def test_branch_files_are_separate(self):
        with tempfile.TemporaryDirectory() as tmp:
            mgr = _mgr(tmp)
            mgr.ground_scene(_tags(entity=['jason'], player_location='tavern'))
            mgr.set_branch('alt')
            self.assertEqual(mgr.scene['player_location'], '')
            mgr.ground_scene(_tags(entity=['jason'], player_location='cave'))
            mgr.set_branch('story')
            self.assertEqual(mgr.scene['player_location'], 'tavern')


if __name__ == '__main__':
    unittest.main()
