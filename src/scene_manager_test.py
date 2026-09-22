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
    """Discard console output so tests stay quiet."""

    def print(self, *args, **kwargs):  # pylint: disable=unused-argument
        """Swallow any Rich console call."""
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

    def test_low_confidence_location_change_is_ignored(self):
        """Staying means staying: >0.7 is required to relocate.

        The reply tagger misreads a room merely mentioned in the prose
        ("the cabin door thumped open behind me") and reports it with low
        confidence. SCENE_STATE is authoritative for the next turn's
        prompt, so that must not move the PC.
        """
        with tempfile.TemporaryDirectory() as tmp:
            mgr = _mgr(tmp)
            mgr.ground_scene(_tags(
                entity=['jason'], player_location='garden', moving_confidence=0.0,
            ))
            self.assertEqual(mgr.scene['player_location'], 'garden')
            mgr.ground_scene(_tags(
                entity=['jason', 'elara'], player_location='cabin',
                moving_confidence=0.1,
            ))
            self.assertEqual(mgr.scene['player_location'], 'garden')
            # The arrival still lands; only the room holds.
            self.assertIn('elara', mgr.scene['entity'])

    def test_high_confidence_location_change_still_applies(self):
        with tempfile.TemporaryDirectory() as tmp:
            mgr = _mgr(tmp)
            mgr.ground_scene(_tags(
                entity=['jason'], player_location='tavern', moving_confidence=0.0,
            ))
            mgr.ground_scene(_tags(
                entity=['jason'], player_location='stables', moving_confidence=0.9,
            ))
            self.assertEqual(mgr.scene['player_location'], 'stables')

    def test_first_location_fills_when_none_known(self):
        with tempfile.TemporaryDirectory() as tmp:
            mgr = _mgr(tmp)
            mgr.ground_scene(_tags(
                entity=['jason'], player_location='garden', moving_confidence=0.0,
            ))
            self.assertEqual(mgr.scene['player_location'], 'garden')

    def test_creature_persists_while_the_room_holds(self):
        """Ambient wildlife carries forward, like the cast does."""
        with tempfile.TemporaryDirectory() as tmp:
            mgr = _mgr(tmp)
            mgr.ground_scene(_tags(
                entity=['jason'], creature=['rat snake'],
                player_location='garden', moving_confidence=0.0,
            ))
            self.assertEqual(mgr.scene['creature'], ['rat snake'])
            # A later turn that omits the creature keeps it in the scene.
            mgr.ground_scene(_tags(
                entity=['jason'], player_location='garden', moving_confidence=0.0,
            ))
            self.assertEqual(mgr.scene['creature'], ['rat snake'])

    def test_creature_never_joins_the_character_roster(self):
        with tempfile.TemporaryDirectory() as tmp:
            mgr = _mgr(tmp)
            mgr.ground_scene(_tags(
                entity=['jason'], creature=['rat snake', 'red fox'],
                player_location='garden', moving_confidence=0.0,
            ))
            self.assertEqual(mgr.scene['known_characters'], ['jason'])
            self.assertEqual(mgr.scene_names('creature'),
                             ['rat snake', 'red fox'])

    def test_creature_is_left_behind_on_a_move(self):
        with tempfile.TemporaryDirectory() as tmp:
            mgr = _mgr(tmp)
            mgr.ground_scene(_tags(
                entity=['jason'], creature=['rat snake'],
                player_location='garden', moving_confidence=0.0,
            ))
            mgr.ground_scene(_tags(
                entity=['jason'], player_location='cabin', moving_confidence=0.9,
            ))
            self.assertEqual(mgr.scene['creature'], [])

    def test_branch_files_are_separate(self):
        with tempfile.TemporaryDirectory() as tmp:
            mgr = _mgr(tmp)
            mgr.ground_scene(_tags(entity=['jason'], player_location='tavern'))
            mgr.set_branch('alt')
            self.assertEqual(mgr.scene['player_location'], '')
            mgr.ground_scene(_tags(entity=['jason'], player_location='cave'))
            mgr.set_branch('story')
            self.assertEqual(mgr.scene['player_location'], 'tavern')

    def test_placeholder_locations_are_never_stored(self):
        """A tagger's '?' never becomes authoritative state or a lie."""
        with tempfile.TemporaryDirectory() as tmp:
            mgr = _mgr(tmp)
            mgr.ground_scene(_tags(
                entity=['jason', 'mira'], player_location='tavern',
                npc_locations=['mira: tavern'], moving_confidence=0.0,
            ))
            # The reply tagger gives up: '?' for both the PC and an NPC.
            grounded = mgr.ground_scene(_tags(
                entity=['jason'], player_location='?',
                npc_locations=['mira: ?', 'bram: ?'],
                moving_confidence=0.0,
            ))
            scene = dict(grounded)
            self.assertEqual(scene['player_location'], 'tavern')
            self.assertNotIn('mira: ?', scene['npc_locations'])
            self.assertNotIn('bram: ?', scene['npc_locations'])
            self.assertIn('mira: tavern', scene['npc_locations'])

    def test_placeholder_player_location_never_moves_the_pc(self):
        """Even high confidence cannot relocate the PC to '?'.

        The reply tagger misreads rooms it merely mentions; SCENE_STATE is
        authoritative for the next turn, so junk strings must not move her.
        """
        with tempfile.TemporaryDirectory() as tmp:
            mgr = _mgr(tmp)
            mgr.ground_scene(_tags(
                entity=['jason'], player_location='garden',
                moving_confidence=0.0,
            ))
            grounded = mgr.ground_scene(_tags(
                entity=['jason'], player_location='?',
                moving_confidence=0.9,
            ))
            self.assertEqual(dict(grounded)['player_location'], 'garden')

    def test_placeholder_player_location_fills_from_next_real_tag(self):
        with tempfile.TemporaryDirectory() as tmp:
            mgr = _mgr(tmp)
            mgr.ground_scene(_tags(
                entity=['jason'], player_location='?',
                moving_confidence=0.0,
            ))
            self.assertEqual(mgr.scene['player_location'], '')
            mgr.ground_scene(_tags(
                entity=['jason'], player_location='cabin',
                moving_confidence=0.0,
            ))
            self.assertEqual(mgr.scene['player_location'], 'cabin')


class SceneLedgerTest(unittest.TestCase):
    """Per-turn scene pages: record, rollback, reset, delete, fork."""

    def test_ground_scene_writes_ledger_pages(self):
        with tempfile.TemporaryDirectory() as tmp:
            mgr = _mgr(tmp)
            mgr.ground_scene(_tags(
                entity=['jason'], player_location='tavern',
            ), turn_num=1)
            mgr.ground_scene(_tags(
                entity=['jason', 'mira'], player_location='tavern',
            ), turn_num=2)
            self.assertEqual(mgr.last_grounded_turn, 2)
            self.assertEqual(mgr.turns['2']['entity'], ['jason', 'mira'])
            self.assertEqual(mgr.turns['1']['entity'], ['jason'])
            # Regenerating turn 2 re-grounds that page from the previous
            # one, so the discarded reply's cast does not bleed through.
            mgr.ground_scene(_tags(
                entity=['jason'], player_location='tavern',
            ), regen=True)
            self.assertEqual(mgr.turns['2']['entity'], ['jason'])
            self.assertEqual(mgr.last_grounded_turn, 2)
            # The ledger round-trips through the JSON file.
            again = _mgr(tmp)
            self.assertEqual(again.last_grounded_turn, 2)
            self.assertEqual(again.turns['1']['entity'], ['jason'])
            # Ledger keys never leak into the grounded meta tags.
            self.assertNotIn('turns', dict(mgr.ground_scene(_tags(
                entity=['jason'], player_location='tavern',
            ))))

    def test_regen_regrounds_from_previous_page(self):
        with tempfile.TemporaryDirectory() as tmp:
            mgr = _mgr(tmp)
            mgr.ground_scene(_tags(
                entity=['jason'], player_location='inn',
            ), turn_num=1)
            mgr.ground_scene(_tags(
                entity=['jason', 'mira'], player_location='vault',
                moving_confidence=0.9,
            ), turn_num=2)
            # Regenerate turn 2: the re-sent query restores page 1 as
            # the merge base, then the rewritten reply grounds on top.
            mgr.ground_scene(_tags(
                entity=['jason'], player_location='inn', moving_confidence=0.0,
            ), regen=True)
            mgr.ground_scene(_tags(
                entity=['jason', 'orpheus'], player_location='inn',
                moving_confidence=0.0,
            ), turn_num=2)
            self.assertEqual(mgr.scene['player_location'], 'inn')
            self.assertEqual(mgr.scene['entity'], ['jason', 'orpheus'])
            self.assertNotIn('mira', mgr.scene['entity'])
            self.assertEqual(mgr.turns['2']['entity'], ['jason', 'orpheus'])

    def test_turn_number_defaults_to_next(self):
        with tempfile.TemporaryDirectory() as tmp:
            mgr = _mgr(tmp)
            mgr.ground_scene(_tags(entity=['jason'], player_location='inn'))
            self.assertEqual(mgr.last_grounded_turn, 1)
            mgr.ground_scene(_tags(entity=['jason'], player_location='inn'))
            self.assertEqual(mgr.last_grounded_turn, 2)

    def test_rollback_restores_page_and_truncates(self):
        with tempfile.TemporaryDirectory() as tmp:
            mgr = _mgr(tmp)
            mgr.ground_scene(_tags(
                entity=['jason'], player_location='inn',
            ), turn_num=1)
            mgr.ground_scene(_tags(
                entity=['jason', 'mira'], player_location='inn',
            ), turn_num=2)
            mgr.ground_scene(_tags(
                entity=['jason', 'mira'], player_location='vault',
                moving_confidence=0.9,
            ), turn_num=3)
            mgr.rollback_to(2)
            self.assertEqual(mgr.scene['player_location'], 'inn')
            self.assertIn('mira', mgr.scene['entity'])
            self.assertEqual(mgr.last_grounded_turn, 2)
            self.assertEqual(sorted(mgr.turns), ['1', '2'])
            # The file on disk agrees.
            with open(os.path.join(tmp, 'ephemeral_scene_story.json'),
                      encoding='utf-8') as handle:
                disk = json.load(handle)
            self.assertEqual(disk['last_grounded_turn'], 2)
            # Rolling back to zero is the empty scene.
            mgr.rollback_to(0)
            self.assertEqual(mgr.scene['player_location'], '')
            self.assertEqual(mgr.scene['entity'], ['jason'])
            self.assertEqual(mgr.last_grounded_turn, 0)
            self.assertEqual(mgr.turns, {})

    def test_rollback_past_ledger_starts_empty(self):
        with tempfile.TemporaryDirectory() as tmp:
            mgr = _mgr(tmp)
            mgr.ground_scene(_tags(
                entity=['jason'], player_location='vault',
            ), turn_num=9)
            mgr.rollback_to(3)
            self.assertEqual(mgr.scene['player_location'], '')
            self.assertEqual(mgr.scene['entity'], ['jason'])
            self.assertEqual(mgr.last_grounded_turn, 0)
            self.assertEqual(mgr.turns, {})
            # The next grounding keys from the rolled-back counter.
            mgr.ground_scene(_tags(entity=['mira'], player_location='dock'))
            self.assertEqual(mgr.last_grounded_turn, 1)

    def test_reset_and_delete_branch_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            mgr = _mgr(tmp)
            mgr.ground_scene(_tags(
                entity=['jason', 'mira'], player_location='inn',
            ), turn_num=1)
            mgr.set_branch('alt')
            self.assertEqual(mgr.scene['player_location'], '')
            mgr.ground_scene(_tags(
                entity=['jason'], player_location='cave',
            ), turn_num=1)
            alt_path = os.path.join(tmp, 'ephemeral_scene_alt.json')
            story_path = os.path.join(tmp, 'ephemeral_scene_story.json')
            self.assertTrue(os.path.exists(alt_path))
            # The branch you are on is never deleted out from under you.
            mgr.delete_branch('alt')
            self.assertTrue(os.path.exists(alt_path))
            mgr.set_branch('story')
            mgr.delete_branch('alt')
            self.assertFalse(os.path.exists(alt_path))
            # Reset empties the current scene and its ledger on disk.
            mgr.reset_branch()
            self.assertEqual(mgr.scene['player_location'], '')
            self.assertEqual(mgr.scene['entity'], ['jason'])
            self.assertEqual(mgr.last_grounded_turn, 0)
            self.assertEqual(mgr.turns, {})
            with open(story_path, encoding='utf-8') as handle:
                disk = json.load(handle)
            self.assertEqual(disk['last_grounded_turn'], 0)

    def test_fork_full_clone_and_cut(self):
        with tempfile.TemporaryDirectory() as tmp:
            mgr = _mgr(tmp)
            mgr.ground_scene(_tags(
                entity=['jason'], player_location='inn',
            ), turn_num=1)
            mgr.ground_scene(_tags(
                entity=['jason', 'mira'], player_location='vault',
                moving_confidence=0.9,
            ), turn_num=2)
            # Full clone: the whole scene travels, ledger and all.
            mgr.fork_branch('story', 'book2')
            with open(os.path.join(tmp, 'ephemeral_scene_book2.json'),
                      encoding='utf-8') as handle:
                clone = json.load(handle)
            self.assertEqual(clone['player_location'], 'vault')
            self.assertEqual(sorted(clone['turns']), ['1', '2'])
            self.assertEqual(clone['last_grounded_turn'], 2)
            # Cut fork: the scene as of page one — the vault never was.
            mgr.fork_branch('story', 'book3', cut_turns=1)
            with open(os.path.join(tmp, 'ephemeral_scene_book3.json'),
                      encoding='utf-8') as handle:
                cut = json.load(handle)
            self.assertEqual(cut['player_location'], 'inn')
            self.assertEqual(cut['last_grounded_turn'], 1)
            self.assertEqual(sorted(cut['turns']), ['1'])

    def test_ledger_prunes_to_cap(self):
        with tempfile.TemporaryDirectory() as tmp:
            mgr = _mgr(tmp)
            mgr.turns = {
                str(t): {'player_location': f'room{t}'}
                for t in range(1, 402)
            }
            mgr._prune_ledger()
            self.assertEqual(len(mgr.turns), 400)
            self.assertNotIn('1', mgr.turns)
            self.assertIn('401', mgr.turns)


if __name__ == '__main__':
    unittest.main()
