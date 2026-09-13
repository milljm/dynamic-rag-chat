"""Run with: python src/prompt_manager_test.py

Avoid `python -m src.prompt_manager_test` — src/__init__.py imports langchain.
"""
from __future__ import annotations

import os
import sys
import tempfile
import unittest
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from langchain_core.prompts import ChatPromptTemplate, PromptTemplate  # noqa: E402

from prompt_manager import PromptManager, REPLACEABLE  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class _Console:
    """Discard Rich console output so tests stay quiet."""

    def print(self, *args, **kwargs):  # pylint: disable=unused-argument
        """Swallow any Rich console call."""
        return None


def _args(assistant=False, vector_dir=''):
    return SimpleNamespace(
        assistant_mode=assistant, debug=False, vector_dir=vector_dir,
    )


def _empty_documents(**overrides):
    docs = {
        'gold_resume': '',
        'search_resume': '',
        'has_documents_index': False,
        'dynamic_files': '',
        'agent_calls': 0,
        'search_fetches': 0,
    }
    docs.update(overrides)
    return docs


class SlotTreeTests(unittest.TestCase):
    """Slot resolution, the disk manifest, overlays, and composition."""

    def test_assistant_plot_is_the_assistant_tree(self):
        pm = PromptManager(_Console(), ROOT, _args(True))
        path = pm.plot_file('assistant', 'system')
        self.assertTrue(path.endswith(os.path.join(
            'assistant', 'heavy', 'canon_system.md')))
        self.assertTrue(os.path.isfile(path))

    def test_story_plot_is_the_story_tree(self):
        pm = PromptManager(_Console(), ROOT, _args(False))
        path = pm.plot_file('story', 'human')
        self.assertTrue(path.endswith(os.path.join(
            'story', 'heavy', 'canon_human.md')))
        self.assertTrue(os.path.isfile(path))

    def test_rejects_bad_slots(self):
        pm = PromptManager(_Console(), ROOT, _args(True))
        with self.assertRaises(ValueError):
            pm.plot_file('other', 'system')
        with self.assertRaises(ValueError):
            pm.plot_file('story', 'footer')

    def test_live_flavor_follows_assistant_mode(self):
        assistant = PromptManager(_Console(), ROOT, _args(True))
        story = PromptManager(_Console(), ROOT, _args(False))
        self.assertEqual(assistant.flavor(), 'assistant')
        self.assertEqual(story.flavor(), 'story')

    def test_slot_reads_and_missing_optional_slot_is_empty(self):
        pm = PromptManager(_Console(), ROOT, _args(True))
        self.assertIn('<NEED_SEARCH>', pm.slot('heavy', 'need_search'))
        self.assertEqual(pm.optional_slot('heavy', 'no_such_fragment'), '')

    def test_manifest_lists_disk_slots(self):
        pm = PromptManager(_Console(), ROOT, _args(False))
        self.assertIn('canon_system', pm.slots('heavy'))
        self.assertIn('tagging_human', pm.slots('pre_processor'))

    def test_assistant_manifest_has_its_own_slots(self):
        pm = PromptManager(_Console(), ROOT, _args(True))
        self.assertIn('need_gold', pm.slots('heavy'))
        self.assertNotIn('polish_system', pm.slots('heavy'))

    def test_reload_rescans_tree(self):
        pm = PromptManager(_Console(), ROOT, _args(True))
        first = pm.slot('heavy', 'canon_system')
        pm.reload()
        self.assertEqual(first, pm.slot('heavy', 'canon_system'))

    def test_overlay_mirrors_tree_and_spares_stock(self):
        with tempfile.TemporaryDirectory() as tmp:
            pm = PromptManager(_Console(), ROOT, _args(True, tmp))
            stock = pm.plot_file('assistant', 'system')
            with open(stock, encoding='utf-8') as handle:
                before = handle.read()
            pm.write_plot('assistant', 'system', 'CUSTOM SYSTEM\n')
            overlay = pm.overlay_path(stock)
            self.assertTrue(overlay.endswith(os.path.join(
                'prompt_overrides', 'assistant', 'heavy', 'canon_system.md')))
            self.assertTrue(os.path.isfile(overlay))
            self.assertEqual(pm.slot('heavy', 'canon_system'), 'CUSTOM SYSTEM\n')
            with open(stock, encoding='utf-8') as handle:
                self.assertEqual(handle.read(), before)
            restored = pm.restore_plot('assistant', 'system')
            self.assertFalse(restored['overlaid'])
            self.assertEqual(pm.slot('heavy', 'canon_system'), before)

    def test_write_plot_rejects_human(self):
        with tempfile.TemporaryDirectory() as tmp:
            pm = PromptManager(_Console(), ROOT, _args(False, tmp))
            with self.assertRaises(ValueError):
                pm.write_plot('story', 'human', 'nope')

    def test_compose_includes_need_search_cookbook(self):
        pm = PromptManager(_Console(), ROOT, _args(True))
        system, _ = pm.compose_assistant_plot(_empty_documents())
        self.assertIn('<NEED_SEARCH>', system)
        self.assertIn('<NEED_SEARCH:NVDA share price>', system)

    def test_compose_omits_need_search_cookbook_on_resume(self):
        pm = PromptManager(_Console(), ROOT, _args(True))
        system, human = pm.compose_assistant_plot(_empty_documents(
            search_resume='Lead in.',
            dynamic_files='=== WEB_SEARCH ===\nhits',
            search_fetches=1,
        ))
        self.assertNotIn('You may emit one live-lookup tag', system)
        self.assertIn('<SEARCH_RESUME_EVENT>', system)
        self.assertIn('LIVE_LOOKUP', system)
        self.assertTrue(human)

    def test_compose_omits_need_search_when_agent_already_capped(self):
        pm = PromptManager(_Console(), ROOT, _args(True))
        system, _ = pm.compose_assistant_plot(_empty_documents(agent_calls=2))
        self.assertNotIn('You may emit one live-lookup tag', system)

def _story_documents(**overrides):
    docs = {
        'ooc_mode_bool': 'FALSE',
        'additional_content': '',
        'prompt_stack': [],
    }
    docs.update(overrides)
    return docs


class StoryStackTests(unittest.TestCase):
    """Story control segregation, OOC routing, and mood selection."""

    def test_ooc_turn_drops_the_story_stack(self):
        pm = PromptManager(_Console(), ROOT, _args(False))
        system, human = pm.compose_story_plot(_story_documents(ooc_mode_bool='TRUE'))
        self.assertIn('<OOC_MODE>', system)
        for block in ('<ROOT_PRIMER>', '<PLAYER_AGENCY>', '<CAMERA>', '<ANTI_ECHO>',
                      '<WRITING_STYLE>', '<NPC_BEHAVIOR>', '<WORLD_RESPONSE>',
                      '<PLOT_ADVANCEMENT>', '<WORLD_INITIATION>',
                      '<RESPONSE_CHECKLIST>'):
            self.assertNotIn(block, system, block)
        self.assertTrue(human)

    def test_story_turn_gets_core_controls_in_order(self):
        pm = PromptManager(_Console(), ROOT, _args(False))
        system, _ = pm.compose_story_plot(_story_documents())
        order = ['<ROOT_PRIMER>', '<PLAYER_AGENCY>', '<CAMERA>', '<ANTI_ECHO>',
                 '<WRITING_STYLE>', '<NPC_BEHAVIOR>', '<WORLD_RESPONSE>',
                 '<PLOT_ADVANCEMENT>', '<WORLD_INITIATION>', '<RESPONSE_CHECKLIST>']
        positions = [system.index(block) for block in order]
        self.assertEqual(positions, sorted(positions))
        self.assertNotIn('<OOC_MODE>', system)

    def test_ooc_turn_is_much_shorter_than_story_turn(self):
        pm = PromptManager(_Console(), ROOT, _args(False))
        story, _ = pm.compose_story_plot(_story_documents())
        ooc, _ = pm.compose_story_plot(_story_documents(ooc_mode_bool='TRUE'))
        self.assertLess(len(ooc), len(story) // 3)

    def test_mood_control_splices_after_npc_behavior(self):
        pm = PromptManager(_Console(), ROOT, _args(False))
        system, _ = pm.compose_story_plot(
            _story_documents(prompt_stack=['mood_combat']))
        self.assertIn('<MOOD_COMBAT>', system)
        self.assertLess(system.index('<NPC_BEHAVIOR>'),
                        system.index('<MOOD_COMBAT>'))
        self.assertLess(system.index('<MOOD_COMBAT>'),
                        system.index('<WORLD_RESPONSE>'))

    def test_unknown_control_is_ignored(self):
        pm = PromptManager(_Console(), ROOT, _args(False))
        system, _ = pm.compose_story_plot(_story_documents(
            prompt_stack=['nope_missing', 'canon', 'ooc', 'mood_anger']))
        self.assertNotIn('nope_missing', system)
        self.assertNotIn('<OOC_MODE>', system)
        self.assertIn('<MOOD_ANGER>', system)

    def test_mood_menu_lists_described_controls_only(self):
        pm = PromptManager(_Console(), ROOT, _args(False))
        menu = dict(pm.mood_menu())
        self.assertIn('mood_combat', menu)
        self.assertTrue(menu['mood_combat'])
        # Core and one-shot files declare no header, so they are never offered.
        for absent in ('canon', 'agency', 'camera', 'ooc', 'addendum', 'polish'):
            self.assertNotIn(absent, menu)

    def test_mood_replaces_core_controls(self):
        pm = PromptManager(_Console(), ROOT, _args(False))
        docs = _story_documents(prompt_stack=['mood_tense'])
        stack = pm.story_stack(docs)
        self.assertIn('mood_tense', stack)
        self.assertNotIn('plot', stack)          # mood_tense replaces: plot, initiation
        self.assertNotIn('initiation', stack)
        system, _ = pm.compose_story_plot(docs)
        self.assertIn('<MOOD_TENSE>', system)
        self.assertNotIn('<PLOT_ADVANCEMENT>', system)
        self.assertNotIn('<WORLD_INITIATION>', system)
        self.assertIn('<NPC_BEHAVIOR>', system)   # not displaced

    def test_core_invariants_are_not_replaceable(self):
        pm = PromptManager(_Console(), ROOT, _args(False))
        for invariant in ('canon', 'agency', 'camera'):
            self.assertNotIn(invariant, REPLACEABLE)
        system, _ = pm.compose_story_plot(_story_documents(prompt_stack=['mood_grief']))
        self.assertIn('<PLAYER_AGENCY>', system)
        self.assertIn('<CAMERA>', system)

    def test_control_header_is_never_sent(self):
        pm = PromptManager(_Console(), ROOT, _args(False))
        system, _ = pm.compose_story_plot(
            _story_documents(prompt_stack=['mood_combat']))
        self.assertNotIn('<!--', system)
        self.assertNotIn('replaces:', system)

    def test_tagging_prompt_drops_the_stack_when_no_menu(self):
        """Reply / import tagging must not be asked to pick a stack.

        ContextManager only supplies a menu for the query direction; the
        section is gated on it, which keeps ~5KB of moods out of the
        tagging prompt for the AI reply.
        """
        pm = PromptManager(_Console(), ROOT, _args(False))
        template = pm.slot('pre_processor', 'tagging_human')
        docs = {'user_name': 'x', 'chat_history': '', 'user_query': 'hi'}
        for menu, armed in ((pm.mood_menu(), True), ([], False)):
            rendered = PromptTemplate(
                template=template, template_format='jinja2',
            ).format(**dict(docs, mood_menu=menu))
            self.assertEqual('prompt_stack' in rendered, armed)
            self.assertEqual('CONTROL_MENU' in rendered, armed)
            self.assertNotIn('{%', rendered)
            self.assertNotIn('{{', rendered)

    def test_families_cover_every_offered_mood(self):
        pm = PromptManager(_Console(), ROOT, _args(False))
        families = [name for name, _ in pm.mood_family_menu()]
        self.assertIn('danger', families)
        covered = set()
        for family in families:
            covered |= {n for n, _ in pm.mood_menu(families=[family])}
        self.assertEqual(covered, {n for n, _ in pm.mood_menu()})

    def test_mood_menu_narrows_by_family(self):
        pm = PromptManager(_Console(), ROOT, _args(False))
        narrowed = [n for n, _ in pm.mood_menu(families=['danger'])]
        self.assertIn('mood_combat', narrowed)
        self.assertNotIn('mood_downtime', narrowed)
        self.assertEqual(pm.mood_menu(families=['no_such_family']), [])

    def test_router_prompt_offers_only_families(self):
        """The router sees 6 families, not 47 moods."""
        pm = PromptManager(_Console(), ROOT, _args(False))
        template = pm.slot('pre_processor', 'mood_router_human')
        docs = {'user_name': 'x', 'chat_history': '', 'user_query': 'hi',
                'family_menu': pm.mood_family_menu()}
        rendered = PromptTemplate(
            template=template, template_format='jinja2').format(**docs)
        block = rendered.split('<FAMILY_MENU>')[1].split('</FAMILY_MENU>')[0]
        listed = [x for x in block.strip().splitlines() if x.startswith('- ')]
        self.assertEqual(len(listed), len(pm.mood_family_menu()))
        self.assertNotIn('mood_', rendered)
        self.assertNotIn('{%', rendered)
        self.assertNotIn('{{', rendered)

    def test_mood_router_formats_into_messages(self):
        """Guards the message path the router actually uses.

        ``PromptTemplate`` has no ``format_messages``; building it that way
        raised AttributeError inside a broad except, which silently disabled
        two-stage routing with no log to prove it.
        """
        pm = PromptManager(_Console(), ROOT, _args(False))
        messages = pm.compose_mood_router({
            'user_name': 'x', 'chat_history': '', 'user_query': 'hi',
            'family_menu': pm.mood_family_menu()})
        self.assertEqual(len(messages), 1)
        self.assertIn('danger', messages[0].content)
        self.assertNotIn('{%', messages[0].content)
        self.assertNotIn('{{', messages[0].content)

    def _tagging_text(self, pm, direction: str, menu) -> str:
        """Render the tagging prompt the way ContextManager does.

        compose_tagging_messages() sets its keys on ``documents`` in place,
        so the caller must format with that same dict (as pre_processor does).
        """
        docs = {'user_name': 'x', 'chat_history': '', 'user_query': 'hi'}
        messages = pm.compose_tagging_messages(docs, direction, menu)
        prompt = ChatPromptTemplate.from_messages(messages)
        rendered = prompt.format_messages(**docs)
        return '\n'.join(str(m.content) for m in rendered)

    def test_compose_tagging_messages_story_query(self):
        """The whole tagging construction, not just the template file."""
        pm = PromptManager(_Console(), ROOT, _args(False))
        text = self._tagging_text(pm, 'query', pm.mood_menu())
        self.assertIn('CONTROL_MENU', text)
        self.assertIn('prompt_stack', text)

    def test_compose_tagging_messages_reply_has_no_stack(self):
        pm = PromptManager(_Console(), ROOT, _args(False))
        text = self._tagging_text(pm, 'response', [])
        self.assertNotIn('prompt_stack', text)
        self.assertIn('your own previous narration', text)

    def test_entity_takes_anything_with_a_name(self):
        """A name is what earns RAG scoping, whatever the thing is."""
        pm = PromptManager(_Console(), ROOT, _args(False))
        text = self._tagging_text(pm, 'query', pm.mood_menu())
        self.assertIn('"creature": [string]', text)
        self.assertIn('## 9) creature', text)
        # A named dog, a named cat, or a named swarm all become entities,
        # because `entity` is the field retrieval filters on.
        self.assertIn('has been given a name', text)
        self.assertIn('Spot', text)
        self.assertIn('Wailing Death', text)
        # ...and an unnamed beast never reaches it.
        self.assertIn('Never an unnamed beast', text)
        self.assertIn('never drives retrieval', text)

    def test_creature_sheet_is_not_humanoid(self):
        """A rat snake must never be asked for a gender or a race."""
        pm = PromptManager(_Console(), ROOT, _args(False))
        prompt = pm.slot('pre_processor', 'creature_human')
        self.assertIn('"demeanour"', prompt)
        self.assertIn('"danger"', prompt)
        self.assertNotIn('"gender"', prompt)
        self.assertNotIn('"race"', prompt)

    def test_creature_sheet_renders(self):
        pm = PromptManager(_Console(), ROOT, _args(False))
        template = pm.slot('pre_processor', 'creature_human')
        rendered = PromptTemplate(
            template=template, template_format='jinja2').format(
                character_name='rat snake', chat_history='A snake crossed my boot.',
                user_name='Aeloria')
        self.assertIn('rat snake', rendered)
        self.assertNotIn('{{', rendered)
        self.assertNotIn('{%', rendered)

    def test_tagging_rules_guard_logged_failures(self):
        """Every assertion here is a real failure seen in a logged turn."""
        pm = PromptManager(_Console(), ROOT, _args(False))
        text = self._tagging_text(pm, 'query', pm.mood_menu())
        # A beast already listed under `present` still belongs in creature,
        # or it stays in entity forever and gets tracked twice.
        self.assertIn('even when PREVIOUS_TURN or SCENE_STATE lists it', text)
        # A voice heard from across the path is not audience, or it becomes a
        # speaker who is not present (and later a character sheet).
        self.assertIn('Only names that are also in entity', text)
        # The location must be a place, not a garment the snake climbed over.
        self.assertIn('never a body part or a garment', text)

    def test_addendum_only_when_content_exists(self):
        pm = PromptManager(_Console(), ROOT, _args(False))
        without, _ = pm.compose_story_plot(_story_documents(explicit=True))
        with_content, _ = pm.compose_story_plot(_story_documents(
            additional_content='explicit rules here', explicit=True))
        self.assertNotIn('<ADDITIONAL_CONTENT>', without)
        self.assertIn('<ADDITIONAL_CONTENT>', with_content)
        # The placeholder is rendered from documents later, in format_messages.
        self.assertIn('{{additional_content}}', with_content)

    def test_nsfw_addendum_never_rides_a_sfw_turn(self):
        """content_rating sfw excludes nsfw.md, however long that file is."""
        pm = PromptManager(_Console(), ROOT, _args(False))
        sfw = _story_documents(
            additional_content='the shipped NOTE TO USER placeholder',
            explicit=False, content_rating='sfw',
        )
        self.assertNotIn('addendum', pm.story_stack(sfw))
        self.assertNotIn('<ADDITIONAL_CONTENT>', pm.compose_story_plot(sfw)[0])

    def test_content_rating_nsfw_arms_the_addendum(self):
        pm = PromptManager(_Console(), ROOT, _args(False))
        armed = _story_documents(
            additional_content='rules', explicit=True, content_rating='nsfw')
        self.assertIn('<ADDITIONAL_CONTENT>', pm.compose_story_plot(armed)[0])
        # Either signal on its own is enough.
        by_rating = _story_documents(
            additional_content='rules', content_rating='nsfw')
        self.assertIn('<ADDITIONAL_CONTENT>', pm.compose_story_plot(by_rating)[0])


if __name__ == '__main__':
    unittest.main()
