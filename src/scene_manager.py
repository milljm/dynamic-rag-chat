""" Manage Scenes stateful properties """
import os
import json
import re
from typing import Optional, Any
from collections import namedtuple
from .ragtag_manager import RAGTag
from .chat_utils import CommonUtils, ChatOptions


PRONOUNS = ['i', 'you', 'him', 'her', 'she', 'they', 'user', 'them', 'me']

class SceneManager:
    """
    Maintain a healthy scene state by grounded entity and
    audience and other ephemeral turn by turn information
    """
    def __init__(self, console, common: CommonUtils, args: ChatOptions):
        self.console = console
        self.common = common
        self.opts = args
        self.scene = self.load_scene()
        self.debug = args.debug

    @staticmethod
    def _ragtag_to_dict(tags: list[RAGTag])->dict:
        return {t.tag: t.content for t in tags}

    @staticmethod
    def _dict_to_ragtag(tags: dict[str,str|list])->list[RAGTag]:
        return [RAGTag(tag=k, content=v) for k, v in tags.items()]

    def _no_scene(self)->dict:
        return {
            'entity'   : [self.opts.user_name.lower()],
            'audience' : [self.opts.user_name.lower()],
            'location' : '',
            'known_characters' : [f'{self.opts.user_name.lower()}: the protagonist'],
            'entities_about' : ''
        }

    def new_scene(self)->dict[str,Any]:
        """ return an empty scene suitable for a new location """
        _scene: dict = self._no_scene()
        _scene['known_characters'] = list(self.scene['known_characters'])
        return _scene

    def load_scene(self)->dict[str, Any]:
        """ Load scene from disk """
        os.makedirs(self.opts.vector_dir, exist_ok=True)
        scene_file = os.path.join(self.opts.vector_dir, 'ephemeral_scene.json')
        if os.path.exists(scene_file):
            try:
                with open(scene_file, 'r', encoding='utf-8') as f:
                    return json.loads(f)
            # pylint: disable=broad-exception-caught   # LLMs are unpredictable
            except Exception:
            # pylint: enable=broad-exception-caught
                pass
        return self._no_scene()

    def save_scene(self, scene: Optional[dict[str, Any]] = None):
        """ Save current scene state to disk """
        data = scene if scene is not None else self._no_scene
        with open(os.path.join(self.opts.vector_dir,
                               'ephemeral_scene.json'),
                               'w',
                               encoding='utf-8') as f:
            json.dump(data, f)

    def _grow_known_characters(self, entities_about:list)->None:
        """
        loop through known characters and discover if we need to
        add anything new incoming from entities_about.
        """
        names = self.common.regex.names # shorthand re compile: r"([A-Za-z'-]+)"
        already_known = set([names.match(item).group(1).lower()
                          for item in self.scene['known_characters']])

        incoming = set([names.match(item).group(1).lower() for item in entities_about])
        additions = already_known.difference(incoming) | incoming.difference(already_known)

        # no changes
        if not additions:
            return

        for entity in additions:
            for about_line in entities_about:
                _entity = names.match(about_line).group(1).lower()
                if entity in _entity:
                    if self.opts.debug:
                        self.console.print(f'ADDING CHARACTER: {entity}',
                                   style=f'color({self.opts.color})', highlight=True)
                    self.scene['known_characters'].append(about_line)

    def ground_scene(self, tags: list[RAGTag], query: str)->list[RAGTag]:
        """
        ### Ground Scene

        Consume incoming RAGTags and return a sanitized/updated grounded RAGTag
        group using previous establish turns. Examples:

        - If during this new turn we did not change locations, then any character
        listed in entities from the previous turn(s) will exist in this new turn regardless
        of the new turn's pre-processor's 'entity values (missing values are common when the
        player doesn't mention their name during their turn).
        - Strip out useless pronouns in entity, audience.
        - Clear entities from the ephemeral memory when location changes (overwrite
        ephemeral scene with new scene data).
        - Loads the ephemeral scene upon chat startup (user loaded the program. Akin to
        loading a save game.)

        *Key init args:*
            .. code-block:: python
                tags: list[RAGTag] # RAGTag object from get_tags(response)
                query: str         # The LLMs response
        *Returns:*
            .. code-block:: python
                returns list[RAGTag]
        """
        scene = dict(self.scene)
        tag_dict = self._ragtag_to_dict(tags)

        # hack to fix missing *required* fields... because you can't rely on LLMs to
        #  always generate the metadata fields you ask them to
        tag_dict['location'] = tag_dict.get('location', '')
        tag_dict['entity'] = tag_dict.get('entity', [self.opts.user_name.lower()])
        tag_dict['audience'] = tag_dict.get('audience', [self.opts.user_name.lower()])
        tag_dict['entities_about'] = tag_dict.get(
            'entities_about',
            [f'{self.opts.user_name.lower()}: the protagonist'])
        if self.opts.debug:
            self.console.print(f'EXISTING EPHEMERAL SCENE:\n{scene}',
                               f'\nINCOMING SCENE:\n{tag_dict}',
                                style=f'color({self.opts.color})', highlight=True)

        # Short handing: s = scene (ephemeral), t = turn (the current response from the LLM)
        scene_tuple = namedtuple('s_diff', ['s', 't'])
        scene_dict = {}
        for v in ['location', 'entity', 'audience', 'entities_about']:
            scene_dict[v] = scene_tuple(s=scene[v], t=tag_dict[v])

        # Build turn entity/audience sets .lower()
        t_e_set = ({s.lower() for s in set(scene_dict['entity'].t)}
                   if scene_dict['entity'].t
                   and scene_dict['entity'].t not in PRONOUNS else set())
        t_a_set = ({s.lower() for s in set(scene_dict['audience'].t)}
                   if scene_dict['audience'].t else set())
        t_ea_set = ({s.lower() for s in set(scene_dict['entities_about'].t)}
                   if scene_dict['entities_about'].t else set())

        # Build scene entity/audience sets .lower()
        s_e_set = {s.lower() for s in set(scene_dict['entity'].s)}
        s_a_set = {s.lower() for s in set(scene_dict['audience'].s)}

        # If the location changes we should clear entity and audience, and populate it again
        # with grounded information, and any entities included among the new turn.
        # TODO: WIP, this is very difficult to rely on exact LLM responses for 'where' we are.
        # We may need to use 'str: query' and regex for movement words (walked, moved, ran, etc)
        # and do an AND logic bool.
        if scene_dict['location'].t != scene_dict['location'].s:
            if self.opts.debug:
                self.console.print(f'LOCATION CHANGE: from:>{scene_dict["location"].s}<',
                                   f' to:>{scene_dict["location"].t}<',
                                   style=f'color({self.opts.color})', highlight=True)
            self.scene.update(self.new_scene())
            self.scene['entity'] = list(set(self.scene['entity']).union(t_e_set))
            self.scene['audience'] = list(set(self.scene['audience']).union(t_a_set))
            self.scene['location'] = scene_dict['location'].t

        # Location remains the same, continue to populate entity/audience with
        # exiting characters from ephemeral data, and add any ones discovered.
        else:
            entity_set = s_e_set.union(t_a_set, s_e_set)
            audience_set = s_a_set.union(t_a_set)
            self.scene['audience'] = list(audience_set)
            self.scene['entity'] = list(entity_set)

        # Update entities_about (always grows), this populates known_characters in templates
        # NOTE: tricky conversion of entities_about --> known_characters happens here. The
        #       reason for this is due to the easier to identify naming convention the LLM sees
        #       as tightly coupled to 'entities' which LLMs are good at discovering. e.g., if
        #       you ask an LLM about 'known_characters' it will struggle vs entities_about.
        self._grow_known_characters(t_ea_set)

        # Update incoming scene data with grounded scene
        tag_dict.update(self.scene)
        self.save_scene(self.scene)
        if self.opts.debug:
            self.console.print(f'FINAL SCENE:\n{tag_dict}',
                                style=f'color({self.opts.color})', highlight=True)
        return self._dict_to_ragtag(tag_dict)
