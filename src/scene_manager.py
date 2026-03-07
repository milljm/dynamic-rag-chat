""" Manage Scenes stateful properties """
import os
import json
from typing import Optional, Any
from .chat_utils import CommonUtils, ChatOptions, RAGTag

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
    def _dict_to_ragtag(tags: dict[str,str|list])->list[RAGTag]:
        return [RAGTag(tag=k, content=v) for k, v in tags.items()]

    def _ragtag_to_scene_dict(self, tags: list[RAGTag])->dict:
        allowed = set(self._no_scene())
        return {t.tag: t.content for t in tags if t.tag in allowed}

    @staticmethod
    def _ragtag_to_dict(tags: list[RAGTag])->dict:
        return {t.tag: t.content for t in tags}

    def _no_scene(self)->dict:
        return {
            'entity'           : [self.opts.user_name.lower()],
            'audience'         : [],
            'known_characters' : [self.opts.user_name.lower()],
            'player_location'  : '',
            'npc_locations'    : [],
        }

    def new_scene(self)->dict[str,Any]:
        """ return an empty scene suitable for a new location """
        _scene: dict = self._no_scene()
        _scene['known_characters'] = list(self.scene['known_characters'])
        return _scene

    def get_scene(self)->dict:
        """ return current scene meta """
        return self.scene

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
        data = scene if scene is not None else self.get_scene()
        with open(os.path.join(self.opts.vector_dir,
                               'ephemeral_scene.json'),
                               'w',
                               encoding='utf-8') as f:
            json.dump(data, f)

    def is_new_character(self, character: str) -> bool:
        """
        Return True and add to growing list of NPCs, if a new NPC is discovered
        """
        entry = (character or "").strip().lower()
        if self.debug:
            self.console.print(f'ENTITY: {entry} NOT IN KNOWN: '
                        f'{entry not in (c.lower() for c in self.scene["known_characters"])}',
                        style=f'color({self.opts.color})', highlight=False)

        if entry not in (c.lower() for c in self.scene['known_characters']):
            self.scene['known_characters'].append(entry)
            self.save_scene()
            return True

        if self.debug:
            self.console.print('NO NEW CHARS DISCOVERED',
                style=f'color({self.opts.color})', highlight=False)
        return False

    def ground_scene(self, tags: list[RAGTag])->list[RAGTag]:
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
                tags: list[RAGTag]
        *Returns:*
            .. code-block:: python
                returns list[RAGTag]
        """
        preceding_scene = self.get_scene()
        current_scene = self._ragtag_to_scene_dict(tags)
        preceding_scene.update(current_scene)

        meta_data = self._ragtag_to_dict(tags)
        needs_update = float(meta_data.get('moving_confidence', 0.5)) > 0.7
        if needs_update:
            preceding_scene = self.new_scene()
            preceding_scene.update(current_scene)
            meta_data.update(preceding_scene)
        self.save_scene(preceding_scene)
        return self._dict_to_ragtag(meta_data)
