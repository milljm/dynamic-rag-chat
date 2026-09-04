"""Manage story-mode scene state: who is here, where, across turns and restarts."""
import os
import re
import json
from typing import Any, Optional
try:
    from .chat_utils import CommonUtils, ChatOptions, RAGTag
except ImportError:
    from chat_utils import CommonUtils, ChatOptions, RAGTag

_PRONOUNS = {
    'i', 'me', 'my', 'mine', 'you', 'your', 'yours',
    'he', 'him', 'his', 'she', 'her', 'hers',
    'they', 'them', 'their', 'theirs', 'we', 'us', 'our',
    'it', 'its', 'someone', 'anyone', 'everybody', 'nobody',
    'pc', 'player',
}
_EMPTY = {'', 'none', 'null', 'unspecified', 'unknown', 'n/a'}
_MOVE_THRESHOLD = 0.7


class SceneManager:
    """
    ### SceneManager

    Story-mode scene state: who is here, where, across turns and
    restarts. Grounds ``entity`` / ``audience`` / location tags so RAG
    filters and the plot prompt agree. Assistant mode does not use this.

    *Class init args:*
        .. code-block:: python
            console: Console
            common: CommonUtils
            args: ChatOptions  # user_name, vector_dir, debug

    *Usage:*
        - per branch:
            .. code-block:: python
                scene.set_branch('story')
                scene.ground_scene(tags)
                scene.save_scene()

        - NPC sheets:
            .. code-block:: python
                if scene.is_new_character(name):
                    ...
    """

    def __init__(self, console, common: CommonUtils, args: ChatOptions):
        self.console = console
        self.common = common
        self.opts = args
        self.branch = 'story'
        self.debug = args.debug
        self.scene = self.load_scene()

    def _player(self) -> str:
        """Lowercased player name."""
        return (self.opts.user_name or 'user').strip().lower()

    def _no_scene(self) -> dict:
        """Empty scene for a brand-new location or first launch."""
        player = self._player()
        return {
            'entity': [player],
            'audience': [],
            'known_characters': [player],
            'player_location': '',
            'npc_locations': [],
        }

    @staticmethod
    def _as_list(value) -> list[str]:
        """Split a tag value into cleaned proper-name tokens."""
        if value is None:
            return []
        if isinstance(value, (list, tuple, set)):
            raw = list(value)
        else:
            raw = re.split(r'[,;|/]', str(value))
        out = []
        seen = set()
        for item in raw:
            name = str(item).strip().lower()
            if not name or name in _EMPTY or name in _PRONOUNS or name in seen:
                continue
            seen.add(name)
            out.append(name)
        return out

    def _names(self, value) -> list[str]:
        """Like _as_list but drop pronoun-only tokens (people, not locations)."""
        return [n for n in self._as_list(value) if n not in _PRONOUNS]

    def _union(self, *parts) -> list[str]:
        """Stable unique concat of name lists."""
        out = []
        seen = set()
        for part in parts:
            for name in self._names(part):
                if name in seen:
                    continue
                seen.add(name)
                out.append(name)
        return out

    def _with_player(self, names: list[str]) -> list[str]:
        """Guarantee the PC is first in the present-entity list."""
        player = self._player()
        rest = [n for n in names if n != player]
        return [player] + rest

    @staticmethod
    def _location(value) -> str:
        """Single location string, lowercase."""
        if isinstance(value, (list, tuple)):
            value = value[0] if value else ''
        return str(value or '').strip().lower()

    def _npc_map(self, value) -> dict[str, str]:
        """Parse `name: place` tokens into a dict."""
        mapping = {}
        for token in self._as_list(value):
            if ':' in token:
                name, _, place = token.partition(':')
                name = name.strip().lower()
                place = place.strip().lower()
                if name and name not in _PRONOUNS:
                    mapping[name] = place or mapping.get(name, '')
            elif token not in _PRONOUNS:
                mapping.setdefault(token, '')
        return mapping

    def _npc_list(self, mapping: dict[str, str]) -> list[str]:
        """Serialize npc location map back to tag values."""
        out = []
        for name, place in mapping.items():
            out.append(f'{name}: {place}' if place else name)
        return out

    def _confidence(self, tags: list[RAGTag]) -> float:
        """moving_confidence from tags, default stay-put."""
        for tag in tags:
            if tag.tag != 'moving_confidence':
                continue
            try:
                return float(tag.content)
            except (TypeError, ValueError):
                return 0.0
        return 0.0

    def _is_relocating(self, incoming: dict, prev: dict, tags: list[RAGTag]) -> bool:
        """True only when confidence is high AND the location string changed."""
        new_loc = self._location(incoming.get('player_location'))
        old_loc = self._location(prev.get('player_location'))
        if not new_loc or new_loc == old_loc:
            return False
        return self._confidence(tags) > _MOVE_THRESHOLD

    def _normalize_scene(self, data: dict) -> dict:
        """Coerce a loaded JSON blob into lists so `for char in entity` is safe."""
        base = self._no_scene()
        if not isinstance(data, dict):
            return base
        base['player_location'] = self._location(data.get('player_location'))
        base['entity'] = self._with_player(self._names(data.get('entity')))
        base['audience'] = self._names(data.get('audience'))
        base['known_characters'] = self._union(
            data.get('known_characters'), base['entity'],
        )
        base['npc_locations'] = self._npc_list(self._npc_map(data.get('npc_locations')))
        return base

    def _ragtag_to_scene_dict(self, tags: list[RAGTag]) -> dict:
        """Pull scene keys out of a tag list."""
        allowed = set(self._no_scene())
        incoming = {}
        for tag in tags:
            if tag.tag in allowed:
                incoming[tag.tag] = tag.content
        return incoming

    @staticmethod
    def _ragtag_to_dict(tags: list[RAGTag]) -> dict:
        """All tags as a plain dict."""
        return {t.tag: t.content for t in tags}

    def _dict_to_ragtag(self, tags: dict[str, str | list]) -> list[RAGTag]:
        """Dict → RAGTag list."""
        return [RAGTag(tag=k, content=v) for k, v in tags.items()]

    def _scene_file(self) -> str:
        """Per-branch scene path."""
        safe = re.sub(r'[^a-zA-Z0-9_-]+', '_', self.branch or 'story')
        safe = safe.strip('_') or 'story'
        return os.path.join(self.opts.vector_dir, f'ephemeral_scene_{safe}.json')

    def new_scene(self) -> dict[str, Any]:
        """Empty scene that keeps the roster of known characters."""
        scene = self._no_scene()
        scene['known_characters'] = list(self.scene.get('known_characters') or [self._player()])
        return scene

    def get_scene(self) -> dict:
        """Return current scene meta (live dict)."""
        return self.scene

    def set_branch(self, branch: str) -> None:
        """Switch scene files when the story branch changes."""
        name = (branch or 'story').strip() or 'story'
        if name == self.branch:
            return
        self.save_scene()
        self.branch = name
        self.scene = self.load_scene()

    def load_scene(self) -> dict[str, Any]:
        """Load scene from disk. Falls back to the legacy un-prefixed file."""
        os.makedirs(self.opts.vector_dir, exist_ok=True)
        legacy = os.path.join(self.opts.vector_dir, 'ephemeral_scene.json')
        for path in (self._scene_file(), legacy):
            if not os.path.exists(path):
                continue
            try:
                with open(path, 'r', encoding='utf-8') as handle:
                    return self._normalize_scene(json.load(handle))
            except (OSError, TypeError, ValueError, json.JSONDecodeError):
                continue
        return self._no_scene()

    def save_scene(self, scene: Optional[dict[str, Any]] = None):
        """Save current scene state to disk and keep memory in sync."""
        data = scene if scene is not None else self.get_scene()
        self.scene = data
        os.makedirs(self.opts.vector_dir, exist_ok=True)
        with open(self._scene_file(), 'w', encoding='utf-8') as handle:
            json.dump(data, handle)

    def is_new_character(self, character: str) -> bool:
        """Return True and record the name if this NPC has not been seen."""
        entry = (character or '').strip().lower()
        if not entry or entry in _PRONOUNS or entry in _EMPTY:
            return False
        known = [c.lower() for c in self.scene.get('known_characters', [])]
        if entry in known:
            return False
        self.scene.setdefault('known_characters', []).append(entry)
        self.save_scene()
        return True

    def _merge_stay(self, prev: dict, incoming: dict) -> dict:
        """Same room: people persist even if the tagger omitted them."""
        scene = {
            'player_location': (
                self._location(incoming.get('player_location'))
                or self._location(prev.get('player_location'))
            ),
            'entity': self._with_player(self._union(prev.get('entity'), incoming.get('entity'))),
            'audience': (
                self._names(incoming['audience'])
                if 'audience' in incoming
                else self._names(prev.get('audience'))
            ),
            'known_characters': self._union(
                prev.get('known_characters'), incoming.get('entity'), prev.get('entity'),
            ),
            'npc_locations': self._npc_list({
                **self._npc_map(prev.get('npc_locations')),
                **self._npc_map(incoming.get('npc_locations')),
            }),
        }
        return scene

    def _merge_move(self, prev: dict, incoming: dict) -> dict:
        """New room: drop who was here, keep the known-character roster."""
        scene = self.new_scene()
        scene['known_characters'] = self._union(
            prev.get('known_characters'), incoming.get('entity'),
        )
        scene['player_location'] = self._location(incoming.get('player_location'))
        scene['entity'] = self._with_player(self._names(incoming.get('entity')))
        scene['audience'] = self._names(incoming.get('audience'))
        scene['npc_locations'] = self._npc_list(self._npc_map(incoming.get('npc_locations')))
        return scene

    def ground_scene(self, tags: list[RAGTag]) -> list[RAGTag]:
        """Sanitize tags against the previous turn and persist the result.

        - People already in the room stay in the room unless location changed.
        - Pronouns never become entity names.
        - The PC is always present.
        - A location change requires both a new player_location string and
          moving_confidence > 0.7. Overconfident taggers no longer wipe the
          cast because the player looked out a window.
        """
        prev = dict(self.scene)
        incoming = self._ragtag_to_scene_dict(tags)
        if self._is_relocating(incoming, prev, tags):
            scene = self._merge_move(prev, incoming)
        else:
            scene = self._merge_stay(prev, incoming)
        scene['known_characters'] = self._union(scene.get('known_characters'), scene.get('entity'))
        self.save_scene(scene)
        meta = self._ragtag_to_dict(tags)
        meta.update(scene)
        if self.debug:
            self.console.print(
                f'SCENE GROUNDED: {scene}',
                style=f'color({self.opts.color})',
                highlight=False,
            )
        return self._dict_to_ragtag(meta)
