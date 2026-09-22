"""Manage story-mode scene state: who is here, where, across turns and restarts."""
import os
import re
import json
import shutil
from copy import deepcopy
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
# Location strings that mean "the tagger had no idea". Persisting them
# puts literal junk ("bram: ?") into SCENE_STATE, which the prompt calls
# authoritative.
_PLACEHOLDER_LOCATIONS = _EMPTY | {'?', '??', '???', 'na', 'tbd', '-', '—'}
_MOVE_THRESHOLD = 0.7
_MAX_LEDGER = 400        # per-turn scene pages kept for rollback/fork


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

        - NPC sheets are minted from the grounded cast by
          ContextManager.create_character(); the sheet file on disk is
          the already-seen gate, not the known-character roster.
    """

    def __init__(self, console, common: CommonUtils, args: ChatOptions):
        self.console = console
        self.common = common
        self.opts = args
        self.branch = 'story'
        self.debug = args.debug
        self.turns: dict[str, dict] = {}
        self.last_grounded_turn = 0
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
            'creature': [],
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

    def scene_names(self, key: str) -> list[str]:
        """Cleaned name list for one scene key (entity, creature, …)."""
        return self._as_list(self.scene.get(key))

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
        """Single location string, lowercase; placeholders become ''."""
        if isinstance(value, (list, tuple)):
            value = value[0] if value else ''
        text = str(value or '').strip().lower()
        if text in _PLACEHOLDER_LOCATIONS:
            return ''
        return text

    def _npc_map(self, value) -> dict[str, str]:
        """Parse `name: place` tokens into a dict."""
        mapping = {}
        for token in self._as_list(value):
            if ':' in token:
                name, _, place = token.partition(':')
                name = name.strip().lower()
                place = self._location(place)
                if name and name not in _PRONOUNS and place:
                    # A placeholder or empty place records nothing at all:
                    # it must not overwrite a real location in the merge,
                    # and it must not persist junk ("bram: ?") as state.
                    mapping[name] = place
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

    def _scene_file(self, branch: Optional[str] = None) -> str:
        """Per-branch scene path."""
        safe = re.sub(r'[^a-zA-Z0-9_-]+', '_', branch or self.branch or 'story')
        safe = safe.strip('_') or 'story'
        return os.path.join(self.opts.vector_dir, f'ephemeral_scene_{safe}.json')

    @staticmethod
    def _as_turn(value) -> int:
        """Best-effort turn number, 0 when absent/unparsable."""
        try:
            return max(0, int(value))
        except (TypeError, ValueError):
            return 0

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
                    data = json.load(handle)
                self.turns = {
                    str(self._as_turn(k)): deepcopy(v)
                    for k, v in (data.get('turns') or {}).items()
                    if self._as_turn(k) and isinstance(v, dict)
                }
                self.last_grounded_turn = self._as_turn(
                    data.get('last_grounded_turn'),
                )
                return self._normalize_scene(data)
            except (OSError, TypeError, ValueError, json.JSONDecodeError):
                continue
        self.turns = {}
        self.last_grounded_turn = 0
        return self._no_scene()

    def save_scene(self, scene: Optional[dict[str, Any]] = None):
        """Save current scene state plus the turn ledger to disk."""
        data = scene if scene is not None else self.get_scene()
        self.scene = data
        os.makedirs(self.opts.vector_dir, exist_ok=True)
        with open(self._scene_file(), 'w', encoding='utf-8') as handle:
            json.dump({
                **data,
                'turns': self.turns,
                'last_grounded_turn': self.last_grounded_turn,
            }, handle)

# ---------- turn ledger / rollback / fork (mirrors PlotManager) ----------

    def _ledger(self) -> dict[int, dict]:
        """Post-turn scene states keyed by turn number."""
        return {
            self._as_turn(k): v
            for k, v in (self.turns or {}).items() if self._as_turn(k)
        }

    def _prune_ledger(self) -> None:
        """Keep the newest _MAX_LEDGER pages; oldest pages fall off."""
        if len(self.turns) <= _MAX_LEDGER:
            return
        keep = sorted(self.turns, key=self._as_turn)[-_MAX_LEDGER:]
        self.turns = {key: self.turns[key] for key in keep}

    def _write_page(self, turn: int, scene: dict) -> None:
        """Record the post-turn scene so any later turn can be undone."""
        self.turns[str(turn)] = deepcopy(scene)
        self._prune_ledger()

    def rollback_to(self, turn: int) -> None:
        """Truncate the scene to its state as of ``turn``.

        The live scene becomes the post-``turn`` ledger page — the
        nearest recorded page at or before ``turn``; with nothing there
        (a brand-new branch, or pages pruned past the rewind point) the
        empty scene — and every later page is dropped, so rewound turns
        never happened.
        """
        turn = max(0, self._as_turn(turn))
        pages = self._ledger()
        eligible = [k for k in pages if k <= turn]
        if eligible:
            self.scene = deepcopy(pages[max(eligible)])
            self.last_grounded_turn = max(eligible)
        else:
            self.scene = self._no_scene()
            self.last_grounded_turn = 0
        self.turns = {
            str(k): deepcopy(v) for k, v in pages.items() if k <= turn
        }
        self.save_scene()

    def reset_branch(self) -> None:
        """Empty the current branch's scene (its history was reset)."""
        self.scene = self._no_scene()
        self.turns = {}
        self.last_grounded_turn = 0
        self.save_scene()

    def delete_branch(self, branch: str) -> None:
        """Drop a deleted branch's scene file. Best effort, never fatal."""
        if not branch or branch == self.branch:
            return
        try:
            os.remove(self._scene_file(branch))
        except OSError:
            pass

    def fork_branch(self, src: str, name: str,
                    cut_turns: Optional[int] = None) -> None:
        """Copy the parent scene into a forked branch.

        Full clone copies the file verbatim — location, cast and turn
        ledger travel with the fork. A cut fork keeps the scene exactly
        as of that page: the post-``cut`` ledger entry (nearest page at
        or before the cut, else the empty scene) with later pages
        pruned. Best effort, never fatal.
        """
        if not name or name == src:
            return
        self.save_scene()
        src_path = self._scene_file(src)
        dst_path = self._scene_file(name)
        if cut_turns is None:
            if os.path.exists(src_path):
                shutil.copyfile(src_path, dst_path)
            return
        if not os.path.exists(src_path):
            return
        try:
            with open(src_path, 'r', encoding='utf-8') as handle:
                data = json.load(handle)
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            return
        cut = max(0, int(cut_turns))
        pages = {
            self._as_turn(k): v
            for k, v in (data.get('turns') or {}).items()
            if self._as_turn(k) and isinstance(v, dict)
        }
        eligible = [k for k in pages if k <= cut]
        if eligible:
            scene = self._normalize_scene(pages[max(eligible)])
            last = max(eligible)
        else:
            scene = self._no_scene()
            last = 0
        ledger = {str(k): deepcopy(v) for k, v in pages.items() if k <= cut}
        os.makedirs(self.opts.vector_dir, exist_ok=True)
        with open(dst_path, 'w', encoding='utf-8') as handle:
            json.dump(
                {**scene, 'turns': ledger, 'last_grounded_turn': last}, handle,
            )

    def _merge_stay(self, prev: dict, incoming: dict) -> dict:
        """Same room: people persist even if the tagger omitted them."""
        scene = {
            # Staying means staying: player_location may be filled in when
            # unknown, but never changed. A real move requires
            # moving_confidence > 0.7 and routes through _merge_move.
            # Without this, a tagger that misreads a room merely mentioned
            # in the prose (a cabin door opening, an NPC's house) silently
            # relocates the PC in SCENE_STATE, which the prompt calls
            # authoritative for the next turn.
            'player_location': (
                self._location(prev.get('player_location'))
                or self._location(incoming.get('player_location'))
            ),
            'entity': self._with_player(self._union(prev.get('entity'), incoming.get('entity'))),
            'creature': self._union(prev.get('creature'), incoming.get('creature')),
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
        scene['creature'] = self._names(incoming.get('creature'))
        scene['audience'] = self._names(incoming.get('audience'))
        scene['npc_locations'] = self._npc_list(self._npc_map(incoming.get('npc_locations')))
        return scene

    def ground_scene(self, tags: list[RAGTag], turn_num: int | None = None,
                     regen: bool = False) -> list[RAGTag]:
        """Sanitize tags against the previous turn and persist the result.

        - People already in the room stay in the room unless location changed.
        - Pronouns never become entity names.
        - The PC is always present.
        - A location change requires both a new player_location string and
          moving_confidence > 0.7. Overconfident taggers no longer wipe the
          cast because the player looked out a window.

        ``turn_num`` keys the per-turn ledger page (from
        ``documents['turn_num']`` on the reply pass); unset it derives
        from ``last_grounded_turn`` — the next turn for a fresh query,
        the same turn when ``regen`` (a regenerate re-grounds the page
        that is already there, after rewinding the live scene to the
        previous page so the discarded reply's cast never bleeds into
        the rewrite).
        """
        turn = self._as_turn(turn_num)
        if not turn:
            turn = (
                self.last_grounded_turn if regen
                else self.last_grounded_turn + 1
            )
        prev = dict(self.scene)
        if regen and turn > 1:
            pages = self._ledger()
            eligible = [k for k in pages if k <= turn - 1]
            if eligible:
                prev = deepcopy(pages[max(eligible)])
        incoming = self._ragtag_to_scene_dict(tags)
        if self._is_relocating(incoming, prev, tags):
            scene = self._merge_move(prev, incoming)
        else:
            scene = self._merge_stay(prev, incoming)
        scene['known_characters'] = self._union(scene.get('known_characters'), scene.get('entity'))
        self._write_page(turn, scene)
        self.last_grounded_turn = max(self.last_grounded_turn, turn)
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
