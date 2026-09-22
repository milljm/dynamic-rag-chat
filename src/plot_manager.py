"""Manage hidden fable state: the journey behind the story the player sees.

The fable is the answer to "what has this journey been about" — the plot
brain behind role-play storytelling that the player never sees. It is
maintained by lightweight LLM calls that run in the same daemon-thread
window that already tags the reply after the heavy LLM finishes streaming,
and persisted per branch (each branch is its own book on the shelf; each
owns its own fable).

Three layers, all Python-owned — the LLM only proposes deltas:
- spine: rolling compression of what the journey has been about.
- loops: open threads (promises, foreshadowing, threats in motion).
- npc_directives / dormant_arcs / director_notes: the hidden layer.

State file: ``{vector_dir}/ephemeral_fable_{branch}.json``.
"""
import json
import os
import re
import shutil
from copy import deepcopy
from typing import Any, Optional

from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.prompts import HumanMessagePromptTemplate
from langchain_openai import ChatOpenAI
from openai import APITimeoutError

try:
    from .chat_utils import CommonUtils, ChatOptions
    from .scene_manager import _EMPTY, _PRONOUNS, SceneManager
except ImportError:
    from chat_utils import CommonUtils, ChatOptions
    from scene_manager import _EMPTY, _PRONOUNS, SceneManager

_SCENE_KEYS = (
    'player_location', 'entity', 'creature', 'audience',
    'npc_locations', 'known_characters',
)
_MAX_SPINE = 1200        # hard cap on the rolling spine, characters
_MAX_LOOPS = 8           # open threads kept hot
_LOOP_TTL = 12           # turns a loop may go untouched before archiving
_MAX_DIRECTIVES = 6      # hidden NPC directives kept
_MAX_NOTES = 400         # director notes, characters
_MAX_LEDGER = 400        # per-turn fable pages kept for rollback/fork

# Scribe loop summaries are fuzzy-matched against stored threads so a
# reworded carry-forward ("goat climbs the fence" vs "thistle climbs the
# fence slat") updates the existing loop instead of minting a twin.
_LOOP_OVERLAP = 0.6      # Jaccard token overlap treated as the same thread
_LOOP_MIN_SHARED = 3     # ... with at least this many shared content words
_LOOP_STOPWORDS = frozenset(
    'a an the this that these those to of and or in on at for with from by '
    'is are was were be been being am as into back out up down over under '
    'again then so but her his their its it he she they them his hers ours '
    'my me i we you your'.split()
)


def _summary_tokens(summary: str) -> set[str]:
    """Content words of a loop summary, stopwords removed."""
    words = set(re.findall(r'[a-z0-9]+', str(summary).lower()))
    return words - _LOOP_STOPWORDS


def _is_same_thread(words: set[str], stored_summary: str) -> bool:
    """True when an incoming summary rewords a stored thread.

    Both a minimum shared-content-word floor and a Jaccard ratio: the
    floor keeps short generic summaries ("hook number 1" vs "hook number
    2") from collapsing into one thread on two shared words alone.
    """
    stored = _summary_tokens(stored_summary)
    shared = words & stored
    if len(shared) < _LOOP_MIN_SHARED:
        return False
    return len(shared) / len(words | stored) >= _LOOP_OVERLAP


class PlotManager:
    """
    ### PlotManager

    Hidden fable state for story mode: spine, open loops, hidden NPC
    truths and director notes, maintained invisibly by light LLM calls
    after the heavy reply streams. Mirrors ``SceneManager``: branch
    scoped JSON on disk, LLM proposes, Python decides.

    *Class init args:*
        .. code-block:: python
            console: Console
            common: CommonUtils
            args: ChatOptions  # user_name, vector_dir, debug
            llm: ChatOpenAI    # lightweight client (pre_llm)
            prompts: PromptManager
            scene: Optional[SceneManager]  # roster/scene grounding

    *Usage:*
        - after the heavy reply (daemon thread):
            .. code-block:: python
                archived = plot.record(documents)

        - before the next heavy prompt:
            .. code-block: python
                documents['fable_brief'] = plot.brief(scene_entities)

        - forking a branch (Rooted: each branch is its own book):
            .. code-block:: python
                plot.fork_branch(src, name, cut_turns)
    """

    # pylint: disable=too-many-positional-arguments, too-many-arguments
    def __init__(self, console, common: CommonUtils, args: ChatOptions,
                 llm: ChatOpenAI, prompts, scene: Optional[SceneManager] = None):
        self.console = console
        self.common = common
        self.opts = args
        self.llm = llm
        self.prompts = prompts
        self.scene = scene
        self.debug = args.debug
        self.branch = 'story'
        self.fable = self.load_fable()

    # ---------- naming / cleaning helpers ----------

    def _player(self) -> str:
        """Lowercased player name."""
        return (self.opts.user_name or 'user').strip().lower()

    @staticmethod
    def _clean_str(value, cap: int) -> str:
        """One flattened, whitespace-collapsed string, hard capped."""
        if value is None:
            return ''
        text = re.sub(r'\s+', ' ', str(value)).strip()
        return text[:cap]

    @staticmethod
    def _clean_names(value) -> list[str]:
        """Cleaned lowercase proper-name tokens, never pronouns."""
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

    @staticmethod
    def _key(summary: str) -> str:
        """Fuzzy-match key for loop summaries."""
        return re.sub(r'[^a-z0-9]+', ' ', str(summary).lower()).strip()

    @staticmethod
    def _as_turn(value) -> int:
        """Best-effort turn number, 0 when absent/unparsable."""
        try:
            return max(0, int(value))
        except (TypeError, ValueError):
            return 0

    @staticmethod
    def _parse_json(content) -> Optional[dict]:
        """Tolerant JSON object extraction from a light LLM response."""
        text = str(content or '').strip()
        start, end = text.find('{'), text.rfind('}')
        if start < 0 or end <= start:
            return None
        try:
            data = json.loads(text[start:end + 1])
        except (TypeError, ValueError, json.JSONDecodeError):
            return None
        return data if isinstance(data, dict) else None

    # ---------- persistence ----------

    def _fable_file(self, branch: Optional[str] = None) -> str:
        """Per-branch fable path."""
        safe = re.sub(r'[^a-zA-Z0-9_-]+', '_', branch or self.branch or 'story')
        safe = safe.strip('_') or 'story'
        return os.path.join(self.opts.vector_dir, f'ephemeral_fable_{safe}.json')

    @staticmethod
    def _empty() -> dict[str, Any]:
        """Fresh fable."""
        return {
            'spine': '',
            'loops': [],
            'npc_directives': {},
            'dormant_arcs': [],
            'director_notes': '',
            'last_fabled_turn': 0,
            'turns': {},
        }

    # pylint: disable-next=too-many-branches  # one guard per legacy shape
    def _normalize(self, data) -> dict[str, Any]:
        """Coerce a loaded JSON blob into the fable shape."""
        fable = self._empty()
        if not isinstance(data, dict):
            return fable
        fable['spine'] = self._clean_str(data.get('spine'), _MAX_SPINE)
        loops = data.get('loops')
        if isinstance(loops, list):
            for item in loops:
                if not isinstance(item, dict):
                    continue
                summary = self._clean_str(item.get('summary'), 240)
                if not summary:
                    continue
                fable['loops'].append({
                    'summary': summary,
                    'planted_turn': self._as_turn(item.get('planted_turn')),
                    'last_seen_turn': self._as_turn(item.get('last_seen_turn')),
                    'entity': self._clean_names(item.get('entity')),
                })
        directives = data.get('npc_directives')
        if isinstance(directives, dict):
            for name, spec in directives.items():
                key = str(name).strip().lower()
                if not key or key in _EMPTY or key in _PRONOUNS:
                    continue
                if isinstance(spec, dict):
                    fable['npc_directives'][key] = {
                        field: self._clean_str(spec.get(field), 200)
                        for field in ('drive', 'secret', 'plan', 'stance')
                        if self._clean_str(spec.get(field), 200)
                    }
        arcs = data.get('dormant_arcs')
        if isinstance(arcs, list):
            fable['dormant_arcs'] = [
                self._clean_str(arc, 200) for arc in arcs
                if self._clean_str(arc, 200)
            ][:2]
        fable['director_notes'] = self._clean_str(
            data.get('director_notes'), _MAX_NOTES,
        )
        fable['last_fabled_turn'] = self._as_turn(data.get('last_fabled_turn'))
        pages = data.get('turns')
        if isinstance(pages, dict):
            fable['turns'] = {
                str(self._as_turn(k)): self._normalize(v)
                for k, v in pages.items()
                if self._as_turn(k)
            }
        # Legacy files kept pre-turn `snapshots`; a snapshot of turn N is
        # the post-turn state of N-1 (and of turn 1 is the empty book, so
        # it has no page).
        snaps = data.get('snapshots')
        if isinstance(snaps, dict):
            for key, value in snaps.items():
                turn = self._as_turn(key)
                if turn > 1 and str(turn - 1) not in fable['turns']:
                    fable['turns'][str(turn - 1)] = self._normalize(value)
        return fable

    def _state(self) -> dict[str, Any]:
        """Fable without the ledger keys (the persistable turn state)."""
        return {
            k: v for k, v in self.fable.items()
            if k not in ('snapshots', 'turns')
        }

    def load_fable(self) -> dict[str, Any]:
        """Load fable from disk, or start empty."""
        os.makedirs(self.opts.vector_dir, exist_ok=True)
        path = self._fable_file()
        if os.path.exists(path):
            try:
                with open(path, 'r', encoding='utf-8') as handle:
                    return self._normalize(json.load(handle))
            except (OSError, TypeError, ValueError, json.JSONDecodeError):
                pass
        return self._empty()

    def save_fable(self, fable: Optional[dict[str, Any]] = None) -> None:
        """Save current fable state to disk."""
        if fable is not None:
            self.fable = fable
        os.makedirs(self.opts.vector_dir, exist_ok=True)
        with open(self._fable_file(), 'w', encoding='utf-8') as handle:
            json.dump(self.fable, handle)

    def set_branch(self, branch: str) -> None:
        """Switch fable files when the story branch changes."""
        name = (branch or 'story').strip() or 'story'
        if name == self.branch:
            return
        self.save_fable()
        self.branch = name
        self.fable = self.load_fable()

    # ---------- turn ledger / rollback / fork (Rooted) ----------

    def _ledger(self) -> dict[int, dict[str, Any]]:
        """Post-turn fable states keyed by turn number."""
        turns = self.fable.get('turns') or {}
        return {
            self._as_turn(k): v for k, v in turns.items() if self._as_turn(k)
        }

    def _prune_ledger(self) -> None:
        """Keep the newest _MAX_LEDGER pages; oldest pages fall off."""
        turns = self.fable.get('turns') or {}
        if len(turns) <= _MAX_LEDGER:
            return
        keep = sorted(turns, key=self._as_turn)[-_MAX_LEDGER:]
        self.fable['turns'] = {key: turns[key] for key in keep}

    def _write_page(self, turn: int) -> None:
        """Record the post-turn fable so any later turn can be undone."""
        self.fable.setdefault('turns', {})[str(turn)] = deepcopy(self._state())
        self._prune_ledger()

    def _restore_pre_turn(self, turn: int) -> None:
        """Rewind the fable to the state before ``turn`` was fabled."""
        turns = self._ledger()
        ledger = {
            str(k): deepcopy(v) for k, v in turns.items() if k <= turn - 1
        }
        eligible = [k for k in turns if k <= turn - 1]
        restored = deepcopy(turns[max(eligible)]) if eligible else self._empty()
        restored['turns'] = ledger
        self.fable = restored

    def rollback_to(self, turn: int) -> None:
        """Truncate the fable to its state as of ``turn``.

        The current fable becomes the post-``turn`` ledger page — the
        nearest recorded page at or before ``turn``; with nothing there
        (a brand-new book, or pages pruned past the rewind point) the
        empty book — and every later page is dropped. Called for rewind,
        delete-last and edit-user so the director never again whispers
        about turns that no longer exist.
        """
        turn = max(0, self._as_turn(turn))
        turns = self._ledger()
        eligible = [k for k in turns if k <= turn]
        restored = deepcopy(turns[max(eligible)]) if eligible else self._empty()
        restored['turns'] = {
            str(k): deepcopy(v) for k, v in turns.items() if k <= turn
        }
        self.fable = restored
        self.save_fable()

    def reset_branch(self) -> None:
        """Empty the current branch's fable (its history was reset)."""
        self.fable = self._empty()
        self.save_fable()

    def delete_branch(self, branch: str) -> None:
        """Drop a deleted branch's fable file. Best effort, never fatal."""
        if not branch or branch == self.branch:
            return
        try:
            os.remove(self._fable_file(branch))
        except OSError:
            pass

    def fork_branch(self, src: str, name: str,
                    cut_turns: Optional[int] = None) -> None:
        """Copy the parent fable into a forked branch.

        Full clone copies the file verbatim — the bookworm opens the book
        at page one, ledger and all. A cut fork keeps the fable exactly
        as of that page: the post-``cut`` ledger entry (nearest page at
        or before the cut, else the empty book) with later pages pruned
        and any loop planted after the cut dropped. Best effort, never
        fatal.
        """
        if not name or name == src:
            return
        self.save_fable()
        src_path = self._fable_file(src)
        dst_path = self._fable_file(name)
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
        state = self._normalize(data)
        pages = state.pop('turns', None) or {}
        turns = {self._as_turn(k): v for k, v in pages.items()}
        eligible = [k for k in turns if k <= cut]
        restored = deepcopy(turns[max(eligible)]) if eligible else self._empty()
        restored['turns'] = {
            str(k): deepcopy(v) for k, v in turns.items() if k <= cut
        }
        restored['loops'] = [
            loop for loop in restored.get('loops', [])
            if self._as_turn(loop.get('planted_turn')) <= cut
        ]
        last = self._as_turn(restored.get('last_fabled_turn'))
        restored['last_fabled_turn'] = min(last, cut) if last else cut
        os.makedirs(self.opts.vector_dir, exist_ok=True)
        with open(dst_path, 'w', encoding='utf-8') as handle:
            json.dump(restored, handle)

    # ---------- the heavy prompt's hidden brief ----------

    def brief(self, present: Optional[list[str]] = None) -> str:
        """Render the hidden brief the heavy LLM weaves in unseen.

        ``present`` is the grounded scene entity list; only NPC directives
        for characters actually in the scene are revealed.
        """
        parts = []
        if self.fable.get('spine'):
            parts.append('The journey so far: ' + self.fable['spine'])
        loops = sorted(
            self.fable.get('loops', []),
            key=lambda loop: self._as_turn(loop.get('last_seen_turn')),
            reverse=True,
        )[:6]
        if loops:
            parts.append(
                'Open threads you planted and must not drop:\n'
                + '\n'.join(f'- {loop["summary"]}' for loop in loops)
            )
        directives = self.fable.get('npc_directives') or {}
        # The player is always in `present`, but their mind is not ours to
        # brief: a stored PC directive reads back as inner monologue.
        player = self._player()
        here = [
            n.strip().lower() for n in (present or [])
            if n.strip().lower() != player
        ]
        lines = []
        for name in here:
            spec = directives.get(name)
            if not spec:
                continue
            bits = ', '.join(f'{k}: {v}' for k, v in spec.items() if v)
            if bits:
                lines.append(f'{name} — {bits}')
        if lines:
            parts.append(
                'Hidden NPC truths (the player does not know these):\n'
                + '\n'.join(lines[:4])
            )
        if self.fable.get('director_notes'):
            parts.append(
                'Director notes for you only: ' + self.fable['director_notes']
            )
        return '\n'.join(parts)

    # ---------- the invisible update pass ----------

    def _ask(self, slot: str, payload: dict) -> Optional[dict]:
        """One light-LLM call against a pre_processor slot; dict or None."""
        template = (
            self.prompts.slot('pre_processor', slot)
            if slot != 'fable_director_human'
            else self.prompts.optional_slot('pre_processor', slot)
        )
        if not template:
            return None
        human_tmpl = PromptTemplate(template=template, template_format='jinja2')
        prompt = ChatPromptTemplate.from_messages(
            [HumanMessagePromptTemplate(prompt=human_tmpl)],
        ).format_messages(**payload)
        if self.debug:
            self.console.print(
                f'FABLE PROMPT ({slot}):\n{prompt}\n\n',
                style=f'color({self.opts.color})', highlight=False,
            )
        try:
            content = self.llm.invoke(prompt).content
        except APITimeoutError:
            return None
        # pylint: disable-next=bare-except  # a daemon thread must not die
        except:
            return None
        self.common.write_debug(f'plot_manager_{slot}', content)
        if self.debug:
            self.console.print(
                f'FABLE RESPONSE ({slot}):\n{content}\n\n',
                style=f'color({self.opts.color})', highlight=False,
            )
        return self._parse_json(content)

    def record(self, documents: dict) -> list[dict]:
        """Fable one turn: scribe the reply, then run the director.

        Runs on the reply-tagging daemon thread, after ``ground_scene``.
        Returns retired loop summaries (with their entity lists) so the
        caller can archive them into RAG. Failures never fatal.
        """
        if self.llm is None or getattr(self.llm, 'model_name', '') == 'None':
            return []
        declared = self._as_turn(documents.get('turn_num'))
        regen = bool(documents.get('regenerate'))
        last = self._as_turn(self.fable.get('last_fabled_turn'))
        if declared and not regen and last >= declared:
            return []
        turn = declared or last + 1
        if regen:
            self._restore_pre_turn(turn)

        scene = {}
        if self.scene is not None:
            scene = {
                key: self.scene.get_scene().get(key, '')
                for key in _SCENE_KEYS
            }
        fable_view = self._state()
        payload = {
            'user_name': self._player(),
            'turn_num': str(turn),
            'user_input': str(documents.get('user_query') or ''),
            'story_reply': str(documents.get('llm_response') or ''),
            'scene_state': json.dumps(scene, ensure_ascii=False),
            'current_fable': json.dumps(fable_view, ensure_ascii=False),
        }
        archived = []
        delta = self._ask('fable_scribe_human', payload)
        if delta:
            archived = self._apply_scribe(delta, turn)
        direction = self._ask('fable_director_human', payload)
        if direction:
            self._apply_director(direction)
        self.fable['last_fabled_turn'] = max(last, turn)
        self._write_page(turn)
        self.save_fable()
        return archived

    def _apply_scribe(self, delta: dict, turn: int) -> list[dict]:
        """Merge the scribe delta. Returns loops retired this turn."""
        spine = self._clean_str(delta.get('spine'), _MAX_SPINE)
        if spine:
            self.fable['spine'] = spine
        archived = []
        touched = set()
        incoming = delta.get('loops')
        if isinstance(incoming, list):
            for item in incoming:
                if not isinstance(item, dict):
                    continue
                summary = self._clean_str(item.get('summary'), 240)
                if not summary:
                    continue
                key = self._key(summary)
                if key in touched:
                    continue
                touched.add(key)
                names = self._clean_names(item.get('entity'))
                status = str(item.get('status') or 'open').strip().lower()
                planted = bool(item.get('planted'))
                found = next(
                    (loop for loop in self.fable['loops']
                     if self._key(loop['summary']) == key),
                    None,
                )
                if found is None:
                    # Paraphrase guard: the scribe rewords a thread it is
                    # carrying forward ("goat climbs the fence" vs "thistle
                    # climbs the fence slat"). Token overlap above
                    # _LOOP_OVERLAP counts as the same thread; the stored
                    # summary stays canonical and only the seen-turn moves.
                    words = _summary_tokens(summary)
                    if words:
                        found = next(
                            (loop for loop in self.fable['loops']
                             if _is_same_thread(words, loop['summary'])),
                            None,
                        )
                if found is not None:
                    found['entity'] = sorted(
                        set(found.get('entity', [])) | set(names),
                    )
                    if status == 'closed':
                        archived.append({
                            'summary': found['summary'],
                            'entity': found.get('entity', []),
                        })
                        self.fable['loops'].remove(found)
                    else:
                        found['last_seen_turn'] = max(
                            self._as_turn(found.get('last_seen_turn')), turn,
                        )
                elif planted:
                    # A new thread planted this turn. Threads the scribe
                    # merely re-lists without claiming a planting stay out:
                    # live runs showed that branch filling every slot with
                    # narrated events ("the goat ate a weed") until no real
                    # thread fit.
                    self.fable['loops'].append({
                        'summary': summary,
                        'planted_turn': turn,
                        'last_seen_turn': turn,
                        'entity': names,
                    })
        # Deterministic TTL + cap, regardless of what the scribe said.
        kept = []
        for loop in self.fable['loops']:
            if (turn - self._as_turn(loop.get('last_seen_turn'))) > _LOOP_TTL:
                archived.append({
                    'summary': loop['summary'],
                    'entity': loop.get('entity', []),
                })
                continue
            kept.append(loop)
        kept.sort(key=lambda loop: self._as_turn(loop.get('last_seen_turn')))
        self.fable['loops'] = kept[-_MAX_LOOPS:]
        return archived

    def _apply_director(self, delta: dict) -> None:
        """Merge hidden-layer deltas, capped and roster-validated."""
        player = self._player()
        known = set()
        if self.scene is not None:
            known = {
                str(name).strip().lower()
                for name in self.scene.get_scene().get('known_characters', [])
            }
        directives = delta.get('npc_directives')
        if isinstance(directives, dict):
            merged = dict(self.fable.get('npc_directives') or {})
            for name, spec in directives.items():
                key = str(name).strip().lower()
                if not key or key in _EMPTY or key in _PRONOUNS:
                    continue
                if key == player:
                    # The player writes their own mind. A director plan for
                    # the PC leaks straight through brief() into the heavy
                    # prompt and comes back out as invented inner monologue.
                    continue
                if known and key not in known:
                    continue
                if not isinstance(spec, dict):
                    continue
                entry = {
                    field: self._clean_str(spec.get(field), 200)
                    for field in ('drive', 'secret', 'plan', 'stance')
                    if self._clean_str(spec.get(field), 200)
                }
                if entry:
                    merged[key] = entry
            # Purge legacy player directives from earlier fable files too.
            merged.pop(player, None)
            if known:
                merged = {
                    k: v for k, v in merged.items() if k in known
                }
            self.fable['npc_directives'] = dict(
                list(merged.items())[:_MAX_DIRECTIVES],
            )
        arcs = delta.get('dormant_arcs')
        if isinstance(arcs, list):
            kept = [
                self._clean_str(arc, 200) for arc in arcs
                if self._clean_str(arc, 200)
            ]
            self.fable['dormant_arcs'] = kept[:2]
        notes = self._clean_str(delta.get('director_notes'), _MAX_NOTES)
        if notes:
            self.fable['director_notes'] = notes
