""" Manage Scenes stateful properties """
import os
import json
import re
from typing import Iterable, Optional, Any, Tuple, List
from collections import namedtuple
from .ragtag_manager import RAGTag
from .chat_utils import CommonUtils, ChatOptions

PRONOUNS = {'i', 'you', 'we', 'him', 'her', 'she', 'teller',
            'they', 'user', 'them', 'me', 'reader', 'asker', 'liability'}

PRONOUN_ALIASES = {'narrator','protagonist','mc','pc','player',
                   'storyteller','dm','gm', 'ai', 'assistant'}

GENERIC_HEAD_WORDS = {
    'aspect','self','dream','feeling','feelings','mood','memory','thought','voice',
    'presence','shadow','figure','stranger','man','woman','child','guard','merchant',
    'traveler','priest','innkeeper','soldier','shopkeeper','victim','witness'
}

MOVE_HINTS = [
    "walk", "step", "stride", "stroll", "march", "pace", "wander",
    "run", "sprint", "dash", "jog", "hurry", "rush",
    "crawl", "creep", "sneak", "tiptoe", "climb", "descend",
    "move", "go", "leave", "exit", "enter", "head", "approach",
    "cross", "pass", "advance", "retreat", "follow",
    "pursue", "chase", "flee", "escape", "travel", "journey",
    "ride", "sail", "row", "fly", "soar", "drift", "glide",
    "slide", "slip", "jump", "leap", "bound", "hop"
]
_HINT_ALT = "|".join(re.escape(h) for h in MOVE_HINTS)
_HINT_REGEX = rf"(?:{_HINT_ALT})\w*"

# direction / target cues after a movement verb
_DIR_RE = re.compile(r"\b(to|toward|towards|into|onto|across|through|down|up|out|back|north|south|east|west|home|away)\b",
                     re.IGNORECASE)

# clause-level (subjectless) movement cues
_ABSOLUTE_RES = [
    re.compile(r"\bhead(?:ed|ing)\b(?:\W+\w+){0,3}?", re.IGNORECASE),    # 'heading', 'headed'
    re.compile(r"\bset\s+(?:off|out)\b", re.IGNORECASE),                 # 'set off/out'
    re.compile(r"\bstart\s+(?:off|out)\b", re.IGNORECASE),               # 'start off/out'
    re.compile(r"\bmake\s+(?:for|my\s+way|our\s+way)\b", re.IGNORECASE), # 'make for', 'make my/our way'
    re.compile(r"\bstrike\s+out\b", re.IGNORECASE),                      # 'strike out'
]

RARE_SENTINEL = '[[RARE_EVENT:USED]]'
RARE_KEYWORDS = ('paralyzed','drugged','captured','bound','kidnapped','stunned','ambushed')

class SceneManager:
    """
    Maintain a healthy scene state by grounded entity and
    audience and other ephemeral turn by turn information
    """
    PRIVATE_SCENE_KEYS = {"beats_since_sex", "linger_beats"}

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
            'entity'           : [self.opts.user_name.lower()],
            'audience'         : [],
            'known_characters' : [self.opts.user_name.lower()],
            'plots'            : {},
            'location'         : '',
            'entities_about'   : [],
            'summary'          : '',
            'scene_mode'       : 'sfw',
            'beats_since_sex'  : 999,
            'linger_beats'     : 0,
            'last_content_rating': 'sfw',
        }

    def _ensure_rare_fields(self):
        s = self.scene
        s.setdefault('rare_event_pending', False)
        s.setdefault('rare_event_used', False)
        s.setdefault('rare_event_cooldown', 0)
        s.setdefault('safe_mode_turns', 0)

    def allow_rare_now(self):
        """Arm a one-turn permission (used by your [RARE NOW] control)."""
        self._ensure_rare_fields()
        if (self.scene['safe_mode_turns'] == 0
            and self.scene['rare_event_cooldown'] == 0):
            self.scene['rare_event_pending'] = True

    def rare_now_addendum(self) -> str:
        """Return the one-turn addendum to append to your HumanPrompt."""
        self._ensure_rare_fields()
        if self.scene['safe_mode_turns'] > 0:
            return ''
        if self.scene['rare_event_pending']:
            return ("\nNOW: A rare involuntary event MAY occur this turn. "
                    "Otherwise, do not force protagonist actions.")
        return ''

    # ---- end-of-turn bookkeeping ----
    def _rare_used(self, model_text: str) -> bool:
        if not model_text:
            return False
        if RARE_SENTINEL in model_text:
            return True
        low = model_text.lower()
        return any(k in low for k in RARE_KEYWORDS)

    def finalize_turn(self, model_text: str, *, cooldown=20) -> None:
        """
        Call once per turn AFTER model output is available.
        Handles:
          - cooldown decrement
          - safe_mode decrement
          - consuming pending rare flag
          - marking used + starting cooldown if triggered
          - persisting scene
        """
        self._ensure_rare_fields()

        # decrement timers
        s = self.scene
        s['safe_mode_turns']      = max(s.get('safe_mode_turns', 0) - 1, 0)
        s['rare_event_cooldown']  = max(s.get('rare_event_cooldown', 0) - 1, 0)

        # consume pending for this turn
        if s.get('rare_event_pending'):
            if self._rare_used(model_text):
                s['rare_event_used'] = True
                s['rare_event_cooldown'] = max(s['rare_event_cooldown'], cooldown)
            s['rare_event_pending'] = False

    # ---- convenience for UI ----
    def should_show_now_button(self) -> bool:
        self._ensure_rare_fields()
        return (
            self.scene['rare_event_cooldown'] == 0
            and self.scene['safe_mode_turns'] == 0
            and not self.scene['rare_event_pending']
        )

    def _sanitize_for_rag(self, data: dict) -> dict:
        """Return a copy of data safe for Chroma field filters:
        - drops PRIVATE_SCENE_KEYS
        - coerces any non-string scalars to strings
        - coerces list items to strings
        """
        out = {}
        for k, v in data.items():
            if k in self.PRIVATE_SCENE_KEYS:
                continue
            if isinstance(v, list):
                out[k] = [str(x) if not isinstance(x, str) else x for x in v]
            elif isinstance(v, (int, float, bool)):
                out[k] = str(v)
            else:
                out[k] = v
        return out

    def _token_len(self, s: str) -> int:
        # Cheap token-ish length: words + quoted fragments
        return len(re.findall(r"[A-Za-z0-9’'_-]+|\"[^\"]*\"", s or ""))

    def _explicit_tag(self, s: str, tag: str) -> bool:
        # e.g. [NSFW] or [SFW]
        return bool(re.search(rf"\[\s*{re.escape(tag)}\s*\]", s or "", re.IGNORECASE))

    def _sexual_signal(self, tag_dict: dict[str, Any]) -> bool:
        # Primary: pre-processor classifier
        cr = (tag_dict.get('content_rating') or self.scene.get('last_content_rating') or 'sfw').lower()
        if cr.startswith('nsfw'):
            return True
        # Secondary: nsfw_reasons
        reasons = tag_dict.get('nsfw_reasons') or []
        return isinstance(reasons, list) and len(reasons) > 0

    def _aftercare_signal(self, query: str, tag_dict: dict[str, Any]) -> bool:
        # Soft cues that the scene is winding down; easily tweakable
        cues = [
            r"\b(aftercare|cleanup|clean\s*up|washed|dressed|cover(?:ed)?(?:\s*up)?|left\s+the\s+room|leaves\s+the\s+room|good\s*night)\b",
            r"\b(catching\s+breath|cooling\s+down|calming\s+down)\b",
        ]
        text = " ".join([
            query or "",
            " ".join(map(str, tag_dict.get('summary', ""))) if isinstance(tag_dict.get('summary'), list) else str(tag_dict.get('summary', "")),
        ])
        return any(re.search(pat, text, re.IGNORECASE) for pat in cues)

    def _extract_name(self, s: str) -> str | None:
        """
        Extract canonical name key from a 'Name: desc' line.
        - Strips outer [ ] if present.
        - Uses self.common.regex.names when available, else a safe fallback.
        - Returns lowercase key or None if nothing usable.
        """
        s = (s or "").strip()
        if not s:
            return None
        if s.startswith('[') and s.endswith(']'):
            s = s[1:-1].strip()
        head = s.split(':', 1)[0].strip()
        rx = getattr(self.common, "regex", None)
        rx = getattr(rx, "names", None) or re.compile(r"([A-Za-z][A-Za-z'\-]+)")
        m = rx.match(head)
        if m:
            return m.group(1).lower()
        return head.lower() if head else None

    def _clean_line(self, s: str) -> str:
        """Trim and drop a single layer of [ ... ] if model emitted bracket-wrapped strings."""
        s = (s or "").strip()
        if s.startswith('[') and s.endswith(']'):
            s = s[1:-1].strip()
        return s

    def _split_head(self, line: str) -> tuple[str, str]:
        """Return (head_original, head_lower). Empty strings if none."""
        s = self._clean_line(line)
        head = s.split(':', 1)[0].strip()
        return head, head.lower()

    def _looks_like_proper_name(self, head_original: str) -> bool:
        """Heuristic: at least one capitalized token; not just generic words."""
        toks = re.findall(r"[A-Za-z][A-Za-z'\-]*", head_original)
        if not toks:
            return False
        if any(t.lower() in GENERIC_HEAD_WORDS for t in toks):
            return False
        return any(t[0].isupper() for t in toks)

    def _extract_name(self, s: str) -> str | None:
        """
        Your existing canonical key extractor, but safe:
        return lowercase canonical key or None.
        """
        s = self._clean_line(s)
        head = s.split(':', 1)[0].strip()
        if not head:
            return None
        # if you have a stricter regex available, use it here
        m = re.match(r"([A-Za-z][A-Za-z'\-]+(?:\s+[A-Za-z][A-Za-z'\-]+)*)", head)
        return m.group(1).lower() if m else head.lower()

    def _name_head(self, line: str) -> str:
        """
        Return the lowercase 'Name' portion before ':' from an entities_about line.
        Strips one layer of [ ... ] if present. '' if none.
        """
        s = (line or "").strip()
        if s.startswith('[') and s.endswith(']'):
            s = s[1:-1].strip()
        head = s.split(':', 1)[0].strip()
        return head.lower()

    def _name_fix(self, s: str) -> str:
        return self.opts.user_name if self.is_pronounish(s) else s

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

    @staticmethod
    def is_pronounish(head: str) -> bool:
        """
        True if the 'Name' before ':' is effectively a pronoun/alias.
        Handles 'the reader', possessives, and reflexives.
        """
        h = (head or "").strip().lower()
        h = re.sub(r"^the\s+", "", h)      # 'the reader' -> 'reader'
        h = re.sub(r"'s$", "", h)          # "user's" -> "user"
        if not h:
            return False
        return (
            h in PRONOUNS
            or h in PRONOUN_ALIASES
            or h.endswith("self")  # myself/yourself/ourselves/themselves/herself/himself
            or h in {"yourselves","ourselves","themselves"}
        )

    @staticmethod
    def movement_analysis(
        dialog: str,
        subjects: Optional[Iterable[str]] = None
    ) -> Tuple[int, float, List[str], bool]:
        """
        Return (hits, score, matches, moved).
        - Subjects = pronouns ∪ provided entity names (case-insensitive).
        - 'head' counts only as 'headed/heading' (avoids 'shake your head').
        - Direction/target tokens boost the score slightly.
        """
        text = dialog
        matches: List[str] = []
        seen_spans = set()
        hits = 0
        strong_hits = 0

        # Build dynamic subjects
        subj_set = {s.strip().lower() for s in PRONOUNS}
        if subjects:
            subj_set |= {str(s).strip().lower() for s in subjects if str(s).strip()}

        total = (len(subj_set) * len(MOVE_HINTS)) if subj_set and MOVE_HINTS else 1
        gap   = r"(?:\W+(?:\w+|'[a-z]+)){0,2}?"  # up to 2 filler tokens (incl contractions)

        # Pass 1: subject-anchored
        for subj in subj_set:
            subj_pat = r"\b" + re.sub(r"\s+", r"\\s+", re.escape(subj)) + r"\b"
            pattern  = rf"{subj_pat}{gap}\W+\b{_HINT_REGEX}\b"
            for m in re.finditer(pattern, text, flags=re.IGNORECASE):
                g = m.group(0)
                # ignore bare 'head' noun
                if (re.search(r"\bhead\b", g, flags=re.IGNORECASE)
                    and not re.search(r"\bhead(?:ed|ing)\b", g, flags=re.IGNORECASE)):
                    continue
                span = (m.start(), m.end())
                if span in seen_spans:
                    continue
                seen_spans.add(span)
                hits += 1
                matches.append(g)
                tail = text[m.end(): m.end() + 80]
                if _DIR_RE.search(tail):
                    strong_hits += 1

        # Pass 2 (optional): clause-level cues
        for rx in _ABSOLUTE_RES:
            for m in rx.finditer(text):
                span = (m.start(), m.end())
                if span in seen_spans:
                    continue
                g = m.group(0)
                window = text[m.end(): m.end() + 80]
                # require a nearby direction/target, except 'make for/my/our way'
                if not _DIR_RE.search(window) and "make " not in g.lower():
                    continue
                seen_spans.add(span)
                hits += 1
                matches.append(g)
                strong_hits += 1

        weighted = strong_hits * 2 + (hits - strong_hits)
        score = weighted / total if total else 0.0
        moved = hits >= 1
        return hits, score, matches, moved

    def is_npc(self, character: str) -> bool:
        """
        Return True if character is a valid NPC/named character
        """
        entry = (character or "").strip().lower()

        # character *is* the player, and not an NPC
        if entry == self.opts.user_name.lower() or self.opts.user_name.lower() in entry:
            return False

        # Combine all the "ignore me" categories into one set
        ignored = PRONOUNS | PRONOUN_ALIASES | GENERIC_HEAD_WORDS
        if not entry or entry in ignored:
            return False
        for _ignore in GENERIC_HEAD_WORDS:
            if entry.find(_ignore) != -1:
                return False
        return True

    def is_new_character(self, character: str) -> bool:
        """
        Return True and add to growing list of NPCs, if a new NPC is discovered
        """
        entry = (character or "").strip().lower()
        if not self.is_npc(character):
            return False

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

    def update_scene_mode(self, tag_dict: dict[str, Any], query: str) -> None:
        """
        Sticky SFW/NSFW with hysteresis and short-input continuity.
        Inputs:
        - tag_dict: includes content_rating/nsfw_reasons from pre-processor
        - query   : raw user text for explicit [NSFW]/[SFW] or aftercare cues
        Persists:
        - self.scene['scene_mode'], ['beats_since_sex'], ['linger_beats'], ['last_content_rating']
        """
        mode = self.scene.get('scene_mode', 'sfw')
        last_cr = (tag_dict.get('content_rating')
                   or self.scene.get('last_content_rating') or 'sfw').lower()
        self.scene['last_content_rating'] = last_cr

        # Inline operator overrides always win
        force_nsfw = self._explicit_tag(query, 'NSFW')
        force_sfw  = self._explicit_tag(query, 'SFW')

        if force_nsfw and mode != 'nsfw':
            self.scene['scene_mode'] = 'nsfw'
            self.scene['beats_since_sex'] = 0
            self.scene['linger_beats'] = 2  # allow a couple beats of short-input continuity
            return

        if force_sfw and mode != 'sfw':
            self.scene['scene_mode'] = 'sfw'
            self.scene['beats_since_sex'] = 999
            self.scene['linger_beats'] = 0
            return

        # Signals
        sexual = self._sexual_signal(tag_dict)
        short_input = self._token_len(query) < 12
        aftercare = self._aftercare_signal(query, tag_dict)

        # State machine
        if mode == 'sfw':
            # Upgrade to nsfw on a sexual signal from classifier OR sustained cues
            if sexual:
                self.scene['scene_mode'] = 'nsfw'
                self.scene['beats_since_sex'] = 0
                self.scene['linger_beats'] = 2
            else:
                # remain sfw
                self.scene['beats_since_sex'] = 999
                self.scene['linger_beats'] = max(0, self.scene.get('linger_beats', 0) - 1)

        else:  # mode == 'nsfw'
            if sexual:
                # reset counters while sexual content continues
                self.scene['beats_since_sex'] = 0
                self.scene['linger_beats'] = 5
            else:
                # no sexual signal this beat
                # short-input continuity: don't downgrade on brief/ambiguous lines
                if short_input and self.scene.get('linger_beats', 0) > 0:
                    self.scene['linger_beats'] -= 1
                    # keep nsfw, do not increment beats_since_sex aggressively
                    self.scene['beats_since_sex'] = min(2, self.scene.get('beats_since_sex', 0) + 1)
                else:
                    self.scene['beats_since_sex'] = self.scene.get('beats_since_sex', 0) + 1

            # Downgrade only when both aftercare-ish cues and 2+ quiet beats
            if aftercare and self.scene['beats_since_sex'] >= 2:
                self.scene['scene_mode'] = 'sfw'
                self.scene['beats_since_sex'] = 999
                self.scene['linger_beats'] = 0

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
        # always generate the metadata fields you ask them to
        tag_dict['location'] = tag_dict.get('location', '')

        tag_dict['entity'] = tag_dict.get('entity', [self.opts.user_name.lower()])
        tag_dict['entity'] = [self._name_fix(s) for s in (tag_dict.get('entity') or [])]

        tag_dict['audience'] = tag_dict.get('audience', [self.opts.user_name.lower()])
        tag_dict['audience'] = [self._name_fix(s) for s in (tag_dict.get('audience') or [])]

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
        t_entities = scene_dict['entity'].t or []
        t_e_set = {s.lower() for s in t_entities} - PRONOUNS
        t_a_set = ({s.lower() for s in set(scene_dict['audience'].t)}
                   if scene_dict['audience'].t else set())

        # Build scene entity/audience sets from ephemeral data
        s_e_set = {s.lower() for s in set(scene_dict['entity'].s)}
        s_a_set = {s.lower() for s in set(scene_dict['audience'].s)}

        hits, score, matches, moved = self.movement_analysis(query, tag_dict['entity'])
        if self.opts.debug:
            self.console.print(
                f'\n\nMOVE SCORE:\n\tHits:{hits}\n\tSore: {score}'
                f'\n\tMatches: {matches}\n\tMoved: {moved}'
                f'\n\tLLM Detect Move: {tag_dict.get("moving", False)}',
                f'\n\tFrom!=To: {scene_dict["location"].t != scene_dict["location"].s}',
                f'\n\tFinal Decision: {((scene_dict["location"].t != scene_dict["location"].s
                                    and moved)
                                    or tag_dict.get('moving', False))}\n\n',
                style=f'color({self.opts.color})', highlight=True
                )

        # Location changed, clear ephemeral with turn data
        if ((scene_dict['location'].t != scene_dict['location'].s
            and moved)
            or tag_dict.get('moving', False)):
            self.scene.update(self.new_scene())
            self.scene['entity'] = list(set(self.scene['entity']).union(t_e_set))
            self.scene['audience'] = list(set(self.scene['audience']).union(t_a_set))
            self.scene['location'] = scene_dict['location'].t

        # Location remains the same append any new turn data to ephemeral data
        else:
            entity_set = s_e_set.union(t_e_set, t_a_set)
            audience_set = s_a_set.union(t_a_set)
            self.scene['audience'] = list(audience_set)
            self.scene['entity'] = list(entity_set)

        # Update incoming scene data with grounded scene
        self.update_scene_mode(tag_dict, query)

        # Surface scene_mode to model and persist
        self.scene['scene_mode'] = self.scene.get('scene_mode', 'sfw')
        tag_dict['scene_mode'] = self.scene['scene_mode']

        # Update incoming scene data with grounded scene
        tag_dict.update(self.scene)
        export_dict = self._sanitize_for_rag(tag_dict)

        if self.opts.debug:
            self.console.print(f'FINAL SCENE (exported):\n{export_dict}',
                       style=f'color({self.opts.color})', highlight=True)

        self.save_scene(self.scene)
        return self._dict_to_ragtag(export_dict)
