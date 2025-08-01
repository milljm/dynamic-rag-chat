""" common utils used by multiple class modules """
from __future__ import annotations
import os
import re
import sys
import pickle
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping
from typing import NamedTuple
import yaml

class RAGTag(NamedTuple):
    """
    namedtuple class constructor
      RAGTag(tag: str, content: str)
    """
    tag: str
    content: str

@dataclass
class StandardAttributes:
    """ Data class to hold immutable project attributes """
    collections: dict   # RAG Collection name to collection id

    @classmethod
    def attributes(cls)->'StandardAttributes':
        """ return project attributes shared throughout project """
        return cls(collections={'user' : 'user_documents',
                                'ai'   : 'ai_documents',
                                'gold' : 'gold_documents'}
                   )

# pylint: disable=too-many-instance-attributes  # thats what dataclasses are for
@dataclass(slots=True, kw_only=True)
class ChatOptions:
    """ Chat arguments dataclass """
    # ---------- “core” options ----------
    host: str = 'http://localhost:11434/v1'
    model: str = 'gemma3:27b'
    completion_tokens: int = 2048
    time_zone: str = 'GMT'
    api_key: str = 'none'
    assistant_mode: bool = False
    no_rags: bool = False
    debug: bool = False
    verbose: bool = False
    light_mode: bool = False
    name: str = 'assistant'

    # ---------- RAG / pre‑ & post‑processing ----------
    preconditioner: str = 'gemma3:1b'
    embeddings: str = 'nomic-embed-text'
    pre_host: str = 'http://localhost:11434/v1'
    emb_host: str = 'http://localhost:11434/v1'
    vector_dir: str = field(default_factory=lambda: str(Path.cwd() / 'vector_data'))
    matches: int = 20

    # ---------- history ----------
    chat_history: int = 1000
    history_sessions: int = 5

    # ---------- UI ----------
    syntax_theme: str = 'fruity'
    color: int = field(init=False)

    # ---------- bulk import ----------
    import_dir: str | bool = False
    import_pdf: str | bool = False
    import_txt: str | bool = False
    import_web: str | bool = False

    # --- post‑processing of derived fields ---
    def __post_init__(self) -> None:
        # derive colour from light/dark mode
        object.__setattr__(self, "color", 245 if self.light_mode else 233)

    _ALIASES = {
        # YAML/config wording        # ChatOptions field
        'llm_server':                'host',
        'pre_llm':                   'preconditioner',
        'embedding_llm':             'embeddings',
        'pre_server':                'pre_host',
        'embedding_server':          'emb_host',
        'history_dir':               'vector_dir',
        'history_matches':           'matches',
        'history_max':               'chat_history',
        'chat_max':                  'chat_history',
        'use_rags':                   'no_rags',
    }

    _INT_FIELDS = {'matches', 'completion_tokens', 'chat_history', 'history_sessions'}
    @classmethod
    def _build(cls, current_dir: str | Path, raw: Mapping[str, Any]) -> "ChatOptions":
        """
        Convert *any* dict‑like object (from YAML or argparse)
        into valid kwargs for the dataclass.
        """
        data: dict[str, Any] = {}
        for key, value in raw.items():
            field_name = cls._ALIASES.get(key, key)
            if field_name in cls._INT_FIELDS:
                value = int(value)
            data[field_name] = value

        # vector directory default needs `current_dir`
        data.setdefault("vector_dir", os.path.join(current_dir, "vector_data"))
        return cls(**data)

    @classmethod
    def from_yaml(cls, current_dir: str | Path) -> "ChatOptions":
        """Load `.chat.yaml` (if present) and merge with defaults."""
        cfg_file = Path(current_dir) / ".chat.yaml"
        raw: dict[str, Any] = {}
        if cfg_file.exists():
            raw = yaml.safe_load(cfg_file.read_text("utf‑8")) or {}
            raw = raw.get("chat", {})
        return cls._build(current_dir, raw)

    @classmethod
    def from_args(cls, current_dir: str | Path, args_namespace) -> "ChatOptions":
        """Build from an `argparse.Namespace`."""
        return cls._build(current_dir, vars(args_namespace))
# pylint: enable=too-many-instance-attributes


@dataclass
class RegExp:
    """ regular expression in use throughout the project """
    model_re = re.compile(r'(\w+)\W+')
    find_prompt  = re.compile(r'(?<=[<m]eta_prompt: ).*?(?=[>)])', re.DOTALL)
    meta_start_re = re.compile(r'{\W*(metadata)\W+:', re.IGNORECASE)
    json_template = re.compile(r'\{+\s*((?:".+?":.+?)+)\s*\}+', re.DOTALL)
    json_style = re.compile(r'```json.*```', re.DOTALL)
    json_malformed = re.compile(r'{+(.*)}', re.DOTALL)
    all_json = re.compile(r'{.*}', re.DOTALL)
    curly_match = re.compile(r'\{\{\s*(.*?)\s*\}\}', re.DOTALL)
    entities = re.compile(r'[;,|\n]+|\s{2,}|(?<!\w)\s(?!\w)', re.DOTALL)
    metadata_key = 'metadata'

class CommonUtils():
    """ method holder for command methods used throughout the project """
    def __init__(self, console, args):
        self.console = console
        self.__set_project_attributes()
        self.opts = args
        self.regex = RegExp()
        if not os.path.exists(args.vector_dir):
            try:
                os.makedirs(args.vector_dir)
            except OSError:
                print(f'Unable to create directory: {args.vector_dir}')
                sys.exit(1)

        # Load from file. If file does not exist then self.scene_meta == self._scene_meta above
        self.scene_meta = self.load_scene()

        # Session's Chat History list
        self.chat_history_session = self.load_chat()

        # Heat Map
        self.heat_map = 0
        self.prompt_map = self.create_heatmap(8000)
        self.cleaned_map = self.create_heatmap(1000)

    def __set_project_attributes(self):
        """ create dataclass with project attributes """
        self.attributes = StandardAttributes.attributes()

    def if_importing(self):
        """ return bool if we are importing documents """
        return (self.opts.import_dir or
                self.opts.import_web or
                self.opts.import_pdf or
                self.opts.import_txt)

    @staticmethod
    def get_aliases():
        """
        Return a dictionary of aliases to be used as aliases when performing field-filtering
        matches in RAG collection.\n
        Exp: `{ 'entities' : 'entity' }`\n,
        will treat metadata tags such as:\n
          \tRAGTag('entities','[john, jane]')\n
        to be considered as:\n
          \tRAGTag('entity', '[john, jane]')
        """
        return {
            "locations": "location",
            "audiences": "audience",
            "items": "items",
            "narrative_arcs": "narrative_arcs",
            "completed_narrative_arcs": "completed_narrative_arcs",
            "entity_locations": "entity_location",
            "keywords_entities": "entity",
        }

    @staticmethod
    def no_scene()->dict:
        """ return an empty scene """
        return {
            # Core Spatial-Temporal Anchors
            'location': 'unknown',                  # e.g., 'Beast cockpit'
            'time': 'unknown',                      # 'day', 'night', 'dusk', 'morning'
            'status': 'unknown',                    # 'in motion', 'camped', 'combat'

            # Character Presence & Perspective
            'entity': [],                           # Characters mentioned
            'audience': [],                         # Characters being spoken to directly
            'entity_location': ['unknown'],         # john backseat, jane frontseat, bo navigation

            # Narrative Flow / Contextual Arc
            'narrative_arc': 'unspecified',         # e.g., 'john_trust_arc'
            'scene_type': 'unspecified',            # e.g., 'banter', 'flashback', 'combat',
            'tone': 'neutral',                      # e.g., 'playful', 'tense'
            'emotion': 'neutral',                   # emotion felt or scene tone
            'focus': ['conversation'],              # narrative function (dialogue, strategy, etc.)
            'narrative_arcs': set(['']),            # e.g., 'head_to_sector8', 'save_world'
            'completed_narrative_arcs': set(['']),  # e.g., 'talk_to_john', 'help_jane'

            # Environmental & Sensory Cues
            'weather': None,                        # e.g., 'storm approaching'
            'sensory_mood': None,                   # e.g., 'dim light and engine hum'

            # Mechanical/Narrative Nudges
            'user_choice': None,                    # User's last clear decision or action
            'last_object_interacted': None,         # e.g., 'radio', 'rifle', 'memory shard'

            # System Controls / Optional
            'time_jump_allowed': False,             # Can the LLM skip forward?
            'scene_locked': False,                  # Prevents new characters from entering scene
            'narrator_mode': False,                 # If True, LLM may use omniscient 3rd-person
        }

    def get_multikeys(self):
        """
        Return non-flat keys from self.no_scene()\n
        e.g., Return what items may contain a list or set
        """
        return {k.strip().lower() for k, v in self.no_scene().items()
                if isinstance(v, (list, set))
                }

    @staticmethod
    def validate_entity_presence(scene: dict)->list[str]:
        """
        Ensure all characters in `entity` are grounded in either `audience` or `entity_location`.
        Returns a list of phantom entities (those not grounded).
        """
        def normalize_entity_list(e_field)->set[str]:
            """
            Accepts a list or comma-delimited string and returns a set of cleaned entity names.
            """
            if isinstance(e_field, str):
                return set(e.strip() for e in e_field.split(',') if e.strip())
            elif isinstance(e_field, list):
                result = set()
                for item in e_field:
                    if isinstance(item, str):
                        result.update(e.strip() for e in item.split(',') if e.strip())
                return result
            return set()
        raw_entities = scene.get('entity', [])
        raw_audience = scene.get('audience', [])
        raw_locations = scene.get('entity_location', [])
        entities = normalize_entity_list(raw_entities)
        audience = normalize_entity_list(raw_audience)
        locations = normalize_entity_list(raw_locations)

        physically_present = set()
        for ent in entities:
            ent_lower = ent.strip().lower()
            for loc in locations:
                if ent_lower in loc.lower():
                    physically_present.add(ent)
        grounded = audience.union(physically_present)
        phantoms = [e for e in entities if e not in grounded]
        return phantoms

    def _clear_scene(self)->dict:
        """ clear the scene """
        self.scene_meta = self.no_scene()

    def scene_tracker_from_tags(self, tags: list[RAGTag])->str:
        """ Build a formatted scene state string based on incoming RAGTags and internal memory """
        tag_dict = {tag.tag: tag.content for tag in tags}
        # Allow for an update to self.no_scene schema
        for key, value in self.no_scene().items():
            if key not in self.scene_meta:
                self.scene_meta[key] = value
        scene = self.scene_meta.copy()
        for key, value in scene.items():
            if key not in self.no_scene():
                self.scene_meta.pop(key)

        scene = self.scene_meta.copy()
        for key in scene:
            incoming = tag_dict.get(key)
            # Skip if incoming is invalid or empty
            if incoming in (None, 'none', 'unknown', [], '', {}, 'null'):
                continue
            # Handle list fields safely
            if isinstance(scene[key], list) and not isinstance(incoming, list):
                incoming = [incoming]
            # Handle unique scene tracking features (like missions and completed_missiones)
            if key == 'narrative_arcs':
                if isinstance(incoming, list):
                    scene[key].update(i for i in incoming if i and i.strip())
                elif isinstance(incoming, str) and incoming.strip():
                    scene[key].add(incoming.strip())
            elif key == 'completed_narrative_arcs':
                if isinstance(incoming, list):
                    scene[key].update(i for i in incoming if i and i.strip())
                elif isinstance(incoming, str) and incoming.strip():
                    scene[key].add(incoming.strip())
                try:
                    _arcs = set(scene[key])
                    for _mission in _arcs:
                        scene['completed_narrative_arcs'].remove(_mission)
                        scene['narrative_arcs'].remove(_mission)
                except KeyError:
                    pass
            else:
                scene[key] = incoming
        # Update internal memory with merged scene state
        self.scene_meta = scene  # No `.copy()` needed
        # Convert to string for LLM injection
        def stringify(k, v):
            if isinstance(v, (list, set)):
                clean = sorted(i for i in v if i and str(i).strip())
                return f'{k}=' + (', '.join(clean) if clean else 'none')
            if v in (None, '', [], {}, 'none', 'unknown'):
                return f'{k}=none'
            return f'{k}={v}'
        phantoms = self.validate_entity_presence(scene)
        scene_str = '#SCENE_STATE: ' + '; '.join(stringify(k, v) for k, v in sorted(scene.items()))
        if phantoms:
            scene_str += (
    '\n\n⚠️ CRITICAL WARNING: The following entities are listed in `entity:` '
    f'but are NOT grounded in `audience:` or `entity_location:`: {", ".join(phantoms)}.\n'
    'They may only be referenced passively. These characters must not speak, act, or appear. '
    'They are not physically present in the scene.\n'
    '🧠 You may reference them emotionally (e.g., thoughts, memories, feelings), but they must '
    'NOT:\n'
    '- Perform actions (e.g., enter the room, move, react)\n'
    '- Speak or interrupt\n'
    '- Be visually or physically described unless in remembered detail\n'
    'They may NOT arrive, emerge, appear, or be discovered mid-scene.\n'
    'Scene location is LOCKED. Character list is FINAL.\n'
    'Any attempt to use these characters as if present is a violation of story continuity.\n'
    '### 🔐 Scene Presence Rules (Active)')
        self.save_scene(self.scene_meta)
        return scene_str

    def sanatize_response(self, response: str, strip: bool = False)->str:
        """ remove emojis, metadata tagging, etc """
        response = self.remove_tags(response)
        if strip:
            response = self.normalize_for_dedup(response)
        return response

    @staticmethod
    def tags_to_dict(tags: list[RAGTag])->dict:
        """ Convert list of RAGTag objects to a dictionary """
        return {tag.tag: tag.content for tag in tags}

    @staticmethod
    def normalize_metadata_for_rag(meta: dict)->dict:
        """ serialize values for RAG meta-fields """
        result = {}
        for key, val in meta.items():
            if isinstance(val, list):
                result[key] = ', '.join(str(v) for v in val)
            elif isinstance(val, bool):
                result[key] = str(val).lower()  # optional: keep as string for uniformity
            elif val is None:
                result[key] = "none"
            else:
                result[key] = str(val)
        return result

    @staticmethod
    def sanitize_json_string(json_string):
        r"""
        Remove any characters with ASCII values less than 32, except for \n, \r, and \t
        """
        json_string = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', json_string)
        json_string = re.sub(r'\n', '', json_string)
        return json_string

    def remove_tags(self, response: str)->str:
        """ remove metadata from response """
        _response = str(response)
        for match in self.regex.all_json.findall(_response):
            _response = _response.replace(f'{match}', '')
        return _response

    @staticmethod
    def parse_tags(meta_tags: dict|list[list[str,str]])->list[RAGTag[str,str]]:
        """ Parse supplied dictionary or list of lists into RAGTags """
        _rag_tags = []
        if isinstance(meta_tags, dict):
            items = meta_tags.items()
        else:
            items = meta_tags  # Assume it's list[list[str, str]]
        for key, value in items:
            if isinstance(value, str):
                # Try to split if it's a multi-item string (comma, semicolon, pipe, etc.)
                split_values = re.split(r'[;,|]\s*', value.strip())
                # Use list if it split into multiple values, else keep as string
                value = split_values if len(split_values) > 1 else split_values[0]
            _rag_tags.append(RAGTag(key, value))
        return _rag_tags

    def get_tags(self, response: str)->list[RAGTag]:
        """ Extract tags in JSON and meta_tag format from the LLM's response """
        _tags = []
        try:
            # JSON-style block. Attempt several kinds of matching, break on the first
            # successful json.loads()
            for match in [self.regex.json_template.search(response),
                          self.regex.json_malformed.search(response),
                          self.regex.curly_match.search(response)]:
                if match:
                    json_str = match.group(1)
                    # print('DEBUG: sanatizing...')
                    # json_str = self.sanatize_response(json_str)
                    try:
                        data = json.loads(f'{{{json_str}}}')
                        data = data[self.regex.metadata_key]
                    except json.decoder.JSONDecodeError:
                        continue
                    _tags.extend(self.parse_tags(data))
                    break
            seen = set()
            deduped = []
            for tag in _tags:
                key = (tag.tag,
                       tuple(tag.content)
                       if isinstance(tag.content, (list, set)) else tag.content)
                if key not in seen:
                    seen.add(key)
                    deduped.append(tag)
            return deduped

        # pylint: disable=broad-exception-caught  # too many ways for this to go wrong
        except Exception as e:
            if self.opts.debug:
                print(f'[get_tags error] {e}')
            return []

    @staticmethod
    def normalize_for_dedup(text: str)->str:
        """ remove emojis and other markdown """
        text = re.sub(r'[\U0001F600-\U0001F64F\u2600-\u26FF\u2700-\u27BF]', '', text)
        return ' '.join(text.lower().split())

    @staticmethod
    def stringify_lists(nested_list)->str:
        """ return a flat string """
        def process(item):
            result = []
            if isinstance(item, list):
                for subitem in item:
                    result.extend(process(subitem))
            else:
                result.append(str(item))
            return result
        flat_strings = process(nested_list)
        return '\n\n'.join(flat_strings)

    def create_heatmap(self, hot_max: int = 0, reverse: bool =False)->dict[int:int]:
        """
        Return a dictionary of ten color ascii codes (values) with the keys representing
        the maximum integer for said color code:
        ./heat_map(10) --> {0: 123, 1: 51, 2: 46, 3: 42, 4: 82, 5: 154,
                            6: 178, 7: 208, 8: 166, 9: 203, 10: 196}
        Options: reverse = True for oppisite effect
        """
        heat = {0: 123} # declare a zero
        colors = [51, 46, 42, 82, 154, 178, 208, 166, 203, 196]
        if self.opts.light_mode:
            heat = {0: 21} # declare a zero
            colors = [19, 26, 30, 28, 65, 58, 94, 130, 124, 196]
        if reverse:
            colors = colors[::-1]
            heat = {0: 196} # declare a zero
        for i in range(10):
            x = int(((i+1)/10) * hot_max)
            heat[x] = colors[i]
        return heat

    def save_scene(self, scene)->None:
        """ persist scenes """
        scene_file = os.path.join(self.opts.vector_dir, 'ephemeral_scene.pkl')
        try:
            with open(scene_file, "wb") as f:
                pickle.dump(scene, f)
        except FileNotFoundError as e:
            print(f'Error saving chat. Check --history-dir\n{e}')

    def load_scene(self)->dict:
        """ Persist chat history (load) """
        scene_file = os.path.join(self.opts.vector_dir, 'ephemeral_scene.pkl')
        try:
            with open(scene_file, "rb") as f:
                loaded_scene = pickle.load(f)
        except FileNotFoundError:
            with open(scene_file, "wb") as f:
                pickle.dump(self.no_scene(), f)
            return self.no_scene()
        except pickle.UnpicklingError as e:
            print(f'Scene file {scene_file} not a pickle file:\n{e}')
            sys.exit(1)
        # pylint: disable=broad-exception-caught  # so many ways to fail, catch them all
        except Exception as e:
            print(f'Warning: Error loading scene file: {e}')
        return loaded_scene

    def save_chat(self)->None:
        """ Persist chat history (save) """
        if self.opts.assistant_mode and not self.opts.no_rags:
            return
        history_file = os.path.join(self.opts.vector_dir, 'chat_history.pkl')
        try:
            with open(history_file, "wb") as f:
                pickle.dump(self.chat_history_session, f)
        except FileNotFoundError as e:
            print(f'Error saving chat. Check --history-dir\n{e}')

    def load_chat(self)->list:
        """ Persist chat history (load) """
        loaded_list = []
        if self.opts.assistant_mode and not self.opts.no_rags:
            return []
        history_file = os.path.join(self.opts.vector_dir, 'chat_history.pkl')
        try:
            with open(history_file, "rb") as f:
                loaded_list = pickle.load(f)
                # trunacate to max ammount
                loaded_list = loaded_list[-self.opts.chat_history:]
        except FileNotFoundError:
            pass
        except pickle.UnpicklingError as e:
            print(f'Chat history file {history_file} not a pickle file:\n{e}')
            sys.exit(1)
        # pylint: disable=broad-exception-caught  # so many ways to fail, catch them all
        except Exception as e:
            print(f'Warning: Error loading chat: {e}')
        return loaded_list

    def save_prompt(self, prompt)->str:
        """ Save the LLMs prompt, overwriting the previous one """
        prompt_file = os.path.join(self.opts.vector_dir, 'llm_prompt.pkl')
        try:
            with open(prompt_file, "wb") as f:
                pickle.dump(prompt, f)
        except FileNotFoundError as e:
            print(f'Error saving LLM prompt. Check --history-dir\n{e}')
        return prompt

    def load_prompt(self)->str:
        """ Persist LLM dynamic prompt (load) """
        prompt_file = os.path.join(self.opts.vector_dir, 'llm_prompt.pkl')
        try:
            with open(prompt_file, "rb") as f:
                prompt_str = pickle.load(f)
        except FileNotFoundError:
            return ''
        except pickle.UnpicklingError as e:
            print(f'Chat history file {prompt_file} not a pickle file:\n{e}')
            sys.exit(1)
        # pylint: disable=broad-exception-caught  # so many ways to fail, catch them all
        except Exception as e:
            print(f'Warning: Error loading chat: {e}')
        return prompt_str

    def check_prompt(self, last_message)->str:
        """ allow the LLM to add to its own system prompt """
        prompt = self.regex.find_prompt.findall(last_message)[-1:]
        if prompt:
            prompt = self.stringify_lists(prompt)
            llm_prompt = self.save_prompt(prompt)
            if self.opts.debug:
                self.console.print(f'PROMPT CHANGE: {llm_prompt}',
                                   style=f'color({self.opts.color})', highlight=True)
            else:
                with open(os.path.join(self.opts.vector_dir, 'debug.log'),
                          'w', encoding='utf-8') as f:
                    f.write(f'PROMPT CHANGE: {llm_prompt}')
        return self.load_prompt()
