""" common utils used by multiple class modules """
import os
import re
import sys
import pickle
import json
from dataclasses import dataclass
from typing import NamedTuple
import yaml

class RAGTag(NamedTuple):
    """
    namedtuple class constructor
      RAGTag(tag: str, content: str)
    """
    tag: str
    content: str

# pylint: disable=too-many-instance-attributes
@dataclass
class ChatOptions:
    """
    Chat arguments dataclass, with a method to initialize using **kwargs.
    """
    debug: bool
    host: str
    model: str
    num_ctx: int
    time_zone: str
    light_mode: bool
    name: str
    verbose: bool
    assistant_mode: bool
    no_rags: bool
    model: str
    preconditioner: str
    embeddings: str
    vector_dir: str
    matches: int
    pre_host: str
    emb_host: str
    api_key: str
    num_ctx: int
    chat_history: int
    history_sessions: int
    syntax_theme: str
    color: int
    import_dir: str
    import_pdf: str
    import_txt: str
    import_web: str

    @classmethod
    def from_yaml(cls, current_dir) -> 'ChatOptions':
        """ read options from yaml file, or set defaults """
        chat_yaml = os.path.join(current_dir, '.chat.yaml')
        _dict = {}
        if os.path.exists(chat_yaml):
            with open(chat_yaml, 'r', encoding='utf-8') as f:
                _dict = yaml.safe_load(f) or {}
        arg_dict = _dict.get('chat', {})  # short hand
        # Create the ChatOptions instance with all required arguments
        return cls(host = arg_dict.get('llm_server', 'http://localhost:11434/v1'),
            light_mode=arg_dict.get('light_mode', False),
            name=arg_dict.get('name', 'assistant'),
            verbose=arg_dict.get('verbose', False),
            assistant_mode=arg_dict.get('assistant_mode', False),
            no_rags=arg_dict.get('use_rags', False),
            model = arg_dict.get('model', 'gemma3:27b'),
            preconditioner = arg_dict.get('pre_llm', 'gemma3:1b'),
            embeddings = arg_dict.get('embedding_llm', 'nomic-embed-text'),
            vector_dir = arg_dict.get('history_dir', os.path.join(current_dir, 'vector_data')),
            matches = int(arg_dict.get('history_matches', 5)), # 5 from each RAG (User & AI)
            pre_host = arg_dict.get('pre_server', 'http://localhost:11434/v1'),
            emb_host = arg_dict.get('embedding_server', 'http://localhost:11434/v1'),
            api_key = arg_dict.get('api_key', 'none'),
            num_ctx = int(arg_dict.get('context_window', 4192)),
            chat_history = int(arg_dict.get('history_max', 1000)),
            history_sessions = int(arg_dict.get('history_sessions', 5)),
            time_zone = arg_dict.get('time_zone', 'GMT'),
            debug = arg_dict.get('debug', False),
            syntax_theme = arg_dict.get('syntax_theme', 'fruity'),
            color = 245 if arg_dict.get('light_mode', False) else 233,
            import_dir = arg_dict.get('import_dir', False),
            import_pdf = arg_dict.get('import_pdf', False),
            import_txt = arg_dict.get('import_txt', False),
            import_web = arg_dict.get('import_web', False),
        )
    @classmethod
    def set_from_args(cls, current_dir, vargs) -> 'ChatOptions':
        """ populate args from ArgumentParser """
        arg_dict = vars(vargs)
        return cls(host = arg_dict.get('host', 'http://localhost:11434/v1'),
            light_mode=arg_dict.get('light_mode', False),
            name=arg_dict.get('name', 'assistant'),
            verbose=arg_dict.get('verbose', False),
            assistant_mode=arg_dict.get('assistant_mode', False),
            no_rags=arg_dict.get('use_rags', False),
            model = arg_dict.get('model', 'gemma3:27b'),
            preconditioner = arg_dict.get('preconditioner', 'gemma3:1b'),
            embeddings = arg_dict.get('embeddings', 'nomic-embed-text'),
            vector_dir = arg_dict.get('history_dir', os.path.join(current_dir, 'vector_data')),
            matches = int(arg_dict.get('matches', 5)), # 5 from each RAG (User & AI)
            pre_host = arg_dict.get('pre_host', 'http://localhost:11434/v1'),
            emb_host = arg_dict.get('emb_host', 'http://localhost:11434/v1'),
            api_key = arg_dict.get('api_key', 'none'),
            num_ctx = int(arg_dict.get('context_window', 4192)),
            chat_history = int(arg_dict.get('chat_max', 1000)),
            history_sessions = int(arg_dict.get('history_sessions', 5)),
            time_zone = arg_dict.get('time_zone', 'GMT'),
            debug = arg_dict.get('debug', False),
            syntax_theme = arg_dict.get('syntax_theme', 'fruity'),
            color = 245 if arg_dict.get('light_mode', False) else 233,
            import_dir = arg_dict.get('import_dir', False),
            import_pdf = arg_dict.get('import_pdf', False),
            import_txt = arg_dict.get('import_txt', False),
            import_web = arg_dict.get('import_web', False),
        )
# pylint: enable=too-many-instance-attributes

@dataclass
class RegExp:
    """ regular expression in use throughout the project """
    model_re = re.compile(r'(\w+)\W+')
    find_prompt  = re.compile(r'(?<=[<m]eta_prompt: ).*?(?=[>)])', re.DOTALL)
    meta_data = re.compile(r'[<]?meta_tags:(.*?);?\s*>', re.DOTALL)
    meta_block = re.compile(r'[<]?meta_tags:.*?\s*>', re.DOTALL)
    meta_start_re = re.compile(r'[\(<\[]\s*meta[_\-:]?', re.IGNORECASE)
    meta_iter = re.compile(r'(\w+):\s*(.+?)(?=[;\n>]|$)')
    json_template = re.compile(r'\{+\s*((?:".+?":.+?)+)\s*\}+', re.DOTALL)
    json_style = re.compile(r'\{\s*(.*?)\s*\}', re.DOTALL)
    json_malformed = re.compile(r'{+(.*)}', re.DOTALL)
    curly_match = re.compile(r'\{\{\s*(.*?)\s*\}\}', re.DOTALL)

class CommonUtils():
    """ method holder for command methods used throughout the project """
    def __init__(self, console, args):
        self.console = console
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
        def normalize_entity_list(field)->set[str]:
            """
            Accepts a list or comma-delimited string and returns a set of cleaned entity names.
            """
            if isinstance(field, str):
                return set(e.strip() for e in field.split(',') if e.strip())
            elif isinstance(field, list):
                result = set()
                for item in field:
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
        """ remove emojis, meta data tagging, etc """
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

    def remove_tags(self, response: str)->str:
        """ remove meta_tags from response """
        _response = str(response)
        for match in self.regex.meta_block.findall(_response):
            _response = _response.replace(f'{match}', '')
        return _response

    def parse_tags(self, flat_pairs: dict | list[tuple[str, str]])->list[RAGTag[str, str]]:
        """Normalize any kind of tag input into RAGTag list of (str, str)"""
        tags = []
        # Normalize input to iterable of (k, v) pairs
        if isinstance(flat_pairs, dict):
            flat_pairs = flat_pairs.items()
        for k, v in flat_pairs:
            canonical_key = self.get_aliases().get(k.strip().lower(), k.strip().lower())
            # Normalize to comma-separated string if multi_key
            if canonical_key in self.get_multikeys():
                if isinstance(v, list):
                    values = ', '.join(str(i).strip() for i in v if str(i).strip())
                elif isinstance(v, str):
                    # If it’s already a string, assume it's comma-separated
                    values = ', '.join(i.strip() for i in v.split(',') if i.strip())
                else:
                    values = str(v).strip() if v is not None else ''
            else:
                values = str(v).strip() if v is not None else ''
            tags.append(RAGTag(canonical_key, values))
        return tags

    @staticmethod
    def sanitize_json_string(json_string):
        r"""
        Remove any characters with ASCII values less than 32, except for \n, \r, and \t
        """
        json_string = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', json_string)
        json_string = re.sub(r'\n', '', json_string)
        return json_string

    def get_tags(self, response: str)->list[RAGTag[str, str]]:
        """ Extract tags in JSON and meta_tag format from the LLM's response """
        _tags = []
        try:
            # JSON-style block. Attempt several kinds of matching, break on the first
            # successful json.loads()
            for match in [self.regex.json_template.search(response),
                          self.regex.json_style.search(response),
                          self.regex.json_malformed.search(response),
                          self.regex.curly_match.search(response)]:
                if match:
                    json_str = match.group(1)
                    json_str = self.sanatize_response(json_str)
                    try:
                        data = json.loads(f'{{{json_str}}}')
                    except json.decoder.JSONDecodeError:
                        continue
                    _tags.extend(self.parse_tags(data))
                    break

            # meta_tag format
            meta_matches = self.regex.meta_data.findall(response)
            if meta_matches:
                flat_pairs = []
                for match in meta_matches:
                    flat_pairs.extend(self.regex.meta_iter.findall(match))
                _tags.extend(self.parse_tags(flat_pairs))

            seen = set()
            deduped = []
            for tag in _tags:
                key = (tag.tag,
                       tuple(tag.content) if isinstance(tag.content, (list, set)) else tag.content)
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
