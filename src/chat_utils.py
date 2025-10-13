""" common utils used by multiple class modules """
from __future__ import annotations
import os
import re
import sys
import pickle
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Optional
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
                                'ai'   : 'ai_documents'}
                   )

# pylint: disable=too-many-instance-attributes  # thats what dataclasses are for
@dataclass(slots=True, kw_only=True)
class ChatOptions:
    """ Chat arguments dataclass """
    # ---------- “core” options ----------
    host: str = 'http://localhost:11434/v1'
    model: str = 'gemma3:27b'
    nsfw_model: Optional[str] = None
    completion_tokens: int = 1000
    time_zone: str = 'GMT'
    api_key: str = 'none'
    assistant_mode: bool = False
    no_rags: bool = False
    debug: bool = False
    verbose: bool = False
    light_mode: bool = False
    prompts_debug: bool = False
    one_shot: bool = False
    name: str = 'assistant'
    user_name: str = 'John'
    temperature: float = 0.5
    top_p: float = 0.95
    repeat_penalty: float = 1.10
    frequency_penalty: float = 0.4
    presence_penalty: float = 0.2
    context_window: int = 32768
    continue_from: int = -1
    sex: str = 'male'
    character_sheet: str = ''

    # ---------- RAG / pre‑ & post‑processing ----------
    preconditioner: str = 'gemma3:1b'
    embeddings: str = 'nomic-embed-text'
    pre_host: str = 'http://localhost:11434/v1'
    emb_host: str = 'http://localhost:11434/v1'
    vector_dir: str = field(default_factory=lambda: str(Path.cwd() / 'vector_data'))
    matches: int = 20

    # ---------- history ----------
    chat_history: int = 10000
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
        object.__setattr__(self, 'color', 245 if self.light_mode else 236)

        # normalize nsfw_model: fallback to model if unset/empty/legacy sentinel
        if not self.nsfw_model or str(self.nsfw_model).strip().lower() in {'', 'none', 'not_set'}:
            object.__setattr__(self, 'nsfw_model', self.model)

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
        Convert *any* dict-like object (from YAML or argparse)
        into valid kwargs for the dataclass.
        """
        data: dict[str, Any] = {}
        for key, value in raw.items():
            field_name = cls._ALIASES.get(key, key)
            if field_name in cls._INT_FIELDS:
                value = int(value)
            data[field_name] = value

        # vector directory default needs `current_dir`
        data.setdefault('vector_dir', os.path.join(current_dir, 'vector_data'))
        return cls(**data)

    @classmethod
    def from_yaml(cls, current_dir: str | Path) -> 'ChatOptions':
        """Load `.chat.yaml` (if present) and merge with defaults."""
        cfg_file = Path(current_dir) / '.chat.yaml'
        raw: dict[str, Any] = {}
        if cfg_file.exists():
            raw = yaml.safe_load(cfg_file.read_text('utf-8')) or {}
            raw = raw.get('chat', {})
        return cls._build(current_dir, raw)

    @classmethod
    def from_args(cls, current_dir: str | Path, args_namespace) -> 'ChatOptions':
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
    safe_name = re.compile(r'[^a-z0-9]+')  # lowercase + underscores
    core = re.compile(r'[^a-z0-9._:-]+') # friendly token
    names = re.compile(r"([A-Za-z'-]+)")
    ooc_prefix = re.compile(r'^\s*(?:OOC:|SYSTEM:|OOC>)', re.I)
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

        # Session's Chat History dictionary
        self.chat_history_session = {}
        self.chat_history_session = self.load_chat()

        # Heat Map
        self.heat_map = 0
        self.prompt_map = self.create_heatmap(int(args.context_window))
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

    def sanitize_response(self, response: str, strip: bool = False)->str:
        """ remove emojis, metadata tagging, etc """
        response = self.remove_tags(response)
        response = self.removed_other(response)
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

    def removed_other(self, response: str)->str:
        """ remove other fluff that the LLM likes to add """
        _response = str(response)
        _response = _response.replace('```', '')
        _response = _response.replace('Metadata:', '')
        _response = _response.replace('Metadata JSON object:', '')
        if not self.opts.assistant_mode:
            _response = _response.replace('json', '')
        return _response

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
                    # print('DEBUG: sanitizing...')
                    # json_str = self.sanitize_response(json_str)
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
    def stringify_lists(nested_list: list|str)->str:
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
        Options: reverse = True for opposite effect
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

    def save_chat(self)->None:
        """ Persist chat history (save) """
        if self.opts.assistant_mode and not self.opts.no_rags:
            return
        if self.opts.continue_from != -1:
            if self.opts.debug:
                self.console.print('CONTINUE_FROM Enabled. Not saving chat',
                                   style=f'color({self.opts.color})', highlight=True)
            return
        history_file = os.path.join(self.opts.vector_dir, 'chat_history.pkl')
        try:
            with open(history_file, "wb") as f:
                pickle.dump(self.chat_history_session, f)
        except FileNotFoundError as e:
            print(f'Error saving chat. Check --history-dir\n{e}')

    def load_chat(self)->dict:
        """ Persist chat history (load) """
        loaded_dict = {'default': [],
                       'current': 'default'}
        if self.opts.assistant_mode and not self.opts.no_rags:
            return loaded_dict | self.chat_history_session
        history_file = os.path.join(self.opts.vector_dir, 'chat_history.pkl')
        try:
            with open(history_file, "rb") as f:
                loaded_dict = pickle.load(f)
        except FileNotFoundError:
            pass
        except pickle.UnpicklingError as e:
            print(f'Chat history file {history_file} not a pickle file:\n{e}')
            sys.exit(1)
        # pylint: disable=broad-exception-caught  # so many ways to fail, catch them all
        except Exception as e:
            print(f'Warning: Error loading chat: {e}')
        return loaded_dict

    def save_thinking(self, thinking_str: str)->None:
        """ Save Thinking """
        thinking_file = os.path.join(self.opts.vector_dir, 'thinking_debug.log')
        with open(thinking_file, 'w', encoding='utf-8') as f:
            f.write(thinking_str)

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

    def write_debug(self, prefix: str, message: str)->None:
        """ Write to vector_data/{prefix}_debug.log """
        sanitized = prefix.replace('/', '-')
        with open(os.path.join(self.opts.vector_dir, f'{sanitized}_debug.log'),
                  'w', encoding='utf-8') as f:
            f.write(str(message))
