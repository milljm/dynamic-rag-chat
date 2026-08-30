""" common utils used by multiple class modules """
from __future__ import annotations
import os
import re
import ast
import sys
import pickle
import json
import datetime
import secrets
import tempfile
import shutil
import fcntl
from contextlib import contextmanager
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Mapping, Optional
from typing import NamedTuple
import pytz
import yaml

def load_pdf(path: str) -> list:
    """Read a PDF with pypdf. Replaces langchain_community.PyPDFLoader."""
    from pypdf import PdfReader
    from langchain_core.documents import Document

    reader = PdfReader(path)
    docs = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ''
        docs.append(
            Document(page_content=text, metadata={'source': str(path), 'page': i})
        )
    return docs

class RAGTag(NamedTuple):
    """
    namedtuple class constructor
      RAGTag(tag: str, content: str|list)
    """
    tag: str
    content: str|list


HISTORY_JSON = 'chat_history.json'
HISTORY_PKL = 'chat_history.pkl'
HISTORY_VERSION = 1
HISTORY_META_KEYS = frozenset({
    'current', 'assistant_mode', 'branch_modes', 'version',
})


def active_branch(assistant_mode: bool, history: dict | None) -> str:
    """Branch for this process.

    ``--assistant-mode`` (or Spur after a mode sync) stays on assistant.
    Bare ``./chat.py`` is story — do not resume Spur's last assistant current.
    """
    hist = history if isinstance(history, dict) else {}
    if assistant_mode:
        return 'assistant'
    current = hist.get('current') or 'story'
    if current == 'assistant':
        return 'story'
    if current in HISTORY_META_KEYS:
        return 'story'
    if hist and not isinstance(hist.get(current), list):
        return 'story'
    return current


def _json_default(obj: Any):
    """Best-effort conversion for leftover pickle types."""
    as_dict = getattr(obj, '_asdict', None)
    if callable(as_dict):
        return as_dict()
    if isinstance(obj, (set, tuple)):
        return list(obj)
    if isinstance(obj, bytes):
        return obj.decode('utf-8', errors='replace')
    if isinstance(obj, Path):
        return str(obj)
    return str(obj)


def _read_json_dict(path: str) -> dict | None:
    """Load a JSON object from `path`, or None if missing/corrupt/not a dict."""
    try:
        with open(path, 'r', encoding='utf-8') as handle:
            data = json.load(handle)
    except FileNotFoundError:
        return None
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError) as exc:
        print(f'Warning: could not read {path}: {exc}')
        return None
    if isinstance(data, dict):
        return data
    print(f'Warning: {path} is a {type(data).__name__}, expected dict')
    return None


def _read_pickle_dict(path: str) -> dict | None:
    """Load a pickle dict from `path` (legacy chat_history.pkl)."""
    try:
        with open(path, 'rb') as handle:
            data = pickle.load(handle)
    except FileNotFoundError:
        return None
    except (pickle.UnpicklingError, EOFError) as exc:
        print(f'Warning: could not read {path}: {exc}')
        return None
    except Exception as exc:  # pylint: disable=broad-exception-caught
        print(f'Warning: Error loading chat: {exc}')
        return None
    if isinstance(data, dict):
        return data
    print(f'Warning: {path} is a {type(data).__name__}, expected dict')
    return None


def _atomic_write_json(path: str, data: dict) -> None:
    """Write JSON via unique tmp + fsync + replace. Keeps path.bak as a copy.

    Concurrent saves (Spur persist + session GET) must not share chat_history.json.tmp
    or replace() the live file away before the tmp exists.
    """
    directory = os.path.dirname(path) or '.'
    os.makedirs(directory, exist_ok=True)
    bak = path + '.bak'
    with _history_lock(path):
        fd, tmp = tempfile.mkstemp(prefix='.chat_history.', suffix='.tmp', dir=directory)
        try:
            with os.fdopen(fd, 'w', encoding='utf-8') as handle:
                json.dump(
                    data, handle, ensure_ascii=False, indent=2, default=_json_default,
                )
                handle.write('\n')
                handle.flush()
                os.fsync(handle.fileno())
            if os.path.isfile(path):
                try:
                    shutil.copy2(path, bak)
                except OSError:
                    pass
            os.replace(tmp, path)
            tmp = ''
        finally:
            if tmp:
                try:
                    os.remove(tmp)
                except OSError:
                    pass


@contextmanager
def _history_lock(path: str):
    """Exclusive lock so two threads cannot interleave save/load-migrate."""
    lock_path = path + '.lock'
    handle = open(lock_path, 'a+', encoding='utf-8')
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        yield
    finally:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        except OSError:
            pass
        handle.close()


def load_history_from_dir(vector_dir: str, *, migrate: bool = True) -> dict | None:
    """Load history from JSON, falling back to legacy pickle.

    When a pickle is the only source and `migrate` is true, write JSON so the
    next boot never needs pickle again. The .pkl file is left in place.
    """
    json_path = os.path.join(vector_dir, HISTORY_JSON)
    pkl_path = os.path.join(vector_dir, HISTORY_PKL)
    loaded = _read_json_dict(json_path)
    source = 'json' if loaded is not None else ''
    if loaded is None:
        loaded = _read_json_dict(json_path + '.bak')
        if loaded is not None:
            print(f'Warning: restored chat history from {json_path}.bak')
            source = 'json'
            try:
                _atomic_write_json(json_path, loaded)
            except OSError:
                pass
    if loaded is None:
        loaded = _read_pickle_dict(pkl_path)
        if loaded is None:
            loaded = _read_pickle_dict(pkl_path + '.bak')
            if loaded is not None:
                print(f'Warning: restored chat history from {pkl_path}.bak')
        if loaded is not None:
            source = 'pickle'
    if loaded is None:
        return None
    loaded.setdefault('version', HISTORY_VERSION)
    if source == 'pickle' and migrate:
        try:
            _atomic_write_json(json_path, loaded)
            print(f'Migrated chat history pickle → {json_path}')
        except OSError as exc:
            print(f'Warning: could not migrate history to JSON: {exc}')
    return loaded


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
    # ---------- Servers
    # host: str = 'http://localhost:11434/v1'
    host: str | None = None
    pre_host: str | None = None
    emb_host: str | None = None
    polisher_host: str | None = None
    entity_host: str | None = None
    agent_host: str | None = None
    vision_host: str | None = None
    casual_host: str | None = None
    coder_host: str | None = None
    structured_host: str | None = None
    general_host: str | None = None
    nsfw_host: str | None = None

    # ---------- Models
    model: str | None = None
    preconditioner: str | None = None
    embeddings: str | None = None
    polisher_llm: Optional[str] = 'None'
    entity_llm: Optional[str] = 'None'
    agent_llm: Optional[str] = 'None'
    vision_llm: Optional[str] = 'None'
    casual_llm: str | None = None
    coder_llm: str | None = None
    structured_llm: str | None = None
    general_llm: str | None = None
    nsfw_llm: str | None = None

    # ---------- model settings
    model_temp: float = 1.0
    model_topp: float = 0.95
    pre_temp: float = 0.7
    pre_topp: float = 0.95
    nsfw_temp: float = 1.0
    nsfw_topp: float = 0.95
    polisher_temp: float = 1.0
    polisher_topp: float = 0.95
    entity_temp: float = 0.7
    entity_topp: float = 0.95
    agent_temp: float = 0.6
    agent_topp: float = 0.95
    vision_temp: float = 0.9
    vision_topp: float = 0.95
    casual_temp: float = 0.9
    casual_topp: float = 0.95
    general_temp: float = 0.9
    general_topp: float = 0.95
    coder_temp: float = 0.7
    coder_topp: float = 0.95
    structured_temp: float = 0.7
    structured_topp: float = 0.95
    completion_tokens: int = 4000
    repeat_penalty: float = 1.10
    frequency_penalty: float = 0.4
    presence_penalty: float = 0.2
    context_window: int = 32768
    disable_thinking: bool = False

    seed: Optional[str] = str(secrets.randbits(32))

    time_zone: str = 'GMT'
    api_key: str = 'none'
    tavily_key: str = 'none'
    sd_server: str = ''
    assistant_mode: bool = False
    no_rags: bool = False
    debug: bool = False
    verbose: bool = False
    light_mode: bool = False

    name: str = 'assistant'
    user_name: str = 'John'
    continue_from: int = -1
    sex: str = 'male'
    character_sheet: str = ''

    # ---------- Context / RAG / pre‑ & post‑processing / agentic use ----------
    vector_dir: str = field(default_factory=lambda: str(Path.cwd() / 'vector_data'))
    matches: int = 2
    chat_history: int = 10000
    history_sessions: int = 10
    unmolested_sessions: int = 4
    polisher_cnt: int = 1
    distrust_confidence: float = 0.6
    lookback: int | None = None

    # ---------- UI ----------
    syntax_theme: str = 'coffee'
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

        # LM Studio / Ollama want *some* string. Empty is a langchain error.
        key = (self.api_key or '').strip()
        if not key or key.lower() in {'none', 'null', 'not_set'}:
            object.__setattr__(self, 'api_key', 'none')

        # Vision/agent/polisher/entity are opt-in. ChatOpenAI rejects model=None;
        # the rest of the code treats the string 'None' as "not configured".
        for field_name in ('polisher_llm', 'entity_llm', 'agent_llm', 'vision_llm'):
            value = getattr(self, field_name)
            if value is None or str(value).strip() == '':
                object.__setattr__(self, field_name, 'None')

        # If the specialized host wasn't explicitly supplied,
        # inherit the main model server.
        host_fields = (
            'pre_host',
            'emb_host',
            'polisher_host',
            'entity_host',
            'agent_host',
            'vision_host',
            'casual_host',
            'coder_host',
            'structured_host',
            'general_host',
            'nsfw_host',
        )

        for field_name in host_fields:
            if not getattr(self, field_name):
                object.__setattr__(self, field_name, self.host)
        # Set Orchestration models to default model if not set
        mode_fields = {
            'casual': ('casual_llm', 'casual_host'),
            'coder': ('coder_llm', 'coder_host'),
            'structured': ('structured_llm', 'structured_host'),
            'general': ('general_llm', 'general_host'),
            'nsfw': ('nsfw_llm', 'nsfw_host'),
            }
        for _, (llm_field, host_field) in mode_fields.items():
            value = getattr(self, llm_field)

            if not value or str(value).strip().lower() in {'', 'none', 'not_set'}:
                object.__setattr__(self, llm_field, self.model)
                object.__setattr__(self, host_field, self.host)

    _ALIASES = {
        # YAML/config wording        # ChatOptions field
        'model_server':              'host',
        'llm_server':                'host',
        'polisher_server':           'polisher_host',
        'agent_server':              'agent_host',
        'vision_server':             'vision_host',
        'pre_llm':                   'preconditioner',
        'embedding_llm':             'embeddings',
        'pre_server':                'pre_host',
        'embedding_server':          'emb_host',
        'entity_server':             'entity_host',
        'nsfw_server':               'nsfw_host',
        'history_dir':               'vector_dir',
        'rag_matches':               'matches',
        'history_max':               'chat_history',
        'chat_max':                  'chat_history',
        'casual_server':             'casual_host',
        'coder_server':              'coder_host',
        'structured_server':         'structured_host',
        'general_server':            'general_host',
        'tavily_key':                'tavily_key',
        'sd_server':                 'sd_server',
        'sd_host':                   'sd_server',
        'stable_diffusion':          'sd_server',
    }

    _INT_FIELDS = {'matches', 'completion_tokens', 'chat_history', 'history_sessions'}
    _IGNORED_FIELDS = {'color', 'use_rags', 'spur', 'spur_rebuild', 'serve'}
    @classmethod
    def _build(cls,
               current_dir: str | Path,
               raw: Mapping[str, Any],
               base: 'ChatOptions | None' = None,) -> 'ChatOptions':
        """
        Convert *any* dict-like object (from YAML or argparse)
        into valid kwargs for the dataclass.
        """
        if base is None:
            base = cls()

        data = asdict(base)
        data.pop('color')
        for key, value in raw.items():
            if key in cls._IGNORED_FIELDS:
                continue
            field_name = cls._ALIASES.get(key, key)
            if field_name not in data:
                continue
            if field_name in cls._INT_FIELDS:
                value = int(value)
            data[field_name] = value

        # vector directory default needs `current_dir`
        data.setdefault('vector_dir', os.path.join(current_dir, 'vector_data'))
        # Old yaml used `use_rags: true` to *enable* RAG (the flag was inverted).
        # `no_rags` now means disable. Honor the old key only when the new one
        # is absent.
        if 'no_rags' not in raw and 'use_rags' in raw:
            data['no_rags'] = not bool(raw.get('use_rags'))
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
    def from_args(cls, current_dir: str | Path, args_namespace, base) -> 'ChatOptions':
        """Build from an `argparse.Namespace`."""
        return cls._build(current_dir, vars(args_namespace), base)
# pylint: enable=too-many-instance-attributes

@dataclass
class RegExp:
    """ regular expression in use throughout the project """
    # model_re = re.compile(r'(\w+)\W+')
    model_re = re.compile(r'([a-zA-Z]+\d*[a-zA-Z]*)[-_]?(\w*)?[-_](\d+[a-z]*)', flags=re.IGNORECASE)
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
    think_re = re.compile(r'<think>.*</think>(.*)', re.DOTALL)
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
        self.chat_history_session = {'story': [],
                                     'assistant': [],
                                     'current': 'assistant' if args.assistant_mode else 'story',
                                     'branch_modes': {},
                                     'assistant_mode': args.assistant_mode,
                                     }
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
    def history_line(msg: dict) -> str:
        """One USER:/AI: line, plus [attached: …] when the turn had files."""
        role = 'USER' if msg.get('role') == 'user' else 'AI'
        content = msg.get('content', '')
        names = []
        for att in msg.get('attachments') or []:
            if isinstance(att, dict) and att.get('name'):
                names.append(str(att['name']))
        if names:
            return f'{role}: {content}\n[attached: {", ".join(names)}]'
        return f'{role}: {content}'

    @staticmethod
    def record_attachment(documents: dict, name: str, text: str = '',
                          kind: str = 'text') -> None:
        """Track a user file so it can be gold-ingested after this turn's retrieve."""
        documents.setdefault('attachment_texts', [])
        documents['attachment_texts'].append({
            'name': name or 'file',
            'text': text or '',
            'kind': kind or 'text',
        })

    _FILE_EXTS = (
        'py', 'md', 'txt', 'json', 'yaml', 'yml', 'csv', 'pdf', 'js', 'ts',
        'tsx', 'jsx', 'html', 'css', 'rs', 'go', 'c', 'cpp', 'h', 'hpp',
        'java', 'sh', 'bash', 'toml', 'ini', 'xml', 'sql', 'ipynb', 'png',
        'jpg', 'jpeg', 'gif', 'webp', 'svg', 'rb', 'php', 'kt', 'swift',
        'r', 'lua', 'vue', 'scss', 'rst', 'tex', 'cfg', 'conf',
    )
    _FILE_NAME_RE = re.compile(
        r'(?<![A-Za-z0-9_.-])'
        r'([A-Za-z][\w.-]*\.(' + '|'.join(_FILE_EXTS) + r'))'
        r'(?![A-Za-z0-9_.-])',
        re.IGNORECASE,
    )

    @classmethod
    def extract_filenames(cls, text: str) -> list[str]:
        """Basenames mentioned in user text (spur-server.py, README.md, …)."""
        if not text:
            return []
        seen = set()
        out = []
        for match in cls._FILE_NAME_RE.finditer(text):
            name = match.group(1).lower()
            if name in seen:
                continue
            seen.add(name)
            out.append(name)
        return out

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
                result[key] = 'none'
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
    def parse_tags(meta_tags: dict|list[list[str,str]])->list[RAGTag]:
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

    @staticmethod
    def extract_first_json(text: str) -> dict | str:
        """
        Extract the first JSON object from LLM output.

        Strategy
        --------
        1. Remove markdown fences
        2. Find first '{'
        3. Extract balanced braces
        4. Try strict JSON parse
        5. Fallback to Python literal parser
        """

        if not text:
            return ''

        # --- remove markdown fences (common with LLMs) ---
        text = re.sub(r'```(?:json)?', '', text)

        # --- normalize python booleans ---
        text = re.sub(r'\bTrue\b', 'true', text)
        text = re.sub(r'\bFalse\b', 'false', text)
        text = re.sub(r'\bNone\b', 'null', text)

        # --- locate first json object ---
        start = text.find('{')
        if start == -1:
            return ''

        depth = 0
        end = None
        for i, ch in enumerate(text[start:], start):
            if ch == '{':
                depth += 1
            elif ch == '}':
                depth -= 1
                if depth == 0:
                    end = i + 1
                    break

        if end is None:
            return ''
        candidate = text[start:end]
        # --- json repairs
        # remove stray double quote after numbers
        candidate = re.sub(r'(:\s*\d+(?:\.\d+)?)"', r'\1', candidate)
        # --- strict JSON parse ---
        try:
            obj = json.loads(candidate)
            return obj if isinstance(obj, dict) else candidate
        except json.JSONDecodeError:
            pass
        # --- python literal fallback (handles single quotes etc) ---
        try:
            obj = ast.literal_eval(candidate)
            if isinstance(obj, dict):
                return obj
        # pylint: disable-next=broad-exception-caught
        except Exception:
            pass
        return candidate

    def get_tags(self, response: str)->list[RAGTag]:
        """ Extract tags in JSON and meta_tag format from the LLM's response """
        _tags = []
        # Sometimes LLMs prioritize Markdown over JSON output, even when you ask for only JSON.
        response = response.replace('\\_', '_')
        # We must remove the reasoning frame to capture the real JSON output block the model was
        # trying to generate
        think_frame = self.regex.think_re.findall(response)
        if think_frame:
            response = think_frame[0]
            if self.opts.debug:
                self.console.print(f'PRE-PROCESSOR REASONING REMOVED RESPONSE:\n{response}\n\n',
                                style=f'color({self.opts.color})', highlight=False)
        matches = self.extract_first_json(response)
        if isinstance(matches, str):
            self.console.print('\nPardon the intrusion, but pre-processor returned non-valid JSON '
                               'results. Please see:\n\n\tvector_data/pre_processor_debug.log\n\t'
                               'vector_data/json_load_debug.log\n\nfor more information (This turn '
                               'was not saved to the RAG).',
                                style=f'color({self.opts.color})', highlight=False)
            self.write_debug('json_load', matches)
            return []
        if matches:
            _tags.extend(self.parse_tags(matches.get('metadata', {})))
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

    def empty_chat_history(self) -> dict:
        """Default branched history. Never a list — callers do .items()."""
        assistant = bool(getattr(self.opts, 'assistant_mode', False))
        return {
            'story': [],
            'assistant': [],
            'current': 'assistant' if assistant else 'story',
            'branch_modes': {},
            'assistant_mode': assistant,
            'version': HISTORY_VERSION,
        }

    def active_branch(self, history: dict | None = None) -> str:
        """See module-level ``active_branch``."""
        hist = history if isinstance(history, dict) else self.load_chat()
        return active_branch(bool(self.opts.assistant_mode), hist)

    def save_chat(self, history)->None:
        """Persist chat history as JSON. Atomic replace + .bak."""
        if self.opts.continue_from != -1:
            if self.opts.debug:
                self.console.print('CONTINUE_FROM Enabled. Not saving chat',
                                   style=f'color({self.opts.color})', highlight=True)
            return
        if not isinstance(history, dict):
            print(f'Error saving chat: history is {type(history).__name__}, not dict')
            return
        history.setdefault('version', HISTORY_VERSION)
        history_file = os.path.join(self.opts.vector_dir, HISTORY_JSON)
        try:
            _atomic_write_json(history_file, history)
            self.chat_history_session = history
        except FileNotFoundError as e:
            print(f'Error saving chat. Check --history-dir\n{e}')
        except Exception as exc:  # pylint: disable=broad-exception-caught
            print(f'Error saving chat: {exc}')

    def load_chat(self)->dict:
        """Load JSON history, migrating pickle if that is all we have."""
        loaded = load_history_from_dir(self.opts.vector_dir, migrate=True)
        if loaded is None:
            if isinstance(self.chat_history_session, dict):
                return self.chat_history_session
            self.chat_history_session = self.empty_chat_history()
            return self.chat_history_session
        self.chat_history_session = loaded
        return self.chat_history_session

    def save_thinking(self, thinking_str: str)->None:
        """ Save Thinking """
        thinking_file = os.path.join(self.opts.vector_dir, 'thinking_debug.log')
        with open(thinking_file, 'w', encoding='utf-8') as f:
            f.write(thinking_str)

    def save_prompt(self, prompt)->str:
        """ Save the LLMs prompt, overwriting the previous one """
        prompt_file = os.path.join(self.opts.vector_dir, 'llm_prompt.pkl')
        try:
            with open(prompt_file, 'wb') as f:
                pickle.dump(prompt, f)
        except FileNotFoundError as e:
            print(f'Error saving LLM prompt. Check --history-dir\n{e}')
        return prompt

    def load_prompt(self)->str:
        """ Persist LLM dynamic prompt (load) """
        prompt_file = os.path.join(self.opts.vector_dir, 'llm_prompt.pkl')
        try:
            with open(prompt_file, 'rb') as f:
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

    @staticmethod
    def get_time(tzone: str)->str:
        """ return the time """
        mdt_timezone = pytz.timezone(tzone)
        my_time = datetime.datetime.now(mdt_timezone)
        _str_fmt = (f'{my_time.year}-{my_time.month}-{my_time.day}'
                   f':{my_time.hour}:{my_time.minute}:{my_time.second}'
                   f' {"AM" if my_time.hour < 12 else "PM"}')
        return _str_fmt
