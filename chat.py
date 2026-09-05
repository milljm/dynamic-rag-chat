#!/usr/bin/env python3
""" Chat Main executable/entry point """
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "langchain==1.3.18",
#     "langchain-core==1.6.1",
#     "langchain-classic==1.0.8",
#     "langchain_ollama==1.1.0",
#     "langchain_openai==1.6.0",
#     "langchain_chroma==1.1.0",
#     "langchain-text-splitters==1.1.2",
#     "chromadb>=1.3.5,<2.0.0",
#     "pydantic>=2.7.4,<3",
#     "posthog",
#     "prompt_toolkit",
#     "rich",
#     "rank_bm25",
#     "pypdf",
#     "pytz",
#     "pillow",
#     "cloudscraper",
#     "beautifulsoup4",
#     "pygments",
#     "jinja2",
#     "duckduckgo-search",
#     "ddgs",
#     "langchain-tavily",
#     "streamlit",
#     "nodejs",
#     "fastapi",
#     "yfinance",
# ]
# ///
import os
import io
import re
import sys
import time
import base64
import argparse
import mimetypes
import shutil
import hashlib
from dataclasses import dataclass, asdict
from copy import deepcopy
from typing import List, Optional
import posthog
import cloudscraper
from rich.console import Console
from rich.theme import Theme
from bs4 import BeautifulSoup
from PIL import Image
from prompt_toolkit import PromptSession
from prompt_toolkit.key_binding import KeyBindings
from src import ContextManager
from src import RAG
from src import RenderWindow
from src import CommonUtils, ChatOptions
from src import ImportData
from src import SceneManager
from src import Orchestration
from src.sd_session import clear_session
from src.chat_utils import (
    load_pdf, HISTORY_META_KEYS, drop_last_assistant, last_user_text,
    purge_rag_entries,
)
posthog.disabled = True
dark_rich_142_styles = {
    'markdown.h1': 'bold #FFFFFF',
    'markdown.h2': 'bold #CCCCCC',
    'markdown.h3': 'bold #999999',
    'markdown.h4': 'italic #777777',
    'markdown.h5': '#555555',
    'markdown.h6': '#333333',
    'markdown.item.bullet': 'yellow',
    'markdown.hr': 'yellow',
    'markdown.table.header': 'bold white',
    'markdown.table.border': 'bright_black',
}
light_rich_142_styles = {
    'markdown.h1': 'bold #000000',
    'markdown.h2': 'bold #333333',
    'markdown.h3': 'bold #666666',
    'markdown.h4': '#888888',
    'markdown.h5': '#AAAAAA',
    'markdown.h6': '#BBBBBB',
    'markdown.item.bullet': 'yellow',
    'markdown.hr': 'yellow',
    'markdown.table.header': 'bold white',
    'markdown.table.border': 'bright_black',
}

console = Console(highlight=True, theme=Theme(dark_rich_142_styles))
current_dir = os.path.dirname(os.path.abspath(__file__))

CMD_LINE = re.compile(r'^[ \t]*\\(?P<cmd>[A-Za-z0-9_\-\?]+)(?:[ \t]+(?P<args>.*))?$')
RARE_TOKENS = (r'[RARE NOW]', r'[RARE USED]', r'[RARE RESET]', r'[SAFE MODE]')
RARE_TOKENS_RE = re.compile('|'.join(re.escape(t) for t in RARE_TOKENS))
INCLUDE_RE = re.compile(r'\{\{([^}]+)\}\}')  # {{/path}} or {{https://url}}

HELP_TEXT = (
    'in-command switches you can use (some are mode-specific):\n\n'
    '\t\\regenerate                  - regenerate last turn\n'
    '\t\\no-context msg              - perform a query with no context\n'
    '\t\\agent msg                   - force agent (web search)\n'
    '\t\\image msg                   - force Stable Diffusion (Automatic1111)\n'
    '\t\\delete-last                 - delete last message from history\n'
    '\t\\turn                        - show turn/status\n'
    '\t\\rewind N                    - rewind to turn N (keep 0..N)\n'
    '\t\\branch NAME@N               - set/fork branch name, if empty list branches;\n'
    '\t                               optional @N to fork from first N turns\n'
    '\t\\dbranch NAME                - delete chat history branch\n'
    '\t\\history [N]                 - show last N user inputs (default 5)\n'
    '\t\\include branch msg          - include branch as attachment\n'
    '\t\\reset                       - resets history/RAG for current branch\n'
    '\n[bold]context injection[/bold]\n'
    '    {{/absolute/path/to/file}}       - include a file as context\n'
    '    {{https://somewebsite.com/}}     - include URL as context\n'
    '\n[bold]story controls[/bold]\n'
    f"    {', '.join(RARE_TOKENS)}\n"
    '\n[bold]keyboard shortcuts (terminal):[/bold]\n\n'
    '    [yellow]Ctrl-W[/yellow] - delete word left of cursor\n'
    '    [yellow]Ctrl-U[/yellow] - delete everything left of cursor\n'
    '    [yellow]Ctrl-K[/yellow] - delete everything right of cursor\n'
    '    [yellow]Ctrl-A[/yellow] - move to beginning of line\n'
    '    [yellow]Ctrl-E[/yellow] - move to end of line\n'
    '    [yellow]Ctrl-L[/yellow] - clear screen\n'
)

def is_message_list(history_branch: list) -> bool:
    """Detect whether we are on the new format."""
    return (bool(history_branch)
            and isinstance(history_branch[0], dict)
            and 'role' in history_branch[0])


def is_history_branch(history: dict, name: str) -> bool:
    """True when `name` is a real message-list branch, not metadata."""
    return name not in HISTORY_META_KEYS and isinstance(history.get(name), list)


def turn_count(messages: list) -> int:
    """Number of complete user turns."""
    if not messages:
        return 0
    if is_message_list(messages):
        return sum(1 for m in messages if m.get('role') == 'user')
    return len(messages)  # old format


def get_last_n_turns(messages: list, n: int) -> list:
    """Return the last n complete turns (user + assistant pairs)."""
    if not messages:
        return []
    if is_message_list(messages):
        # take last 2*n messages, but never go past the beginning
        return messages[-(n * 2):]
    return messages[-n:]


def slice_to_turn(messages: list, turn_n: int) -> list:
    """Keep only the first `turn_n` turns (1-based)."""
    if not messages:
        return []
    if is_message_list(messages):
        return messages[:turn_n * 2]
    return messages[:turn_n]


def delete_last_turn(messages: list) -> list:
    """Remove the last complete turn."""
    if not messages:
        return messages
    if is_message_list(messages):
        # remove last assistant + last user (if present)
        if messages and messages[-1].get('role') == 'assistant':
            messages.pop()
        if messages and messages[-1].get('role') == 'user':
            messages.pop()
        return messages
    messages.pop()
    return messages

@dataclass
class ParsedInput:
    """ In-line command options dataclass """
    # What the user actually wants to “say” to the model (after stripping controls)
    clean_text: str
    # Commands like \rewind, \turn, \no-context, etc. (only one command per line supported;
    # if you want multiple, call handle_command() repeatedly from your UI)
    command: Optional[str]
    args: str
    # Inline story toggles like [RARE NOW], [SAFE MODE] etc.
    rare_controls: List[str]
    # Context includes found: absolute paths or URLs
    includes: List[str]

@dataclass
class SessionContext:
    """
    Common Objects used through out the project

    common = CommonUtils
    rag = RAG
    context = ContextManager
    renderer = RenderWindow
    scene = SceneManager

    """
    common: CommonUtils
    rag: RAG
    context: ContextManager
    renderer: RenderWindow
    scene: SceneManager
    orchestration: Orchestration

    @classmethod
    def from_args(cls, c_console, c_args)->'SessionContext':
        """ instance and return session dataclass """
        _orchestration = Orchestration(console, c_args)
        _common = CommonUtils(c_console, c_args)
        _scene = SceneManager(console, _common, c_args)
        _rag = RAG(c_console, _common, c_args)
        _context = ContextManager(console, _common, _rag, _scene, current_dir, c_args)
        _renderer = RenderWindow(console, _common, _context, current_dir, _orchestration, c_args)
        return cls(common=_common,
                   rag=_rag,
                   context=_context,
                   renderer=_renderer,
                   scene=_scene,
                   orchestration=_orchestration)

def parse_user_input(raw: str) -> ParsedInput:
    """ Parse incoming command string and return a ParsedInput dataclass """
    line = raw.strip()

    # 1) Extract a leading command like \rewind 12
    command, c_args = None, ''
    m = CMD_LINE.match(line.splitlines()[0]) if line else None
    if m:
        command = m.group(1).lower()
        c_args = (m.group(2) or '').strip()
        # remove the first line entirely (the \cmd line)
        rest = line.splitlines()[1:]
        line = '\n'.join(rest).strip()

    # 2) Extract RARE tokens anywhere in the remaining text
    rare_controls = RARE_TOKENS_RE.findall(line)
    if rare_controls:
        line = RARE_TOKENS_RE.sub('', line).strip()

    # 3) Extract includes like {{/abs/path}} or {{https://...}}
    includes = INCLUDE_RE.findall(line)
    if includes:
        line = INCLUDE_RE.sub('', line).strip()

    return ParsedInput(
        clean_text=line,
        command=command,
        args=c_args,
        rare_controls=rare_controls,
        includes=includes,
    )

class CustomWidthFormatter(argparse.RawTextHelpFormatter):
    """Help formatter that keeps flags readable at width 100."""
    def __init__(self, prog):
        """Pin help position and total width for the CLI."""
        super().__init__(prog, max_help_position=40, width=100)

class Chat():
    """ Begin initializing variables classes. Call .chat() to begin """
    def __init__(self, o_session, _args):
        self.opts: ChatOptions = _args
        self.session: SessionContext = o_session
        self.scraper = cloudscraper.create_scraper()

        if _args.assistant_mode:
            self.chat_branch = 'assistant'
        else:
            self.chat_branch = self.session.common.active_branch()
        self._initialize_startup_tasks()

    def _initialize_startup_tasks(self):
        """ run startup routines """
        o_opts = self.opts # shorthand
        if o_opts.debug:
            console.print('[italic dim grey30]Debug mode enabled. I will re-read the '
                               'prompt files each time.[/]')
        if o_opts.assistant_mode:
            extra = ' RAGs disabled (--no-rags).' if o_opts.no_rags else ''
            console.print(f'[italic dim grey30]Assistant mode enabled.{extra}[/]')

    def prepare_turn(self, raw_user_input: str, extras: dict | None = None):
        """
        Returns everything Streamlit needs for one turn.
        """
        documents, meta_data = self.get_documents(raw_user_input, extras)
        return documents, meta_data

    def _branch_exists(self, history, name):
        return is_history_branch(history, name)

    def _slice_upto(self, lst, n):
        n = max(0, min(n, len(lst)))
        return lst[:n]

    def qwen_prompt(self)->str:
        """ return the contents of qwen.md """
        qwen_file = os.path.join(self.opts.vector_dir, '../', 'prompts', 'qwen.md')
        if os.path.exists(qwen_file):
            with open(qwen_file, 'r', encoding='utf-8') as f:
                return f.read()
        return ''

    @staticmethod
    def set_lightmode_aware(light: bool)->str:
        """ inject a light-mode aware prompt command """
        if light:
            return ('Reminder: The user is using a high luminance background. Therefore, try'
                    ' and only use dark emojis which will provide high-contrast')
        return ('Reminder: The user is using a low luminance background. Therefore, try'
                    ' and only use bright emojis which will provide high-contrast')

    def token_counter(self, documents: dict)->any:
        """ report each document token counts """
        for key, value in documents.items():
            yield (key, self.session.context.token_retriever(value))

    def token_manager(self, documents: dict,
                            token_reduction: int)->tuple[int,int]:
        """ Handle token counts and token colors for statistical printing """
        tokens = 0
        for _, token_cnt in self.token_counter(documents):
            tokens += token_cnt

        # Set timers, and completion token counter, colors...
        self.session.common.heat_map = self.session.common.create_heatmap(tokens,
                                                                          reverse=True)
        cleaned_color = [v for k,v in
                         self.session.common.create_heatmap(tokens * 8).items()
                         if k<=token_reduction][-1:][0]

        return (tokens, cleaned_color)

    def get_character_sheet(self)->str:
        """ return contents of character sheet if supplied a path to one """
        if self.opts.character_sheet:
            if os.path.exists(self.opts.character_sheet):
                with open(self.opts.character_sheet, 'r', encoding='utf-8') as f:
                    return f.read()
        return ''

    def get_documents(self, user_input, extras: dict | None = None)->tuple[dict,list]:
        """
        Populate documents, the object which is fed to prompt formaters, and
        ultimately is what makes up the context for the LLM
        """
        documents = dict()
        pre_process_time = time.time()
        history = self.session.common.load_chat()
        prev_msgs = get_last_n_turns(history[self.chat_branch], 1)
        documents.update(
            {'user_query'         : user_input,
             'model'              : self.opts.model,
             'dynamic_files'      : '',
             'include_branch'     : '',
             'dynamic_images'     : [],
             'attachment_texts'   : [],
             'attached_files_note': '',
             'gold_resume'        : '',
             'search_resume'      : '',
             'documents_index'    : '',
             'turn_num'           : turn_count(history[self.chat_branch]) + 1,
             'history_sessions'   : self.opts.history_sessions,
             'name'               : self.opts.name,
             'user_name'          : self.opts.user_name,
             'pro_object'         : 'him' if self.opts.sex == 'male' else 'her',
             'pro_subject'        : 'he' if self.opts.sex == 'male' else 'she',
             'possessive_adj'     : 'his' if self.opts.sex == 'male' else 'her',
             'possessive_pronoun' : 'his' if self.opts.sex == 'male' else 'hers',
             'character_sheet'    : self.get_character_sheet(),
             'date_time'          : self.session.common.get_time(self.opts.time_zone),
             'pre_process_time'   : pre_process_time,
             'light_mode'         : self.set_lightmode_aware(self.opts.light_mode),
             'previous'           : prev_msgs,
             'history'            : history,
             'entities'           : [],
             'explicit'           : False,
             'qwen_prompts'       : self.qwen_prompt(),
             }
            )
        if extras:
            documents.update(extras)

        (documents,
         pre_t,
         post_t,
         meta_data) = self.session.context.handle_context(documents)

        # Heat Map
        (prompt_tokens, cleaned_color) = self.token_manager(documents, max(0, pre_t - post_t))

        # Get total token estimate of context
        performance_summary = ''
        for k, v in self.token_counter(documents):
            performance_summary += f'{k}:{v}\n'
        performance = (f'Total Tokens: {prompt_tokens}\n'
                       f'Duplicate Tokens removed: {max(0, pre_t - post_t)}\n'
                       'Maximum response length: '
                       f'{self.opts.completion_tokens}')

        pre_process_time = (time.time() - pre_process_time)

        # Fill documents with other useful information used downstream. This is a bit dirty
        # but, some of this information may become useful to provide to the LLM itself.
        documents.update({
             'performance'      : performance,
             'completion_tokens': self.opts.completion_tokens,
             'pre_process_time' : pre_process_time,
             'prompt_tokens'    : prompt_tokens,
             'token_savings'    : max(0, pre_t - post_t),
             'cleaned_color'    : cleaned_color,
             }
            )
        return (documents, meta_data)

    def load_content_as_context(self, user_input: str) -> dict:
        """Parse user_input for all occurrences of {{ /path/to/file }}"""
        documents = {
            'dynamic_images': [],
            'dynamic_files': '',
            'user_query': user_input,
            'attachment_texts': [],
        }
        included_files = self.session.common.regex.curly_match.findall(user_input)

        def read_file(file_path: str) -> str:
            """Helper to read file contents or fetch from URL."""
            if os.path.exists(file_path):  # Local file
                return self._process_local_file(file_path)
            elif file_path.startswith('http'):  # URL
                return self._process_url(file_path)
            else:
                return None

        for included_file in included_files:
            file_data = read_file(included_file)
            if file_data:
                data, icon = file_data
                _file = os.path.basename(included_file)
                documents['user_query'] = documents['user_query'].replace(included_file,
                                                                          f'{_file} {icon} ✅')
                if icon == '🖼️':  # Image
                    documents['dynamic_images'].append(data)
                    self.session.common.record_attachment(
                        documents, _file, kind='image',
                    )
                else:
                    documents['dynamic_files'] += f'\n=== {_file} ===\n{data}\n\n'
                    self.session.common.record_attachment(
                        documents, _file, text=str(data), kind='text',
                    )

            else:
                documents['user_query'] = documents['user_query'].replace(included_file,
                                                                           f'{included_file} ❌')

        return documents

    def _process_local_file(self, included_file: str) -> tuple:
        """Process local files based on their mime type."""
        mime_format = mimetypes.guess_type(included_file)[0]
        data = ''
        icon = '📁'  # Default icon
        if mime_format:
            mime, _format = mime_format.split('/')
            if mime == 'image':
                icon, data = self._process_image(included_file, _format)
            elif _format == 'pdf':
                icon, data = self._process_pdf(included_file)
            else:
                if _format == 'html':
                    icon = '🌍'
                elif mime == 'text':
                    icon = '📄'
                data = self._process_text(included_file)
        return data, icon

    def _process_text(self, included_file: str):
        """Read a local text/html file as UTF-8."""
        with open(included_file, 'r', encoding='utf-8') as f:
            data = f.read()
        return data

    def _process_image(self, included_file: str, _format: str) -> tuple:
        """Process image files."""
        icon = '🖼️'
        with Image.open(included_file) as img:
            img = img.convert('RGB')
            buffered = io.BytesIO()
            img.save(buffered, format=_format)
            data = base64.b64encode(buffered.getvalue()).decode('utf-8')
        return icon, data

    def _process_pdf(self, included_file: str) -> tuple:
        """Process PDF files."""
        icon = '📕'
        data = ''.join(doc.page_content for doc in load_pdf(included_file))
        return icon, data

    def _process_url(self, url: str) -> tuple:
        """Process URL links."""
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
                   'AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'}
        try:
            response = self.scraper.get(url, headers=headers, timeout=10)
            if response.status_code == 200:
                soup = BeautifulSoup(response.content, 'html.parser')
                text = ' '.join(soup.get_text().split())
                return (text[:80_000], '🌍')
            return (f'Error fetching URL ({response.status_code})', '❌')
        except Exception:  # pylint: disable=broad-exception-caught
            return None

    def no_context(self, user_input)->tuple:
        """ perform search without any context involved """
        prompt_tokens = self.session.context.token_retriever(user_input) # short hand
        collections = self.session.common.attributes.collections # short hand
        self.session.common.heat_map = self.session.common.create_heatmap(prompt_tokens,
                                                            reverse=True)
        cleaned_color = [v for k,v in
                         self.session.common.create_heatmap(prompt_tokens / 2).items()
                         if k<=0][-1:][0]
        # pylint: disable=consider-using-f-string  # no, this is how it is done
        documents = {'no_context'               : True,
                     'user_query'               : user_input,
                     'name'                     : self.opts.name,
                     'user_name'                : self.opts.user_name,
                     'model'                    : self.opts.model,
                     'chat_history'             : '',
                     'previous'                 : '',
                     'dynamic_files'            : '',
                     'include_branch'           : '',
                     'dynamic_images'           : [],
                     'attachment_texts'         : [],
                     'attached_files_note'      : '',
                     'gold_resume'              : '',
                     'search_resume'            : '',
                     'documents_index'          : '',
                     'turn_num'                 : 0,
                     'history_sessions'         : 0,
                     collections['ai']          : '',
                     collections['user']        : '',
                     collections['gold']        : '',
                     'content_type'             : '',
                     'context'                  : '',
                     'explicit'                 : False,
                     'additional_content'       : '',
                     'date_time'                : self.session.common.get_time(self.opts.time_zone),
                     'completion_tokens'        : self.opts.completion_tokens,
                     'pre_process_time'         : '{:.1f}'.format(0),
                     'performance'              : '',
                     'light_mode'               : self.set_lightmode_aware(self.opts.light_mode),
                     'llm_prompt'               : '',
                     'prompt_tokens'            : prompt_tokens,
                     'token_savings'            : 0,
                     'cleaned_color'            : cleaned_color,
                     'qwen_prompts'             : self.qwen_prompt(),
                     }
        return documents

    def _cmd_delete_last(self, history: dict) -> None:
        """Drop the last assistant/user turn from the current branch."""
        try:
            msgs = history[self.chat_branch]
            snapshot = list(msgs)
            history[self.chat_branch] = delete_last_turn(msgs)
            kept = {id(item) for item in history[self.chat_branch]}
            dropped = [item for item in snapshot if id(item) not in kept]
            purge_rag_entries(self.session.rag, dropped)
            self.session.common.save_chat(history)
            self.session.renderer.clear_ooc()
            clear_session(str(self.opts.vector_dir))
            console.print('[green]Deleted last turn.[/green]', highlight=False)
        except IndexError:
            console.print('[yellow]History empty.[/yellow]')

    def _cmd_rewind(self, history: dict, arg: str) -> None:
        """Keep turns 1..N on the current branch."""
        try:
            n = int(arg)
            cur = history[self.chat_branch]
            total = turn_count(cur)
            if not 1 <= n <= total:
                console.print(f'[red]usage: \\rewind N  (1 ≤ N ≤ {total})[/red]')
                return
            kept = slice_to_turn(cur, n)
            purge_rag_entries(self.session.rag, cur[len(kept):])
            history[self.chat_branch] = kept
            self.session.common.save_chat(history)
            console.print(
                f'[green]Rewound to turn {n} of {total}.[/green]',
                highlight=False,
            )
            self.session.renderer.clear_ooc()
            clear_session(str(self.opts.vector_dir))
        except ValueError:
            console.print('[red]usage: \\rewind N[/red]')

    def _cmd_dbranch(self, history: dict, arg: str) -> None:
        """Delete a named history branch and its on-disk RAG."""
        if self.opts.assistant_mode:
            console.print('[red]Cannot manage branches in assistant mode[/red]')
            return
        protected = {'story', 'assistant', 'current', 'assistant_mode',
                     'branch_modes', 'version'}
        for branch in list(history.keys()):
            if not is_history_branch(history, branch):
                continue
            if arg == self.chat_branch:
                console.print(
                    '[red]Cannot delete current branch you are on. '
                    'Use "/reset" instead',
                )
                return
            if arg in protected:
                console.print(f'[red]Cannot delete {arg} branch. (protected)[/red]')
                return
            if arg == branch and arg != 'current':
                history.pop(arg)
                self.session.rag.wipe_branch_stores(arg)
                self.session.common.save_chat(history)
                console.print(f'[green]Deleted: [/green]{arg}', highlight=False)
                return

    def _cmd_reset(self, history: dict) -> None:
        """Clear history and RAG for the current branch."""
        self.session.rag.wipe_branch_stores(self.chat_branch)
        history[self.chat_branch] = []
        console.print(f'[green]Reset: [/green]{self.chat_branch}', highlight=False)
        self.session.common.save_chat(history)
        clear_session(str(self.opts.vector_dir))

    def _list_branches(self, history: dict) -> None:
        """Print branch names with turn counts and a preview of the last message."""
        branches = sorted(k for k in history.keys() if is_history_branch(history, k))
        maxlen = max((len(n) for n in branches), default=0)
        if self.chat_branch in branches:
            branches.remove(self.chat_branch)
            branches.insert(0, self.chat_branch)
        for name in branches:
            count = turn_count(history[name])
            preview = ''
            if count > 0:
                last = history[name][-1].get('content', '')
                last = ' '.join(last.split())
                preview = last[:40] + ('…' if len(last) > 40 else '')
                preview = f'[dim]{preview}[/dim]'
            if name == self.chat_branch:
                console.print(
                    f'\t➡ [green]{name:<{maxlen}}[/green] : '
                    f'[{count:>3}] {preview}',
                    highlight=False,
                )
            else:
                console.print(
                    f'\t  {name:<{maxlen}} : [{count:>3}] {preview}',
                    highlight=False,
                )

    def _create_or_switch_branch(self, history: dict, spec: str) -> None:
        """Switch to NAME, or fork the current branch (optionally NAME@N)."""
        raw = spec.strip()
        if self.opts.assistant_mode:
            console.print(
                '[red]Not possible while in assistant mode. If you wish '
                'to use this feature,\nrun the streamlit version instead'
                ':[/red]\n\n\tstreamlit run streamlit_chat.py -- '
                '--assistant-mode',
                highlight=False,
            )
            return
        if raw in {'current', 'assistant', 'assistant_mode', 'branch_modes', 'version'}:
            console.print('[red]Invalid branch name.[/red]', highlight=False)
            return
        if '@' in raw:
            name, n_str = raw.split('@', 1)
            try:
                cut = int(n_str) * 2
            except ValueError:
                console.print('[red]usage: \\branch NAME[@N][/red]', highlight=False)
                return
        else:
            name, cut = raw, None
        if self._branch_exists(history, name):
            if name == self.chat_branch:
                console.print(f'[green]Already on branch:[/green] {name}', highlight=False)
            else:
                self.chat_branch = name
                history['current'] = name
                console.print(f'[green]Switched to :[/green] {name}', highlight=False)
            self.session.common.save_chat(history)
            return
        src = self.chat_branch
        base = history[src]
        new_list = deepcopy(self._slice_upto(
            base, cut if cut is not None else len(base),
        ))
        history[name] = new_list
        history['current'] = name
        self.chat_branch = name
        self.session.common.save_chat(history)
        self.session.renderer.clear_ooc()
        try:
            if cut is None:
                self.session.rag.clone_collection(src, name, overwrite=False)
            else:
                self.session.rag.build_collection_from_texts(
                    name, new_list, overwrite=True,
                )
            console.print(f'[green]Branched to:[/green] {name}', highlight=False)
        except Exception as exc:  # pylint: disable=broad-exception-caught
            console.print(
                f'[red]RAG sync failed for "{name}":[/red] {exc}',
                highlight=False,
            )

    def _cmd_branch(self, history: dict, arg: str) -> None:
        """List branches, or create/switch to NAME[@N]."""
        if not arg:
            self._list_branches(history)
            return
        self._create_or_switch_branch(history, arg)

    def _cmd_history(self, history: dict, arg: str) -> None:
        """Print the last N turns of the current branch."""
        try:
            n = int(arg or '5')
        except ValueError:
            n = 5
        messages = history[self.chat_branch]
        turns = get_last_n_turns(messages, n)
        if not turns:
            console.print('[yellow]No history yet.[/yellow]')
            return
        total_turns = turn_count(messages)
        turn_num = total_turns - (len(turns) // 2)
        for msg in turns:
            if not isinstance(msg, dict):
                console.print(f'\n\n{msg}')
                continue
            role = msg.get('role', '?').lower()
            content = msg.get('content', '')
            if role == 'user':
                turn_num += 1
                console.print()
                console.print(f'[bold cyan]⬇  TURN {turn_num}  ⬇[/bold cyan]')
                console.print(f'[bold]USER:[/bold] {content}')
            elif role == 'assistant':
                console.print(f'\n[bold]AI:[/bold] {content}')
            else:
                console.print(f'\n[dim]{role.upper()}:[/dim] {content}')
        console.print()

    def _cmd_regenerate(self, history: dict):
        """Drop the last assistant reply and re-send the previous user text.

        Returns a ParsedInput to send, or None if there is no user turn.
        """
        msgs = history.get(self.chat_branch) or []
        dropped = []
        if msgs and isinstance(msgs[-1], dict) and msgs[-1].get('role') == 'assistant':
            dropped.append(msgs[-1])
        drop_last_assistant(msgs)
        purge_rag_entries(self.session.rag, dropped)
        history[self.chat_branch] = msgs
        text = last_user_text(msgs)
        if not text:
            console.print('[yellow]Nothing to regenerate.[/yellow]')
            return None
        self.session.common.save_chat(history)
        return parse_user_input(text)
    def _dispatch_command(self, parsed, history: dict):
        """Handle slash commands.

        Returns (parsed, skip_model). skip_model True means the TUI loop
        should continue without calling the LLM.
        """
        cmd, arg = parsed.command, parsed.args
        skip_and_done = {
            'delete-last': lambda: self._cmd_delete_last(history),
            'turn': lambda: console.print(turn_count(history[self.chat_branch])),
            'rewind': lambda: self._cmd_rewind(history, arg),
            'dbranch': lambda: self._cmd_dbranch(history, arg),
            'reset': lambda: self._cmd_reset(history),
            'branch': lambda: self._cmd_branch(history, arg),
            'history': lambda: self._cmd_history(history, arg),
        }
        if cmd in skip_and_done:
            skip_and_done[cmd]()
            return parsed, True
        if cmd in ('no-context', 'include', 'agent', 'image'):
            if not self.opts.assistant_mode:
                console.print('[red]Only available while in assistant mode.[/red]')
                return parsed, True
            return parsed, False
        if cmd == 'regenerate':
            nxt = self._cmd_regenerate(history)
            if nxt is None:
                return parsed, True
            return nxt, False
        console.print(f'[red]Unknown command:[/red] \\{cmd}')
        return parsed, True

    def _prepare_turn_documents(self, parsed, history: dict, raw: str,
                                regenerate: bool = False):
        """Build the documents dict for this user turn, or None on failure."""
        extras = {}
        if regenerate:
            extras['regenerate'] = True
        if parsed.includes:
            extras['has_files'] = True
            extras['attached_filenames'] = CommonUtils.collect_filenames(
                parsed.includes,
            )
        meta_data = []
        if parsed.command in ('no-context', 'agent', 'image'):
            if parsed.command == 'no-context':
                documents = self.no_context(parsed.args or parsed.clean_text)
            else:
                documents, meta_data = self.get_documents(
                    parsed.args or parsed.clean_text, extras,
                )
            documents['in_line_commands'] = f'Meta: [{parsed.command}]'
        else:
            documents, meta_data = self.get_documents(parsed.clean_text, extras)
        if not documents:
            console.print(
                '[red]There was an error while running pre-processor work.[/red]'
                'In many cases, re-submitting your query again solves the issue.',
            )
            return None, None
        if parsed.command == 'include':
            val = parsed.args
            if val not in history:
                console.print(f'[red]Unknown branch[/red] \\{val}')
                return None, None
            include_branch = ' '.join(history[val][-self.opts.history_sessions:])
            documents['include_branch'] = str(include_branch)
        if parsed.command == 'agent':
            documents['use_agent'] = True
            documents['agent_ran'] = False
        if parsed.command == 'image':
            documents['use_sd'] = True
            documents['sd_ran'] = False
        elif self.opts.assistant_mode:
            clear_session(str(self.opts.vector_dir))
        if parsed.includes:
            inc_docs = self.load_content_as_context(
                ' '.join(f'{{{{{x}}}}}' for x in parsed.includes),
            )
            documents.update(inc_docs)
            documents['user_query'] = (
                f'{raw} \n\nattachments:{documents["user_query"]}'
            )
        self.session.context.ingest_user_attachments(documents, meta_data)
        return documents, meta_data

    def chat(self):
        """Prompt the user for questions and stream replies."""
        c_session = PromptSession()
        kb = KeyBindings()

        @kb.add('escape', 'enter')
        def handle_submit(event):
            """Send the multiline buffer (Esc+Enter)."""
            event.current_buffer.validate_and_handle()

        @kb.add('enter')
        def handle_newline(event):
            """Insert a newline instead of submitting."""
            event.current_buffer.insert_text('\n')

        console.print(
            '💬 Press [italic red]Esc Enter[/italic red] to send message, '
            r'[red]\?[/red] [italic red]Esc Enter[/italic red] for help, '
            '[italic red]Ctrl-C[/italic red] to quit.\n',
        )
        try:
            while True:
                raw = c_session.prompt(
                    '>>> ', multiline=True, key_bindings=kb,
                ).strip()
                if not raw:
                    continue
                if raw == r'\?':
                    console.print(HELP_TEXT)
                    continue
                history = self.session.common.load_chat()
                parsed = parse_user_input(raw)
                regenerate = parsed.command == 'regenerate'
                if parsed.command:
                    parsed, skip = self._dispatch_command(parsed, history)
                    if skip:
                        continue
                documents, meta_data = self._prepare_turn_documents(
                    parsed, history, raw, regenerate=regenerate,
                )
                if not documents:
                    continue
                if regenerate:
                    documents['regenerate'] = True
                self.session.renderer.live_stream(documents, meta_data)
        except (KeyboardInterrupt, EOFError):
            sys.exit()

def seed_from_string(user_input: str) -> int:
    """ generate a valid 32bit int based on incoming text """
    return int.from_bytes(hashlib.sha256(user_input.encode('utf-8')).digest()[:4], 'big')

def verify_args(p_args):
    """ verify arguments are correct """
    # The issue added to the feature tracker: nothing to verify yet
    return p_args

def _default_lookup(defaults, use_defaults):
    """Return ChatOptions field lookup, or SUPPRESS for the YAML pre-parse."""
    if use_defaults:
        return lambda name: getattr(defaults, name)
    return lambda _name: argparse.SUPPRESS


def _add_llm_group(parser, title, prefix, dests, D):
    """Add the standard --*-llm/--*-server/--*-temp/--*-top_p group.

    dests maps keys 'llm', 'host', 'temp', 'topp' to ChatOptions field names.
    """
    group = parser.add_argument_group(title)
    group.add_argument(
        f'--{prefix}-llm', metavar='', dest=dests['llm'], type=str,
        default=D(dests['llm']), help='Model (default: %(default)s)',
    )
    group.add_argument(
        f'--{prefix}-server', metavar='', dest=dests['host'], type=str,
        default=D(dests['host']), help='Server address (default: %(default)s)',
    )
    group.add_argument(
        f'--{prefix}-temp', metavar='', dest=dests['temp'], type=float,
        default=D(dests['temp']), help='Temperature (default: %(default)s)',
    )
    group.add_argument(
        f'--{prefix}-top_p', metavar='', dest=dests['topp'], type=float,
        default=D(dests['topp']), help='top_p (default: %(default)s)',
    )
    return group


def _add_core_model_args(parser, D):
    """Story, preconditioner, and embedding CLI flags (irregular dest names)."""
    story = parser.add_argument_group('Story Model Options')
    story.add_argument('--model', metavar='', default=D('model'),
                       help='Model (default: %(default)s)')
    story.add_argument('--model-server', metavar='', dest='host', type=str,
                       default=D('host'), help='Server address (default: %(default)s)')
    story.add_argument('--model-temp', metavar='', type=float, default=D('model_temp'),
                       help='Temperature (default: %(default)s)')
    story.add_argument('--model-top_p', metavar='', dest='model_topp', type=float,
                       default=D('model_topp'), help='top_p (default: %(default)s)')

    pre_model = parser.add_argument_group(
        'Preconditioner Model (lightweight model) Options',
    )
    pre_model.add_argument('--pre-llm', metavar='', dest='preconditioner', type=str,
                           default=D('preconditioner'),
                           help='Model (default: %(default)s)')
    pre_model.add_argument('--pre-server', metavar='', dest='pre_host', type=str,
                           default=D('pre_host'),
                           help='Server address (default: %(default)s)')
    pre_model.add_argument('--pre-temp', metavar='', type=float, default=D('pre_temp'),
                           help='Temperature (default: %(default)s)')
    pre_model.add_argument('--pre-top_p', metavar='', type=float, dest='pre_topp',
                           default=D('pre_topp'), help='top_p (default: %(default)s)')

    embedding_model = parser.add_argument_group('Embedding Model Options')
    embedding_model.add_argument(
        '--embedding-llm', metavar='', dest='embeddings', type=str,
        default=D('embeddings'), help='Model (default: %(default)s)',
    )
    embedding_model.add_argument(
        '--embedding-server', metavar='', dest='emb_host', type=str,
        default=D('emb_host'), help='Server address (default: %(default)s)',
    )

    rerank = parser.add_argument_group(
        'Rerank Model Options (optional RAG cross-encoder)',
    )
    rerank.add_argument(
        '--rerank-llm', metavar='', dest='rerank_llm', type=str,
        default=D('rerank_llm'),
        help='Optional reranker (default: %(default)s). Blank = skip.',
    )
    rerank.add_argument(
        '--rerank-server', metavar='', dest='rerank_host', type=str,
        default=D('rerank_host'),
        help='Rerank server (default: inherits --model-server)',
    )
    rerank.add_argument(
        '--rerank-timeout', metavar='', dest='rerank_timeout', type=float,
        default=D('rerank_timeout'),
        help='Seconds to wait on /v1/rerank (default: %(default)s)',
    )


def _add_optional_model_args(parser, D):
    """Optional orchestrated models (polisher through structured)."""
    polisher = _add_llm_group(
        parser, 'Polisher Model Options (optionally polish output)', 'polisher',
        {'llm': 'polisher_llm', 'host': 'polisher_host',
         'temp': 'polisher_temp', 'topp': 'polisher_topp'},
        D,
    )
    polisher.add_argument(
        '--polisher-cnt', metavar='', default=D('polisher_cnt'),
        help='The number of passes to polish final content (default: %(default)s)\n'
             'Warning: Models tend to balloon out of proportions. Start low.',
    )
    groups = (
        ('NSFW Model Options (optional)', 'nsfw',
         {'llm': 'nsfw_llm', 'host': 'nsfw_host',
          'temp': 'nsfw_temp', 'topp': 'nsfw_topp'}),
        ('NPC Character Creation Model Options (optional)', 'entity',
         {'llm': 'entity_llm', 'host': 'entity_host',
          'temp': 'entity_temp', 'topp': 'entity_topp'}),
        ('Vision Model Options (optional)', 'vision',
         {'llm': 'vision_llm', 'host': 'vision_host',
          'temp': 'vision_temp', 'topp': 'vision_topp'}),
        ('Agentic Model Options (optional)', 'agent',
         {'llm': 'agent_llm', 'host': 'agent_host',
          'temp': 'agent_temp', 'topp': 'agent_topp'}),
        ('Casual Model Options (optional)', 'casual',
         {'llm': 'casual_llm', 'host': 'casual_host',
          'temp': 'casual_temp', 'topp': 'casual_topp'}),
        ('General Model Options (optional)', 'general',
         {'llm': 'general_llm', 'host': 'general_host',
          'temp': 'general_temp', 'topp': 'general_topp'}),
        ('Coder Model Options (optional)', 'coder',
         {'llm': 'coder_llm', 'host': 'coder_host',
          'temp': 'coder_temp', 'topp': 'coder_topp'}),
        ('Analysis/Reasoning Model Options (optional)', 'structured',
         {'llm': 'structured_llm', 'host': 'structured_host',
          'temp': 'structured_temp', 'topp': 'structured_topp'}),
    )
    for title, prefix, dests in groups:
        _add_llm_group(parser, title, prefix, dests, D)


def _add_user_and_api_args(parser, D):
    """User identity and API key flags."""
    user_args = parser.add_argument_group('User Options')
    user_args.add_argument('--name', metavar='', default=D('name'), type=str,
                           help="Your assistant's name (default: %(default)s)")
    user_args.add_argument('--user-name', metavar='', default=D('user_name'),
                           type=str,
                           help="Your character's name (default: %(default)s)")
    user_args.add_argument('--sex', metavar='', default=D('sex'), type=str,
                           help="Your character's sex (helps with pronouns) "
                                '(default: %(default)s)')
    user_args.add_argument(
        '--character-sheet', metavar='', default=D('character_sheet'), type=str,
        help='Your character sheet (default: %(default)s)',
    )
    user_args.add_argument('--time-zone', metavar='', default=D('time_zone'),
                           type=str,
                           help="Your assistant's time zone (default: %(default)s)")

    api_args = parser.add_argument_group('API / Service Options')
    api_args.add_argument('--api-key', metavar='', default=D('api_key'), type=str,
                          help='Your API Key (default: REDACTED)')
    api_args.add_argument('--tavily-key', metavar='', default=D('tavily_key'),
                          type=str, help='Your Tavily API Key (default: REDACTED)')
    api_args.add_argument(
        '--sd-server', metavar='', dest='sd_server', type=str,
        default=D('sd_server'),
        help='Automatic1111 URL (http://host:7860). Blank disables.',
    )
    api_args.add_argument(
        '--sd-model', metavar='', dest='sd_model', type=str,
        default=D('sd_model'),
        help='Automatic1111 checkpoint title. Blank keeps whatever A1111 has loaded.',
    )


def _add_context_args(parser, D):
    """RAG, history, and import flags."""
    context_args = parser.add_argument_group('Context / RAG / History Options')
    context_args.add_argument(
        '--rag-matches', metavar='', dest='matches', type=int,
        default=D('matches'),
        help="Number of results to pull from *each* RAG (USER's, and AI's)\n"
             '(default: %(default)s)',
    )
    context_args.add_argument(
        '--history-sessions', metavar='', type=int,
        default=D('history_sessions'),
        help='Chat history responses available in context\n'
             '(default: %(default)s)',
    )
    context_args.add_argument(
        '--unmolested-sessions', metavar='', type=int,
        default=D('unmolested_sessions'),
        help='Chat history responses available before staggering occurs.\n'
             'Set to 0 to disable (default: %(default)s)',
    )
    context_args.add_argument(
        '--lookback', metavar='', type=int, default=D('lookback'),
        help='Max turns to use when computing unmolested-sessions.\n'
             'A value of None = exponential decay of entire history\n'
             '(default: %(default)s)',
    )
    context_args.add_argument(
        '--history-dir', metavar='', dest='vector_dir', type=str,
        default=D('vector_dir'),
        help='Your RAG and Chat History directory. It is dynamically\n'
             'generated and therefore safe to delete if you wish to physically\n'
             'remove all past information.\n(default: %(default)s)',
    )

    import_args = parser.add_argument_group('Import Options')
    gold = (
        'Use --assistant-mode to populate the assistant GOLD RAG.'
    )
    import_args.add_argument('--import-pdf', metavar='', type=str,
                             help='Path to PDF to pre-populate GOLD RAG.\n' + gold)
    import_args.add_argument('--import-txt', metavar='', type=str,
                             help='Path to TXT to pre-populate GOLD RAG.\n' + gold)
    import_args.add_argument('--import-web', metavar='', type=str,
                             help='URL to pre-populate GOLD RAG.\n' + gold)
    import_args.add_argument(
        '--import-dir', metavar='', type=str,
        help='Path to recursively find and import assorted files\n'
             '(*.md, *.html, *.txt, *.pdf, *.py).\n'
             'Use --assistant-mode to populate the assistant GOLD RAG\n'
             'with *.* file patterns.',
    )


def _add_runtime_args(parser, D, use_defaults: bool):
    """Interface, behavior, generation, display, agent, and debug flags."""
    ui_args = parser.add_argument_group('Interface Options')
    ui_args.add_argument('--light-mode', action='store_true',
                         default=D('light_mode'),
                         help='Use a color scheme suitable for light-background terminals.')
    ui_args.add_argument('--spur', action='store_true',
                         default=False if use_defaults else argparse.SUPPRESS,
                         help='Open the Spur browser UI (one process: adapter + built UI).')
    ui_args.add_argument('--spur-rebuild', action='store_true',
                         default=False if use_defaults else argparse.SUPPRESS,
                         help='Force a rebuild of the Spur UI before serving.')
    ui_args.add_argument('--serve', action='store_true',
                         default=False if use_defaults else argparse.SUPPRESS,
                         help='With --spur: bind 0.0.0.0 so an iPad on the LAN can open Spur.')
    ui_args.add_argument('-v', '--verbose', action='store_true',
                         default=D('verbose'),
                         help='Do not hide what the model is thinking\n'
                              '(if the model supports thinking).')

    behavior_args = parser.add_argument_group('Behavior Options')
    behavior_args.add_argument(
        '--assistant-mode', action='store_true', default=D('assistant_mode'),
        help='Switch to Assistant Mode',
    )
    behavior_args.add_argument(
        '--disable-thinking', action='store_true',
        default=D('disable_thinking'),
        help='Do not utilize reasoning, even if the model supports it.\n'
             '(default: %(default)s)',
    )
    behavior_args.add_argument(
        '--no-rags', action='store_true', dest='no_rags',
        default=D('no_rags'),
        help='Disable RAG retrieve/store. Tagging and model routing still run.\n'
             '(default: %(default)s)',
    )
    behavior_args.add_argument(
        '--use-rags', action='store_true', dest='use_rags',
        default=argparse.SUPPRESS,
        help=argparse.SUPPRESS,
    )

    generation_args = parser.add_argument_group('Generation Options')
    generation_args.add_argument(
        '--repeat-penalty', metavar='', type=float, default=D('repeat_penalty'),
        help='Model repeat penalty (default: %(default)s)',
    )
    generation_args.add_argument(
        '--frequency-penalty', metavar='', type=float,
        default=D('frequency_penalty'),
        help='Model frequency penalty (default: %(default)s)',
    )
    generation_args.add_argument(
        '--presence-penalty', metavar='', type=float,
        default=D('presence_penalty'),
        help='Model presence penalty (default: %(default)s)',
    )
    generation_args.add_argument('--seed', metavar='', type=str, default=D('seed'),
                                 help='Model(s) seed (default: %(default)s)')
    generation_args.add_argument(
        '--context-window', metavar='', type=int, default=D('context_window'),
        help='Does nothing except beautify the color map of "context".\n'
             'Enter the maximum context window set on the server.\n'
             '(default: %(default)s)',
    )
    generation_args.add_argument(
        '--completion-tokens', metavar='', dest='completion_tokens',
        type=int, default=D('completion_tokens'),
        help='The maximum tokens the LLM can respond with\n'
             '(default: %(default)s)',
    )

    display_args = parser.add_argument_group('Display / Formatting Options')
    display_args.add_argument(
        '--syntax-style', metavar='', dest='syntax_theme', type=str,
        default=D('syntax_theme'),
        help='Your desired syntax-highlight theme (default: %(default)s).\n'
             'See https://pygments.org/styles/ for available themes.',
    )

    agent_args = parser.add_argument_group('Agentic Tool Options')
    agent_args.add_argument(
        '--distrust-confidence', metavar='', type=float,
        default=D('distrust_confidence'),
        help="How much do you distrust the model's self-assessment.\n"
             'Lower = fewer searches; higher = more searches.\n'
             '(0.0 = never, 1.0 = always; default: %(default)s)',
    )

    debug_args = parser.add_argument_group('Debugging Options')
    debug_args.add_argument('-d', '--debug', action='store_true',
                            default=D('debug'),
                            help='Print preconditioning message, prompt, etc.')


def _add_arguments(parser: argparse.ArgumentParser, defaults, *, use_defaults: bool) -> None:
    """Register all CLI options. If use_defaults=False, suppress defaults."""
    lookup = _default_lookup(defaults, use_defaults)
    _add_core_model_args(parser, lookup)
    _add_optional_model_args(parser, lookup)
    _add_user_and_api_args(parser, lookup)
    _add_context_args(parser, lookup)
    _add_runtime_args(parser, lookup, use_defaults)

def parse_args(argv, yaml_opts):
    """Two-stage parse so help shows effective defaults: CLI > YAML > dataclass."""
    about = """
 Terminal-native AI chat with dynamic RAG-powered memory. Your conversations,
 lore, and context stay organized automatically — so your LLM never forgets.

 Navigate to https://github.com/milljm/dynamic-rag-chat for more information.
"""
    epilog = f"""
example:
Story Mode:
  ./{os.path.basename(__file__)} --model gemma3-27b \\
     --pre-llm gemma3-1b \\
     --embedding-llm nomic-embed-text \\
     --model-server http://localhost:11434/v1

Assistant Mode:
  ./{os.path.basename(__file__)} --assistant-mode \\
     --model gemma3-27b \\
     --pre-llm gemma3-1b \\
     --embedding-llm nomic-embed-text \\
     --model-server http://localhost:11434/v1

Spur (browser UI, same flags):
  ./{os.path.basename(__file__)} --spur
  ./{os.path.basename(__file__)} --spur --assistant-mode
  ./{os.path.basename(__file__)} --spur --serve

That's a lot of available options! The only required options are --model,
--pre-llm, and --embedding-llm. Everything else is optional.

Store your options in `.chat.yaml` to run `./chat.py` without command-line
arguments. See `.chat.yaml.example` for details.
"""

    # -------- Stage 1: pre-parse (suppress defaults, ignore -h) --------
    pre = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    _add_arguments(pre, yaml_opts, use_defaults=False)   # no defaults => only capture user-supplied
    partial, _ = pre.parse_known_args(argv)
    merged = asdict(yaml_opts)
    merged.update({k: v for k, v in vars(partial).items() if v is not None})

    # pylint: disable-next=protected-access   # I must do so...
    _opts = ChatOptions._build(current_dir, merged, yaml_opts)

    # -------- Stage 2: real parser with merged defaults --------
    parser = argparse.ArgumentParser(
        description=about,
        epilog=epilog,
        formatter_class=CustomWidthFormatter,
        allow_abbrev=False,
    )
    # give it a proper -h/--help
    # _add_arguments(parser, argparse.Namespace(**merged), use_defaults=True)
    _add_arguments(parser, _opts, use_defaults=True)

    return verify_args(parser.parse_args(argv))

if __name__ == '__main__':
    opts = ChatOptions.from_yaml(current_dir)
    args = parse_args(sys.argv[1:], opts)
    if getattr(args, 'spur', False) or getattr(args, 'serve', False):
        from spur_launch import launch as _launch_spur
        sys.exit(_launch_spur(sys.argv[1:]))
    _opts = ChatOptions.from_args(current_dir, args, opts)
    dark_rich_142_styles = Theme({
            'markdown.h1': 'bold #FFFFFF',
            'markdown.h2': 'bold #CCCCCC',
            'markdown.h3': 'bold #999999',
            'markdown.h4': 'italic #777777',
            'markdown.h5': '#555555',
            'markdown.h6': '#333333',
            'markdown.item.bullet': 'yellow',
            'markdown.hr': 'yellow',
            'markdown.table.header': 'bold white',
            'markdown.table.border': 'bright_black',
        })
    light_rich_142_styles = Theme({
            'markdown.h1': 'bold #000000',
            'markdown.h2': 'bold #333333',
            'markdown.h3': 'bold #666666',
            'markdown.h4': 'bold italic #888888',
            'markdown.h5': 'italic #888888',
            'markdown.h6': '#888888',
            'markdown.item.bullet': 'dark_orange',
            'markdown.hr': 'dark_orange',
            'markdown.table.header': 'bold black',
            'markdown.table.border': 'bright_black',
            'markdown.code': 'black on #e6e6e6',
        })

    console = Console(theme=light_rich_142_styles if _opts.light_mode else dark_rich_142_styles)
    _opts.seed = seed_from_string(_opts.seed)
    session = SessionContext.from_args(console, _opts)
    import_data = ImportData(session)
    try:
        if args.import_txt:
            if os.path.exists(args.import_txt):
                import_data.store_text(args)
            else:
                print(f'Error: The file at {args.import_txt} does not exist.')
                sys.exit(1)
        if args.import_pdf:
            if os.path.exists(args.import_pdf):
                import_data.extract_text_from_pdf(args)
            else:
                print(f'Error: The file at {args.import_pdf} does not exist.')
                sys.exit(1)
        if args.import_web:
            import_data.extract_text_from_web(args)
            sys.exit(0)
        if args.import_dir:
            if os.path.exists(args.import_dir):
                import_data.extract_text_from_dir(args)
                sys.exit(0)
        chat = Chat(session, _opts)
        chat.chat()
    except KeyboardInterrupt:
        sys.exit()
