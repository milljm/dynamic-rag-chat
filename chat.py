#!/usr/bin/env python3
""" Chat Main executable/entry point """
# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "langchain==0.3.24",
#     "langchain-core==0.3.59",
#     "langchain_ollama==0.3.2",
#     "langchain_openai==0.3.16",
#     "langchain_chroma==0.2.3",
#     "langchain-community==0.3.23",
#     "chromadb==0.6.3",
#     "pydantic==2.11.3",
#     "posthog==4.0.0",
#     "prompt_toolkit",
#     "rich",
#     "rank_bm25",
#     "pypdf",
#     "pytz",
#     "pillow"
#     "requests",
#     "beautifulsoup4",
#     "pygments",
#     "jinja2",
#     "duckduckgo-search",
#     "ddgs",
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
import glob
import hashlib
from dataclasses import dataclass, asdict
from copy import deepcopy
from typing import Dict, Any, List, Optional
import requests
from rich.console import Console
from rich.theme import Theme
from bs4 import BeautifulSoup
from PIL import Image
from prompt_toolkit import PromptSession
from prompt_toolkit.key_binding import KeyBindings
from langchain_community.document_loaders import PyPDFLoader
from src import ContextManager
from src import RAG
from src import RenderWindow
from src import CommonUtils, ChatOptions
from src import ImportData
from src import SceneManager
console = Console(highlight=True)
current_dir = os.path.dirname(os.path.abspath(__file__))

CMD_LINE = re.compile(r"^[ \t]*\\(?P<cmd>[A-Za-z0-9_\-\?]+)(?:[ \t]+(?P<args>.*))?$")
RARE_TOKENS = (r"[RARE NOW]", r"[RARE USED]", r"[RARE RESET]", r"[SAFE MODE]")
RARE_TOKENS_RE = re.compile("|".join(re.escape(t) for t in RARE_TOKENS))
INCLUDE_RE = re.compile(r"\{\{([^}]+)\}\}")  # {{/path}} or {{https://url}}

HELP_TEXT = (
    "in-command switches you can use:\n\n"
    "\t\\regenerate                  - regenerate last turn\n"
    "\t\\no-context msg              - perform a query with no context\n"
    "\t\\agent msg                   - enable agent (web search)\n"
    "\t\\delete-last                 - delete last message from history\n"
    "\t\\turn                        - show turn/status\n"
    "\t\\rewind N                    - rewind to turn N (keep 0..N)\n"
    "\t\\branch NAME@N               - set/fork branch name, if empty list branches;\n"
    "\t                               optional @N to fork from first N turns\n"
    "\t\\dbranch NAME                - delete chat history branch\n"
    "\t\\history [N]                 - show last N user inputs (default 5)\n"
    "\t\\include branch              - include branch as attachment\n"
    "\t\\reset                       - resets history/RAG for current branch\n"
    "\n[bold]context injection[/bold]\n"
    "    {{/absolute/path/to/file}}       - include a file as context\n"
    "    {{https://somewebsite.com/}}     - include URL as context\n"
    "\n[bold]story controls[/bold]\n"
    f"    {', '.join(RARE_TOKENS)}\n"
    "\n[bold]keyboard shortcuts (terminal):[/bold]\n\n"
    "    [yellow]Ctrl-W[/yellow] - delete word left of cursor\n"
    "    [yellow]Ctrl-U[/yellow] - delete everything left of cursor\n"
    "    [yellow]Ctrl-K[/yellow] - delete everything right of cursor\n"
    "    [yellow]Ctrl-A[/yellow] - move to beginning of line\n"
    "    [yellow]Ctrl-E[/yellow] - move to end of line\n"
    "    [yellow]Ctrl-L[/yellow] - clear screen\n"
)

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

    @classmethod
    def from_args(cls, c_console, c_args)->'SessionContext':
        """ instance and return session dataclass """
        _common = CommonUtils(c_console, c_args)
        _scene = SceneManager(console, _common, c_args)
        _rag = RAG(c_console, _common, c_args)
        _context = ContextManager(console, _common, _rag, _scene, current_dir, c_args)
        _renderer = RenderWindow(console, _common, _context, current_dir, c_args)
        return cls(common=_common,
                   rag=_rag,
                   context=_context,
                   renderer=_renderer,
                   scene=_scene)


def parse_user_input(raw: str) -> ParsedInput:
    """ Parse incoming command string and return a ParsedInput dataclass """
    line = raw.strip()

    # 1) Extract a leading command like \rewind 12
    command, c_args = None, ""
    m = CMD_LINE.match(line.splitlines()[0]) if line else None
    if m:
        command = m.group(1).lower()
        c_args = (m.group(2) or "").strip()
        # remove the first line entirely (the \cmd line)
        rest = line.splitlines()[1:]
        line = "\n".join(rest).strip()

    # 2) Extract RARE tokens anywhere in the remaining text
    rare_controls = RARE_TOKENS_RE.findall(line)
    if rare_controls:
        line = RARE_TOKENS_RE.sub("", line).strip()

    # 3) Extract includes like {{/abs/path}} or {{https://...}}
    includes = INCLUDE_RE.findall(line)
    if includes:
        line = INCLUDE_RE.sub("", line).strip()

    return ParsedInput(
        clean_text=line,
        command=command,
        args=c_args,
        rare_controls=rare_controls,
        includes=includes,
    )

def apply_rare_controls(found: List[str], scene: Dict[str, Any]):
    """ RARE Event Handler """
    scene.setdefault('rare_event_pending', False)
    scene.setdefault('rare_event_used', False)
    scene.setdefault('rare_event_cooldown', 0)
    scene.setdefault('safe_mode_turns', 0)

    if ('[RARE NOW]' in found
        and scene.get('safe_mode_turns', 0) == 0
        and scene.get('rare_event_cooldown', 0) == 0):
        scene['rare_event_pending'] = True
    if '[RARE USED]' in found:
        scene['rare_event_used'] = True
        scene['rare_event_pending'] = False
        scene['rare_event_cooldown'] = max(scene.get('rare_event_cooldown', 0), 20)
    if '[RARE RESET]' in found:
        scene.update({
            'rare_event_pending': False,
            'rare_event_used': False,
            'rare_event_cooldown': 0,
            'safe_mode_turns': 0
        })
    if '[SAFE MODE]' in found:
        scene['safe_mode_turns'] = max(scene.get('safe_mode_turns', 0), 30)

def now_addendum(scene: Dict[str, Any]) -> str:
    """ drop in rare event LLM trigger command (literally worded) """
    if scene.get('safe_mode_turns', 0) > 0:
        return ''
    if scene.get('rare_event_pending'):
        return "\nNOW: A rare involuntary event MAY occur this turn." \
        " Otherwise, do not force protagonist actions."
    return ''

class Chat():
    """ Begin initializing variables classes. Call .chat() to begin """
    def __init__(self, o_session, _args):
        self.opts: ChatOptions = _args
        self.session: SessionContext = o_session

        if _args.assistant_mode:
            self.chat_branch = 'assistant'
        else:
            self.chat_branch = self.session.common.load_chat()['current']
        self._initialize_startup_tasks()

    def _initialize_startup_tasks(self):
        """ run startup routines """
        opts = self.opts # shorthand
        if opts.debug:
            console.print('[italic dim grey30]Debug mode enabled. I will re-read the '
                               'prompt files each time.[/]')
        elif opts.prompts_debug:
            console.print('[italic dim grey30]prompts-debug enabled. I will re-read the '
                               'prompt files each time.[/]')
        if opts.assistant_mode and not opts.no_rags:
            console.print('[italic dim grey30]Assistant mode enabled. RAGs disabled'
                          ' (--use-rags to enable).[/]')
        elif opts.no_rags and opts.assistant_mode:
            console.print('[italic dim grey30]Assistant mode enabled.[/]')

    def _branch_exists(self, history, name):
        return name in history and name != 'current'

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

    def get_documents(self, user_input)->tuple[dict,int,int,int,float]:
        """
        Populate documents, the object which is fed to prompt formaters, and
        ultimately is what makes up the context for the LLM
        """
        documents = dict()
        pre_process_time = time.time()
        history = self.session.common.load_chat()
        previous = history[self.chat_branch][-2:-1:]
        documents.update(
            {'user_query'         : user_input,
             'model'              : self.opts.model,
             'dynamic_files'      : '',
             'include_branch'     : '',
             'dynamic_images'     : [],
             'turn_num'           : len(history[self.chat_branch])+1,
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
             'previous'           : previous,
             'history'            : history,
             'entities'           : [],
             'explicit'           : False,
             'qwen_prompts'       : self.qwen_prompt(),
             }
            )

        (documents,
         pre_t,
         post_t) = self.session.context.handle_context(documents)

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
        return documents

    def load_content_as_context(self, user_input: str) -> dict:
        """Parse user_input for all occurrences of {{ /path/to/file }}"""
        documents = {'dynamic_images': [], 'dynamic_files': '', 'user_query': user_input}
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
                if icon == "🖼️":  # Image
                    documents['dynamic_images'].append(data)
                else:
                    documents['dynamic_files'] += f'\n=== {_file} ===\n{data}\n\n'

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
                    icon = "🌍"
                elif mime == 'text':
                    icon = "📄"
                with open(included_file, 'r', encoding='utf-8') as f:
                    data = f.read()
        return data, icon

    def _process_image(self, included_file: str, _format: str) -> tuple:
        """Process image files."""
        icon = "🖼️"
        with Image.open(included_file) as img:
            img = img.convert("RGB")
            buffered = io.BytesIO()
            img.save(buffered, format=_format)
            data = base64.b64encode(buffered.getvalue()).decode("utf-8")
        return icon, data

    def _process_pdf(self, included_file: str) -> tuple:
        """Process PDF files."""
        icon = "📕"
        data = ""
        loader = PyPDFLoader(included_file)
        pages = []
        for page in loader.lazy_load():
            pages.append(page)
        page_texts = list(map(lambda doc: doc.page_content, pages))
        for page_text in page_texts:
            data += page_text
        return icon, data

    def _process_url(self, url: str) -> tuple:
        """Process URL links."""
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
                   'AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'}
        try:
            response = requests.get(url, headers=headers, timeout=10)
            if response.status_code == 200:
                soup = BeautifulSoup(response.content, 'html.parser')
                return soup.get_text(), "🌍"
            return f"Error {response.status_code}", "❌"
        # pylint: disable=bare-except  # too many ways this can go wrong
        except:
            return None, None
        # pylint: enable=bare-except  # too many ways this can go wrong

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
                     'pre_process_time'         : '{:.1f}s'.format(0),
                     'performance'              : '',
                     'light_mode'               : self.set_lightmode_aware(self.opts.light_mode),
                     'llm_prompt'               : '',
                     'prompt_tokens'            : prompt_tokens,
                     'token_savings'            : 0,
                     'cleaned_color'            : cleaned_color,
                     'qwen_prompts'             : self.qwen_prompt(),
                     }
        return documents

    def chat(self):
        """ Prompt the User for questions, and begin! """
        c_session = PromptSession()
        kb = KeyBindings()
        @kb.add('escape', 'enter')
        def _(event):
            buffer = event.current_buffer
            buffer.validate_and_handle()
        @kb.add('enter')
        def _(event):
            buffer = event.current_buffer
            buffer.insert_text('\n')

        console.print('💬 Type message then press [italic red]Esc+Enter[/italic red] to send '
                      r'(multi-line), [red]\? Esc+Enter[/red] for help, '
                      '[italic red]Ctrl+C[/italic red] to quit.\n')

        try:
            while True:
                raw = c_session.prompt(">>> ",
                                       multiline=True,
                                       key_bindings=kb).strip()
                if not raw:
                    continue

                if raw == r'\?':
                    console.print(HELP_TEXT)
                    continue
                history = self.session.common.load_chat()
                parsed = parse_user_input(raw)
                # Global commands that do not call the model:
                if parsed.command:
                    cmd, arg = parsed.command, parsed.args
                    if cmd == "delete-last":
                        try:
                            _ = history[self.chat_branch].pop()
                            self.session.common.save_chat(history)
                            self.session.renderer.clear_ooc()
                            console.print("[green]Deleted last.[/green]", highlight=False)
                        except IndexError:
                            console.print("[yellow]History empty.[/yellow]")
                        continue
                    elif cmd == "turn":
                        console.print(max(1,len(history[self.chat_branch])))
                        continue
                    elif cmd == "rewind":
                        try:
                            n = int(arg)
                            cur = history[self.chat_branch]
                            total = len(cur)
                            if not (1 <= n <= total):
                                console.print(f"[red]usage: \\rewind N  (1 ≤ N ≤ {total})[/red]")
                                continue
                            # keep the first n turns (1-based absolute index)
                            history[self.chat_branch] = cur[:n]
                            self.session.common.save_chat(history)
                            console.print(f"[green]Rewound to turn {n} of {total}.[/green]",
                                           highlight=False)

                            if history[self.chat_branch]:
                                print(f"\n⬇ CURRENT (TURN {len(history[self.chat_branch])}) ⬇\n"
                                    f"{history[self.chat_branch][-1]}")
                            self.session.renderer.clear_ooc()
                        except ValueError:
                            console.print("[red]usage: \\rewind N[/red]")
                        continue
                    elif cmd == "dbranch":
                        if self.opts.assistant_mode:
                            console.print("[red]Cannot manage branches in assistant mode[/red]")
                            continue
                        for branch in history.keys():
                            if branch == 'current':
                                continue
                            if arg == self.chat_branch:
                                console.print("[red]Cannot delete current branch you are on. "
                                              "Use '/reset' instead")
                                break
                            if arg == 'default':
                                console.print("[red]Cannot delete default branch.[/red]")
                                break
                            if arg == 'assistant':
                                console.print("[red]Cannot delete assistant branch.[/red]")
                                break
                            if arg == branch and arg != 'current':
                                history.pop(arg)
                                for path in glob.glob(
                                    f'{self.opts.vector_dir}{os.path.sep}{arg}*'):
                                    if os.path.isdir(path):
                                        console.print(
                                            f'[green]Deleting:[/green] {path}')
                                        shutil.rmtree(path)

                                # Delete Chroma collection corresponding to branch name
                                self.session.rag.delete_collection(arg)
                                self.session.common.save_chat(history)
                                console.print(f"[green]Deleted: [/green]{arg}", highlight=False)
                                break
                        continue
                    elif cmd == "reset":
                        self.session.rag.delete_collection(self.chat_branch)
                        history[self.chat_branch] = []
                        for path in glob.glob(
                            f'{self.opts.vector_dir}{os.path.sep}{self.chat_branch}*'):
                            if os.path.isdir(path):
                                console.print(f'[green]Deleting:[/green] {path}')
                                shutil.rmtree(path)
                        console.print(f"[green]Reset: [/green]{self.chat_branch}",
                                      highlight=False)
                        self.session.common.save_chat(history)
                        continue
                    elif cmd == "branch":
                        if not arg:
                            branches = sorted([k for k in history.keys() if k != 'current'])
                            maxlen = max((len(n) for n in branches), default=0)

                            if self.chat_branch in branches:
                                branches.remove(self.chat_branch)
                                branches.insert(0, self.chat_branch)

                            for name in branches:
                                count = len(history[name])
                                preview = ""
                                if count > 0:
                                    last = history[name][-1].replace("\n", " ")
                                    user = last.find('USER:')
                                    preview = last[user:40+user] + ("…" if len(last) > 40 else "")
                                    preview = f"[dim]{preview}[/dim]"  # <= dim effect

                                if name == self.chat_branch:
                                    console.print(
                                        f"\t➡ [green]{name:<{maxlen}}[/green] : "
                                        f"[{count:>3}] {preview}",
                                        highlight=False
                                    )
                                else:
                                    console.print(
                                        f"\t  {name:<{maxlen}} : [{count:>3}] {preview}",
                                        highlight=False
                                    )
                            continue
                        # parse "name@N" to branch from first N turns of current branch
                        # examples: "\branch testing@5" or just "\branch testing"
                        raw = arg.strip()
                        if self.opts.assistant_mode:
                            console.print("[red]Not allowed while in assistant mode.[/red]",
                                          highlight=False)
                            continue
                        if raw == "current" or raw == "" or raw == "assistant":  # guard
                            console.print("[red]Invalid branch name.[/red]", highlight=False)
                            continue

                        if "@" in raw:
                            name, n_str = raw.split("@", 1)
                            try:
                                cut = int(n_str)
                            except ValueError:
                                console.print("[red]usage: \\branch NAME[@N][/red]",
                                               highlight=False)
                                continue
                        else:
                            name, cut = raw, None

                        # switching if it exists, else create from current up to cut (or full)
                        if self._branch_exists(history, name):
                            if name == self.chat_branch:
                                console.print(f"[green]Already on branch:[/green] {name}",
                                               highlight=False)
                            else:
                                self.chat_branch = name
                                history['current'] = name
                                console.print(f"[green]Switched to :[/green] {name}",
                                               highlight=False)
                            self.session.common.save_chat(history)
                            continue

                        # create new branch from current
                        src = self.chat_branch
                        base = history[src]
                        new_list = deepcopy(self._slice_upto(
                                            base, cut if cut is not None else len(base)))
                        history[name] = new_list
                        history['current'] = name
                        self.chat_branch = name
                        self.session.common.save_chat(history)
                        self.session.renderer.clear_ooc()
                        # ---------------- RAG sync for the new branch ----------------
                        try:
                            if cut is None:
                                # exact fork of current branch's RAG
                                self.session.rag.clone_collection(src, name, overwrite=False)
                            else:
                                # rebuild target RAG from the truncated history we just created
                                self.session.rag.build_collection_from_texts(
                                    name, new_list, overwrite=True)

                            console.print(f"[green]Branched to:[/green] {name}", highlight=False)
                        # pylint: disable=broad-exception-caught
                        except Exception as e:
                        # pylint: enable=broad-exception-caught
                            console.print(f"[red]RAG sync failed for '{name}':[/red] "
                                          f"{e}", highlight=False)
                            # optional: rollback history on failure
                            # history.pop(name, None)
                            # history['current'] = src
                            # self.chat_branch = src
                            # self.session.common.save_chat()
                        # ----------------------------------------------------------------
                        continue
                    elif cmd == "history":
                        try:
                            n = int(arg or "5")
                        except ValueError:
                            n = 5
                        turns = history[self.chat_branch][-n:]
                        total = len(history[self.chat_branch])
                        start = total - len(turns)
                        for _, v in enumerate(turns, start=start+1):
                            print(f'\n\n{v}')
                        continue
                    elif cmd in ("no-context", "include", "agent"):
                        if not self.opts.assistant_mode:
                            console.print("[red]Only available while in assistant mode.[/red]")
                            continue
                    elif cmd in ('regenerate'):
                        try:
                            last = history[self.chat_branch].pop()
                            match = re.findall(r'USER:(.*\n\n)', last)
                            self.session.common.save_chat(history)
                            parsed = parse_user_input(match[0])
                            # pass
                        except IndexError:
                            console.print("[yellow]History empty.[/yellow]")
                            continue
                    else:
                        console.print(f"[red]Unknown command:[/red] \\{cmd}")
                        continue

                # Apply inline RARE controls (toggle flags), never seen by the model
                apply_rare_controls(parsed.rare_controls, self.session.scene.get_scene())

                # Build documents (with or without context)
                if parsed.command in ('no-context', 'agent'):
                    if parsed.command == 'no-context':
                        documents = self.no_context(parsed.args or parsed.clean_text)
                    else:
                        documents = self.get_documents(parsed.args or parsed.clean_text)
                    documents['in_line_commands'] = f'Meta: [{parsed.command}]'
                else:
                    documents = self.get_documents(parsed.clean_text)
                if not documents:
                    console.print("[red]There was an error while running pre-processor work.[/red]"
                                  "In many cases, re-submitting your query again solves the issue.")
                    continue

                if parsed.command == "include":
                    val = parsed.args
                    if val in history:
                        include_branch = ' '.join(history[val][-self.opts.history_sessions:])
                    else:
                        console.print(f"[red]Unknown branch[/red] \\{val}")
                        continue
                    documents['include_branch'] = str(include_branch)

                if parsed.command == "agent":
                    documents['use_agent'] = True
                    documents['agent_ran'] = False

                # Add any inline includes as context (files/URLs)
                if parsed.includes:
                    inc_docs = self.load_content_as_context(
                        " ".join(f"{{{{{x}}}}}" for x in parsed.includes))
                    documents.update(inc_docs)
                    documents['user_query'] = f'{raw} \n\nattachments:{documents["user_query"]}'

                # Inject NOW addendum if armed
                sys_addon = now_addendum(self.session.scene.get_scene())
                if sys_addon:
                    # renderer should append this to system prompt
                    documents['system_addendum'] = sys_addon

                # Handoff to renderer
                # Your renderer should read documents['system_addendum']
                # (if any) and append to the system prompt
                self.session.renderer.live_stream(documents)

        except KeyboardInterrupt:
            sys.exit()
        except EOFError:
            sys.exit()

def seed_from_string(user_input: str) -> int:
    """ generate a valid 32bit int based on incoming text """
    return int.from_bytes(
        hashlib.sha256(user_input.encode("utf-8")).digest()[:4],
        "big"
    )

def verify_args(p_args):
    """ verify arguments are correct """
    # The issue added to the feature tracker: nothing to verify yet
    return p_args

def _add_arguments(parser: argparse.ArgumentParser, defaults, *, use_defaults: bool) -> None:
    """Register all CLI options. If use_defaults=False, suppress defaults (for pre-parse)."""
    D = (lambda name: getattr(defaults, name)) if use_defaults else (lambda _: argparse.SUPPRESS)

    parser.add_argument('--model', metavar='', default=D('model'),
                        help='LLM Model (default: %(default)s)')
    parser.add_argument('--polisher', metavar='', default=D('polisher'),
                        help='Polisher LLM Model (default: %(default)s)'
                        ' (optional, used to "polish" the final output, with something like'
                        ' Midnight Miqu)')
    parser.add_argument('--polisher-cnt', metavar='', default=D('polisher_cnt'),
                        help='The number passes to polish final content (default: %(default)s)'
                        ' Warning: Models tend to ballon out of proportions. Start low.')
    parser.add_argument('--nsfw-model', metavar='', default=D('nsfw_model'),
                        help='NSFW LLM Model (default: %(default)s)')
    parser.add_argument('--pre-llm', metavar='', dest='preconditioner', default=D('preconditioner'),
                        type=str, help='Summarizer/Tagging Preconditioner LLM '
                        '(default: %(default)s)')
    parser.add_argument('--entity-llm', metavar='', dest='entity_llm',
                        default=D('entity_llm'),
                        type=str, help='Entity/Character Sheet LLM (default: %(default)s)')
    parser.add_argument('--embedding-llm', metavar='', dest='embeddings',
                        default=D('embeddings'),
                        type=str, help='LLM Embedding Model (default: %(default)s)')
    parser.add_argument('--agent-llm', metavar='', dest='agent_llm',
                        default=D('agent_llm'),
                        type=str, help='LLM Agent Tooling Model (default: %(default)s)')
    parser.add_argument('--summarizer-llm', metavar='', dest='summarizer_llm',
                        default=D('summarizer_llm'),
                        type=str, help='LLM Agent Tooling Model (default: %(default)s)')
    parser.add_argument('--vision-llm', metavar='', dest='vision_llm',
                        default=D('vision_llm'),
                        type=str, help='LLM Vision Model (default: %(default)s)')

    parser.add_argument('--llm-server', metavar='', dest='host', default=D('host'),
                        type=str, help='OpenAI API server address (default: %(default)s)')
    parser.add_argument('--polisher-server', metavar='', dest='polisher_host',
                        default=D('polisher_host'),
                        type=str, help='OpenAI API server address (default: %(default)s)')
    parser.add_argument('--pre-server', metavar='', dest='pre_host', default=D('pre_host'),
                        type=str, help='OpenAI API server address (default: %(default)s)')
    parser.add_argument('--entity-server', metavar='', dest='entity_host', default=D('entity_host'),
                        type=str, help='OpenAI API server address (default: %(default)s)')
    parser.add_argument('--embedding-server', metavar='', dest='emb_host', default=D('emb_host'),
                        type=str, help='OpenAI API server address (default: %(default)s)')
    parser.add_argument('--agent-server', metavar='', dest='agent_host', default=D('agent_host'),
                        type=str, help='OpenAI API server address (default: %(default)s)')
    parser.add_argument('--summarizer-server', metavar='', dest='summarizer_host',
                        default=D('summarizer_host'),
                        type=str, help='OpenAI API server address (default: %(default)s)')
    parser.add_argument('--vision-server', metavar='', dest='vision_host',
                        default=D('vision_host'),
                        type=str, help='OpenAI API server address (default: %(default)s)')


    parser.add_argument('--api-key', metavar='', default=D('api_key'),
                        type=str, help='You API Key (default: REDACTED)')
    parser.add_argument('--name', metavar='', default=D('name'),
                        type=str, help='Your assistants name (default: %(default)s)')
    parser.add_argument('--user-name', metavar='', default=D('user_name'),
                        type=str, help='Your characters name (default: %(default)s)')
    parser.add_argument('--sex', metavar='', default=D('sex'),
                        type=str, help='Your characters sex (helps with pro-nouns)'
                        ' (default: %(default)s)')
    parser.add_argument('--character-sheet', metavar='', default=D('character_sheet'),
                        type=str, help='Your character sheet (default: %(default)s)')
    parser.add_argument('--time-zone', metavar='', default=D('time_zone'),
                        type=str, help='your assistants name (default: %(default)s)')

    # Chat History/RAG matches
    parser.add_argument('--rag-matches', metavar='', dest='matches', default=D('matches'), type=int,
                        help='Number of results to pull from *each* RAG (there are 3 RAGs)'
                        ' (default: %(default)s)')
    parser.add_argument('--history-sessions', metavar='', default=D('history_sessions'), type=int,
                        help='Chat history responses available in context (overridden by'
                        ' --one-shot if enabled) (default: %(default)s)')
    parser.add_argument('--one-shot', action='store_true', default=D('one_shot'),
                        help='Summarize --history-sessions into single message for one-shot type'
                        ' LLMs.')
    parser.add_argument('--one-shot-history', metavar='', default=D('one_shot_history'), type=int,
                        help='Chat history responses available in context when --one-shot'
                        ' is enabled (default: %(default)s)')

    # imports
    parser.add_argument('--import-pdf', metavar='', type=str,
                        help='Path to pdf to pre-populate GOLD RAG (--assistant-mode to populate '
                        'assistant GOLD RAG)')
    parser.add_argument('--import-txt', metavar='', type=str,
                        help='Path to txt to pre-populate GOLD RAG (--assistant-mode to populate '
                        'assistant GOLD RAG)')
    parser.add_argument('--import-web', metavar='', type=str,
                        help='URL to pre-populate GOLD RAG (--assistant-mode to populate '
                        'assistant GOLD RAG)')
    parser.add_argument('--import-dir', metavar='', type=str,
                        help=('Path to recursively find and import assorted files (*.md *.html, '
                              '*.txt, *.pdf, *.py) (--assistant-mode to populate assistant GOLD '
                              'RAG with *.* file patterns)'))

    # flags (bools)
    parser.add_argument('--light-mode', action='store_true', default=D('light_mode'),
                        help='Use a color scheme suitable for light background terminals')
    parser.add_argument('--assistant-mode', action='store_true', default=D('assistant_mode'),
                        help='Do not utilize story-telling mode prompts or the RAGs. Do not save '
                        'chat history to disk')
    parser.add_argument('--disable-thinking', action='store_true', default=D('disable_thinking'),
                        help='Do not utilize reasoning, even if the model supports it '
                        '(default: %(default)s)')
    parser.add_argument('--use-rags', action='store_true', default=D('no_rags'),
                        help='Use RAGs regardless of assistant-mode (no effect when not also using '
                        'assistant-mode)')
    parser.add_argument('-d', '--debug', action='store_true', default=D('debug'),
                        help='Print preconditioning message, prompt, etc')
    parser.add_argument('-v', '--verbose', action='store_true', default=D('verbose'),
                        help='Do not hide what the model is thinking (if the model supports '
                        'thinking)')
    parser.add_argument('--prompts-debug', action='store_true', default=D('prompts_debug'),
                        help='re-read the prompt files every turn')
    parser.add_argument('--no-think-tag', action='store_true', default=D('no_think_tag'),
                        help='Use this if your model fails to produce a <think> tag before it'
                        ' begins reasoning')

    parser.add_argument('--history-dir', metavar='', dest='vector_dir', default=D('vector_dir'),
                        type=str, help='History directory (default: %(default)s)')

    parser.add_argument('--temperature', metavar='', type=float, default=D('temperature'),
                        help='Model temperature (default: %(default)s)')
    parser.add_argument('--repeat-penalty', metavar='', type=float,
                        default=D('repeat_penalty'),
                        help='Model repeat penalty (default: %(default)s)')
    parser.add_argument('--top-p', metavar='', type=float, default=D('top_p'),
                        help='Model top_p (default: %(default)s)')
    parser.add_argument('--frequency-penalty', metavar='', type=float,
                        default=D('frequency_penalty'),
                        help='Model frequency penalty (default: %(default)s)')
    parser.add_argument('--presence-penalty', metavar='', type=float,
                        default=D('presence_penalty'),
                        help='Model presence penalty (default: %(default)s)')
    parser.add_argument('--seed', metavar='', type=str,
                        default=D('seed'),
                        help='Model(s) seed (default: %(default)s)')

    parser.add_argument('--context-window', metavar='', type=int,
                        default=D('context_window'),
                        help=('Does nothing, except to beautify the color map of "context". '
                              'Input here, what your max context window is being set on the server '
                              '(default: %(default)s)'))

    parser.add_argument('--completion-tokens', metavar='', dest='completion_tokens',
                        default=D('completion_tokens'), type=int,
                        help='The maximum tokens the LLM can respond with (default: %(default)s)')
    parser.add_argument('--syntax-style', metavar='', dest='syntax_theme',
                        default=D('syntax_theme'), type=str,
                        help=('Your desired syntax-highlight theme (default: %(default)s). '
                              'See https://pygments.org/styles/ for available themes'))

def parse_args(argv, yaml_opts):
    """Two-stage parse so help shows effective defaults: CLI > YAML > dataclass."""
    about = """
A tool capable of dynamically creating/instancing RAG
collections using quick 1B parameter summarizers to 'tag'
items of interest that will be fed back into the context
window for your favorite heavy-weight LLM to draw upon.

This allows for long-term memory, and fast relevant
content generation.
"""
    epilog = f"""
example:
  ./{os.path.basename(__file__)} --model gemma3-27b --pre-llm gemma3-1b --embedding-llm nomic-embed-text

Chat can read a .chat.yaml file to import your arguments. See .chat.yaml.example for details.
"""

    # -------- Stage 1: pre-parse (suppress defaults, ignore -h) --------
    pre = argparse.ArgumentParser(add_help=False)
    _add_arguments(pre, yaml_opts, use_defaults=False)   # no defaults => only capture user-supplied
    partial, _ = pre.parse_known_args(argv)
    merged = asdict(yaml_opts)
    merged.update({k: v for k, v in vars(partial).items() if v is not None})

    # -------- Stage 2: real parser with merged defaults --------
    parser = argparse.ArgumentParser(
        description=about,
        epilog=epilog,
        formatter_class=argparse.RawTextHelpFormatter
    )
    # give it a proper -h/--help
    _add_arguments(parser, argparse.Namespace(**merged), use_defaults=True)

    return verify_args(parser.parse_args(argv))

if __name__ == '__main__':
    args = parse_args(sys.argv[1:], ChatOptions.from_yaml(current_dir))
    _opts = ChatOptions.from_args(current_dir, args)
    light_mode_theme = Theme({
            "markdown.code": "black on #e6e6e6",
    })
    if _opts.light_mode:
        console = Console(theme=light_mode_theme)
    _opts.seed = seed_from_string(_opts.seed)
    session = SessionContext.from_args(console, _opts)
    import_data = ImportData(session)
    try:
        if args.import_txt:
            if os.path.exists(args.import_txt):
                import_data.store_text(args.import_txt)
            else:
                print(f'Error: The file at {args.import_txt} does not exist.')
                sys.exit(1)
        if args.import_pdf:
            if os.path.exists(args.import_pdf):
                import_data.extract_text_from_pdf(args.import_pdf)
            else:
                print(f"Error: The file at {args.import_pdf} does not exist.")
                sys.exit(1)
        if args.import_web:
            import_data.extract_text_from_web(args.import_web)
            sys.exit(0)
        if args.import_dir:
            if os.path.exists(args.import_dir):
                import_data.extract_text_from_dir(args.import_dir)
                sys.exit(0)
        chat = Chat(session, _opts)
        chat.chat()
    except KeyboardInterrupt:
        sys.exit()
