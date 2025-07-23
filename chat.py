#!/usr/bin/env python3
""" Chat Main executable/entry point """
# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "langchain",
#     "langchain-core",
#     "langchain_ollama",
#     "langchain_openai,
#     "langchain_chroma",
#     "langchain-community",
#     "prompt_toolkit",
#     "rich",
#     "rank_bm25",
#     "pypdf",
#     "pytz",
#     "pillow"
#     "requests",
#     "beautifulsoup4",
#     "pygments",
# ]
# ///
import os
import io
import sys
import time
import base64
import argparse
import datetime
import mimetypes
from dataclasses import dataclass
import pytz
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
console = Console(highlight=True)
current_dir = os.path.dirname(os.path.abspath(__file__))

@dataclass
class SessionContext:
    """
    Common Objects used through out the project

    common = CommonUtils
    rag = RAG
    context = ContextManager
    renderer = RenderWindow

    """
    common: CommonUtils
    rag: RAG
    context: ContextManager
    renderer: RenderWindow

    @classmethod
    def from_args(cls, c_console, c_args)->'SessionContext':
        """ instance and return session dataclass """
        _common = CommonUtils(c_console, c_args)
        _rag = RAG(c_console, _common, c_args)
        _context = ContextManager(console, _common, _rag, current_dir, c_args)
        _renderer = RenderWindow(console, _common, _context, current_dir, c_args)
        return cls(common=_common,
                   rag=_rag,
                   context=_context,
                   renderer=_renderer)

class Chat():
    """ Begin initializing variables classes. Call .chat() to begin """
    def __init__(self, o_session, _args):
        self.opts: ChatOptions = _args
        self.session: SessionContext = o_session
        self._initialize_startup_tasks()

    def _initialize_startup_tasks(self):
        """ run starup routines """
        opts = self.opts # shorthand
        if opts.debug:
            console.print('[italic dim grey30]Debug mode enabled. I will re-read the '
                               'prompt files each time.[/]')
        if opts.assistant_mode and not opts.no_rags:
            console.print('[italic dim grey30]Assistant mode enabled. RAGs disabled, Chat '
                               'History will not persist.[/]')
        elif opts.no_rags and opts.assistant_mode:
            console.print('[italic dim grey30]Assistant mode enabled.[/]')

    @staticmethod
    def get_time(tzone: str)->str:
        """ return the time """
        mdt_timezone = pytz.timezone(tzone)
        my_time = datetime.datetime.now(mdt_timezone)
        _str_fmt = (f'{my_time.year}-{my_time.month}-{my_time.day}'
                   f':{my_time.hour}:{my_time.minute}:{my_time.second}'
                   f' {"AM" if my_time.hour < 12 else "PM"}')
        return _str_fmt

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
            yield (key, self.session.context.token_retreiver(value))

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

    def get_documents(self, user_input)->tuple[dict,int,int,int,float]:
        """
        Populate documents, the object which is fed to prompt formaters, and
        ultimately is what makes up the context for the LLM
        """
        pre_process_time = time.time()
        (documents,
         pre_t,
         post_t) = self.session.context.handle_context([user_input])

        # Heat Map
        (prompt_tokens, cleaned_color) = self.token_manager(documents, max(0, pre_t - post_t))

        # Get total token estimate of context
        performance_summary = ''
        for k, v in self.token_counter(documents):
            performance_summary += f'{k}:{v}\n'
        performance = (f'Total Tokens: {prompt_tokens}\n'
                       f'Duplicate Tokens removed: {max(0, pre_t - post_t)}\n'
                       'My maximum Context Window size: '
                       f'{self.opts.completion_tokens}')

        # pylint: disable=consider-using-f-string  # no, {:.f} this is how it is done
        pre_process_time = '{:.1f}s'.format(time.time() - pre_process_time)

        # Fill documents with other useful information used downstream. This is a bit dirty
        # but, some of this information may become useful to provide to the LLM itself. And
        # so we include it here. Each item bellow becomes available within the LLM prompt
        # template system like so:
        #   My name is {name}. The date is {date_time}. It took {pre_process_time}... etc etc
        documents.update(
            {'user_query'       : user_input,
             'dynamic_files'    : '',
             'dynamic_images'   : [],
             'performance'      : performance,
             'history_sessions' : self.opts.history_sessions,
             'name'             : self.opts.name,
             'date_time'        : self.get_time(self.opts.time_zone),
             'completion_tokens': self.opts.completion_tokens,
             'pre_process_time' : pre_process_time,
             'light_mode'       : self.set_lightmode_aware(self.opts.light_mode),
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
                    documents['dynamic_files'] += f'{data}\n\n'

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
            elif _format == 'html':
                icon = "🌍"
            elif mime == 'text':
                icon = "📄"
            else:
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
        prompt_tokens = self.session.context.token_retreiver(user_input) # short hand
        collections = self.session.common.attributes.collections # short hand
        self.session.common.heat_map = self.session.common.create_heatmap(prompt_tokens,
                                                            reverse=True)
        cleaned_color = [v for k,v in
                         self.session.common.create_heatmap(prompt_tokens / 2).items()
                         if k<=0][-1:][0]
        # pylint: disable=consider-using-f-string  # no, this is how it is done
        documents = {'user_query'        : user_input,
                     'name'              : self.opts.name,
                     'chat_history'      : '',
                     'dynamic_files'     : '',
                     'dynamic_images'    : [],
                     'history_sessions'  : self.opts.history_sessions,
                     collections['ai']   : '',
                     collections['user'] : '',
                     collections['gold'] : '',
                     'content_type'      : '',
                     'context'           : '',
                     'date_time'         : self.get_time(self.opts.time_zone),
                     'completion_tokens' : self.opts.completion_tokens,
                     'pre_process_time'  : '{:.1f}s'.format(0),
                     'performance'       : '',
                     'light_mode'        : self.set_lightmode_aware(self.opts.light_mode),
                     'llm_prompt'        : '',
                     'prompt_tokens'     : prompt_tokens,
                     'token_savings'     : 0,
                     'cleaned_color'     : cleaned_color,
                     }
        return documents

    def chat(self):
        """ Prompt the User for questions, and begin! """
        c_session = PromptSession()
        kb = KeyBindings()
        @kb.add('enter')
        def _(event):
            buffer = event.current_buffer
            buffer.insert_text('\n')
        console.print('💬 Press [italic red]Esc+Enter[/italic red] to send (multi-line), '
                      r'[red]\? Esc+Enter[/red] for help, '
                      '[italic red]Ctrl+C[/italic red] to quit.\n')
        try:
            while True:
                user_input = c_session.prompt(">>> ", multiline=True, key_bindings=kb).strip()
                if not user_input:
                    continue
                if user_input == r'\?':
                    console.print('in-command switches you can use:\n\n\t\\no-context '
                                  '[italic]msg[/italic] (perform a query with no context)\n'
                                  '\t{{/absolute/path/to/file}}   - Include a file as context\n'
                                  '\t{{https://somewebsite.com/}} - Include URL as context')
                    continue

                if user_input.find(r'\no-context') >=0:
                    user_input = user_input.replace(r'\no-context ', '')
                    documents = self.no_context(user_input)
                    documents['in_line_commands'] = '\nMeta commands: [no-context, ]'

                else:
                    # Grab our lovely context
                    documents = self.get_documents(user_input)

                # add in-line content
                documents.update(self.load_content_as_context(user_input))

                # handoff to rich live
                self.session.renderer.live_stream(documents)

        except KeyboardInterrupt:  # ctl-c
            sys.exit()
        except EOFError:  # ctl-d
            sys.exit()

def verify_args(p_args):
    """ verify arguments are correct """
    # The issue added to the feature tracker: nothing to verify yet
    return p_args

def parse_args(argv, opts):
    """ parse arguments """
    about = """
A tool capable of dynamically creating/instancing RAG
collections using quick 1B parameter summarizers to 'tag'
items of interest that will be fed back into the context
window for your favorite heavy-weight LLM to draw upon.

This allows for long-term memory, and fast relevent
content generation.
"""
    # pylint: disable=line-too-long
    epilog = f"""
example:
  ./{os.path.basename(__file__)} --model gemma3-27b --pre-llm gemma3-1b --embedding-llm nomic-embed-text

Chat can read a .chat.yaml file to import your arguments. See .chat.yaml.example for details.
    """
    # pylint: enable=line-too-long
    parser = argparse.ArgumentParser(description=f'{about}',
                                     epilog=f'{epilog}',
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--model', default=opts.model, metavar='',
                        help='LLM Model (default: %(default)s)')
    parser.add_argument('--pre-llm', metavar='', dest='preconditioner',
                        default=opts.preconditioner,
                        type=str, help='pre-processor LLM (default: %(default)s)')
    parser.add_argument('--embedding-llm', metavar='', dest='embeddings',
                        default=opts.embeddings,
                        type=str, help='LM embedding model (default: %(default)s)')
    parser.add_argument('--history-dir', metavar='', dest='vector_dir', default=opts.vector_dir,
                        type=str, help='History directory (default: %(default)s)')
    parser.add_argument('--llm-server', metavar='', dest='host', default=opts.host, type=str,
                        help='OpenAI API server address (default: %(default)s)')
    parser.add_argument('--pre-server', metavar='', dest='pre_host', default=opts.pre_host,
                        type=str, help='OpenAI API server address (default: %(default)s)')
    parser.add_argument('--embedding-server', metavar='', dest='emb_host', default=opts.emb_host,
                        type=str, help='OpenAI API server address (default: %(default)s)')
    parser.add_argument('--api-key', metavar='', default=opts.api_key, type=str,
                        help='You API Key (default: REDACTED)')
    parser.add_argument('--name', metavar='', default=opts.name, type=str,
                        help='Your assistants name (default: %(default)s)')
    parser.add_argument('--time-zone', metavar='', default=opts.time_zone,
                        type=str, help='your assistants name (default: %(default)s)')
    parser.add_argument('--history-matches', metavar='', dest='matches', default=opts.matches,
                        type=int,
                        help='Number of results to pull from each RAG (default: %(default)s)')
    parser.add_argument('--history-sessions', metavar='', default=opts.history_sessions,
                        type=int,
                        help='Chat history responses availble in context (default: %(default)s)')
    parser.add_argument('--history-max', metavar='', dest='chat_max', default=opts.chat_history,
                        type=int,
                        help='Chat history responses to save to disk (default: %(default)s)')
    parser.add_argument('--completion-tokens', metavar='', dest='completion_tokens',
                        default=opts.completion_tokens, type=int,
                        help='The maximum tokens the LLM can respond with (default: %(default)s)')
    parser.add_argument('--syntax-style', metavar='', dest='syntax_theme',
                        default=opts.syntax_theme, type=str,
                        help=('Your desired syntax-highlight theme (default: %(default)s).'
                              ' See https://pygments.org/styles/ for available themes'))
    parser.add_argument('--import-pdf', metavar='', type=str,
                        help='Path to pdf to pre-populate main RAG')
    parser.add_argument('--import-txt', metavar='', type=str,
                        help='Path to txt to pre-populate main RAG')
    parser.add_argument('--import-web', metavar='', type=str,
                        help='URL to pre-populate main RAG')
    parser.add_argument('--import-dir', metavar='', type=str,
                        help=('Path to recursively find and import assorted files (*.md *.html, '
                        '*.txt, *.pdf)'))
    parser.add_argument('--light-mode', action='store_true', default=opts.light_mode,
                        help='Use a color scheme suitible for light background terminals')
    parser.add_argument('--assistant-mode', action='store_true', default=opts.assistant_mode,
                        help='Do not utilize story-telling mode prompts or the RAGs. Do not save'
                        ' chat history to disk')
    parser.add_argument('--use-rags', action='store_true', default=opts.no_rags,
                        help='Use RAGs regardless of assistant-mode (no effect when not also using'
                        ' assistent-mode)')
    parser.add_argument('-d', '--debug', action='store_true', default=opts.debug,
                        help='Print preconditioning message, prompt, etc')
    parser.add_argument('-v','--verbose', action='store_true', default=opts.verbose,
                        help='Do not hide what the model is thinking (if the model supports'
                        ' thinking)')

    return verify_args(parser.parse_args(argv))

if __name__ == '__main__':
    args = parse_args(sys.argv[1:], ChatOptions.from_yaml(current_dir))
    _opts = ChatOptions.from_args(current_dir, args)
    light_mode_theme = Theme({
            "markdown.code": "black on #e6e6e6",
    })
    if _opts.light_mode:
        console = Console(theme=light_mode_theme)
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
