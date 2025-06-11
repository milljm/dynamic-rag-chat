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
#     "pypdf",
#     "pillow"
#     "requests",
#     "beautifulsoup4",
# ]
# ///
import os
import io
import sys
import time
import base64
import argparse
import datetime
from dataclasses import dataclass
import yaml
import pytz
import requests
from rich.console import Console
from bs4 import BeautifulSoup
from PIL import Image
from prompt_toolkit import PromptSession
from prompt_toolkit.key_binding import KeyBindings
import pypdf # for error handling of PyPDFLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from src import ContextManager
from src import RAG, RAGTagManager
from src import RenderWindow
from src import CommonUtils
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
    rag_tag: RAGTagManager
    context: ContextManager
    renderer: RenderWindow

    @classmethod
    def from_args(cls, c_console, vargs)->'SessionContext':
        """ instance and return session dataclass """
        args_dict = vars(vargs)
        _common = CommonUtils(c_console, **args_dict)
        _rag = RAG(c_console, _common, **args_dict)
        _rag_tag = RAGTagManager(console, _common, **args_dict)
        _context = ContextManager(console, _common, _rag, _rag_tag, current_dir, **args_dict)
        _renderer = RenderWindow(console, _common, _context, current_dir, **args_dict)
        return cls(common=_common,
                   rag=_rag,
                   rag_tag=_rag_tag,
                   context=_context,
                   renderer=_renderer)

class Chat():
    """ Begin initializing variables classes. Call .chat() to begin """
    # pylint: disable=too-many-instance-attributes
    def __init__(self, o_session, **kwargs):
        self.session = o_session
        self.console = console
        self.debug = kwargs['debug']
        self.host = kwargs['host']
        self.model = kwargs['model']
        self.num_ctx = kwargs['num_ctx']
        self.time_zone = kwargs['time_zone']
        self.light_mode = kwargs['light_mode']
        self.name = kwargs['name']
        self.verbose = kwargs['verbose']
        self.chat_sessions = kwargs['chat_sessions']
        if self.debug:
            self.console.print('[italic dim grey30]Debug mode enabled. I will re-read the '
                               'prompt files each time[/]\n')

    @staticmethod
    def get_time(tzone):
        """ return the time """
        mdt_timezone = pytz.timezone(tzone)
        my_time = datetime.datetime.now(mdt_timezone)
        _str_fmt = (f'{my_time.year}-{my_time.month}-{my_time.day}'
                   f':{my_time.hour}:{my_time.minute}:{my_time.second}'
                   f' {"AM" if my_time.hour < 12 else "PM"}')
        return _str_fmt

    @staticmethod
    def set_lightmode_aware(light):
        """ inject a light-mode aware prompt command """
        if light:
            return ('Reminder: The user is using a high luminance background. Therefore, try'
                    ' and only use dark emojis which will provide high-contrast')
        return ('Reminder: The user is using a low luminance background. Therefore, try'
                    ' and only use bright emojis which will provide high-contrast')

    def token_counter(self, documents: dict):
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

        # pylint: disable=consider-using-f-string  # no, this is how it is done
        pre_process_time = '{:.1f}s'.format(time.time() - pre_process_time)

        token_savings = max(0, pre_t - post_t)
        documents.update(
            {'llm_prompt'      : self.session.common.llm_prompt,
             'user_query'      : user_input,
             'dynamic_files'   : '',
             'dynamic_images'  : [],
             'chat_sessions'   : self.chat_sessions,
             'name'            : self.name,
             'date_time'       : self.get_time(self.time_zone),
             'num_ctx'         : self.num_ctx,
             'pre_process_time': pre_process_time,
             'light_mode'      : self.set_lightmode_aware(self.light_mode)}
        )
        # Stringify everything (except images)
        for k, v in documents.items():
            if k in ['dynamic_images']:
                continue
            documents[k] = self.session.common.stringify_lists(v)

        # Do heat map stuff
        (prompt_tokens, cleaned_color) = self.token_manager(documents,
                                                            token_savings)

        performance_summary = ''
        for k, v in self.token_counter(documents):
            performance_summary += f'{k}:{v}\n'

        # Supply the LLM with its own performances
        documents['performance'] = (f'Total Tokens: {prompt_tokens}\n'
                                    f'Duplicate Tokens removed: {max(0, pre_t - post_t)}\n'
                                    f'My maximum Context Window size: {self.num_ctx}')

        return (documents, token_savings, prompt_tokens, cleaned_color, pre_process_time)

    def load_content_as_context(self, user_input):
        """ parse user_input for all occurrences of {{ /path/to/file }} """
        (documents,
        token_savings,
        prompt_tokens,
        cleaned_color,
        pre_process_time) = self.get_documents(user_input)
        included_files = self.session.common.json_template.findall(user_input)
        for included_file in included_files:
            if os.path.exists(included_file):
                if (included_file.lower().endswith('.png')
                    or included_file.lower().endswith('.jpeg')):
                    with Image.open(included_file) as img:
                        buffered = io.BytesIO()
                        if included_file.endswith('.png'):
                            img.save(buffered, format="PNG")
                        elif included_file.endswith('.jpeg'):
                            img.save(buffered, format="JPEG")
                        image_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
                        documents['dynamic_images'].append(image_base64)
                else:
                    with open(included_file, 'r', encoding='utf-8') as f:
                        _tmp = f.read()
                        documents['dynamic_files'] = f'{documents.get("dynamic_files",
                                                                      "")}{_tmp}\n'
                        documents['user_query'] = documents['user_query'].replace(included_file,
                            f'{os.path.basename(included_file)} ✅')
            elif included_file.startswith('http'):
                response = requests.get(included_file, timeout=300)
                if response.status_code == 200:
                    soup = BeautifulSoup(response.content, 'html.parser')
                    documents['dynamic_files'] = (f'{documents.get("dynamic_files", "")}'
                                                  f'{soup.get_text()}\n')
                    documents['user_query'] = documents['user_query'].replace(included_file,
                        f'{included_file} ✅')
            else:
                documents['user_query'] = documents['user_query'].replace(included_file,
                        f'{included_file} ❌')
        return (documents, token_savings, prompt_tokens, cleaned_color, pre_process_time)

    def no_context(self, user_input)->tuple:
        """ perform search without any context involved """
        # pylint: disable=consider-using-f-string  # no, this is how it is done
        documents = {'user_query'      : user_input,
                     'name'            : self.name,
                     'chat_history'    : '',
                     'dynamic_files'   : '',
                     'dynamic_images'  : [],
                     'chat_sessions'   : self.chat_sessions,
                     'ai_documents'    : '',
                     'user_documents'  : '',
                     'context'         : '',
                     'date_time'       : self.get_time(self.time_zone),
                     'num_ctx'         : self.num_ctx,
                     'pre_process_time': '{:.1f}s'.format(0),
                     'performance'     : '',
                     'light_mode'      : self.set_lightmode_aware(self.light_mode),
                     'llm_prompt'      : ''}
        preprocessing = 0
        token_savings = 0
        cleaned_color = 0
        prompt_tokens = self.session.context.token_retreiver(user_input)
        cleaned_color = 0
        self.session.common.heat_map = self.session.common.create_heatmap(prompt_tokens,
                                                            reverse=True)
        cleaned_color = [v for k,v in
                            self.session.common.create_heatmap(prompt_tokens / 2).items()
                            if k<=token_savings][-1:][0]
        return (documents, token_savings, prompt_tokens, cleaned_color, preprocessing)

    def chat(self):
        """ Prompt the User for questions, and begin! """
        c_session = PromptSession()
        kb = KeyBindings()
        @kb.add('enter')
        def _(event):
            buffer = event.current_buffer
            buffer.insert_text('\n')
        self.console.print('💬 Press [italic red]Esc+Enter[/italic red] to send (multi-line), '
                      r'[red]\? Esc+Enter[/red] for help, '
                      '[italic red]Ctrl+C[/italic red] to quit.\n')
        try:
            while True:
                user_input = c_session.prompt(">>> ", multiline=True, key_bindings=kb).strip()
                if not user_input:
                    continue
                if user_input == r'\?':
                    self.console.print('in-command switches you can use:\n\n\t\\no-context '
                                  '[italic]msg[/italic] (perform a query with no context)\n'
                                  '\t{{/absolute/path/to/file}}   - Include a file as context\n'
                                  '\t{{https://somewebsite.com/}} - Include URL as context')
                    continue

                if user_input.find(r'\no-context') >=0:
                    user_input = user_input.replace('\no-context ', '')
                    (documents,
                     token_savings,
                     prompt_tokens,
                     cleaned_color,
                     preprocessing) = self.no_context(user_input)

                if self.session.common.json_template.findall(user_input):
                    (documents,
                    token_savings,
                    prompt_tokens,
                    cleaned_color,
                    preprocessing) = self.load_content_as_context(user_input)

                else:
                    # Grab our lovely context
                    (documents,
                    token_savings,
                    prompt_tokens,
                    cleaned_color,
                    preprocessing) = self.get_documents(user_input)

                # handoff to rich live
                self.session.renderer.live_stream(documents,
                                                  token_savings,
                                                  prompt_tokens,
                                                  cleaned_color,
                                                  preprocessing)

        except KeyboardInterrupt:
            sys.exit()

def verify_args(p_args):
    """ verify arguments are correct """
    # The issue added to the feature tracker: nothing to verify yet
    return p_args

# pylint: disable=too-many-locals  # would force the user to use unfriendly names
def parse_args(argv):
    """ parse arguments """
    about = """
A tool capable of dynamically creating/instancing RAG
collections using quick 1B parameter summarizers to 'tag'
items of interest that will be fed back into the context
window for your favorite heavy-weight LLM to draw upon.

This allows for long-term memory, and fast relevent
content generation.
"""
    epilog = f"""
example:
  ./{os.path.basename(__file__)} -m gemma3-27b -p gemma3-1b -e nomic-embed-text

Chat can read a .chat.yaml file to import your arguments.
See .chat.yaml.example for details.
    """
    # Allow loading a users options pre-set in a .chat.yaml file
    rc_file = os.path.join(current_dir, '.chat.yaml')
    options = {}
    if os.path.exists(rc_file):
        with open(rc_file, 'r', encoding='utf-8') as f:
            options = yaml.safe_load(f) or {}
    arg_dict = options.get('chat', {})
    model = arg_dict.get('model', '')
    preconditioner = arg_dict.get('pre_llm', 'gemma-3-1B-it-QAT-Q4_0')
    embeddings = arg_dict.get('embedding_llm', 'nomic-embed-text-v1.5.f16')
    vector_dir = arg_dict.get('history_dir', None)
    matches = int(arg_dict.get('history_matches', 5)) # 5 from each RAG (User & AI)
    host = arg_dict.get('llm_server', 'http://localhost:11434/v1')
    pre_host = arg_dict.get('pre_server', host)
    emb_host = arg_dict.get('embedding_server', host)
    api_key = arg_dict.get('api_key', None)
    num_ctx = arg_dict.get('context_window', 4192)
    chat_history = arg_dict.get('history_max', 1000)
    history_sessions = arg_dict.get('history_sessions', 5)
    name = arg_dict.get('name', 'assistant')
    time_zone = arg_dict.get('time_zone', 'GMT')
    debug = arg_dict.get('debug', False)
    light = arg_dict.get('light_mode', False)
    if vector_dir is None:
        vector_dir = os.path.join(current_dir, 'vector_data')
    parser = argparse.ArgumentParser(description=f'{about}',
                                     epilog=f'{epilog}',
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--model', default=model, metavar='',
                        help='LLM Model (default: %(default)s)')
    parser.add_argument('--pre-llm', metavar='', dest='preconditioner', default=preconditioner,
                        type=str, help='pre-processor LLM (default: %(default)s)')
    parser.add_argument('--embedding-llm', metavar='', dest='embeddings', default=embeddings,
                        type=str, help='LM embedding model (default: %(default)s)')
    parser.add_argument('--history-dir', metavar='', dest='vector_dir', default=vector_dir,
                        type=str, help='history directory (default: %(default)s)')
    parser.add_argument('--llm-server', metavar='', dest='host', default=host, type=str,
                        help='OpenAI API server address (default: %(default)s)')
    parser.add_argument('--pre-server', metavar='', dest='pre_host', default=pre_host, type=str,
                        help='OpenAI API server address (default: %(default)s)')
    parser.add_argument('--embedding-server', metavar='', dest='emb_host', default=emb_host,
                        type=str, help='OpenAI API server address (default: %(default)s)')
    parser.add_argument('--api-key', metavar='', default=api_key, type=str,
                        help='You API Key (default: %(default)s)')
    parser.add_argument('--name', metavar='', dest='name', default=name, type=str,
                        help='your assistants name (default: %(default)s)')
    parser.add_argument('--time-zone', metavar='', dest='time_zone', default=time_zone, type=str,
                        help='your assistants name (default: %(default)s)')
    parser.add_argument('--history-matches', metavar='', dest='matches', default=matches, type=int,
                        help='Number of results to pull from each RAG (default: %(default)s)')
    parser.add_argument('--history-sessions', metavar='', dest='chat_sessions',
                        default=history_sessions, type=int,
                        help='Chat history responses availble in context (default: %(default)s)')
    parser.add_argument('--history-max', metavar='', dest='chat_max', default=chat_history,
                        type=int,
                        help='Chat history responses to save to disk (default: %(default)s)')
    parser.add_argument('--context-window', metavar='', dest='num_ctx', default=num_ctx, type=int,
                        help='the maximum context window size (default: %(default)s)')
    parser.add_argument('--import-pdf', metavar='', type=str,
                        help='Path to pdf to pre-populate main RAG')
    parser.add_argument('--import-txt', metavar='', type=str,
                        help='Path to txt to pre-populate main RAG')
    parser.add_argument('--import-web', metavar='', type=str,
                        help='URL to pre-populate main RAG')
    parser.add_argument('--import-dir', metavar='', type=str,
                        help='Path to recursively find and import assorted files (*.md *.html)')
    parser.add_argument('--light-mode', action='store_true', default=light,
                        help='Use a color scheme suitible for light background terminals')
    parser.add_argument('-d', '--debug', action='store_true', default=debug,
                        help='Print preconditioning message, prompt, etc')
    parser.add_argument('-v','--verbose', action='store_true', default=debug,
                        help='Do not hide what the model is thinking (if the model supports'
                        ' thinking)')

    return verify_args(parser.parse_args(argv))

def store_chunks(d_session: SessionContext,
                 pages: list[str])->None:
    """ iterate over list, tagging content before storing to RAG """
    for ind, text in enumerate(pages):
        print(f'\t\tmetadata tagging chunk {ind+1}/{len(pages)}')
        _, meta_tags, _  = d_session.context.pre_processor(text)
        _normal = d_session.common.normalize_for_dedup(text)
        d_session.rag.store_data(_normal, tags_metadata=meta_tags)

# pylint: disable=redefined-outer-name #  we exit immediately
def extract_text_from_pdf(d_session: SessionContext,
                          v_args: argparse.ArgumentParser)->None:
    """ Store imported PDF text directly into the RAG """
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=300)
    print(f'Importing document: {v_args.import_pdf}')
    loader = PyPDFLoader(v_args.import_pdf)
    pages = []
    try:
        for page in loader.lazy_load():
            pages.append(page)
        page_texts = list(map(lambda doc: doc.page_content, pages))
        for p_cnt, page_text in enumerate(page_texts):
            if page_text:
                print(f'\tPage {p_cnt+1}/{len(page_texts)}')
                texts = text_splitter.split_text(page_text)
                store_chunks(d_session, texts)
            else:
                print(f'\tPage {p_cnt+1}/{len(page_texts)} blank')
    except pypdf.errors.PdfStreamError as e:
        print(f'Error loading PDF:\n\n\t{e}\n\nIs this a valid PDF?')
        sys.exit(1)
    sys.exit()

def store_text(d_session: SessionContext,
               v_args: argparse.ArgumentParser)->None:
    """ Store imported text file directly into the RAG """
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=300)
    print(f'Importing document: {v_args.import_txt}')
    with open(v_args.import_txt, 'r', encoding='utf-8') as file:
        document_content = file.read()
        texts = text_splitter.split_text(document_content)
        store_chunks(d_session, texts)
    sys.exit()

def extract_text_from_markdown(d_session: SessionContext,
                               v_args: argparse.ArgumentParser)->None:
    """
    walk recursively through provided directory and import *.md, *.html, *.txt
    file directly into the RAG
    """
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=300)
    for fdir, _, files in os.walk(v_args.import_dir):
        for file in files:
            if file.endswith('.md') or file.endswith('.html') or file.endswith('.txt'):
                _file = os.path.join(fdir, file)
                with open(_file, 'r', encoding='utf-8') as file:
                    document_content = file.read()
                if _file.endswith('.html'):
                    soup = BeautifulSoup(document_content, 'html.parser')
                    document_content = soup.get_text()
                print(f'Importing document: {_file}')
                texts = text_splitter.split_text(document_content)
                store_chunks(d_session, texts)
    sys.exit()

def extract_text_from_web(d_session: SessionContext,
                          v_args: argparse.ArgumentParser)->None:
    """ extract plain text from web address """
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=300)
    response = requests.get(v_args.import_web, timeout=300)
    if response.status_code == 200:
        print(f"Document loaded from: {v_args.import_web}")
        soup = BeautifulSoup(response.content, 'html.parser')
        text = soup.get_text()
        texts = text_splitter.split_text(text)
        store_chunks(d_session, texts)
    else:
        print(f'Error obtaining webpage: {response.status_code}\n{response.raw}')
        sys.exit(1)

# pylint: enable=redefined-outer-name
if __name__ == '__main__':
    args = parse_args(sys.argv[1:])
    session = SessionContext.from_args(console, args)
    try:
        if args.import_txt:
            if os.path.exists(args.import_txt):
                store_text(session, args)
            else:
                print(f'Error: The file at {args.import_txt} does not exist.')
                sys.exit(1)
        if args.import_pdf:
            if os.path.exists(args.import_pdf):
                extract_text_from_pdf(session, args)
            else:
                print(f"Error: The file at {args.import_pdf} does not exist.")
                sys.exit(1)
        if args.import_web:
            extract_text_from_web(session, args)
            sys.exit(0)
        if args.import_dir:
            if os.path.exists(args.import_dir):
                extract_text_from_markdown(session, args)
                sys.exit(0)
        chat = Chat(session, **vars(args))
        chat.chat()
    except KeyboardInterrupt:
        sys.exit()
