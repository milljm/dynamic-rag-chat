#!/usr/bin/env python3
""" Chat Main executable/entry point """
# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "langchain",
#     "langchain-core",
#     "langchain_ollama",
#     "langchain_chroma",
#     "langchain-community",
#     "prompt_toolkit",
#     "rich",
#     "pypdf",
# ]
# ///
import os
import sys
import time
import argparse
import datetime
import yaml
import pytz
from rich.console import Console
from prompt_toolkit import PromptSession
from prompt_toolkit.key_binding import KeyBindings
from context_manager import ContextManager
from ragtag_manager import RAG
from prompt_manager import PromptManager
from render_window import RenderWindow
from chat_utils import CommonUtils
console = Console(highlight=True)
current_dir = os.path.dirname(os.path.abspath(__file__))

# pylint: disable=too-many-instance-attributes
class Chat(PromptManager):
    """ Begin initializing variables classes. Call .chat() to begin """
    def __init__(self, **kwargs):
        super().__init__(console)
        self.debug = kwargs['debug']
        self.host = kwargs['host']
        self.model = kwargs['model']

        self.num_ctx = kwargs['num_ctx']
        self.time_zone = kwargs['time_zone']
        self.common = CommonUtils(console, **kwargs)
        self.renderer = RenderWindow(console, self.common, **kwargs)
        self.prompts = PromptManager(console, debug=self.debug)
        self.cm = ContextManager(console, self.common, **kwargs)

        if self.debug:
            self.console.print('[italic dim grey50]Debug mode enabled. I will re-read the '
                               'prompt files each time[/]\n')
        # Contruct prompts
        self.prompts.build_prompts()

        # Class variables
        self.name = kwargs['name']
        self.verbose = kwargs['verbose']

    @staticmethod
    def get_time(tzone):
        """ return the time """
        mdt_timezone = pytz.timezone(tzone)
        my_time = datetime.datetime.now(mdt_timezone)
        _str_fmt = (f'{my_time.year}-{my_time.month}-{my_time.day}'
                   f':{my_time.hour}:{my_time.minute}:{my_time.second}'
                   f' {"AM" if my_time.hour < 12 else "PM"}')
        return _str_fmt

    def token_counter(self, documents: dict):
        """ report each document token counts """
        for key, value in documents.items():
            yield (key, self.cm.token_retreiver(value))

    def token_manager(self, documents: dict,
                            token_reduction: int)->tuple[int,int]:
        """ Handle token counts and token colors for statistical printing """
        tokens = 0
        for _, token_cnt in self.token_counter(documents):
            tokens += token_cnt

        # Set timers, and completion token counter, colors...
        self.common.heat_map = self.common.create_heatmap(tokens, reverse=True)
        cleaned_color = [v for k,v in self.common.create_heatmap(tokens / 4,
                                                          reverse=True).items()
                            if k<=token_reduction][-1:][0]

        return (tokens, cleaned_color)

    def get_documents(self, user_input)->tuple[dict,int,int,int]:
        """
        Populate documents, the object which is fed to prompt formaters, and
        ultimately is what makes up the context for the LLM
        """
        pre_process_time = time.time()
        (documents,
            pre_t,
            post_t) = self.cm.handle_context([user_input,
                                              self.common.chat_history_session[-10:]])

        # pylint: disable=consider-using-f-string  # no, this is how it is done
        pre_process_time = '{:.1f}s'.format(time.time() - pre_process_time)

        token_savings = max(0, pre_t - post_t)
        documents['llm_prompt'] = self.common.llm_prompt
        documents['user_query'] = user_input
        documents['name'] = self.name
        documents['chat_history'] = self.common.chat_history_session[-10:]
        documents['date_time'] = self.get_time(self.time_zone)
        documents['num_ctx'] = self.num_ctx
        documents['pre_process_time'] = pre_process_time
        # Stringify everything
        for k, v in documents.items():
            documents[k] = self.common.stringify_lists(v)

        # Do heat map stuff
        (prompt_tokens, cleaned_color) = self.token_manager(documents,
                                                            token_savings)

        performance_summary = ''
        for k, v in self.token_counter(documents):
            performance_summary += f'{k}:{v}\n'

        # Supply the LLM with its own performances
        documents['performance'] = (f'Total tokens gathered from all RAG sources: {pre_t}\n'
                                    f'Duplicates Removed: {max(0, pre_t - post_t)}\n'
                                    'Breakdown from each token generating '
                                    f'source:\n{performance_summary}\n')

        return (documents, token_savings, prompt_tokens, cleaned_color)

    def chat(self):
        """ Prompt the User for questions, and begin! """
        session = PromptSession()
        kb = KeyBindings()
        @kb.add('enter')
        def _(event):
            buffer = event.current_buffer
            buffer.insert_text('\n')
        console.print("💬 Press [italic red]Esc+Enter[/italic red] to send"
                      " (multi-line, copy/paste safe) [italic red]Ctrl+C[/italic red]"
                      " to quit.\n")
        try:
            while True:
                user_input = session.prompt(">>> ", multiline=True, key_bindings=kb).strip()
                if not user_input:
                    continue
                if user_input == r'\?':
                    console.print('in-command switches you can use:\n\n\t\\no-context '
                                  '[italic]msg[/italic] (perform a query with no context)')
                    continue

                if user_input.find(r'\no-context') >=0:
                    user_input = user_input.replace('\no-context ', '')
                    (documents,
                    token_savings,
                    prompt_tokens,
                    cleaned_color) = self.get_documents(user_input)
                    documents['chat_history'] = []
                    documents['ai_documents'] = []
                    documents['user_documents'] = []
                    documents['history_documents'] = []
                    prompt_tokens = self.cm.token_retreiver(user_input)
                else:
                    # Grab our lovely context
                    (documents,
                    token_savings,
                    prompt_tokens,
                    cleaned_color) = self.get_documents(user_input)

                # handoff to rich live
                self.renderer.live_stream(documents, token_savings, prompt_tokens, cleaned_color)

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
    matches = int(arg_dict.get('history_matches', 10))
    host = arg_dict.get('server', 'localhost:11434')
    num_ctx = arg_dict.get('context_window', 2048)
    chat_history = arg_dict.get('chat_history_max', 1000)
    name = arg_dict.get('name', 'assistant')
    time_zone = arg_dict.get('time_zone', 'GMT')
    debug = arg_dict.get('debug', False)
    if vector_dir is None:
        vector_dir = os.path.join(current_dir, 'vector_data')
    parser = argparse.ArgumentParser(description=f'{about}',
                                     epilog=f'{epilog}',
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('-m', '--model', default=model, metavar='',
                        help='LLM Model (default: %(default)s)')
    parser.add_argument('-p', '--pre-llm', metavar='', nargs='?', dest='preconditioner',
                        default=preconditioner, type=str,
                        help='pre-processor LLM (default: %(default)s)')
    parser.add_argument('-e', '--embedding-llm', metavar='', nargs='?', dest='embeddings',
                        default=embeddings, type=str,
                        help='LM embedding model (default: %(default)s)')
    parser.add_argument('--history-dir', metavar='', nargs='?', dest='vector_dir',
                        default=vector_dir, type=str,
                        help='history directory (default: %(default)s)')
    parser.add_argument('--history-matches', metavar='', nargs='?', dest='matches',
                        default=matches, type=int,
                        help='Number of results to pull from each RAG (default: %(default)s)')
    parser.add_argument('--server', metavar='', nargs='?', dest='host',
                        default=host, type=str,
                        help='ollama server address (default: %(default)s)')
    parser.add_argument('-n','--name', metavar='', nargs='?', dest='name',
                        default=name, type=str,
                        help='your assistants name (default: %(default)s)')
    parser.add_argument('-t','--time-zone', metavar='', nargs='?', dest='time_zone',
                        default=time_zone, type=str,
                        help='your assistants name (default: %(default)s)')
    parser.add_argument('--chat-history-max', metavar='', nargs='?', dest='chat_max',
                        default=chat_history, type=int,
                        help='Chat history responses to save to disk (default: %(default)s)')
    parser.add_argument('--context-window', metavar='', nargs='?', dest='num_ctx',
                        default=num_ctx, type=int,
                        help='the maximum context window size (default: %(default)s)')
    parser.add_argument('--import-pdf', metavar='', nargs='?', type=str,
                        help='Path to pdf to pre-populate main RAG')
    parser.add_argument('--import-txt', metavar='', nargs='?', type=str,
                        help='Path to txt to pre-populate main RAG')
    parser.add_argument('--debug', action='store_true', default=debug,
                        help='Print preconditioning message, prompt, etc')
    parser.add_argument('-v','--verbose', action='store_true', default=debug,
                        help='Do not hide what the model is thinking (if the model supports'
                        ' thinking)')

    return verify_args(parser.parse_args(argv))

if __name__ == '__main__':
    args = parse_args(sys.argv[1:])
    rag = RAG(console, CommonUtils(console,
                                   vector_dir=args.vector_dir,
                                   chat_max=0), host=args.host,
              embeddings=args.embeddings,
              vector_dir=args.vector_dir,
              debug=args.debug)
    if args.import_txt:
        doc_path = args.import_txt
        if os.path.exists(doc_path):
            with open(doc_path, 'r', encoding='utf-8') as file:
                document_content = file.read()
                rag.store_data(document_content)
            print(f"Document loaded from: {doc_path}")
        else:
            print(f"Error: The file at {doc_path} does not exist.")
        sys.exit()
    if args.import_pdf:
        pdf_file = args.import_pdf
        if os.path.exists(pdf_file):
            rag.extract_text_from_pdf(pdf_file)
        else:
            print(f"Error: The file at {pdf_file} does not exist.")
        sys.exit()
    chat = Chat(**vars(args))
    chat.chat()
