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
import re
import os
import sys
import time
import pickle
import argparse
import threading
import datetime
from threading import Thread
import yaml
from langchain.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from rich.live import Live
from rich.markdown import Markdown
from rich.text import Text
from rich.console import Group
from rich.console import Console
from prompt_toolkit import PromptSession
from prompt_toolkit.key_binding import KeyBindings
from ContextManager import ContextManager
from RAGTagManager import RAG
from PromptManager import PromptManager
console = Console(highlight=True)
current_dir = os.path.dirname(os.path.abspath(__file__))

class AnimationThread(Thread):
    """ Allow pulsing animation to run as a thread """
    def __init__(self, owner):
        super().__init__()
        self.owner = owner

    def run(self):
        while self.owner.animation_active:
            self.owner.update_animation()
            time.sleep(0.5)

# pylint: disable=too-many-instance-attributes
class Chat(PromptManager):
    """ Begin initializing variables classes. Call .chat() to begin """
    def __init__(self, **kwargs):
        super().__init__(console)
        self.debug = kwargs['debug']
        self.host = kwargs['host']
        self.model = kwargs['model']
        self.history_dir = kwargs['vector_dir']
        self.num_ctx = kwargs['num_ctx']
        self.prompts = PromptManager(console, debug=self.debug)
        self.cm = ContextManager(console, **kwargs)
        self.llm = ChatOllama(base_url=self.host,
                              model=self.model,
                              temperature=1.0,
                              repeat_penalty=1.1,
                              num_ctx=self.num_ctx,
                              streaming=True)
        self.prompts.build_prompts()
        # Class variables
        self.name = kwargs['name']
        self.verbose = kwargs['verbose']
        self.model_re = re.compile(r'(\w+)\W+')
        self.heat_map = 0
        self.prompt_map = self.create_heatmap(5000)
        self.cleaned_map = self.create_heatmap(1000)
        self.chat_max = kwargs['chat_max']
        self.chat_history_session = self.load_chat(self.history_dir)
        self.llm_prompt = self.load_prompt(self.history_dir)

        # Thinking animation
        self.pulsing_chars = ["⠇", "⠋", "⠙", "⠸", "⠴", "⠦"]
        self.do_once = False
        self.pulse_index = 0
        self.thinking = False
        self.animation_active = False
        self.animation_thread = threading.Thread(target=self.update_animation)

        # self changing prompt
        self.find_prompt = re.compile(r'llm_prompt[\{:"](.*)[\.;"\}]', re.DOTALL)

    @staticmethod
    def create_heatmap(hot_max: int = 0, reverse: bool =False)->dict[int:int]:
        """
        Return a dictionary of ten color ascii codes (values) with the keys representing
        the maximum integer for said color code:
        ./heat_map(10) --> {0: 123, 1: 51, 2: 46, 3: 42, 4: 82, 5: 154,
                            6: 178, 7: 208, 8: 166, 9: 203, 10: 196}
        Options: reverse = True for oppisite effect
        """
        heat = {0: 123} # declare a zero
        colors = [51, 46, 42, 82, 154, 178, 208, 166, 203, 196]
        if reverse:
            colors = colors[::-1]
            heat = {0: 196} # declare a zero
        for i in range(10):
            x = int(((i+1)/10) * hot_max)
            heat[x] = colors[i]
        return heat

    @staticmethod
    def save_chat(history_path, chat_history) ->None:
        """ Persist chat history (save) """
        history_file = os.path.join(history_path, 'chat_history.pkl')
        try:
            with open(history_file, "wb") as f:
                pickle.dump(chat_history, f)
        except FileNotFoundError as e:
            print(f'Error saving chat. Check --history-dir\n{e}')

    def load_chat(self, history_path)->list:
        """ Persist chat history (load) """
        loaded_list = []
        history_file = os.path.join(history_path, 'chat_history.pkl')
        try:
            with open(history_file, "rb") as f:
                loaded_list = pickle.load(f)
                # trunacate to max ammount
                loaded_list = loaded_list[-self.chat_max:]
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
        prompt_file = os.path.join(self.history_dir, 'llm_prompt.pkl')
        try:
            with open(prompt_file, "wb") as f:
                pickle.dump(prompt, f)
        except FileNotFoundError as e:
            print(f'Error saving LLM prompt. Check --history-dir\n{e}')
        return prompt

    @staticmethod
    def load_prompt(history_path)->str:
        """ Persist LLM dynamic prompt (load) """
        prompt_file = os.path.join(history_path, 'llm_prompt.pkl')
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

    def reveal_thinking(self, chunk, show: bool = False):
        """ return thinking chunk if verbose """
        content = chunk.content
        if self.thinking and '</think>' in chunk.content:
            chunk.content = ''
            self.stop_thinking()
        elif not self.thinking and '<think>' in chunk.content:
            self.start_thinking()
            chunk.content = 'AI thinking...'
        elif self.thinking:
            chunk.content = content if show else ''
        return chunk

    def start_thinking(self):
        """ method to start thinking animation """
        if hasattr(self, 'animation_thread') and self.animation_thread.is_alive():
            self.animation_thread.join(timeout=0.1)
        self.thinking = True
        self.do_once = True
        self.animation_active = True
        self.animation_thread = AnimationThread(self)
        self.animation_thread.daemon = True
        self.animation_thread.start()

    def stop_thinking(self):
        """ method to stop thinking animation """
        self.thinking = False
        self.animation_active = False

    def update_animation(self):
        """ a threaded method to run the thinking animation """
        while self.animation_active:
            time.sleep(0.1)  # Adjust speed (0.1 seconds per frame)
            self.pulse_index = (self.pulse_index + 1) % len(self.pulsing_chars)
            self.render_chat()

    # Stream response as chunks
    def stream_response(self, documents: dict):
        """ Parse LLM Prompt """
        prompts = self.prompts
        # pylint: disable=no-member # dynamic prompts (see self.__build_prompts)
        system_prompt = (prompts.get_prompt(f'{prompts.plot_prompt_file}_system.txt')
                     if self.debug else prompts.plot_prompt_system)

        human_prompt = (prompts.get_prompt(f'{prompts.plot_prompt_file}_human.txt')
                     if self.debug else prompts.plot_prompt_human)
        # pylint: enable=no-member
        prompt_template = ChatPromptTemplate.from_messages([
                ("system", system_prompt),
                ("human", human_prompt)
            ])

        if self.debug:
            console.print(f'LLM DOCUMENTS: {documents.keys()}\n{documents["performance"]}\n')
        prompt = prompt_template.format_messages(**documents)
        # pylint: enable=no-member
        if self.debug:
            console.print(f'HEAVY LLM PROMPT (llm.stream()):\n{prompt}\n\n',
                          style='color(233)',
                          highlight=False)
        for chunk in self.llm.stream(prompt):
            chunk = self.reveal_thinking(chunk, self.verbose)
            yield chunk

    def render_footer(self, time_taken: float = 0, **kwargs)->Text:
        """ Handle the footer statistics """
        prompt_tokens = kwargs['prompt_tokens']
        token_count = kwargs['token_count']
        cleaned_color = kwargs['cleaned_color']
        token_savings = kwargs['token_savings']
        # Calculate real-time heat maps
        context_size = [v for k,v in self.prompt_map.items() if k<=prompt_tokens][-1:][0]
        produced = [v for k,v in self.heat_map.items() if token_count>=k][-1:][0]

        # Calculate Tokens/s
        if time_taken > 0:
            tokens_per_second = token_count / time_taken
        else:
            tokens_per_second = 0

        # Implement a thinking emoji
        emoji = f' {self.pulsing_chars[self.pulse_index]} ' if self.thinking else ' '

        # Create the footer text with model info, time, and token count
        model = '-'.join(self.model_re.findall(self.model)[:3])
        footer = Text('\n', style='color(233)')
        footer.append(f'{model}', style='color(202)')
        footer.append(emoji, style='color(51)')
        footer.append('Time:', style='color(233)')
        footer.append(f'{time_taken:.2f}', style='color(94)')
        footer.append('s Tokens(cleaned:', style='color(233)')
        footer.append(f'{token_savings}', style=f'color({cleaned_color})')
        footer.append(' context:', style='color(233)')
        footer.append(f'{prompt_tokens}', style=f'color({context_size})')
        footer.append(' completion:', style='color(233)')
        footer.append(f'{token_count}', style=f'color({produced})')
        footer.append(f') {tokens_per_second:.1f}T/s', style='color(233)')
        return footer

    # Compose the full chat display with footer (model name, time taken, token count)
    def render_chat(self, current_stream: str = "")->Text|Markdown:
        """ render and return markdown/syntax """
        if self.thinking and self.verbose:
            chat_content = Text(current_stream, style='color(233)')
        else:
            chat_content = Markdown(current_stream)

        return chat_content

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
        self.heat_map = self.create_heatmap(tokens, reverse=True)
        cleaned_color = [v for k,v in self.create_heatmap(tokens / 4,
                                                          reverse=True).items()
                            if k<=token_reduction][-1:][0]
        return (tokens, cleaned_color)

    @staticmethod
    def response_count(response)->int:
        """
        Attempt to return a token count in response. Caveats: Some models 'think'
        before responding. Allow this response to not count against the token/s
        performance. Make an assumption: Any return should be considered as 1 token
        at minimum. See the for loop in self.stream_response for details why response
        is empty.
        """
        if response:
            return len(response.split())
        return 1

    def check_prompt(self, last_message):
        """ allow the LLM to add to its own system prompt """
        prompt = self.find_prompt.findall(last_message)[-1:]
        if prompt:
            prompt = self.cm.stringify_lists(prompt)
            prompt = ' '.join(prompt.split()).replace('"', '').replace('\'', '')
            self.llm_prompt = self.save_prompt(prompt)

    def get_documents(self, user_input)->tuple[dict,int,int,int]:
        """
        Populate documents, the object which is fed to prompt formaters, and
        ultimately is what makes up the context for the LLM
        """
        (documents,
            pre_t,
            post_t) = self.cm.handle_context([user_input,
                                              self.chat_history_session[-10:]])
        token_savings = max(0, pre_t - post_t)
        documents['llm_prompt'] = self.llm_prompt
        documents['query'] = user_input
        documents['name'] = self.name
        documents['chat_history'] = self.chat_history_session[-10:]
        utc_time = datetime.datetime.now(datetime.UTC)
        documents['date_time'] = (f'{utc_time.year}-{utc_time.month}-{utc_time.day}'
                                    f':{utc_time.hour}:{utc_time.minute}:{utc_time.second}'
                                    f'.{utc_time.microsecond}')
        documents['num_ctx'] = self.num_ctx
        # Stringify everything
        for k, v in documents.items():
            documents[k] = self.cm.stringify_lists(v)

        # Do heat map stuff
        (prompt_tokens, cleaned_color) = self.token_manager(documents,
                                                            token_savings)

        performance_summary = ''
        for k, v in self.token_counter(documents):
            performance_summary += f'{k}:{v}\n'

        # Supply the LLM with its own performances
        documents['performance'] = (f'Total tokens gathered from all RAG sources: {pre_t}\n'
                                    f'Duplicates Removed: {max(0, pre_t - post_t)}\n'
                                    f'Breakdown from each RAG:\n{performance_summary}\n')

        return (documents, token_savings, prompt_tokens, cleaned_color)

    def live_stream(self, documents: dict,
                          token_savings: int,
                          prompt_tokens: int,
                          cleaned_color: int)->None:
        """ Handle the Rich Live updating process """
        current_response = ''
        footer_meta = {'token_savings' : token_savings,
                       'prompt_tokens' : prompt_tokens,
                       'cleaned_color' : cleaned_color,
                       'token_count'   : 0}

        start_time = time.time()
        query = Markdown(f'**You:** {documents["query"]}\n\n---\n\n')
        with Live(refresh_per_second=20, console=console) as live:
            live.console.clear(home=True)
            live.update(query)
            console.print(f'\nSubmitting {footer_meta["prompt_tokens"]} context tokens to LLM,'
                          ' awaiting repsonse...', style='dim grey37')
            for piece in self.stream_response(documents):
                current_response += piece.content
                footer_meta['token_count'] += self.response_count(piece.content)
                response = self.render_chat(current_response)
                footer = self.render_footer(time.time()-start_time, **footer_meta)
                # create our theme in the following order
                rich_content = Group(query, response, footer)
                # replace 'thinking' output with Mode's Markdown response
                if isinstance(response, Markdown) and self.do_once:
                    self.do_once = False
                    # Reset (erase) the thinking output
                    current_response = ''
                    rich_content = Group(query, response, footer)
                live.update(rich_content)

        # Finish by saving chat history, finding and storing new RAG/Tags
        self.chat_history_session.append(f'DATE TIMESTAMP:{documents["date_time"]}'
                                         f'\nUSER:{documents["query"]}\n'
                                         f'AI:{current_response}\n\n')
        self.cm.handle_context([current_response],
                                direction='store')
        self.save_chat(self.history_dir, self.chat_history_session, )
        self.check_prompt(current_response)

    def chat(self):
        """ Prompt the User for questions, and begin! """
        session = PromptSession()
        kb = KeyBindings()
        @kb.add('enter')
        def _(event):
            buffer = event.current_buffer
            buffer.insert_text('\n')
        console.print("💬 Press [italic grey100]Esc+Enter[/italic grey100] to send"
                      " (multi-line, copy/paste safe) [italic grey100]Ctrl+C[/italic grey100]"
                      " to quit.\n")
        try:
            while True:
                user_input = session.prompt(">>> ", multiline=True, key_bindings=kb).strip()
                if not user_input:
                    continue

                # Grab our lovely context
                (documents,
                 token_savings,
                 prompt_tokens,
                 cleaned_color) = self.get_documents(user_input)

                # handoff to rich live
                self.live_stream(documents, token_savings, prompt_tokens, cleaned_color)

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
    rag = RAG(console, host=args.host,
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
