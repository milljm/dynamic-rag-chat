#!/usr/bin/env python3
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
console = Console(highlight=True)
current_dir = os.path.dirname(os.path.abspath(__file__))

class Chat():
    """ Begin initializing variables classes. Call .chat() to begin """
    def __init__(self, **kwargs):
        for arg, value in kwargs.items():
            setattr(self, arg, value)
        # pylint: disable=no-value-for-parameter
        self.cm = ContextManager(console, **kwargs)

        self.llm = ChatOllama(host=self.host,
                              model=self.model,
                              streaming=True)

        # Class variables
        # TODO: we can generate this much cleaner...
        self.heat_map = {0: 123, 300: 51, 500: 46, 700: 42, 2000: 82, 3000: 154,
                         3500: 178, 4000: 208, 4200: 166, 4500: 203, 4700: 197, 5000: 196}
        self.chat_history_md = ''
        self.chat_history_session = []
        self.build_prompts()

    def build_prompts(self):
        """
        A way to manage a growing number of prompt templates dynamic RAG/Tagging
        might introduce...

          {key : value} pairs become self.key : contents-of-file
          filenaming convention: {value}_system.txt / {value}_human.txt
        """
        if self.debug:
            console.print('[italic dim grey50]Debug mode enabled. I will re-read the '
                          'prompt files each time[/]\n')
        prompt_files = {
            'pre_prompt':  'pre_conditioner_prompt',
            'post_prompt': 'tagging_prompt',
            'plot_prompt': 'plot_prompt'
        }
        for prompt_key, prompt_base in prompt_files.items():
            setattr(self, f'{prompt_key}_file', os.path.join(current_dir, prompt_base))
            setattr(self, f'{prompt_key}_system', self.get_prompt(f'{prompt_base}_system.txt'))
            setattr(self, f'{prompt_key}_human', self.get_prompt(f'{prompt_base}_human.txt'))

    def get_prompt(self, path):
        """ Keep the prompts as files for easier manipulation """
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as prompt:
                return prompt.read()
        else:
            print(f'Prompt not found! I expected to find it at:\n\n\t{path}')
            sys.exit(1)

    # Stream response as chunks
    def stream_response(self, query: str, context: str):
        """ Parse LLM Promp """
        # pylint: disable=no-member # dynamic prompts (see self.__build_prompts)
        sys_prmpt = (self.__get_prompt(f'{self.plot_prompt_file}_system.txt')
                     if self.debug else self.plot_prompt_system)

        hum_prmpt = (self.__get_prompt(f'{self.plot_prompt_file}_human.txt')
                     if self.debug else self.plot_prompt_human)
        # pylint: enable=no-member
        prompt = ChatPromptTemplate.from_messages([
                ("system", sys_prmpt),
                ("human", hum_prmpt)
            ])
        messages = prompt.format_messages(history=' '.join(self.chat_history_session[-3:]),
                                          context=context,
                                          question=query)
        # pylint: enable=no-member
        if self.debug:
            console.print(f'RAW PROMPT (llm.stream()):\n{messages}\n\n', style='color(233)')
        for chunk in self.llm.stream(messages):
            yield chunk

    # Compose the full chat display with footer (model name, time taken, token count)
    def render_chat(self, current_stream: str = "",
                    time_taken: float = 0,
                    token_count: int = 0,
                    prompt_tokens: int = 0,
                    token_reduction: int = 0) -> Group:
        """ render and return markdown/syntax """
        # Create the full chat content using Markdown
        full_md = f'{self.chat_history_md}\n\n{current_stream}'

        # Calculate heat map
        heat = [v for k,v in self.heat_map.items() if k<=prompt_tokens][-1:][0]
        produced = [v for k,v in self.heat_map.items()
                    if max(0, 5000-(token_count*8))>=k][-1:][0]

        # Calculate Tokens/s
        if time_taken > 0:
            tokens_per_second = token_count / time_taken
        else:
            tokens_per_second = 0

        # Create the footer text with model info, time, and token count
        footer = Text('Model: ', style='color(233)')
        footer.append(f'{self.model_name} ', style='color(202)')
        footer.append('| Time: ', style='color(233)')
        footer.append(f'{time_taken:.2f}', style='color(94)')
        footer.append('s | Tokens: Cleaned/', style='color(233)')
        footer.append(f'{token_reduction}', style='color(27)')
        footer.append(' Context/', style='color(233)')
        footer.append(f'{prompt_tokens}', style=f'color({heat})')
        footer.append(' Tokens/', style='color(233)')
        footer.append(f'{token_count}', style=f'color({produced})')
        footer.append(f' ({tokens_per_second:.1f}T/s)', style='color(234)')

        # Render the chat content as Markdown (no panel, just the content)
        chat_content = Markdown(full_md)

        # Return everything as a Group (no borders, just the content)
        return Group(chat_content, footer)

    def chat(self):
        """ Prompt the User for questions, and begin! """
        session = PromptSession()

        # Set up key bindings
        kb = KeyBindings()
        @kb.add('enter')
        def _(event):
            buffer = event.current_buffer
            if buffer.document.text.strip().endswith("\n\n"):
                event.app.exit(result=buffer.document.text.strip())
            else:
                buffer.insert_text('\n')
        console.print("💬 Press [italic grey100]Esc+Enter[/italic grey100] to send"
                      " (multi-line, copy/paste safe) [italic grey100]Ctrl+C[/italic grey100]"
                      " to quit.\n")
        try:
            while True:
                user_input = session.prompt(">>> ", multiline=True, key_bindings=kb).strip()
                if not user_input:
                    continue
                #user_input = input("\n🧠 Ask something (or type 'exit'): ")
                #if user_input.strip().lower() == "exit":
                #    break

                # Grab our lovely context
                (context, token_reduction) = self.cm.handle_context(data=user_input)

                # Gather all prompt tokens, to display as statitics
                prompt_tokens = self.cm.token_retreiver([user_input])
                if self.chat_history_session[-3:]:
                    prompt_tokens = self.cm.token_retreiver(self.chat_history_session[-3:])
                context_tokens = self.cm.token_retreiver(context.split())

                if self.debug:
                    console.print(f'HISTORY:\n{self.chat_history_session[-3:]}\n',
                                  f'HISTORY TOKENS: {prompt_tokens}\n\n',
                                  f'CONTEXT:\n{context}\n',
                                  f'CONTEXT TOKENS: {context_tokens}\n\n',
                                  style='color(233)')

                prompt_tokens += context_tokens
                console.print(f'Process {prompt_tokens} context tokens...', style='dim grey37')

                # Set timers, and completion token counter
                start_time = time.time()
                current_response = ""
                token_count = 0
                with Live(refresh_per_second=20) as live:
                    self.chat_history_md = f"""**You:** *{user_input}*

---
"""
                    #live.update(self.render_chat(self.chat_history_md))
                    for piece in self.stream_response(user_input, context):
                        #print(dir(piece))
                        #if piece.finish_reason == "token_limit":
                        #    completion_count = piece.token_count
                        current_response += piece.content
                        # Update token count (a rough estimate based on the size of the chunk)
                        token_count += len(piece.content.split())
                        live.update(self.render_chat(current_response,
                                                    time.time()-start_time,
                                                    token_count,
                                                    prompt_tokens,
                                                    token_reduction))

                self.chat_history_session.append(current_response)
                self.cm.handle_context(current_response, direction='store')

        except KeyboardInterrupt:
            sys.exit()

def verify_args(p_args):
    """ verify arguments are correct """
    # The issue added to the feature tracker: nothing to verify yet
    return p_args

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
  ./{os.path.basename(__file__)} gemma3-27B
    """
    parser = argparse.ArgumentParser(description=f'{about}',
                                     epilog=f'{epilog}',
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('model', default='',
                         help='Your heavy LLM Model ~27B to whatever you can afford')
    parser.add_argument('--pre-llm', metavar='', nargs='?', dest='preconditioner',
                        default='gemma-3-1b-it-Q4_K_M',
                        type=str, help='1B-2B LLM model for preprocessor work '
                        '(default: %(default)s)')
    parser.add_argument('--embedding_llm', metavar='', nargs='?', dest='embeddings',
                        default='nomic-embed-text-v1.5.Q8_0.gguf:latest',
                        type=str, help='LM embedding model (default: %(default)s)')
    parser.add_argument('--history-dir', metavar='', nargs='?', dest='vector_dir',
                         default=os.path.join(current_dir, 'vector_data'), type=str,
                         help='a writable path for RAG (default: %(default)s)')
    parser.add_argument('--history-matches', metavar='', nargs='?', dest='matches',
                         default=3, type=int,
                         help='Number of results to pull from each RAG (default: %(default)s)')
    parser.add_argument('--server', metavar='', nargs='?', dest='host',
                         default='172.16.155.4:11434', type=str,
                         help='ollama server address (default: %(default)s)')
    parser.add_argument('--import-pdf', metavar='', nargs='?', type=str,
                         help='Path to pdf to pre-populate main RAG')
    parser.add_argument('--import-txt', metavar='', nargs='?', type=str,
                         help='Path to txt to pre-populate main RAG')
    parser.add_argument('--debug', action='store_true', default=False,
                        help='Print preconditioning message, prompt, etc')

    return verify_args(parser.parse_args(argv))

if __name__ == '__main__':
    args = parse_args(sys.argv[1:])
    print(args)
    if args.import_txt:
        doc_path = args.import_txt
        if os.path.exists(doc_path):
            with open(doc_path, 'r', encoding='utf-8') as file:
                document_content = file.read()
                rag = RAG(console, args.server,
                          args.embedding_llm,
                          args.history_dir,
                          args.debug)
                rag.store_data(document_content)
            print(f"Document loaded from: {doc_path}")
        else:
            print(f"Error: The file at {doc_path} does not exist.")
        sys.exit()
    if args.import_pdf:
        pdf_file = args.import_pdf
        if os.path.exists(pdf_file):
            rag = RAG(console, args.server,
                      args.embedding_llm,
                      args.history_dir,
                      args.debug)
            rag.extract_text_from_pdf(pdf_file)
        else:
            print(f"Error: The file at {pdf_file} does not exist.")
        sys.exit()
    chat = Chat(**vars(args))
    chat.chat()
