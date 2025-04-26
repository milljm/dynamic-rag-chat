#!/usr/bin/env python3
""" simple chat with rag """
import os
import sys
import asyncio
import logging
import argparse
import threading
from prompt_toolkit import PromptSession
from prompt_toolkit.key_binding import KeyBindings
from rich.console import Console
from rich.live import Live
from rich.markup import escape
from langchain_chroma import Chroma
from langchain_ollama import ChatOllama
from langchain_ollama import OllamaEmbeddings
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.callbacks.base import AsyncCallbackHandler
from langchain.prompts import ChatPromptTemplate
from richify import richify
logging.getLogger("chromadb").setLevel(logging.ERROR)
current_dir = os.path.dirname(os.path.abspath(__file__))
console = Console()
live = Live(console=console, refresh_per_second=10, vertical_overflow="visible")

#class RichStreamingHandler(AsyncCallbackHandler):
#    """
#    Responsible for pretty console output during streaming.
#    """
#    async def on_llm_new_token(self, token: str, **kwargs):
#        console.print(f"{token}", end="", soft_wrap=True)

class LiveStreamingHandler(AsyncCallbackHandler):
    """
    Responsible for pretty console output during streaming.
    """
    def __init__(self):
        self.buffer = ''

    async def on_llm_new_token(self, token: str, **kwargs):
        pass

    #async def on_llm_new_token(self, token: str, **kwargs):
    #    """ whatever """
    #    self.buffer += token
    #    with Live(Markdown(self.buffer), console=console, auto_refresh=True) as live:
    #        live.update(Markdown(self.buffer))

class DynamicRAGManager():
    """
    Responsible for producing key:value pairs on 'ideas' to return as a naminging convention for
    RAG collection identification.
    """
    def update_rag_async(self, model, url, context, prompt):
        """ regular expression through message and attempt to create key:value tuples """
        pre_llm = OllamModel(model=model, url=url)
        prompt_template = ChatPromptTemplate.from_messages([
            ("system", prompt),
            ("human", '')
        ])
        prompt_template.format_messages(context=context, question='')
        results = pre_llm.query_llm_oneshot(f'{prompt}\n\n{context}', print_footer=False).content
        with open('testing_output.txt', 'w', encoding='utf-8') as f:
            f.write(f'model:\n{model}\n\nresponse:\n{context}\n\nprompt:\n{prompt}\n\nresults:\n{results}')
        return

class VectorData():
    """ Responsible for dealing with vector database operations """
    def __init__(self, vector_dir='',
                 embedding='',
                 url='http://localhost:11434',
                 collection_name='plot',
                 debug=False):
        self.vector_dir = vector_dir
        self.embedding = embedding
        self.url = url
        self.collection_name = collection_name
        self.debug = debug

    def __get_embeddings(self):
        embeddings = OllamaEmbeddings(base_url=self.url, model=self.embedding)
        chroma_db = Chroma(persist_directory=self.vector_dir,
                           embedding_function=embeddings,
                           collection_name=self.collection_name)
        return chroma_db

    def store_data(self, data, chunk_size=1000, chunk_overlap=200):
        """
        Stores chunks into vector data base (1000 chunk size, 200 overlap)
        Syntax:
            store_data(data=str, chunk_size=int, chunk_overlap=int)
        """
        splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size,
                                                  chunk_overlap=chunk_overlap)
        docs = splitter.create_documents([data])
        chroma = self.__get_embeddings()
        chroma.add_documents(docs)

    def retrieve_data(self, query, k=5):
        """
        Return vector data as a list. Syntax:
            retrieve_data(query=str, k=int)->list
              query: your question
              k:     matches to return
        """
        chroma = self.__get_embeddings()
        results = []
        results: list[Document] = chroma.similarity_search(query, k)
        return results

class OllamModel():
    """ Responsible for handling calls to the Ollama LLM """
    def __init__(self, model, url='http://localhost:11434', debug=False):
        self.model = model
        self.url = url
        self.chat_history = []
        self.debug = debug

    def __print_footer(self, response, tight=True):
        """
        Just a little flourish. Any self-respecting user of LLMs
        wants to know this stuff :)
        """
        ms = response.response_metadata['total_duration'] / 1000000
        level = "italic dim grey100" if self.debug else "italic dim grey7"
        g = []
        for k, v in response.usage_metadata.items():
            g.append(f'{k}/{v}')
        if tight:
            model = escape(response.response_metadata["model"])
            tokens = response.usage_metadata["total_tokens"]
            console.print(
            f"[{level}]summarization pre-processor {ms:.2f}/ms: {model} total_tokens:{tokens}[/]"
            )
            console.print('', style='')  # Reset after if needed
        else:
            console.print(f"\n[{level}]{ms:.2f}/ms {' '.join(g)}[/]")

    def query_llm_oneshot(self, query, print_footer=True):
        """ query the llm with a message, without streaming """
        llm = ChatOllama(model=self.model,
                         temperature=0.,
                         base_url=self.url,
                         streaming=False)
        response = llm.invoke(query, stop=["\n\n", "###", "Conclusion"])
        if print_footer:
            self.__print_footer(response, print_footer)
        return response

    async def query_llm(self, query, pre_model=None, pre_url=None, post_prompt=None):
        """ query the llm with a message, stream the message """
        handler = LiveStreamingHandler()
        llm = ChatOllama(model=self.model,
                              temperature=1,
                              base_url=self.url,
                              streaming=True,
                              callbacks=[handler])
        console.print(f'[italic #FF8C00]{self.model}[/italic #FF8C00]:\n')
        response = await llm.ainvoke(query)
        self.chat_history.append(response.content)
        self.__print_footer(response, tight=False)
        # And the magical dynamic RAG creation starts here
        update_rags = DynamicRAGManager()
        threading.Thread(target=update_rags.update_rag_async,
                         args=(pre_model,pre_url,response.content,post_prompt,)).start()

class Chat():
    """
    Entry point. Instances LLMs, RAG, prompts.
    """
    def __init__(self, **kwargs):
        try:
            self.llm_model = kwargs['llm']
            self.preconditioner = kwargs['pre_llm']
            self.embedding_model = kwargs['embedding_llm']
            self.vector_dir = kwargs['history_dir']
            self.history_matches = kwargs['history_matches']
            self.server = f'http://{kwargs["server"]}'
            self.debug = kwargs['debug']
        except KeyError as e:
            print(f'Incorrectly supplied arguments. See --help: {e}')
            sys.exit(1)

        self.llm = OllamModel(self.llm_model, url=self.server, debug=self.debug)
        self.pre_llm = OllamModel(model=self.preconditioner, url=self.server, debug=self.debug)
        self.__build_prompts()

    def __build_prompts(self):
        """ a way to manage a growing number of prompts """
        if self.debug:
            console.print('\n[italic dim grey50]Debug mode enabled. I will re-read the '
                          'prompt files each time[/]')
        prompt_files = {
            'pre_prompt': 'pre_conditioner_prompt',
            'post_prompt': 'tagging_prompt',
            'plot_prompt': 'plot_prompt'
        }
        for prompt_key, prompt_base in prompt_files.items():
            setattr(self, f'{prompt_key}_file', os.path.join(current_dir, prompt_base))
            setattr(self, f'{prompt_key}_system', self.__get_prompt(f'{prompt_base}_system.txt'))
            setattr(self, f'{prompt_key}_human', self.__get_prompt(f'{prompt_base}_human.txt'))

    def __get_prompt(self, path):
        """ Keep the prompts as files for easier manipulation """
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as prompt:
                return prompt.read()
        else:
            print(f'Prompt not found! I expected to find it at:\n\n\t{path}')
            sys.exit(1)

    def instance_rag(self, collection_name='plot'):
        """
        As the name implies, return a RAG with collection name established.
        Defaults to the 'plot' collection.
        """
        return VectorData(vector_dir=self.vector_dir,
                                     embedding=self.embedding_model,
                                     url=self.server,
                                     debug=self.debug,
                                     collection_name=collection_name)

    def pre_processor(self, query):
        """ lightweight LLM as a summarization pre-processor """
        rag = self.instance_rag()
        docs = rag.retrieve_data(query)
        context = "\n\n".join(doc.page_content for doc in docs if doc.page_content.strip())
        if self.debug:
            console.print(f'[italic dim grey7][DEBUG VECTOR]:\n{context}'
                            '[/italic dim grey7]\n')
        # pylint: disable=no-member # dynamic prompts (see self.__build_prompts)
        prompt_template = ChatPromptTemplate.from_messages([
                    ("system", (self.__get_prompt(f'{self.pre_prompt_file}_system.txt')
                                if self.debug else self.pre_prompt_system)),
                    ("human", (self.__get_prompt(f'{self.pre_prompt_file}_human.txt')
                               if self.debug else self.pre_prompt_human))
                ])
        # pylint: enable=no-member
        return prompt_template.format_messages(context=context, question='')

    def run_async_task(self, question: str):
        """ stream the output """
        prompt = self.pre_processor(question)
        if self.debug:
            console.print(f'[italic dim grey7][DEBUG PRECONDITIONED PROMPT]:\n{prompt}'
                            '[/italic dim grey7]\n')
        context = self.pre_llm.query_llm_oneshot(prompt).content
        if self.debug:
            console.print(f'[italic dim grey7][DEBUG PRECONDITIONED CONTEXT]:\n{context}'
                            '[/italic dim grey7]\n')
        # pylint: disable=no-member # dynamic prompts (see self.__build_prompts)
        prompt_template = ChatPromptTemplate.from_messages([
        ("system", (self.__get_prompt(f'{self.plot_prompt_file}_system.txt')
                    if self.debug else self.plot_prompt_system)),
        ("human", (self.__get_prompt(f'{self.plot_prompt_file}_human.txt')
                   if self.debug else self.plot_prompt_human))
            ])
        # pylint: enable=no-member
        history = self.llm.chat_history[-self.history_matches:]
        if self.debug:
            console.print(f'\n[italic dim grey7][DEBUG CHAT HISTORY]:\n{history}'
                              '[/italic dim grey7]\n')
        prompt = prompt_template.format_messages(history=history,
                                                 context=context,
                                                 question=question)
        # pylint: disable=no-member # dynamic prompts (see self.__build_prompts)
        post_prompt = (self.__get_prompt(f'{self.post_prompt_file}_system.txt')
                                if self.debug else self.post_prompt_system)
        # pylint: enable=no-member
        if self.debug:
            console.print(f'\n[italic dim grey7][DEBUG PROMPT]:\n{prompt}[/italic dim grey7]\n')
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None
        if loop and loop.is_running():
            # If there's already a loop, create a task
            return asyncio.ensure_future(self.llm.query_llm(prompt,
                                                            self.preconditioner,
                                                            self.server,
                                                            post_prompt))
        else:
            # If no loop is running, start one
            asyncio.run(self.llm.query_llm(prompt,
                                           self.preconditioner,
                                           self.server,
                                           post_prompt))

    def run(self):
        """Chat with the LLM using a fancy prompt_toolkit interface until Ctrl+C."""
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
        console.print("💬 Chat started. Press [italic grey100]Esc+Enter[/italic grey100] to send."
                      " [italic grey100]Ctrl+C[/italic grey100] to quit.\n")
        try:
            while True:
                question = session.prompt(">>> ", multiline=True, key_bindings=kb).strip()
                if not question:
                    continue
                self.run_async_task(question)
                if self.llm.chat_history:
                    last_response = self.llm.chat_history[-1:][0]
                    rag = self.instance_rag()
                    rag.store_data(data=last_response)

        except KeyboardInterrupt:
            print("\n👋 Exiting chat.")
            sys.exit()

def verify_args(args):
    """ verify arguments are correct """
    # The issue added to the feature tracker: nothing to verify yet
    return args

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
    parser.add_argument('llm', default='',
                         help='Your heavy LLM Model ~27B to whatever you can afford')
    parser.add_argument('--pre-llm', metavar='', nargs='?', default='gemma-3-1b-it-Q4_K_M',
                        type=str, help='1B-2B LLM model for preprocessor work '
                        '(default: %(default)s)')
    parser.add_argument('--embedding_llm', metavar='', nargs='?', default='nomic-embed-text',
                        type=str, help='LM embedding model (default: %(default)s)')
    parser.add_argument('--history_dir', metavar='', nargs='?',
                         default=os.path.join('.', 'vector_data'), type=str,
                         help='a writable path for RAG (default: %(default)s)')
    parser.add_argument('--history-matches', metavar='', nargs='?', default=3, type=int,
                        help='Number of results to pull from each RAG (default: %(default)s)')
    parser.add_argument('--server', metavar='', nargs='?', default='localhost:11434', type=str,
                        help='ollama server address (default: %(default)s)')
    parser.add_argument('--import-pdf', metavar='', nargs='?', type=str,
                         help='Path to pdf to pre-populate main RAG')
    parser.add_argument('--debug', action='store_true', default=False,
                        help='Print preconditioning message, prompt, etc')

    return verify_args(parser.parse_args(argv))

if __name__ == '__main__':
    args = parse_args(sys.argv[1:])
    chat = Chat(**vars(args))
    chat.run()
