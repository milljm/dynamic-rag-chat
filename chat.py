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
import logging
import threading
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import ChatPromptTemplate
from langchain.schema import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain_ollama import ChatOllama
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from rich.live import Live
from rich.markdown import Markdown
from rich.text import Text
from rich.console import Group
from rich.console import Console
from prompt_toolkit import PromptSession
from prompt_toolkit.key_binding import KeyBindings
current_dir = os.path.dirname(os.path.abspath(__file__))
console = Console(highlight=True)

# Silence initial RAG database being empty
logging.getLogger("chromadb").setLevel(logging.ERROR)

class DynamicRAGManager():
    """
    Dynamic RAG/Tag System

    The idea is to use the responses from the heavy-weight LLM, run it through
    a series of quick summarization queries with a light-weight 1B parameter LLM
    to 'tag' things of interest. The hope is that all this information will
    quickly pool into a tightly aware vector database, the more the heavy weight
    LLM is used. Each 'tag' will spawn a new RAG (collection), that the
    pre-conditioner prompt will quickly retrieve and thus, fill our context with
    **very** relevant data.

    Example: The leight-weight model tagged bob: {bob: is a king}. We are then going
    to create (or append) to a new collection 'bob', with every thing about said
    person. Thus eventually pulling in odd information about 'bob' the pre-processor
    might not have matched.

    {person: is a level 12 ranger...}
    {location: sumerset isles...}
    {weird: weird things mentioned about bob here...}
    """
    def update_rag_async(self, base_url, model, prompt_template, debug=False)->str:
        """ regular expression through message and attempt to create key:value tuples """
        pre_llm = OllamModel(base_url=base_url)
        results = pre_llm.llm_query(model, prompt_template).content
        if debug:
            console.print(f'RAG/Tag Prompt:\n{prompt_template}\n\nRAG/Tag Results:\n{results}')
        with open('testing_output.txt', 'w', encoding='utf-8') as f:
            f.write(f'model:\n{model}\n\n'
                    f'prompt:\n{prompt_template}\n\n'
                    f'results:\n{results}')
        return results

class OllamModel():
    """
    Responsible for dealing directly with LLMs,
    out side of the realm of the Chat class
    """
    def __init__(self, base_url):
        self.base_url = base_url

    def llm_query(self, model, prompt_template, temp=0.3)->object:
        """ query the llm with a message, without streaming """
        llm = ChatOllama(model=model,
                         temperature=temp,
                         base_url=self.base_url,
                         streaming=False)
        response = llm.invoke(prompt_template, stop=["\n\n", "###", "Conclusion"])
        return response

class RAG():
    """ Responsible for RAG operations """
    def __init__(self, base_url, embeddings, vector_dir, debug=False):
        self.base_url = base_url
        self.embeddings = embeddings
        self.vector_dir = vector_dir
        self.debug = debug
        self.chroma = Chroma(persist_directory=self.vector_dir)

    def __get_embeddings(self):
        embeddings = OllamaEmbeddings(base_url=self.base_url, model=self.embeddings)
        chroma_db = Chroma(persist_directory=self.vector_dir,
                            embedding_function=embeddings,
                            collection_name='default_for_now')
        return chroma_db

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

    def store_data(self, data, chunk_size=1000, chunk_overlap=200):
        """ store data into the RAG """
        splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size,
                                                  chunk_overlap=chunk_overlap)
        docs = splitter.create_documents([data])
        chroma = self.__get_embeddings()
        chroma.add_documents(docs)
        if self.debug:
            console.print(f'CHUNKS STORED: {len(docs)}')

    def extract_text_from_pdf(self, pdf_path):
        """ extract text from PDFs """
        loader = PyPDFLoader(pdf_path)
        pages = []
        for page in loader.lazy_load():
            pages.append(page)
        page_texts = list(map(lambda doc: doc.page_content, pages))
        for page_text in page_texts:
            if page_text:
                self.store_data(page_text)

class Chat():
    """ Testing """
    def __init__(self, **kwargs):
        try:
            self.model_name = kwargs['llm']
            self.preconditioner = kwargs['pre_llm']
            self.embedding_model = kwargs['embedding_llm']
            self.vector_dir = kwargs['history_dir']
            self.history_matches = kwargs['history_matches']
            self.host = f'http://{kwargs["server"]}'
            self.debug = kwargs['debug']
        except KeyError as e:
            print(f'Incorrectly supplied arguments. See --help: {e}')
            sys.exit(1)
        self.llm = ChatOllama(host=self.host,
                              model=self.model_name,
                              streaming=True)
        # Color tone the prompt tokens
        self.heat_map = {0: 123, 300: 51, 500: 46, 700: 42, 2000: 82, 3000: 154, 3500: 178, 4000: 208, 4200: 166, 4500: 203, 4700: 197, 5000: 196}
        self.chat_history_md = ''
        self.chat_history_session = []
        self.rag = RAG(self.host, self.embedding_model, self.vector_dir)
        self.console = Console(highlight=True)
        self.__build_prompts()

    def __build_prompts(self):
        """
        A way to manage a growing number of prompt templates dynamic RAG/Tagging
        might introduce...

          {key : value} pairs become self.key : contents-of-file
          filenaming convention: {value}_system.txt / {value}_human.txt
        """
        if self.debug:
            console.print('\n[italic dim grey50]Debug mode enabled. I will re-read the '
                          'prompt files each time[/]')
        prompt_files = {
            'pre_prompt':  'pre_conditioner_prompt',
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
            console.print(f'PROMPTS: {messages}', style='color(233)')
        for chunk in self.llm.stream(messages):
            yield chunk

    # Compose the full chat display with footer (model name, time taken, token count)
    def render_chat(self, current_stream: str = "",
                    time_taken: float = 0,
                    token_count: int = 0,
                    prompt_tokens: int = 0) -> Group:
        """ render and return markdown/syntax """
        # Create the full chat content using Markdown
        full_md = f'{self.chat_history_md}\n\n{current_stream}'

        heat = [v for k,v in self.heat_map.items() if k<=prompt_tokens][-1:][0]
        produced = [v for k,v in self.heat_map.items() 
                    if max(0, 5000-(token_count*8))>=k][-1:][0]
        # Create the footer text with model info, time, and token count
        footer = Text('Model: ', style='color(233)')
        footer.append(f'{self.model_name} ', style='color(202)')
        footer.append('| Time: ', style='color(233)')
        footer.append(f'{time_taken:.2f}', style='color(94)')
        footer.append('s | Prompt Tokens:', style='color(233)')
        footer.append(f' {prompt_tokens}', style=f'color({heat})')
        footer.append('| Tokens:', style='color(233)')
        footer.append(f' {token_count}', style=f'color({produced})')

        # Render the chat content as Markdown (no panel, just the content)
        chat_content = Markdown(full_md)

        # Return everything as a Group (no borders, just the content)
        return Group(chat_content, footer)


    def pre_processor(self, query)->str:
        """
        lightweight LLM as a summarization/tagging pre-processor
        """
        docs = self.rag.retrieve_data(query, self.history_matches)
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
        prompt = prompt_template.format_messages(context=context, question='')
        pre_llm = OllamModel(base_url=self.host)
        return pre_llm.llm_query(self.preconditioner, prompt).content

    def post_processing(self, response):
        """
        Dynamic RAG/Tag generation /Threaded non-blocking
        """
        # pylint: disable=no-member # dynamic prompts (see self.__build_prompts)
        post_prompt = (self.__get_prompt(f'{self.post_prompt_file}_system.txt')
                        if self.debug else self.post_prompt_system)
        prompt_template = ChatPromptTemplate.from_messages([
                    ("system", post_prompt),
                    ("human", '{context}')
                ])
        prompt = prompt_template.format_messages(context=response, question='')
        # pylint: enable=no-member
        update_rags = DynamicRAGManager()
        threading.Thread(target=update_rags.update_rag_async,
                         args=(self.host,
                               self.preconditioner,
                               prompt),
                         kwargs={'debug': self.debug}).start()

    def get_context(self, user_input)->str:
        """ process to handle all the lovely context """
        # Retrieve context from RAG for the current question
        documents = self.rag.retrieve_data(user_input, self.history_matches)

        # Retrieve data from pre-processor and query the RAG once more
        pre_query = self.pre_processor(user_input)
        documents.extend(self.rag.retrieve_data(pre_query, self.history_matches))

        # Lambda function to extract page_content from each document, then
        # a set to remove duplicates (eases up prompt tokens sometimes)
        context = set(list(map(lambda doc: doc.page_content, documents)))

        # If the context is empty (no documents found), provide a fallback message
        if not context:
            context = ""

        # LLMs prefer strings separated by \n\n
        context = '\n\n'.join(context)

        return context

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
                context = self.get_context(user_input)

                # Gather all prompt tokens, to display as statitics
                prompt_tokens = len([*context, *self.chat_history_session[-3:]])
                console.print(f'Process {prompt_tokens} context tokens...', style='dim grey11')
                if self.debug:
                    console.print(f'HISTORY:\n{self.chat_history_session[-3:]}\n\n'
                            f'CONTEXT:\n\n{context}\n\n'
                            f'TOTAL TOKENS: {prompt_tokens}',
                            style='color(233)')

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
                                                    prompt_tokens))

                self.chat_history_session.append(current_response)
                self.rag.store_data(current_response)
                # dynamic RAG creation starts here (non-blocking)
                self.post_processing(current_response)

        except KeyboardInterrupt:
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
    parser.add_argument('--pre-llm', metavar='', nargs='?', default='gemma-3-1b',
                        type=str, help='1B-2B LLM model for preprocessor work '
                        '(default: %(default)s)')
    parser.add_argument('--embedding_llm', metavar='', nargs='?',
                        default='nomic-embed-text-v1.5.f16',
                        type=str, help='LM embedding model (default: %(default)s)')
    parser.add_argument('--history-dir', metavar='', nargs='?',
                         default=os.path.join(current_dir, 'vector_data'), type=str,
                         help='a writable path for RAG (default: %(default)s)')
    parser.add_argument('--history-matches', metavar='', nargs='?', default=3, type=int,
                        help='Number of results to pull from each RAG (default: %(default)s)')
    parser.add_argument('--server', metavar='', nargs='?', default='localhost:11434', type=str,
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
    if args.import_txt:
        doc_path = args.import_txt
        if os.path.exists(doc_path):
            with open(doc_path, 'r', encoding='utf-8') as file:
                document_content = file.read()
                rag = RAG(args.server, args.embedding_llm, args.history_dir, args.debug)
                rag.store_data(document_content)
            print(f"Document loaded from: {doc_path}")
        else:
            print(f"Error: The file at {doc_path} does not exist.")
        sys.exit()
    if args.import_pdf:
        pdf_path = args.import_pdf
        if os.path.exists(pdf_path):
            rag = RAG(args.server, args.embedding_llm, args.history_dir, args.debug)
            rag.extract_text_from_pdf(pdf_path)
        else:
            print(f"Error: The file at {pdf_path} does not exist.")
        sys.exit()
    chat = Chat(**vars(args))
    chat.chat()
