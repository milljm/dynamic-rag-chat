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
import re
import sys
import time
import argparse
import logging
import threading
import hashlib
from collections import namedtuple
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
    def __init__(self):
        self.tag_pattern = re.compile(r'{\s*([a-zA-Z0-9_-]+)\s*:\s*([^\}]+)\s*}')

    def update_rag(self, base_url, model, prompt_template, debug=False)->str:
        """ regular expression through message and attempt to create key:value tuples """
        pre_llm = OllamModel(base_url=base_url)
        results = pre_llm.llm_query(model, prompt_template).content
        rag_tags = self.get_tags(results, debug=debug)

        if debug:
            console.print(f'RAG/Tag Results:\n{results}', style='color(233)')
        # debuging the output of rag/tagging for now
        with open('testing_output.txt', 'w', encoding='utf-8') as f:
            f.write(f'model:\n{model}\n\n'
                    f'prompt:\n{prompt_template}\n\n'
                    f'results:\n{results}\n\n'
                    f'rag_tags:\n{rag_tags}')
        return results

    def get_tags(self, content, debug=False) -> list:
        """Convert content into tags to be used for collection identification."""
        rag_tags = []
        tagging = namedtuple('RAGTAG', ('tag', 'content'))

        # Use the precompiled regex pattern
        matches = self.tag_pattern.findall(content)
        if debug:
            console.print(f'DEBUG RAG/TAG MATCHES:\n{matches}\n', style='color(233)')
        # Add matches to rag_tags
        for match in matches:
            rag_tags.append(tagging(match[0], match[1]))

        return rag_tags

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

    @staticmethod
    def _normalize_collection_name(name: str,
                                   min_length: int = 3,
                                   max_length: int = 63,
                                   pad_char: str = 'x') -> str:
        """ padd/sanatize the could-be-invalid collection names """
        # Replace all invalid characters with dashes
        name = re.sub(r'[^a-zA-Z0-9_-]', '-', name)

        # Remove leading/trailing non-alphanumerics to meet start/end rule
        name = re.sub(r'^[^a-zA-Z0-9]+', '', name)
        name = re.sub(r'[^a-zA-Z0-9]+$', '', name)

        # Replace multiple dashes/underscores if needed (optional cleanup)
        name = re.sub(r'[-_]{2,}', '-', name)

        # Avoid names that look like IP addresses
        if re.fullmatch(r'\d{1,3}(\.\d{1,3}){3}', name):
            name = f"col-{name.replace('.', '-')}"

        # Enforce length limits
        if len(name) < min_length:
            name = name.ljust(min_length, pad_char)
        elif len(name) > max_length:
            name = name[:max_length]

        return name

    def _get_embeddings(self, collection):
        collection = self._normalize_collection_name(collection)
        embeddings = OllamaEmbeddings(base_url=self.base_url, model=self.embeddings)
        chroma_db = Chroma(persist_directory=self.vector_dir,
                            embedding_function=embeddings,
                            collection_name=collection)
        return chroma_db

    def retrieve_data(self, query, collection, matches=5):
        """
        Return vector data as a list. Syntax:
            retrieve_data(query=str, collection=str, matches=int)->list
                query: your question
                k:     matches to return
        """
        chroma = self._get_embeddings(collection)
        results = []
        results: list[Document] = chroma.similarity_search(query, matches)
        return results

    def store_data(self, data, collection='ai_response', chunk_size=300, chunk_overlap=150):
        """ store data into the RAG """
        splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size,
                                                  chunk_overlap=chunk_overlap)
        docs = splitter.create_documents([data])
        chroma = self._get_embeddings(collection)
        chroma.add_documents(docs)
        if self.debug:
            console.print(f'CHUNKS STORED: {len(docs)}', style='color(233)')

    def extract_text_from_pdf(self, pdf_path):
        """ extract text from PDFs """
        loader = PyPDFLoader(pdf_path)
        pages = []
        for page in loader.lazy_load():
            pages.append(page)
        page_texts = list(map(lambda doc: doc.page_content, pages))
        for page_text in page_texts:
            if page_text:
                self.store_data(page_text, 'ai_response')

class Chat():
    """ Begin initializing variables classes. Call .chat() to begin """
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
        self.heat_map = {0: 123, 300: 51, 500: 46, 700: 42, 2000: 82, 3000: 154,
         3500: 178, 4000: 208, 4200: 166, 4500: 203, 4700: 197, 5000: 196}
        self.chat_history_md = ''
        self.chat_history_session = []
        self.rag = RAG(self.host, self.embedding_model, self.vector_dir)
        self.console = Console(highlight=True)
        self.rag_token_reduction = 0
        self.__build_prompts()

    def __build_prompts(self):
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
            console.print(f'RAW PROMPT (llm.stream()):\n{messages}\n\n', style='color(233)')
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
        footer.append('s | Token Reduction :', style='color(233)')
        footer.append(f' {self.rag_token_reduction}', style='color(27)')
        footer.append(' | Prompt Tokens:', style='color(233)')
        footer.append(f' {prompt_tokens}', style=f'color({heat})')
        footer.append(' | Tokens:', style='color(233)')
        footer.append(f' {token_count}', style=f'color({produced})')

        # Render the chat content as Markdown (no panel, just the content)
        chat_content = Markdown(full_md)

        # Return everything as a Group (no borders, just the content)
        return Group(chat_content, footer)

    def pre_processor(self, query, collection='default')->tuple:
        """
        lightweight LLM as a summarization/tagging pre-processor
        """
        pre_llm = OllamModel(base_url=self.host)
        rag_tagger = DynamicRAGManager()
        docs = self.rag.retrieve_data(query, collection, matches=self.history_matches)
        context = "\n\n".join(doc.page_content for doc in docs if doc.page_content.strip())
        if self.debug:
            console.print(f'DEBUG VECTOR:\n{context}', style='color(233)')
        # pylint: disable=no-member # dynamic prompts (see self.__build_prompts)
        prompt_template = ChatPromptTemplate.from_messages([
                    ("system", (self.__get_prompt(f'{self.pre_prompt_file}_system.txt')
                                if self.debug else self.pre_prompt_system)),
                    ("human", (self.__get_prompt(f'{self.pre_prompt_file}_human.txt')
                               if self.debug else self.pre_prompt_human))
                ])
        # pylint: enable=no-member
        prompt = prompt_template.format_messages(context=context, question='')
        content = pre_llm.llm_query(self.preconditioner, prompt).content
        tags = rag_tagger.get_tags(content, debug=self.debug)
        return (content, tags)

    def post_processing(self, response):
        """
        Send LLM's resoonse off for post processing as a thread. This allows the
        user to begin formulating a response (and even sending a new message
        before this has completed). This step will begin to shine as more and
        more vectors are established, allowing the pre-processing step to draw
        in more nuanced context.
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
        RAGTag = DynamicRAGManager()
        threading.Thread(target=RAGTag.update_rag,
                         args=(self.host,
                               self.preconditioner,
                               prompt),
                         kwargs={'debug': self.debug}).start()

    @staticmethod
    def token_retreiver(context_list: list)->int:
        """ iterate over list and do a word count (token) """
        _token_cnt = 0
        if isinstance(context_list, list):
            for sentence in context_list:
                _token_cnt += len(sentence.split())
        return _token_cnt

    @staticmethod
    def normalize_for_dedup(text: str) -> str:
        """ remove emojis and other markdown """
        text = re.sub(r'[\U0001F600-\U0001F64F\u2600-\u26FF\u2700-\u27BF]', '', text)
        text = re.sub(r'[^\w\s]', '', text)
        return ' '.join(text.lower().split())

    def deduplicate_texts(self, texts: list[str]) -> list[str]:
        """ attempt to remove fuzzy duplicate sentences """
        seen = set()
        unique = []
        for text in texts:
            norm = self.normalize_for_dedup(text)
            key = hashlib.md5(norm.encode()).hexdigest()
            if key not in seen:
                seen.add(key)
                unique.append(text)
        return unique

    def handle_context(self, data, direction='query')->str:
        """ Method to handle all the lovely context """
        # Retrieve context from AI and User RAG
        if direction == 'query':
            documents = []
            for collection in ['ai_response', 'user_queries']:
                documents.extend(self.rag.retrieve_data(data,
                                                        collection=collection,
                                                        matches=self.history_matches))

            # Retrieve data from fast pre-processor and query the RAG once more
            # This is where things get interesting. Has a knack for bringing
            # in otherwise missed but relevant context at the cost of ~1-2 seconds.
            (pre_query, rag_tags) = self.pre_processor(data, 'ai_response')
            for tag in rag_tags:
                documents.extend(self.rag.retrieve_data(tag.content,
                                                        collection=tag.tag,
                                                        matches=self.history_matches))

            documents.extend(self.rag.retrieve_data(pre_query,
                                                    collection='ai_response',
                                                    matches=self.history_matches))

            # Lambda function to extract page_content from each document, then
            # a set() to remove any duplicates(you'd be surpised how many tokens this saves).
            self.rag_token_reduction = 0
            _pre = self.token_retreiver(list(map(lambda doc: doc.page_content,documents)))
            context = list(set(list(map(lambda doc: doc.page_content, documents))))
            context = self.deduplicate_texts(context)
            _post = self.token_retreiver(context)
            self.rag_token_reduction = (_pre - _post)
            if self.rag_token_reduction and self.debug:
                console.print(f'RAG TOKEN REDUCTION:\n{_pre} --> '
                              f'{_post} = {self.rag_token_reduction}\n\n', style='color(233)')

            # If the context is empty (no documents found), well, wow. Nothing.
            if not context:
                return ''

            # LLMs prefer strings separated by \n\n
            return '\n\n'.join(context)

        # Store Context to AI RAG
        # Aggresively fragment the response from heavy-weight LLM responses
        # We can afford to do this due to the dynamic RAG/Tagging in post_processing
        self.rag.store_data(data, collection='ai_response', chunk_size=150, chunk_overlap=50)
        # dynamic RAG creation starts here (non-blocking)
        self.post_processing(data)

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
                context = self.handle_context(data=user_input)

                # Gather all prompt tokens, to display as statitics
                prompt_tokens = 0
                if self.chat_history_session[-3:]:
                    prompt_tokens = self.token_retreiver(self.chat_history_session[-3:])
                context_tokens = self.token_retreiver(context.split())

                if self.debug:
                    console.print(f'HISTORY:\n{self.chat_history_session[-3:]}\n',
                                  f'HISTORY TOKENS: {prompt_tokens}\n\n',
                                  f'CONTEXT:\n{context}\n',
                                  f'CONTEXT TOKENS: {context_tokens}\n\n',
                                  style='color(233)')

                prompt_tokens += context_tokens
                console.print(f'Process {prompt_tokens} context tokens...', style='dim grey11')

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
                                                    prompt_tokens))

                self.chat_history_session.append(current_response)
                self.handle_context(current_response, direction='store')

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
    parser.add_argument('--pre-llm', metavar='', nargs='?', default='gemma-3-1b-it-Q4_K_M',
                        type=str, help='1B-2B LLM model for preprocessor work '
                        '(default: %(default)s)')
    parser.add_argument('--embedding_llm', metavar='', nargs='?',
                        default='nomic-embed-text-v1.5.Q8_0.gguf:latest',
                        type=str, help='LM embedding model (default: %(default)s)')
    parser.add_argument('--history-dir', metavar='', nargs='?',
                         default=os.path.join(current_dir, 'vector_data'), type=str,
                         help='a writable path for RAG (default: %(default)s)')
    parser.add_argument('--history-matches', metavar='', nargs='?', default=3, type=int,
                        help='Number of results to pull from each RAG (default: %(default)s)')
    parser.add_argument('--server', metavar='', nargs='?', default='172.16.155.4:11434', type=str,
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
