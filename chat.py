#!/usr/bin/env python3
""" simple chat with rag """
import os
import sys
import asyncio
import argparse
from langchain_chroma import Chroma
from langchain_ollama import ChatOllama
from langchain_ollama import OllamaEmbeddings
from langchain.schema import Document
from langchain.schema import AIMessage
from langchain.callbacks.base import AsyncCallbackHandler
from langchain.prompts import ChatPromptTemplate

class StreamingHandler(AsyncCallbackHandler):
    async def on_llm_new_token(self, token: str, **kwargs):
        print(token, end="", flush=True)

class VectorData():
    """ Responsible for dealing with vector database operations """
    def __init__(self, vector_dir='', embedding='', url='http://localhost:11434'):
        self.vector_dir = vector_dir
        self.embedding = embedding
        self.url =url

    def __get_embeddings(self):
        embeddings = OllamaEmbeddings(base_url=self.url, model=self.embedding)
        chroma_db = Chroma(persist_directory=self.vector_dir,
                           embedding_function=embeddings)
        return chroma_db

    def store_data(self, data, chunk_size=1000, overlap=200):
        """
        Stores chunks into vector data base (1000 chunk size, 200 overlap)
        Syntax:
            store_data(data=str, chunk_size=int, overlap=int)
        """
        chunks_dict = []
        chunks_str =[]
        for i in range(0, len(data) - chunk_size + 1, chunk_size - overlap):
            chunks_dict.append({'text' : data[i:i + chunk_size]})
            chunks_str.append(data[i:i + chunk_size])

        documents = [Document(page_content=doc, metadata=meta) for doc, meta in zip(chunks_str,
                                                                                    chunks_dict)]
        chroma = self.__get_embeddings()
        chroma.add_documents(documents)

    def retrieve_data(self, query, k=5):
        """
        Return vector data as a list. Syntax:
            retrieve_data(query=str, k=int)->list
              query: your question
              k:     matches to return
        """
        chroma = self.__get_embeddings()
        results: list[Document] = chroma.similarity_search(query, k)
        for doc in results:
            yield doc.page_content

class OllamModel():
    """ Responsible for handling calls to the Ollama LLM """
    def __init__(self, model, url='http://localhost:11434'):
        self.model = model
        self.url = url
        self.chat_history = []

    def query_llm_oneshot(self, query):
        """ do not use streaming (used to precondition the context) """
        llm = ChatOllama(model=self.model,
                              temperature=0.5,
                              base_url=self.url,
                              streaming=False)
        return llm.invoke(query)

    async def query_llm(self, query):
        """ query the llm with a message """
        handler = StreamingHandler()
        llm = ChatOllama(model=self.model,
                              temperature=1,
                              base_url=self.url,
                              streaming=True,
                              callbacks=[handler])
        print('AI:\n')
        response = await llm.ainvoke(query)
        self.chat_history.append(AIMessage(content=response.content))

class Chat():
    """
    Entry point. requires:
    chat = Chat({'vector_dir':      '/some/path',
                 'llm_model':       'model',
                 'embedding_model': 'model',
                 'question':        '',
                 'history_matches': 'int'})
    (chat = Chat(**kwargs))
    """
    def __init__(self, **kwargs):
        try:
            self.vector_dir = kwargs['history_dir']
            self.llm_model = kwargs['llm']
            self.embedding_model = kwargs['embedding_llm']
            self.question = kwargs['question']
            self.preconditioner = kwargs.get('preconditioner', None)
            self.history_matches = kwargs['history_matches']
            self.server = f'http://{kwargs["server"]}'
        except KeyError as e:
            print(f'Incorrectly supplied arguments. See --help: {e}')
            sys.exit(1)
        self.llm = OllamModel(self.llm_model, url=self.server)
        self.vector_data = VectorData(vector_dir=self.vector_dir,
                                      embedding=self.embedding_model,
                                      url=self.server)
        if self.preconditioner is not None:
            self.pre_llm = OllamModel(model=self.preconditioner, url=self.server)

    def pre_condition(self, query):
        """ Run RAG through preconditioning """
        context = ''
        for match in range(self.history_matches):
            context += '\n\n'.join(self.vector_data.retrieve_data(query))
        prompt_template = ChatPromptTemplate.from_messages([
                    ("system", "concisely summarize the following context, removing "
                               "duplicates, and leaving details alone, without leaving comments"),
                    ("human", "concisely summarize the following context:\n{context}\n\n")
                ])
        return prompt_template.format_messages(context=context, question='')

    def run_async_task(self, question: str):
        """ stream the output """
        context = ''
        if self.pre_condition is not None:
            prompt = self.pre_condition(question)
            context = self.pre_llm.query_llm_oneshot(prompt).content
        prompt_template = ChatPromptTemplate.from_messages([
        ("system", "Answer the users question"),
        ("human", "Chat history:\n{history}\n\nContext:\n{context}\n\nQuestion:{question}")
            ])
        history = self.llm.chat_history
        print(f'DBUGE:{history}')

        prompt = prompt_template.format_messages(history=history,
                                                 context=context,
                                                 question=question)
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            # If there's already a loop, create a task
            return asyncio.ensure_future(self.llm.query_llm(prompt))
        else:
            # If no loop is running, start one
            asyncio.run(self.llm.query_llm(prompt))

    def run(self):
        """ chat with the LLM until the user ctrl-c """
        TOKS = '\033[38;5;22m'
        TYPE = '\033[38;5;94m'
        DARK = '\033[38;5;238m'
        RESET = '\033[0m'
        print('ctl-c to exit')
        try:
            while True:
                question = input("\n> ")
                #response = self.pre_llm.query_llm_oneshot(query)
                #g = []
                #for k, v in response.usage_metadata.items():
                #    g.append(f'{TYPE}{k}{RESET}/{TOKS}{v}{RESET}')
                #print(f'\n{DARK}{response.response_metadata["model"]}:{RESET}'
                #      f'\n{response.content}\n')
                #print(' '.join(g))
                self.run_async_task(question)

        except KeyboardInterrupt:
            sys.exit()

def verify_args(args):
    """ verify arguments are correct """
    #if not os.path.exists(args.database) or not os.access(args.database, os.W_OK):
    #    print(f'{args.database} does not exist, or is not read/writable')
    return args

def parse_args(argv):
    """ parse arguments """
    about = """A simple chat tool with RAG support"""
    epilog = f"""
example:
  ./{os.path.basename(__file__)} gemma3-27B nomic-embed-text /Users/you/chat_history 'Hello!'
    """
    parser = argparse.ArgumentParser(description=f'{about}',
                                     epilog=f'{epilog}',
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('llm', help='LLM model')
    parser.add_argument('embedding_llm', help='LLM embedding model')
    parser.add_argument('history_dir', help='/some/writable/path for history')
    parser.add_argument('question', help='Your query/question/instruction (have fun!)')
    parser.add_argument('--preconditioner', metavar='', nargs='?', help='Optional LLM model to '
                        'refine context from RAG)')
    parser.add_argument('--history-matches', metavar='', nargs='?', default=5, type=int,
                        help='Number of results to pull from RAG as context (default: 5)')
    parser.add_argument('--server', metavar='',nargs='?', default='localhost:11434',
                        help='ollama server address (default: localhost:11434)')
    return verify_args(parser.parse_args(argv))

if __name__ == '__main__':
    args = parse_args(sys.argv[1:])
    chat = Chat(**vars(args))
    chat.run()

    #print(args)
    #vector_data = VectorData(args.history, args.embedding_llm, args.server)
    #for i in vector_data.retrieve_data(args.question, args.history_matches):
    #    print(i)
    #vector_data.store_data(args.question)
