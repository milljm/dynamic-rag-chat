"""
RAGTagManager aims at handling the RAGs and the Collection(s) process (tagging)
"""
import os
import re
import logging
from typing import NamedTuple
from langchain.retrievers import ParentDocumentRetriever
from langchain.storage import LocalFileStore
from langchain.storage._lc_store import create_kv_docstore
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_openai import OpenAIEmbeddings
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from .chat_utils import CommonUtils, ChatOptions  # for Type Hinting
# Silence initial RAG database being empty
logging.getLogger("chromadb").setLevel(logging.ERROR)

class RAGTag(NamedTuple):
    """
    namedtuple class constructor
      RAGTag(tag: str, content: str)
    """
    tag: str
    content: str

class RAGTagManager():
    """
    Dynamic RAG/Tag System
    capture incomming tags the LLM is producing and convert them into usable
    key:value pairs.
    """
    def __init__(self, console, common: CommonUtils, args: ChatOptions):
        self.console = console
        self.common  = common
        self.debug = args.debug
        self.light_mode = args.light_mode
        self.opts = args

    def update_rag(self, response: str, collection: str='ai_documents')->None:
        """ regular expression through message and attempt to create key:value tuples """
        list_rag_tags: list[RAGTag[str, str]] = self.common.get_tags(response)
        # Update the scene
        self.common.scene_tracker_from_tags(list_rag_tags)
        if self.debug:
            self.console.print(f'META TAGS PARSED: {list_rag_tags}',
                               style=f'color({self.opts.color})',
                               highlight=False)
        else:
            with open(os.path.join(self.opts.vector_dir, 'debug.log'),
                      'w', encoding='utf-8') as f:
                f.write(f'META TAGS PARSED: {list_rag_tags}')
        rag = RAG(self.console, self.common, self.opts)
        # New way: Of course its practically built in. Note to self: Never pretend
        # to think you are planting a flag somewhere when it comes to coding.
        rag.store_data(response, tags_metadata=list_rag_tags, collection=collection)

class RAG():
    """ Responsible for RAG operations """
    def __init__(self, console, common: CommonUtils, args: ChatOptions):
        self.console = console
        self.common = common
        self.opts = args

        # hack for now, until Ollama supports v1/embeddings?
        if self.opts.emb_host.find(':11434') != -1:
            # https://someaddress.com/v1 --> someaddress:11434
            ugh = re.findall(r'([\w+\.-]+:[0-9]+)', self.opts.emb_host)[0]
            self.embeddings = OllamaEmbeddings(base_url=ugh,
                                               model=self.opts.embeddings)
        else:
            self.embeddings = OpenAIEmbeddings(base_url=self.opts.emb_host,
                                               model=self.opts.embeddings,
                                               api_key=self.opts.api_key)

        self.parent_splitter = RecursiveCharacterTextSplitter(chunk_size=2000,
                                                              chunk_overlap=1000,
                                                              separators=['\n\n'])
        self.child_splitter = RecursiveCharacterTextSplitter(chunk_size=100,
                                                             chunk_overlap=50,
                                                             separators=['.'])

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

    def parent_retriever(self, collection: str)->ParentDocumentRetriever:
        """ Return ParentDocumentRetriever for provided collection """
        collection = self._normalize_collection_name(collection)
        fs = LocalFileStore(os.path.join(self.opts.vector_dir, collection))
        store = create_kv_docstore(fs)
        return ParentDocumentRetriever(
                    vectorstore=self.vector_store(collection),
                    docstore=store,
                    child_splitter=self.child_splitter,
                    parent_splitter=self.parent_splitter)

    def vector_store(self, collection: str)->Chroma:
        """ Return our Chroma Collections Database """
        collection = self._normalize_collection_name(collection)
        chroma = Chroma(persist_directory=self.opts.vector_dir,
                        embedding_function=self.embeddings,
                        collection_name=collection)
        return chroma

    def retrieve_data(self, query: str,
                            collection: str,
                            meta_data: dict = None,
                            matches=5)->list[Document]:
        """
        Return matching documents
        """
        parent_retriever = self.parent_retriever(collection)
        vector_store = self.vector_store(collection)

        results = []
        try:
            results: list[Document] = vector_store.similarity_search(query,
                                                                    matches,
                                                                    filter=meta_data)
        # pylint: disable=bare-except  # rare to fail, and trying to capture it
        except:
            self.console.print(f'ERROR RETRIEVING: from {collection},'
                               f'meta: {meta_data}',
                               style=f'color({self.opts.color})')
        # pylint: enable=bare-except

        if results:
            results.extend(parent_retriever.invoke(query))
        if self.opts.debug:
            self.console.print(f'RETRIEVED DOCS: from {collection} '
                               f'meta: {meta_data}\n',
                               f'\n{results}\n\n',
                               style=f'color({self.opts.color})')
        else:
            with open(os.path.join(self.opts.vector_dir, 'debug.log'),
                      'w', encoding='utf-8') as f:
                f.write(f'RETRIEVED FROM {collection}: {meta_data}, {results}')
        return results

    def store_data(self, data,
                         tags_metadata: list[RAGTag] = None,
                         collection: str = 'ai_documents')->None:
        """ store data into the RAG """
        # Remove meta_data tagging information from data
        data = self.common.sanatize_response(data, strip=True)
        if tags_metadata is None:
            tags_metadata = {}
        meta_dict = dict(tags_metadata)
        meta_dict = self.common.normalize_metadata_for_rag(meta_dict)
        if self.opts.debug:
            self.console.print(f'STORE DATA:\n{data}\n\nTAGS:\n{meta_dict}',
                               style=f'color({self.opts.color})',
                               highlight=False)
        else:
            with open(os.path.join(self.opts.vector_dir, 'debug.log'),
                      'w', encoding='utf-8') as f:
                f.write(f'STORE DATA: TAGS:{meta_dict}, {data}')
        doc = Document(data, metadata=meta_dict)
        retriever = self.parent_retriever(collection)
        try:
            retriever.add_documents([doc])
        # pylint: disable=bare-except  # Sometimes this can fail for a variety of reasons
        except:
            print(f'\nERROR STORING DATA:\n{data}\n\nTAGS:\n{meta_dict}\n\n'
                  'Check for malformed TAGS (no list items is usually the culprit)')
        # pylint: enable=bare-except
