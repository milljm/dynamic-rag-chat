"""
rag_manager aims at handling RAG operations
"""
import os
import re
import logging
from langchain_community.retrievers import BM25Retriever
from langchain import retrievers  # for Type Hinting
from langchain.retrievers import EnsembleRetriever
from langchain.retrievers import ParentDocumentRetriever
from langchain.storage import LocalFileStore
from langchain.storage._lc_store import create_kv_docstore
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_openai import OpenAIEmbeddings
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from .chat_utils import CommonUtils, ChatOptions, RAGTag  # for Type Hinting
# Silence initial RAG database being empty
logging.getLogger("chromadb").setLevel(logging.ERROR)

class RAG():
    """
    ### RAG

    Responsible for RAG operations.

    *Class init args:*
        .. code-block:: python
            console: Rich.console  # Top level Rich console object
            common: CommonUtils    # Needed for all the metadata tagging/regex involved
            args: ChatOptions      # Arguments in the form of ChatOption dataclass

    *Usage:*
        - instance RAG:
            .. code-block:: python
                rag = RAG(console, common, args)

        - retrieve data from RAG:
            .. code-block:: python
                rag.retrieve(query, collection, metadatas={})

        - store data to RAG:
            .. code-block:: python
                rag.store(query, collection, metadatas={})

    - *query:* Required. string containing user's question.
    - *collection:* Required. RAG collection to pull/write from/to.
    - *metadatas:* Optional. Use field-filtering matching if set.
    """
    def __init__(self, console, common: CommonUtils, args: ChatOptions):
        self.console = console
        self.common = common
        self.opts = args
        self.retriever_id = 0

        # hack for now, until Ollama supports v1/embeddings?
        if self.opts.emb_host.find(':11434') != -1:
            # https://someaddress.com/v1 --> someaddress:11434
            ugh = re.findall(r'([\w+\.-]+:[0-9]+)', self.opts.emb_host)[0]
            self.embeddings = OllamaEmbeddings(base_url=ugh, model=self.opts.embeddings)
        else:
            self.embeddings = OpenAIEmbeddings(base_url=self.opts.emb_host,
                                               model=self.opts.embeddings,
                                               api_key=self.opts.api_key)

        self.parent_splitter = RecursiveCharacterTextSplitter(chunk_size=2000,
                                                              chunk_overlap=1000,
                                                              separators=['\n\n'])
        self.child_splitter = RecursiveCharacterTextSplitter(chunk_size=500,
                                                             chunk_overlap=250,
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

    def _parent_retriever(self, collection: str)->ParentDocumentRetriever:
        """ Return ParentDocumentRetriever for provided collection """
        collection = self._normalize_collection_name(collection)
        fs = LocalFileStore(os.path.join(self.opts.vector_dir, collection))
        store = create_kv_docstore(fs)
        return ParentDocumentRetriever(
                    vectorstore=self._vector_store(collection),
                    docstore=store,
                    child_splitter=self.child_splitter,
                    parent_splitter=self.parent_splitter)

    def _vector_store(self, collection: str)->Chroma:
        """ Return our Chroma Collections Database """
        collection = self._normalize_collection_name(collection)
        chroma = Chroma(persist_directory=self.opts.vector_dir,
                        embedding_function=self.embeddings,
                        collection_name=collection)
        return chroma

    def _ensemble_retriever(self, retriever: list[object],
                                  weights: list[float],
                                  query: str)->list[Document]:
        """
        ### Ensemble Retriever

        Combined search results with weights for each retriever object.

        *Key init args:*
            .. code-block:: python
                retrievers: list[retrievers]  # a list of LangChain retrievers
                weights: list[float]          # weights for each retriever supplied
                query: str                    # the users query
        *Returns:*
            .. code-block:: python
                return list[Document]
        """
        ensemble_retriever = EnsembleRetriever(retrievers=retriever, weights=weights)
        return ensemble_retriever.invoke(query)

    def _bm25_retriever(self, documents: list[Document])->BM25Retriever:
        """
        ### BM25 Retriever

        Returns a BM25 retriever object for use with ensemble_retriever.

        *Key init args:*
            .. code-block:: python
                documents: list[Document]  # list of Document objects
        *Returns:*
            .. code-block:: python
                return retrievers
        """
        _retriever = BM25Retriever.from_documents(documents)
        _retriever.k = self.opts.matches
        return _retriever

    def _chroma_retriever(self, collection: str, kwargs)->retrievers:
        """
        ### Chroma Retriever

        Returns a Chroma retriever object for use with ensemble_retriever.

        *Key init args:*
            .. code-block:: python
                documents: list[Document]  # list of Document objects
                id: int                    # An id to tag this retriever with
        *Returns:*
            .. code-block:: python
                return retrievers
        """
        chroma = self._vector_store(collection)
        _retriever = chroma.as_retriever(search_type='similarity',
                                         search_kwargs=kwargs)
        return _retriever

    def retrieve(self, query: str, collection: str, metadatas: dict=None)->list[Document]:
        """
        ### Retrieve Documents

        Return a list of LangChain Document objects from the RAG `collection` based on
        `query`. A combination of searches will ensue: field-filtering, similarity, contextual
        BM25 (in that order), returning the ensemble of all results (duplicates including).
        It will be necessary to perform any weight/deduplication processes on these results
        afterwards.

        *key init args:*
            .. code-block:: python
                query: str       # the users query
                collection: str  # the RAG collection to draw from
                metadatas: dict  # if set, will perform field-filtering match
        """
        if not metadatas:
            metadatas = None

        # Chroma field-filtering retriever
        ff_retriever = self._chroma_retriever(collection,
                                              {'k': self.opts.matches, 'filter': metadatas})

        # Chroma simuliarity retriever
        ss_retriever = self._chroma_retriever(collection,
                                              {'k': self.opts.matches})

        documents = self._ensemble_retriever([ff_retriever, ss_retriever],
                                             [0.5, 0.5],
                                             query)

        # BM25 results
        if not documents:
            return documents
        bm25_retriever = self._bm25_retriever(documents)
        documents = bm25_retriever.invoke(query)
        # grab parent document
        documents.extend(self._parent_retriever(collection).invoke(query))
        return documents

    def store_data(self, data,
                         tags_metadata: list[RAGTag[str,str|list]] = None,
                         collection: str = '')->None:
        """ store data into the RAG with optional metadata tagged with it """
        if not collection:
            collection = self.common.attributes.collections['ai']
        # Remove meta_data tagging information from data
        data = self.common.sanatize_response(data, strip=True)
        if tags_metadata is None:
            tags_metadata = {}
        meta_dict = dict(tags_metadata)
        meta_dict = self.common.normalize_metadata_for_rag(meta_dict)
        if self.opts.debug:
            self.console.print(f'STORE DATA >>>{data}<<<\n\nTAGS:\n{meta_dict}',
                               style=f'color({self.opts.color})',
                               highlight=False)
        else:
            with open(os.path.join(self.opts.vector_dir, 'debug.log'),
                      'w', encoding='utf-8') as f:
                f.write(f'STORE DATA: TAGS:{meta_dict}, {data}')
        doc = Document(data, metadata=meta_dict)
        retriever = self._parent_retriever(collection)
        try:
            retriever.add_documents([doc])
        # pylint: disable=bare-except  # Sometimes this can fail for a variety of reasons
        except:
            print(f'\nERROR STORING DATA:\n{data}\n\nTAGS:\n{meta_dict}\n\n'
                  'Check for malformed TAGS (no list items is usually the culprit)')
        # pylint: enable=bare-except