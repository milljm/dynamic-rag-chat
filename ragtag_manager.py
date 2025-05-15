"""
RAGTagManager aims at handling the RAGs and the Collection(s) process (tagging)
"""
import sys
import re
import logging
import json
from typing import NamedTuple
import pypdf # for error handling of PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
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
    def __init__(self, console, common, **kwargs):
        self.console = console
        self.common = common
        self.kwargs = kwargs
        self.debug = kwargs['debug']
        # store all reg in chat_utils...
        self.meta_data = common.meta_data
        self.meta_iter = common.meta_iter
        self.json_style = common.json_style

    def update_rag(self, response, collection: str='ai_documents', debug=False)->None:
        """ regular expression through message and attempt to create key:value tuples """
        list_rag_tags = self.get_tags(response, debug=debug)
        if debug:
            self.console.print(f'META TAGS PARSED: {list_rag_tags}',
                               style='color(233)',
                               highlight=False)
        rag = RAG(self.console, self.common, **self.kwargs)
        # New way: Of course its practically built in. Note to self: Never pretend
        # to think you are planting a flag somewhere when it comes to coding.
        rag.store_data(response, tags_metadata=list_rag_tags, collection=collection)

    def parse_tags(self, tag_input: dict | list[tuple[str, str]]) -> list[RAGTag]:
        """Normalize any kind of tag input into RAGTag list."""
        tags = []
        for key, val in dict(tag_input).items():
            if val is None or (isinstance(val, str) and val.lower() in {"null", "none", ""}):
                continue
            if isinstance(val, list):
                val = ",".join(str(v).strip() for v in val if v)
            tags.append(RAGTag(key, str(val).strip()))
        return tags

    def get_tags(self, response: str, debug=False) -> list[RAGTag]:
        """Extract tags from either JSON or meta_tag format in the LLM response."""
        try:
            # Check for JSON-style block
            json_match = self.json_style.search(response)
            if json_match:
                data = json.loads(json_match.group(1))
                return self.parse_tags(data)

            # Fallback to meta_tag format
            meta_matches = self.meta_data.findall(response)
            if meta_matches:
                flat_pairs = []
                for match in meta_matches:
                    flat_pairs.extend(self.meta_iter.findall(match))
                return self.parse_tags(flat_pairs)

            return []

        # pylint: disable=broad-exception-caught  # too many ways for this to go wrong
        except Exception as e:
            if debug:
                print(f'[get_tags error] {e}')
            return []


class RAG():
    """ Responsible for RAG operations """
    def __init__(self, console, common, **kwargs):
        self.console = console
        self.common = common
        self.host = kwargs['host']
        self.embeddings = kwargs['embeddings']
        self.vector_dir = kwargs['vector_dir']
        self.debug = kwargs['debug']

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

    def _get_embeddings(self, collection)->Chroma:
        collection = self._normalize_collection_name(collection)
        embeddings = OllamaEmbeddings(base_url=self.host, model=self.embeddings)
        chroma_db = Chroma(persist_directory=self.vector_dir,
                            embedding_function=embeddings,
                            collection_name=collection)
        return chroma_db

    def retrieve_data(self, query: str,
                            collection: str,
                            meta_data: dict = None,
                            matches=5)->list[Document]:
        """
        Return vector data as a list. Syntax:
            retrieve_data(query=str, collection=str, matches=int)->list
                query: your question
                k:     matches to return
        """
        chroma = self._get_embeddings(collection)
        results = []
        results: list[Document] = chroma.similarity_search(query, matches, filter=meta_data)
        if self.debug:
            self.console.print(f'CHUNKS RETRIEVED FROM {collection}:\n'
                               f'maximum retrievable matches:{matches}\n',
                               f'field-filters:{meta_data}:\n'
                               f'Results found:{len(results)}\n'
                               f'RAW RESULTS:{results}\n\n',
                                style='color(233)',
                                highlight=False)
        return results

    def store_data(self, data,
                         tags_metadata: list[RAGTag] = None,
                         collection: str = 'ai_documents',
                         chunk_size=150, chunk_overlap=75)->None:
        """ store data into the RAG """
        if tags_metadata is None:
            tags_metadata = {}
        meta_dict = dict(tags_metadata)
        doc = Document(page_content=data, metadata=meta_dict)
        splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size,
                                                  chunk_overlap=chunk_overlap)
        docs = splitter.split_documents([doc])
        chroma = self._get_embeddings(collection)
        chroma.add_documents(docs)
        if self.debug:
            self.console.print(f'CHUNKS STORED TO COLLECTION:{collection}: {len(docs)} '
                               f'/ chunk size {chunk_size} '
                               f'/ olverlap {chunk_overlap} '
                               f'meta_data: {meta_dict}'
                               f'\nDOCS:{docs}\n',
                               style='color(233)',
                               highlight=False)

    def extract_text_from_pdf(self, pdf_path)->None:
        """ extract text from PDFs """
        loader = PyPDFLoader(pdf_path)
        pages = []
        try:
            for page in loader.lazy_load():
                pages.append(page)
            page_texts = list(map(lambda doc: doc.page_content, pages))
            for page_text in page_texts:
                if page_text:
                    self.store_data(page_text, collection = 'ai_documents')

        except pypdf.errors.PdfStreamError as e:
            print(f'Error loading PDF:\n\n\t{e}\n\nIs this a valid PDF?')
            sys.exit(1)
