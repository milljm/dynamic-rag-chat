
"""
RAGTagManager aims at handling the RAGs and the Collection(s) process (tagging)
"""
import sys
import re
import logging
from collections import namedtuple
import pypdf # for error handling of PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
# Silence initial RAG database being empty
logging.getLogger("chromadb").setLevel(logging.ERROR)

class RAGTagManager():
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
    def __init__(self, console, **kwargs):
        self.console = console
        self.kwargs = kwargs
        self.debug = kwargs['debug']
        self.tag_pattern = re.compile(r'{\s*([a-zA-Z0-9_-]+)\s*:\s*([^\}]+)\s*}')
        self.find_all = re.compile(r'LLM-SAVE:(.*)', re.DOTALL)
        self.split_pattern = re.compile(r'\s*[;,]\s*')

    def find_tags(self, response: str)->list[tuple]:
        """ parse LLMs response for tags """
        # Match the LLM-SAVE: prefix followed by key:value pairs
        match = self.find_all.findall(response)
        if not match:
            return []  # Return empty list if pattern is not found
        content = match[0]  # Extract everything after 'LLM-SAVE:'
        # Split the string by semicolons or commas, allowing optional whitespace
        parts = self.split_pattern.split(content)
        result = []
        for part in parts:
            if not part.strip():
                continue  # Skip empty entries
            key_value = part.split(':', 1)  # Split on the first colon only
            if len(key_value) == 2:
                key, value = key_value
                result.append((key.strip(), value.strip()))
        return result

    def update_rag(self, response, collection: str='ai_documents', debug=False)->None:
        """ regular expression through message and attempt to create key:value tuples """
        rag_tags = self.get_tags(response, debug=debug)
        rag = RAG(self.console, **self.kwargs)
        # New way: Of course its practically built in. Note to self: Never pretend
        # to think you are planting a flag somewhere when it comes to coding.
        rag.store_data(response, tags_metadata=rag_tags, collection=collection)

    def get_tags(self, response, debug=False) -> list[namedtuple]:
        """Convert content into tags to be used for collection identification."""
        rag_tags = []
        tagging = namedtuple('RAGTAG', ('tag', 'content'))
        # If the LLM happened to output in actual {key:value} format
        matches = self.tag_pattern.findall(response)

        # Our instructed way laid forth in plot_prompt_system.txt
        matches.extend(self.find_tags(response))

        if debug:
            self.console.print(f'RAG/Tag MATCHES:\n{matches}\n\nresponse:'
                               f'\n{response}[END]\n\n', style='color(233)')
        # Add matches to rag_tags
        for match in matches:
            rag_tags.append(tagging(match[0], match[1]))
        return rag_tags

class RAG():
    """ Responsible for RAG operations """
    def __init__(self, console, **kwargs):
        self.console = console
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
            self.console.print(f'CHUNKS RETRIEVED FROM {collection} with filter:'
                               f'matches: {matches}',
                               f' {meta_data}:\n{results}\n\n',
                                style='color(233)',
                                highlight=False)
        return results

    def store_data(self, data,
                         tags_metadata: list[namedtuple] = None,
                         collection: str = 'ai_documents',
                         chunk_size=200, chunk_overlap=100)->None:
        """ store data into the RAG """
        meta_dict = {}
        if tags_metadata:
            for tag in tags_metadata:
                meta_dict[tag.tag] = tag.content
        doc = Document(page_content=data, metadata=meta_dict)
        splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size,
                                                  chunk_overlap=chunk_overlap)
        docs = splitter.split_documents([doc])
        chroma = self._get_embeddings(collection)
        chroma.add_documents(docs)
        if self.debug:
            self.console.print(f'CHUNKS STORED TO collection:{collection}: {len(docs)} '
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
                    self.store_data(page_text, 'ai_documents')
        except pypdf.errors.PdfStreamError as e:
            print(f'Error loading PDF:\n\n\t{e}\n\nIs this a valid PDF?')
            sys.exit(1)
