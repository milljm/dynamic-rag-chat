
"""
RAGTagManager aims at handling the RAGs and the Collection(s) process (tagging)
"""
import re
import logging
from collections import namedtuple
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from OllamaModel import OllamaModel
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

    def update_rag(self, base_url, model, prompt_template, debug=False)->str:
        """ regular expression through message and attempt to create key:value tuples """
        pre_llm = OllamaModel(base_url)
        results = pre_llm.llm_query(model, prompt_template).content
        rag_tags = self.get_tags(results, debug=debug)
        rag = RAG(self.console, **self.kwargs)

        # we are most interested in names
        collections = [value for key, value in rag_tags if key in ['name', 'npc']]
        for collection in collections:
            if self.debug:
                self.console.print(f'PRIORITY TAG:\n{collection}\n\n', style='color(233)')
            for tag in rag_tags:
                k, v = tag
                rag.store_data(f'{k}:{v}', collection=collection)
            if self.debug:
                self.console.print(f'RAG/Tag Results:\n{rag_tags}', style='color(233)')
        # Handle them in the normal way as well
        for tag in rag_tags:
            k, v = tag
            rag.store_data(v, collection=k)
        return results

    def get_tags(self, content, debug=False) -> list:
        """Convert content into tags to be used for collection identification."""
        rag_tags = []
        tagging = namedtuple('RAGTAG', ('tag', 'content'))

        # Use the precompiled regex pattern
        matches = self.tag_pattern.findall(content)
        if debug:
            self.console.print(f'RAG/Tag MATCHES:\n{matches}\n', style='color(233)')
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

    def _get_embeddings(self, collection):
        collection = self._normalize_collection_name(collection)
        embeddings = OllamaEmbeddings(base_url=self.host, model=self.embeddings)
        chroma_db = Chroma(persist_directory=self.vector_dir,
                            embedding_function=embeddings,
                            collection_name=collection)
        return chroma_db

    def retrieve_data(self, query, collection, matches=5):
        """
        Return vector data as a list. Syntax:
            retrieve_data(query=str, collection=str, matches=int)->list
                query:      your query
                collection: a collection to pull from
                matches:    matches to return
        """
        chroma = self._get_embeddings(collection)
        results = []
        results: list[Document] = chroma.similarity_search(query, matches)
        if self.debug:
            self.console.print(f'CHUNKS RETRIEVED:\n{collection}:{results}\n\n',
                                style='color(233)')
        return results

    def store_data(self, data, collection='ai_response', chunk_size=300, chunk_overlap=150):
        """ store data into the RAG """
        splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size,
                                                  chunk_overlap=chunk_overlap)
        docs = splitter.create_documents([data])
        chroma = self._get_embeddings(collection)
        chroma.add_documents(docs)
        if self.debug:
            self.console.print(f'CHUNKS STORED: {len(docs)}', style='color(233)')

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
