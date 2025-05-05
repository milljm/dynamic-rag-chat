"""
ContextManager aims at handeling everything relating to the context
being supplied to the LLM. It utilizing several methods:

    Emoji removal
    Fuzzy match sentences
    list[] -> set() removes any matches from the RAG.
"""
import re
import os
import threading
from collections import namedtuple
from langchain.schema import Document
from langchain.prompts import ChatPromptTemplate
from RAGTagManager import RAGTagManager, RAG
from OllamaModel import OllamaModel
from PromptManager import PromptManager
current_dir = os.path.dirname(os.path.abspath(__file__))
class ContextManager(PromptManager):
    """ A collection of methods aimed at producing/reducing the context """
    def __init__(self, console, **kwargs):
        super().__init__(console)
        self.console = console
        self.prompts = PromptManager(console, debug=self.debug)
        self.rag = RAG(console, **kwargs)
        self.rag_tagger = RAGTagManager(console, **kwargs)
        self.host = kwargs['host']
        self.matches = kwargs['matches']
        self.preconditioner = kwargs['preconditioner']
        self.debug = kwargs['debug']
        self.prompts.build_prompts()

    @staticmethod
    def token_retreiver(context: str)->int:
        """ iterate over list and do a word count (token) """
        _token_cnt = 0
        if isinstance(context, list):
            for sentence in context:
                _token_cnt += len(sentence.split())
        else:
            _token_cnt += len(context.split(' '))
        return _token_cnt

    @staticmethod
    def normalize_for_dedup(text: str) -> str:
        """ remove emojis and other markdown """
        text = re.sub(r'[\U0001F600-\U0001F64F\u2600-\u26FF\u2700-\u27BF]', '', text)
        text = re.sub(r'[^\w\s]', '', text)
        return ' '.join(text.lower().split())

    @staticmethod
    def stringigy_lists(nested_list)->str:
        """ return a flat string """
        def process(item):
            result = []
            if isinstance(item, list):
                for subitem in item:
                    result.extend(process(subitem))
            else:
                result.append(str(item))
            return result
        flat_strings = process(nested_list)
        return '\n\n'.join(flat_strings)

    def pre_processor(self, query)->tuple[str,list[namedtuple]]:
        """
        lightweight LLM as a tagging pre-processor
        """
        prompts = self.prompts
        pre_llm = OllamaModel(self.host)
        # pylint: disable=no-member # dynamic prompts (see self.__build_prompts)
        system_prompt = (prompts.get_prompt(f'{prompts.tag_prompt_file}_system.txt')
                            if self.debug else prompts.tag_prompt_system)
        human_prompt = (prompts.get_prompt(f'{prompts.tag_prompt_file}_human.txt')
                        if self.debug else prompts.tag_prompt_human)

        prompt_template = ChatPromptTemplate.from_messages([
                    ("system", system_prompt),
                    ("human", human_prompt)
                ])
        prompt = prompt_template.format_messages(context=query)
        if self.debug:
            self.console.print(f'PRE-PROCESSOR PROMPT:\n{prompt}\n\n',
                                style='color(233)', highlight=False)
        # pylint: enable=no-member
        content = pre_llm.llm_query(self.preconditioner, prompt).content
        if self.debug:
            self.console.print(f'PRE-PROCESSOR RESPONSE:\n{content}\n\n',
                                style='color(233)', highlight=False)
        tags = self.rag_tagger.get_tags(content, debug=self.debug)
        return (content, tags)

    def post_process(self, response)->None:
        """ Start a thread to process LLMs response """
        threading.Thread(target=self.rag_tagger.update_rag, args=(response,),
                         kwargs={'debug': self.debug},
                         daemon=True).start()

    def gather_context(self, query: str,
                             collection: str,
                             response: str = '')->list[Document]:
        """
        perform gathering routines based on incoming collection
        """
        documents = []
        (_, tags) = self.pre_processor(f'{query}\n\n{response}')
        # Search for 'important' tags, and query those collections for 'data'
        # This is going to be highly relevant data, so expand the matches x4
        for meta_data in tags:
            if self.debug:
                self.console.print(f'GATHER CONTEXT: meta_data:{meta_data}'
                                   f'collection: {collection}',
                                   style='color(233)')
            meta = dict({meta_data.tag:meta_data.content})
            # this will be the most pertinent information, so grab a ton of data
            documents.extend(self.rag.retrieve_data(query,
                                                    collection,
                                                    meta_data=meta,
                                                    matches=max(1, int(self.matches*4))))
            if self.debug:
                self.console.print('DOCUMENT GATHER QUERY:'
                                   f'query: {query}'
                                   f'collection: {collection}'
                                   f'documents: {documents}',
                                   style='color(233)')
            # relax the matches for trival meta_data content. May bring in some other nuance
            documents.extend(self.rag.retrieve_data(meta_data.content,
                                                    collection,
                                                    matches=(max(1, int(self.matches/4)))))
            if self.debug:
                self.console.print('DOCUMENT GATHER META:'
                                   f'meta_data: {meta_data.content}'
                                   f'collection: {collection}'
                                   f'documents: {documents}',
                                   style='color(233)')
        return documents

    def handle_context(self, data_set: list,
                             last_response: str = '',
                             direction='query')->tuple[dict[str,list], int]:
        """ Method to handle all the lovely context """
        # Retrieve context from AI and User RAG
        if direction == 'query':
            pre_tokens = 0
            post_tokens = 0
            collection_list = ['ai_documents', 'user_documents', 'history_documents']
            documents = {key: [] for key in collection_list}
            for data in data_set:
                if not data:
                    continue
                query = self.stringigy_lists(data) # lists -> strings (in case its not)
                storage = []
                for collection in collection_list:
                    # General RAG retreival on default collections
                    storage.extend(self.rag.retrieve_data(query,
                                                          collection,
                                                          matches=max(1, int(self.matches/4))))
                    # Extensive RAG retreival
                    storage.extend(self.gather_context(query,
                                                       collection,
                                                       response=last_response))
                    # Record pre-token counts
                    pages = list(map(lambda doc: doc.page_content, storage))
                    for page in pages:
                        pre_tokens += self.token_retreiver(page)

                    # Put it together, removing duplicates, then back to strings again
                    documents[collection] = (list(set(pages)))

                    # Record post-token counts
                    for page in documents[collection]:
                        post_tokens += self.token_retreiver(page)

                    if self.debug:
                        self.console.print(f'CONTEXT RETRIEVAL:\n{documents}\n\n')
            documents['user_documents'] = []
            documents['history_documents'] = []
            # Store the users query to their RAG, now that we are done pre-processing
            # (so as not to bring back identical information in their query)
            # A little unorthodox, but the first item in the list is ther user's query
            self.rag.store_data(self.stringigy_lists(data_set[0]),
                                collection='user_documents',
                                chunk_size=100,
                                chunk_overlap=50)
            # Return data collected
            return (documents, pre_tokens, post_tokens)
        # Store data (non-blocking)
        #self.rag_tagger.update_rag(str(data_set[0]))
        return self.post_process(data_set[0])
