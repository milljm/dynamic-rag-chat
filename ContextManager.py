"""
ContextManager aims at handeling everything relating to the context
being supplied to the LLM. It utilizing several methods:

    Emoji removal
    Fuzzy match sentences
    list[] -> set() removes any matches from the RAG.
"""
import re
import os
import sys
import hashlib
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

    def pre_processor(self, query,
                            pre_behavior='',
                            collection='default')->tuple[str,list[namedtuple]]:
        """
        lightweight LLM as a summarization/tagging pre-processor
        """
        prompts = self.prompts
        pre_llm = OllamaModel(self.host)
        # pylint: disable=no-member # dynamic prompts (see self.__build_prompts)
        if pre_behavior == 'tag':
            system_prompt = (prompts.get_prompt(f'{prompts.tag_prompt_file}_system.txt')
                             if self.debug else prompts.tag_prompt_system)
            human_prompt = (prompts.get_prompt(f'{prompts.tag_prompt_file}_human.txt')
                            if self.debug else prompts.tag_prompt_human)
            docs = self.rag.retrieve_data(query, collection, matches=self.matches)
            # short circuit, as there is nothing to do
            if not docs:
                return ('', [])
            context = self.stringigy_lists([doc.page_content for
                                            doc in docs if doc.page_content.strip()])
            # context = self.stringigy_lists(query)
        else:
            system_prompt = (prompts.get_prompt(f'{prompts.pre_prompt_file}_system.txt')
                             if self.debug else prompts.pre_prompt_system)
            human_prompt = (prompts.get_prompt(f'{prompts.pre_prompt_file}_human.txt')
                            if self.debug else prompts.pre_prompt_human)
            context = self.stringigy_lists(query)

        prompt_template = ChatPromptTemplate.from_messages([
                    ("system", system_prompt),
                    ("human", human_prompt)
                ])
        prompt = prompt_template.format_messages(context=context)
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

    def post_processing(self, response)->None:
        """
        Send LLM's response off for post processing as a thread. This allows the
        user to begin formulating a response (and even sending a new message
        before this has completed). This step will begin to shine as more and
        more vectors are established, allowing the pre-processing step to draw
        in more nuanced context.
        """
        prompts = self.prompts
        # pylint: disable=no-member # dynamic prompts (see self.__build_prompts)
        post_prompt = (prompts.get_prompt(f'{prompts.tag_prompt_file}_system.txt')
                        if self.debug else prompts.tag_prompt_system)
        prompt_template = ChatPromptTemplate.from_messages([
                    ("system", post_prompt),
                    ("human", '{context}')
                ])
        prompt = prompt_template.format_messages(context=response, question='')
        if self.debug:
            self.console.print(f'POST PROCESS PROMPT TEMPLATE:\n{prompt}\n\n',
                               style='color(233)', highlight=False)
        # pylint: enable=no-member
        threading.Thread(target=self.rag_tagger.update_rag,
                         args=(self.host, self.preconditioner, prompt),
                         kwargs={'debug': self.debug}).start()

    def gather_context(self, query: str, collection: str)->list[Document]:
        """ perform gathering routines based on incoming collection """
        documents = []
        (_, tags) = self.pre_processor(query,
                                       pre_behavior='tag')
        # Search for 'important' tags, and query those collections for 'data'
        # This is going to be highly relevant data, so expand the matches x2
        priorities = [value for key, value in tags if key in ['name', 'npc', 'item']]
        for priority in priorities:
            for rag_tuple in tags:
                # this will be the most pertinent information, so grab a ton of data
                documents.extend(self.rag.retrieve_data(query,
                                                        rag_tuple.content,
                                                        matches=self.matches*4))
                # relax the matches for trival rag_tuple.tag. May bring in some other nuance
                documents.extend(self.rag.retrieve_data(rag_tuple.content,
                                                        priority,
                                                        matches=(max(1,int(self.matches/4)))))
                # Search once more in the default RAG for 'john' regardless of 'data'
                # (brings in older chat history about 'john') (relax matches)
                documents.extend(self.rag.retrieve_data(rag_tuple.content,
                                                        collection,
                                                        matches=max(1,int(self.matches/2))))
        return documents

    def handle_context(self, data_set: list,
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
                data = self.stringigy_lists(data)
                storage = []
                for collection in collection_list:
                    # General RAG retreival on default collections
                    storage.extend(self.rag.retrieve_data(data,
                                                          collection,
                                                          matches=self.matches))
                    # Extensive RAG retreival
                    storage.extend(self.gather_context(data, collection))

                    # Record pre-token counts
                    pages = list(map(lambda doc: doc.page_content, storage))
                    for page in pages:
                        pre_tokens += self.token_retreiver(page)

                    # Put it together, removing duplicates, then back to strings again
                    documents[collection] = (list(set(pages)))

                    # Record post-token counts
                    for page in documents[collection]:
                        post_tokens += self.token_retreiver(page)

            # Store the users query to their RAG, now that we are done pre-processing
            # (so as not to bring back identical information in their query)
            # A little unorthodox, but the first item in the list is ther user's query
            self.rag.store_data(self.stringigy_lists(data_set[0]),
                                collection='user_documents',
                                chunk_size=100,
                                chunk_overlap=50)
            return (documents, pre_tokens, post_tokens)

        # Store Context to AI RAG
        # Aggresively fragment the response from heavy-weight LLM responses
        # We can afford to do this due to the dynamic RAG/Tagging in post_processing
        self.rag.store_data(self.stringigy_lists(data_set[0]),
                            collection='ai_documents',
                            chunk_size=150,
                            chunk_overlap=50)
        # dynamic RAG creation starts here (non-blocking)
        return self.post_processing(data_set[0])
