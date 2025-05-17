"""
ContextManager aims at handeling everything relating to the context
being supplied to the LLM. It utilizing several methods:

    Emoji removal
    Fuzzy match sentences
    list[] -> set() removes any matches from the RAG.
"""
import threading
from langchain.schema import Document
from langchain.prompts import ChatPromptTemplate
from .ragtag_manager import RAGTagManager, RAG, RAGTag
from .ollama_model import OllamaModel
from .prompt_manager import PromptManager
from .filter_builder import FilterBuilder

class ContextManager(PromptManager):
    """ A collection of methods aimed at producing/reducing the context """
    def __init__(self, console, common, current_dir, **kwargs):
        super().__init__(console, current_dir)
        self.console = console
        self.common = common
        self.host = kwargs['host']
        self.matches = kwargs['matches']
        self.preconditioner = kwargs['preconditioner']
        self.debug = kwargs['debug']
        self.name = kwargs['name']
        self.chat_sessions = kwargs['chat_sessions']
        self.rag = RAG(console, self.common, **kwargs)
        self.rag_tagger = RAGTagManager(console, self.common, **kwargs)
        self.prompts = PromptManager(self.console,
                                     current_dir,
                                     model=self.preconditioner,
                                     debug=self.debug)
        self.filter_builder = FilterBuilder()
        self.prompts.build_prompts()

    @staticmethod
    def token_retreiver(context: str|list[str])->int:
        """ iterate over string or list of strings and do a word count (token) """
        _token_cnt = 0
        if isinstance(context, list):
            for sentence in context:
                _token_cnt += len(sentence.split())
        else:
            _token_cnt += len(context.split(' '))
        return _token_cnt

    def pre_processor(self, query: str)->tuple[str,list[RAGTag]]:
        """
        lightweight LLM as a tagging pre-processor
        """
        pre_llm = OllamaModel(self.host)
        prompts = self.prompts
        query = self.common.normalize_for_dedup(query)
        # pylint: disable=no-member # dynamic prompts (see self.__build_prompts)
        human_prompt = (prompts.get_prompt(f'{prompts.tag_prompt_file}_human.txt')
                        if self.debug else prompts.tag_prompt_human)
        # pylint: enable=no-member
        prompt_template = ChatPromptTemplate.from_messages([
                    ("human", human_prompt)
                ])
        prompt = prompt_template.format_messages(context=query)
        if self.debug:
            self.console.print(f'PRE-PROCESSOR PROMPT:\n{prompt}\n\n',
                                style='color(233)', highlight=False)
        content = pre_llm.llm_query(self.preconditioner, prompt).content
        if self.debug:
            self.console.print(f'PRE-PROCESSOR RESPONSE:\n{content}\n\n',
                                style='color(233)', highlight=False)

        tags = self.common.get_tags(content, debug=self.debug)
        return (content, tags)

    def post_process(self, response)->None:
        """ Start a thread to process LLMs response """
        threading.Thread(target=self.rag_tagger.update_rag, args=(response,),
                         kwargs={'debug': self.debug},
                         daemon=True).start()

    @staticmethod
    def stagger_history(history_size, max_elements=20):
        """ stagger chat history """
        indices = []
        current = history_size - 1
        step = 1
        while current >= 0 and len(indices) < max_elements:
            indices.append(current)
            current -= step
            step += 1
        return sorted(indices)

    def get_chat_history(self)->list:
        """
        Limit Chat History to 550 * int(chat_sessions) tokens (defaults = 2750 tokens)
        """
        splice_list = self.stagger_history(len(self.common.chat_history_session),
                                           self.chat_sessions)
        spliced = [self.common.chat_history_session[x] for x in splice_list]
        abridged = []
        _tk_cnt = 0
        max_tokens = 550 * max(1, int(self.chat_sessions)) # Defaults = 2750 tokens
        for response in spliced:
            _tk_cnt += self.token_retreiver(response)
            if _tk_cnt > max_tokens:
                if self.debug:
                    self.console.print(f'MAX CHAT TOKENS: {_tk_cnt}',
                                       style='color(233)',
                                       highlight=False)
                return abridged
            abridged.append(response)
        return abridged

    def gather_context(self, query: str,
                             collection: str,
                             tags: list[RAGTag[str,str]])->list[Document]:
        """
        Perform metadata field filtering matching
        """
        filter_dict = self.filter_builder.build(tags)
        # Combined filter retrieval (highly relevant information)
        documents = self.rag.retrieve_data(query,
                                           collection,
                                           meta_data=filter_dict,
                                           matches=self.matches)
        return documents

    def handle_context(self, data_set: list,
                             direction='query')->tuple[dict[str,list], int]:
        """ Method to handle all the lovely context """
        # Retrieve context from AI and User RAG and Chat History
        if direction == 'query':
            pre_tokens = 0
            post_tokens = 0
            collection_list = ['ai_documents', 'user_documents']
            documents = {key: [] for key in collection_list}
            documents['chat_history'] = self.get_chat_history()
            if data_set:
                query = self.common.stringify_lists(data_set[0])
                # Try to tagify the users query
                (_, meta_tags) = self.pre_processor(query)
                if self.debug:
                    self.console.print(f'TAG RETREIVAL:\n{meta_tags}\n\n',
                                       style='color(233)',
                                       highlight=False)
                for collection in collection_list:
                    storage = []
                    # Extensive RAG retreival: field filter dictionary, highly relevant
                    storage.extend(self.gather_context(query,
                                                       collection,
                                                       meta_tags))

                    # General RAG retreival: if Extensive retrieval above is low, figure the
                    # difference of allowed maximum, and use that number for matches without
                    # field filter searches
                    balance = self.matches - len(storage)
                    if self.debug:
                        self.console.print(f'BALANCE: {balance}', style='color(233)')
                    storage.extend(self.rag.retrieve_data(query,
                                                          collection,
                                                          matches=int(max(1, balance))))

                    # Record pre-token counts
                    pages = list(map(lambda doc: doc.page_content, storage))
                    for page in pages:
                        pre_tokens += self.token_retreiver(page)

                    # Remove duplicates RAG matches
                    documents[collection] = (list(set(pages)))

                    # Record post-token counts
                    for page in documents[collection]:
                        post_tokens += self.token_retreiver(page)

                if self.debug:
                    self.console.print(f'CONTEXT RETRIEVAL:\n{documents}\n\n',
                                       style='color(233)',
                                       highlight=False)

            # Store the users query to their RAG, now that we are done pre-processing
            # (so as not to bring back identical information in their query)
            # A little unorthodox, but the first item in the list is ther user's query
            self.rag.store_data(self.common.stringify_lists(data_set[0]),
                                tags_metadata=meta_tags,
                                collection='user_documents')
            # Return data collected
            return (documents, pre_tokens, post_tokens)
        # Store data (non-blocking)
        return self.post_process(data_set[0])
